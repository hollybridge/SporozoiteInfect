"""
Mesh-based Sporozoite Model for High-Resolution Simulation

This module implements sporozoites as deformable polyhedral meshes that can
interact with implicit dermal tissue forces and constraints.
"""

import numpy as np
import vtk
from typing import List, Tuple, Dict
import random
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class DeformableSporozoiteMesh:
    """
    Represents a sporozoite as a deformable polyhedral mesh
    """
    
    def __init__(self, sporozoite_id: int, initial_position: np.ndarray):
        self.id = sporozoite_id
        self.center_position = np.array(initial_position, dtype=float)
        
        # Biological parameters
        self.length = random.uniform(10.0, 15.0)  # micrometers
        self.diameter = random.uniform(0.8, 1.2)  # micrometers
        self.motility = random.uniform(0.6, 1.0)
        self.stiffness = random.uniform(0.3, 0.7)  # deformation resistance
        
        # Create mesh geometry
        self.vertices = None
        self.faces = None
        self.vertex_normals = None
        self.rest_lengths = None  # for spring constraints
        
        # Physical state
        self.viability = 1.0
        
        # Movement parameters
        self.direction = np.random.uniform(0, 2*np.pi)
        self.undulation_phase = random.uniform(0, 2*np.pi)
        self.undulation_frequency = random.uniform(0.5, 1.5)  # Hz
        
        self._create_sporozoite_mesh()
    
    def _create_sporozoite_mesh(self):
        """Create initial curved rod mesh representing sporozoite"""
        # Create a curved cylindrical mesh
        n_segments = 20  # longitudinal segments
        n_radial = 8     # radial segments
        
        vertices = []
        faces = []
        
        # Generate curved spine of the sporozoite
        t_values = np.linspace(0, 1, n_segments)
        spine_points = []
        
        for t in t_values:
            # Curved spine with natural sporozoite curvature
            x = t * self.length - self.length/2
            y = np.sin(t * np.pi) * self.length * 0.1  # natural curve
            z = 0
            spine_points.append([x, y, z])
        
        spine_points = np.array(spine_points)
        
        # Create cross-sections perpendicular to spine
        for i, spine_point in enumerate(spine_points):
            # Calculate local coordinate system
            if i == 0:
                tangent = spine_points[i+1] - spine_points[i]
            elif i == len(spine_points) - 1:
                tangent = spine_points[i] - spine_points[i-1]
            else:
                tangent = spine_points[i+1] - spine_points[i-1]
            
            tangent = tangent / np.linalg.norm(tangent)
            
            # Create perpendicular vectors
            if abs(tangent[2]) < 0.9:
                normal = np.cross(tangent, [0, 0, 1])
            else:
                normal = np.cross(tangent, [1, 0, 0])
            normal = normal / np.linalg.norm(normal)
            binormal = np.cross(tangent, normal)
            
            # Radius varies along length (tapered ends)
            radius_factor = np.sin(np.pi * i / (n_segments - 1))
            radius = self.diameter/2 * radius_factor
            
            # Create circular cross-section
            for j in range(n_radial):
                angle = 2 * np.pi * j / n_radial
                local_point = (normal * np.cos(angle) + binormal * np.sin(angle)) * radius
                vertex = spine_point + local_point + self.center_position
                vertices.append(vertex)
        
        # Create faces (triangulated surface)
        for i in range(n_segments - 1):
            for j in range(n_radial):
                # Current and next radial indices
                curr_ring = i * n_radial
                next_ring = (i + 1) * n_radial
                curr_j = j
                next_j = (j + 1) % n_radial
                
                # Two triangles per quad
                v1 = curr_ring + curr_j
                v2 = curr_ring + next_j
                v3 = next_ring + next_j
                v4 = next_ring + curr_j
                
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        # Add end caps
        center_start = len(vertices)
        vertices.append(spine_points[0] + self.center_position)
        center_end = len(vertices)
        vertices.append(spine_points[-1] + self.center_position)
        
        # Start cap
        for j in range(n_radial):
            next_j = (j + 1) % n_radial
            faces.append([center_start, j, next_j])
        
        # End cap
        last_ring_start = (n_segments - 1) * n_radial
        for j in range(n_radial):
            next_j = (j + 1) % n_radial
            faces.append([center_end, last_ring_start + next_j, last_ring_start + j])
        
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        self.original_vertices = self.vertices.copy()  # rest state
        
        # Calculate vertex normals
        self._calculate_vertex_normals()
        
        # Set up spring constraints for deformation
        self._setup_spring_constraints()
        
        # Initialize forces and velocity arrays
        self.forces = np.zeros_like(self.vertices)
        self.velocity = np.zeros_like(self.vertices)  # per-vertex velocities
    
    def _calculate_vertex_normals(self):
        """Calculate vertex normals for the mesh"""
        vertex_normals = np.zeros_like(self.vertices)
        
        for face in self.faces:
            if len(face) >= 3:
                v0, v1, v2 = self.vertices[face[:3]]
                face_normal = np.cross(v1 - v0, v2 - v0)
                face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)
                
                for vertex_idx in face:
                    vertex_normals[vertex_idx] += face_normal
        
        # Normalize vertex normals
        for i, normal in enumerate(vertex_normals):
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                vertex_normals[i] = normal / norm
        
        self.vertex_normals = vertex_normals
    
    def _setup_spring_constraints(self):
        """Set up spring constraints between connected vertices"""
        # Find connected vertex pairs from faces
        edges = set()
        for face in self.faces:
            for i in range(len(face)):
                for j in range(i+1, len(face)):
                    edge = tuple(sorted([face[i], face[j]]))
                    edges.add(edge)
        
        self.edge_list = list(edges)
        
        # Calculate rest lengths
        self.rest_lengths = []
        for v1, v2 in self.edge_list:
            rest_length = np.linalg.norm(self.vertices[v1] - self.vertices[v2])
            self.rest_lengths.append(rest_length)
        
        self.rest_lengths = np.array(self.rest_lengths)
    
    def apply_tissue_forces(self, tissue_field, dt: float):
        """Apply forces from implicit tissue field"""
        # Sample tissue properties at each vertex
        for i, vertex in enumerate(self.vertices):
            # Get tissue resistance at this location
            resistance = tissue_field.get_resistance_at_point(vertex)
            flow_field = tissue_field.get_flow_field_at_point(vertex)
            
            # Apply resistance force (opposite to vertex velocity)
            if np.linalg.norm(self.velocity[i]) > 0:
                resistance_force = -resistance * self.velocity[i] * 0.1
                self.forces[i] += resistance_force
            
            # Apply flow forces
            self.forces[i] += flow_field * 0.05
    
    def apply_undulation(self, time: float):
        """Apply undulatory motion characteristic of sporozoites"""
        phase = self.undulation_phase + time * self.undulation_frequency * 2 * np.pi
        
        # Apply sinusoidal undulation along the body
        for i, vertex in enumerate(self.vertices):
            relative_pos = vertex - self.center_position
            
            # Calculate distance along sporozoite body (approximate)
            body_position = np.dot(relative_pos, [1, 0, 0])  # project onto x-axis
            normalized_position = body_position / (self.length/2) if self.length > 0 else 0
            
            # Undulation force perpendicular to body axis
            undulation_amplitude = 0.1 * self.motility
            undulation_y = undulation_amplitude * np.sin(phase + normalized_position * 2 * np.pi)
            undulation_z = undulation_amplitude * np.cos(phase + normalized_position * 2 * np.pi) * 0.3
            
            undulation_force = np.array([0, undulation_y, undulation_z]) * 10.0
            self.forces[i] += undulation_force
    
    def apply_internal_constraints(self):
        """Apply spring forces to maintain mesh structure"""
        spring_constant = self.stiffness * 50.0
        
        for j, (v1, v2) in enumerate(self.edge_list):
            current_vec = self.vertices[v2] - self.vertices[v1]
            current_length = np.linalg.norm(current_vec)
            rest_length = self.rest_lengths[j]
            
            if current_length > 1e-8:
                # Spring force
                force_magnitude = spring_constant * (current_length - rest_length)
                force_direction = current_vec / current_length
                
                spring_force = force_direction * force_magnitude
                
                # Apply equal and opposite forces
                self.forces[v1] += spring_force * 0.5
                self.forces[v2] -= spring_force * 0.5
    
    def update_motion(self, dt: float, time: float):
        """Update sporozoite motion and deformation"""
        # Clear forces
        self.forces.fill(0.0)
        
        # Apply undulation
        self.apply_undulation(time)
        
        # Apply internal spring constraints
        self.apply_internal_constraints()
        
        # Update vertices using explicit integration
        mass = 0.1  # arbitrary mass units
        damping = 0.8
        
        # Simple explicit integration
        acceleration = self.forces / mass
        self.vertices += self.velocity * dt + 0.5 * acceleration * dt**2
        self.velocity = self.velocity * damping + acceleration * dt
        
        # Update center position
        self.center_position = np.mean(self.vertices, axis=0)
        
        # Recalculate normals
        self._calculate_vertex_normals()
        
        # Random direction changes (chemotaxis/random walk)
        if random.random() < 0.1:
            self.direction += random.uniform(-np.pi/6, np.pi/6)
        
        # Apply directional force
        directional_force = np.array([np.cos(self.direction), np.sin(self.direction), 0])
        directional_force *= self.motility * 2.0
        
        # Apply to all vertices
        for i in range(len(self.vertices)):
            self.forces[i] += directional_force
    
    def to_vtk_polydata(self) -> vtk.vtkPolyData:
        """Convert mesh to VTK PolyData for visualization"""
        polydata = vtk.vtkPolyData()
        
        # Add points
        points = vtk.vtkPoints()
        for vertex in self.vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
        polydata.SetPoints(points)
        
        # Add faces
        polys = vtk.vtkCellArray()
        for face in self.faces:
            poly = vtk.vtkPolygon()
            poly.GetPointIds().SetNumberOfIds(len(face))
            for i, vertex_id in enumerate(face):
                poly.GetPointIds().SetId(i, vertex_id)
            polys.InsertNextCell(poly)
        polydata.SetPolys(polys)
        
        # Add vertex data
        # Viability
        viability_array = vtk.vtkFloatArray()
        viability_array.SetName("Viability")
        for _ in self.vertices:
            viability_array.InsertNextValue(self.viability)
        polydata.GetPointData().SetScalars(viability_array)
        
        # Velocity magnitude
        velocity_array = vtk.vtkFloatArray()
        velocity_array.SetName("VelocityMagnitude")
        for i in range(len(self.vertices)):
            vel_mag = np.linalg.norm(self.velocity[i])
            velocity_array.InsertNextValue(vel_mag)
        polydata.GetPointData().AddArray(velocity_array)
        
        # Motility
        motility_array = vtk.vtkFloatArray()
        motility_array.SetName("Motility")
        for _ in self.vertices:
            motility_array.InsertNextValue(self.motility)
        polydata.GetPointData().AddArray(motility_array)
        
        # Sporozoite ID
        id_array = vtk.vtkIntArray()
        id_array.SetName("SporozoiteID")
        for _ in self.vertices:
            id_array.InsertNextValue(self.id)
        polydata.GetPointData().AddArray(id_array)
        
        return polydata
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box of the mesh"""
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        return min_coords, max_coords
    
    def is_viable(self) -> bool:
        """Check if sporozoite is still viable"""
        return self.viability > 0.1
    
    def reduce_viability(self, amount: float):
        """Reduce viability due to environmental stress"""
        self.viability = max(0.0, self.viability - amount)