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
import sys
import os

# Import configuration
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from simulation_config import SimulationConfig

class DeformableSporozoiteMesh:
    """
    Represents a sporozoite as a deformable polyhedral mesh
    """
    
    def __init__(self, sporozoite_id: int, initial_position: np.ndarray, config=None):
        self.id = sporozoite_id
        self.center_position = np.array(initial_position, dtype=float)
        self.config = config or SimulationConfig()
        
        # Biological parameters using config
        self.length = random.uniform(*self.config.SPOROZOITE_LENGTH_RANGE)
        self.diameter = random.uniform(*self.config.SPOROZOITE_DIAMETER_RANGE)
        self.motility = random.uniform(*self.config.MOTILITY_RANGE)
        self.stiffness = random.uniform(*self.config.STIFFNESS_RANGE)
        
        # Movement parameters using config
        self.direction = np.random.uniform(0, 2*np.pi)
        self.undulation_phase = random.uniform(0, 2*np.pi)
        self.undulation_frequency = random.uniform(*self.config.UNDULATION_FREQUENCY_RANGE)
        self.max_speed = random.uniform(*self.config.MAX_SPEED_RANGE)
        
        # Physical state
        self.viability = 1.0
        
        # DEBUG: Print sporozoite initialization parameters
        print(f"\n=== SPOROZOITE {sporozoite_id} INITIALIZATION DEBUG ===")
        print(f"  ID: {self.id}")
        print(f"  Initial position: {self.center_position}")
        print(f"  Length: {self.length:.3f}")
        print(f"  Diameter: {self.diameter:.3f}")
        print(f"  Motility: {self.motility:.3f}")
        print(f"  Stiffness: {self.stiffness:.3f}")
        print(f"  Direction: {self.direction:.3f} rad ({np.degrees(self.direction):.1f}°)")
        print(f"  Undulation phase: {self.undulation_phase:.3f} rad")
        print(f"  Undulation frequency: {self.undulation_frequency:.3f} Hz")
        print(f"  Max speed: {self.max_speed:.3f}")
        print(f"  Initial viability: {self.viability}")
        
        # Create mesh geometry
        self.vertices = None
        self.faces = None
        self.vertex_normals = None
        self.rest_lengths = None  # for spring constraints
        
        self._create_sporozoite_mesh()
    
    def _create_sporozoite_mesh(self):
        """Create initial curved rod mesh representing sporozoite - FIXED VERSION"""
        # Create a curved cylindrical mesh using config parameters
        n_segments = self.config.LONGITUDINAL_SEGMENTS
        n_radial = self.config.RADIAL_SEGMENTS
        
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
        
        # Create cross-sections perpendicular to spine - FIXED VERSION
        for i, spine_point in enumerate(spine_points):
            # Calculate local coordinate system
            if i == 0:
                tangent = spine_points[i+1] - spine_points[i]
            elif i == len(spine_points) - 1:
                tangent = spine_points[i] - spine_points[i-1]
            else:
                tangent = spine_points[i+1] - spine_points[i-1]
            
            tangent_length = np.linalg.norm(tangent)
            if tangent_length > 1e-8:
                tangent = tangent / tangent_length
            else:
                tangent = np.array([1, 0, 0])  # default direction
            
            # Create perpendicular vectors - FIXED
            if abs(tangent[2]) < 0.9:
                normal = np.cross(tangent, [0, 0, 1])
            else:
                normal = np.cross(tangent, [1, 0, 0])
            
            normal_length = np.linalg.norm(normal)
            if normal_length > 1e-8:
                normal = normal / normal_length
            else:
                normal = np.array([0, 1, 0])  # default normal
            
            binormal = np.cross(tangent, normal)
            binormal_length = np.linalg.norm(binormal)
            if binormal_length > 1e-8:
                binormal = binormal / binormal_length
            else:
                binormal = np.array([0, 0, 1])  # default binormal
            
            # Radius varies along length (tapered ends)
            radius_factor = np.sin(np.pi * i / (n_segments - 1))
            radius_factor = max(0.1, radius_factor)  # Minimum radius to prevent collapse
            radius = self.diameter/2 * radius_factor
            
            # Create circular cross-section - FIXED: Actually create different points!
            for j in range(n_radial):
                angle = 2 * np.pi * j / n_radial
                
                # Create point on circle using normal and binormal vectors
                local_point = (normal * np.cos(angle) + binormal * np.sin(angle)) * radius
                
                # Position relative to center (NOT spine_point + center_position)
                vertex = spine_point + local_point  # Relative to origin
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
        vertices.append(spine_points[0])  # Don't add center_position yet
        center_end = len(vertices)
        vertices.append(spine_points[-1])  # Don't add center_position yet
        
        # Start cap
        for j in range(n_radial):
            next_j = (j + 1) % n_radial
            faces.append([center_start, j, next_j])
        
        # End cap
        last_ring_start = (n_segments - 1) * n_radial
        for j in range(n_radial):
            next_j = (j + 1) % n_radial
            faces.append([center_end, last_ring_start + next_j, last_ring_start + j])
        
        # Convert to numpy and position correctly
        vertices = np.array(vertices)
        
        # NOW add the center position to all vertices
        for i in range(len(vertices)):
            vertices[i] += self.center_position
        
        self.vertices = vertices
        self.faces = np.array(faces)
        self.original_vertices = self.vertices.copy()  # rest state
        
        # DEBUG: Print mesh creation info
        print(f"MESH CREATION DEBUG for sporozoite {self.id}:")
        print(f"  Created {len(vertices)} vertices, {len(faces)} faces")
        print(f"  Vertex range X: [{np.min(vertices[:, 0]):.3f}, {np.max(vertices[:, 0]):.3f}]")
        print(f"  Vertex range Y: [{np.min(vertices[:, 1]):.3f}, {np.max(vertices[:, 1]):.3f}]")
        print(f"  Vertex range Z: [{np.min(vertices[:, 2]):.3f}, {np.max(vertices[:, 2]):.3f}]")
        print(f"  First 3 vertices:")
        for i in range(min(3, len(vertices))):
            print(f"    Vertex {i}: {vertices[i]}")
        
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
        """Set up spring constraints between connected vertices - FIXED VERSION"""
        # Only connect adjacent vertices, not every vertex to every other vertex in faces
        edges = set()
        
        # Add edges along rings (radial connections)
        n_segments = self.config.LONGITUDINAL_SEGMENTS
        n_radial = self.config.RADIAL_SEGMENTS
        
        for i in range(n_segments):
            ring_start = i * n_radial
            for j in range(n_radial):
                v1 = ring_start + j
                v2 = ring_start + ((j + 1) % n_radial)
                if v1 < len(self.vertices) - 2 and v2 < len(self.vertices) - 2:  # Exclude end caps
                    edges.add(tuple(sorted([v1, v2])))
        
        # Add edges along length (longitudinal connections)
        for i in range(n_segments - 1):
            for j in range(n_radial):
                v1 = i * n_radial + j
                v2 = (i + 1) * n_radial + j
                if v1 < len(self.vertices) - 2 and v2 < len(self.vertices) - 2:  # Exclude end caps
                    edges.add(tuple(sorted([v1, v2])))
        
        self.edge_list = list(edges)
        
        # Calculate rest lengths
        self.rest_lengths = []
        for v1, v2 in self.edge_list:
            if v1 < len(self.vertices) and v2 < len(self.vertices):
                rest_length = np.linalg.norm(self.vertices[v1] - self.vertices[v2])
                self.rest_lengths.append(rest_length)
        
        self.rest_lengths = np.array(self.rest_lengths)
    
    def apply_tissue_forces(self, tissue_field, dt: float):
        """Apply forces from implicit tissue field - SIMPLIFIED FOR RIGID BODY"""
        # Only apply forces to center of mass, not individual vertices
        center_resistance = tissue_field.get_resistance_at_point(self.center_position)
        center_flow = tissue_field.get_flow_field_at_point(self.center_position)
        
        # These forces will be applied in update_motion as modifications to center movement
        self.tissue_resistance_force = -center_resistance * 0.1  # Reduce tissue effects
        self.tissue_flow_force = center_flow * 0.05
    
    def apply_undulation(self, time: float):
        """Apply very gentle undulatory motion - SIMPLIFIED VERSION"""
        phase = self.undulation_phase + time * self.undulation_frequency * 2 * np.pi
        
        # Apply VERY subtle undulation only
        for i, vertex in enumerate(self.vertices):
            relative_pos = vertex - self.center_position
            
            # Calculate distance along sporozoite body (approximate)
            body_position = np.dot(relative_pos, [1, 0, 0])  # project onto x-axis
            normalized_position = body_position / (self.length/2) if self.length > 0 else 0
            
            # MUCH smaller undulation forces
            undulation_y = self.config.UNDULATION_AMPLITUDE * np.sin(phase + normalized_position * 2 * np.pi)
            undulation_z = self.config.UNDULATION_AMPLITUDE * np.cos(phase + normalized_position * 2 * np.pi) * 0.1
            
            undulation_force = np.array([0, undulation_y, undulation_z]) * self.config.UNDULATION_FORCE_SCALE * self.motility
            self.forces[i] += undulation_force
    
    def update_motion(self, dt: float, time: float):
        """Update sporozoite motion - COMPLETELY RIGID APPROACH WITH SIMPLIFIED DEBUGGING"""
        # Store original relative positions to maintain exact shape
        if not hasattr(self, 'original_relative_positions'):
            self.original_relative_positions = []
            for vertex in self.vertices:
                relative_pos = vertex - self.center_position
                self.original_relative_positions.append(relative_pos.copy())
        
        # Store previous position for movement calculation
        prev_center = self.center_position.copy()
        
        # Calculate center of mass movement (pure translation)
        directional_force = np.array([np.cos(self.direction), np.sin(self.direction), 0])
        directional_force *= self.motility * self.config.DIRECTIONAL_FORCE_STRENGTH
        
        # Add tiny amount of undulation to center position only
        phase = self.undulation_phase + time * self.undulation_frequency * 2 * np.pi
        undulation_displacement = np.array([0, 
                                          self.config.UNDULATION_AMPLITUDE * np.sin(phase) * 0.1,
                                          self.config.UNDULATION_AMPLITUDE * np.cos(phase) * 0.05])
        
        # Update center position (rigid body motion)
        acceleration = directional_force / self.config.MASS
        velocity_change = acceleration * dt
        displacement = velocity_change * dt + undulation_displacement * dt
        
        self.center_position += displacement
        
        # COMPLETELY RIGID MOVEMENT - restore exact original shape
        for i in range(len(self.vertices)):
            if i < len(self.original_relative_positions):
                # Maintain EXACT original relative positions - no deformation at all
                new_vertex = self.center_position + self.original_relative_positions[i]
                self.vertices[i] = new_vertex
        
        # Update velocity array for VTK output (all vertices move together)
        center_velocity = displacement / dt if dt > 0 else np.array([0.0, 0.0, 0.0])
        for i in range(len(self.velocity)):
            self.velocity[i] = center_velocity
        
        # Random direction changes
        if random.random() < self.config.RANDOM_DIRECTION_PROBABILITY:
            direction_change = random.uniform(-self.config.MAX_DIRECTION_CHANGE, 
                                            self.config.MAX_DIRECTION_CHANGE)
            self.direction += direction_change
        
        # Recalculate normals
        self._calculate_vertex_normals()
    
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
        
        # DEBUG: Print sporozoite ID being written to VTK
        print(f"VTK DEBUG: Writing sporozoite ID {self.id} to polydata with {len(self.vertices)} vertices")
        
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
        
        # Sporozoite ID - FIXED WITH DEBUGGING
        id_array = vtk.vtkIntArray()
        id_array.SetName("SporozoiteID")
        for vertex_idx in range(len(self.vertices)):
            id_array.InsertNextValue(self.id)
            # DEBUG: Print first few ID assignments
            if vertex_idx < 3:
                print(f"  VTK DEBUG: Vertex {vertex_idx} assigned ID {self.id}")
        polydata.GetPointData().AddArray(id_array)
        
        # DEBUG: Verify the ID array was created correctly
        retrieved_id_array = polydata.GetPointData().GetArray("SporozoiteID")
        if retrieved_id_array:
            print(f"  VTK DEBUG: ID array created with {retrieved_id_array.GetNumberOfTuples()} values")
            if retrieved_id_array.GetNumberOfTuples() > 0:
                first_id = retrieved_id_array.GetValue(0)
                print(f"  VTK DEBUG: First ID value in array: {first_id}")
        else:
            print(f"  VTK ERROR: Failed to create SporozoiteID array!")
        
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