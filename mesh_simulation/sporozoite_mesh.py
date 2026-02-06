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
        
        # NEW: Péclet number and activity-flow coupling parameters - ADJUSTED FOR BIOLOGICAL REALISM
        # Sporozoite diffusivity should be higher to reflect active swimming capability
        # Typical active matter: D ~ v²τ where v~10-50 μm/s, τ~0.1-1 s → D~10-2500 μm²/s
        self.diffusion_coefficient = 0.1 * self.motility  # Scale with motility, typical range 10-50 μm²/s change back to 50
        self.peclet_number = 0.0  # Will be calculated dynamically
        self.activity_level = self.motility  # Base activity level
        self.flow_regime = "activity_dominated"  # Can be "activity_dominated" or "flow_dominated"
        
        # NEW: Dynamic deformation coupling parameters
        self.base_spring_constant = self.config.SPRING_CONSTANT_BASE
        self.activity_induced_flexibility = 0.0  # How much activity reduces stiffness
        self.flow_induced_flexibility = 0.0     # How much flow reduces stiffness
        
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
            x = t * self.length - self.length/2 # center at origin otherwise 0 to 1 [t]
            y = np.sin(t * np.pi) * self.length * 0.1  # natural curve and scale with length
            z = 0
            spine_points.append([x, y, z])
        
        spine_points = np.array(spine_points)
        
        # Store spine points for LJ potential calculations
        self.original_spine_points = spine_points.copy()  # Relative to center
        
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
        
        # Store original relative positions BEFORE adding center position
        self.original_relative_positions = vertices.copy()
        
        # NOW add the center position to all vertices
        for i in range(len(vertices)):
            vertices[i] += self.center_position
        
        self.vertices = vertices
        self.faces = np.array(faces)
        self.original_vertices = self.vertices.copy()  # rest state
        
        # Calculate vertex normals
        self._calculate_vertex_normals()
        
        # Set up spring constraints for deformation
        self._setup_spring_constraints()
        
        # Initialize forces and velocity arrays
        self.forces = np.zeros_like(self.vertices)
        self.velocity = np.zeros_like(self.vertices)  # per-vertex velocities
    
    def get_current_centerline_points(self) -> np.ndarray:
        """Get current centerline points in global coordinates"""
        if hasattr(self, 'original_spine_points'):
            # Transform original spine points to current position/orientation
            current_spine = []
            for point in self.original_spine_points:
                # Apply current center position
                global_point = self.center_position + point
                current_spine.append(global_point)
            return np.array(current_spine)
        else:
            # Fallback: return just the center position
            return np.array([self.center_position])
    
    def get_centerline_segment_data(self) -> List[Dict]:
        """Get centerline segments with radii for LJ potential calculations"""
        centerline_points = self.get_current_centerline_points()
        segments = []
        
        n_segments = len(centerline_points)
        for i in range(n_segments):
            # Calculate local radius (varies along length)
            t = i / (n_segments - 1) if n_segments > 1 else 0
            radius_factor = np.sin(np.pi * t) if t > 0 and t < 1 else 0.1
            radius_factor = max(0.1, radius_factor)  # Minimum radius
            local_radius = self.diameter/2 * radius_factor
            
            segment_data = {
                'position': centerline_points[i],
                'radius': local_radius,
                'segment_id': i
            }
            segments.append(segment_data)
        
        return segments
    
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
    
    def calculate_volume(self) -> float:
        """
        Calculate the volume of the sporozoite mesh using the divergence theorem.
        For a closed polyhedral mesh, volume = (1/3) * sum(face_area * dot(face_center, face_normal))
        
        Returns:
            Volume in cubic micrometers (μm³)
        """
        total_volume = 0.0
        
        for face in self.faces:
            if len(face) < 3:
                continue
                
            # Get vertices of the face
            face_vertices = [self.vertices[i] for i in face if i < len(self.vertices)]
            
            if len(face_vertices) < 3:
                continue
            
            # Calculate face center
            face_center = np.mean(face_vertices, axis=0)
            
            # Calculate face normal and area using cross product for triangular faces
            if len(face_vertices) == 3:
                # Triangle face
                v0, v1, v2 = face_vertices
                edge1 = v1 - v0
                edge2 = v2 - v0
                face_normal = np.cross(edge1, edge2)
                face_area = 0.5 * np.linalg.norm(face_normal)
                
                # Normalize normal
                if face_area > 1e-12:
                    face_normal = face_normal / (2.0 * face_area)  # Already divided by 2 for triangle
                else:
                    continue
                    
            elif len(face_vertices) >= 4:
                # Polygon face - triangulate and sum
                face_area = 0.0
                face_normal = np.zeros(3)
                
                # Use fan triangulation from first vertex
                v0 = face_vertices[0]
                for i in range(1, len(face_vertices) - 1):
                    v1 = face_vertices[i]
                    v2 = face_vertices[i + 1]
                    
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    triangle_normal = np.cross(edge1, edge2)
                    triangle_area = 0.5 * np.linalg.norm(triangle_normal)
                    
                    face_area += triangle_area
                    face_normal += triangle_normal
                
                # Normalize total normal
                if face_area > 1e-12:
                    face_normal = face_normal / (2.0 * face_area)
                else:
                    continue
            
            # Apply divergence theorem: V += (1/3) * face_area * dot(face_center, face_normal)
            contribution = (1.0/3.0) * face_area * np.dot(face_center, face_normal)
            total_volume += contribution
        
        # Ensure positive volume (normal orientation might be reversed)
        total_volume = abs(total_volume)
        
        return total_volume
    
    def get_approximate_volume(self) -> float:
        """
        Calculate approximate volume using prolate spheroid formula as fallback.
        V = (4/3) * π * a * b² where a = length/2, b = diameter/2
        
        Returns:
            Approximate volume in cubic micrometers (μm³)
        """
        a = self.length / 2.0  # Semi-major axis
        b = self.diameter / 2.0  # Semi-minor axis
        
        # Prolate spheroid volume formula
        volume = (4.0/3.0) * np.pi * a * b * b
        
        return volume
    
    def calculate_peclet_number(self, flow_velocity: np.ndarray) -> float:
        """Calculate Péclet number: Pe = UL/D where U=flow speed, L=characteristic length, D=diffusivity"""
        flow_speed = np.linalg.norm(flow_velocity)
        characteristic_length = self.length  # Use sporozoite length as characteristic scale
        
        # Avoid division by zero
        if self.diffusion_coefficient > 1e-8:
            peclet_number = flow_speed * characteristic_length / self.diffusion_coefficient
        else:
            peclet_number = 0.0
            
        return peclet_number
    
    def update_flow_regime(self, flow_velocity: np.ndarray, threshold_pe: float = 1.0):
        """Update flow regime based on Péclet number"""
        self.peclet_number = self.calculate_peclet_number(flow_velocity)
        
        if self.peclet_number > threshold_pe:
            self.flow_regime = "flow_dominated"  # Pe >> 1: advection dominates
        else:
            self.flow_regime = "activity_dominated"  # Pe << 1: sporozoite activity dominates
    
    def calculate_activity_deformation_coupling(self) -> float:
        """Calculate how sporozoite activity affects its deformability"""
        # Higher activity -> more flexible (lower spring constant)
        # Model: active sporozoites reduce their cytoskeletal stiffness to enable movement
        
        if self.flow_regime == "activity_dominated":
            # In activity-dominated regime, sporozoite controls its own deformation
            # High activity leads to increased flexibility for swimming
            activity_flexibility_factor = self.activity_level * 0.8  # Max 80% reduction in stiffness
        else:
            # In flow-dominated regime, sporozoite tries to maintain structure against flow
            # Reduced flexibility to resist deformation
            activity_flexibility_factor = self.activity_level * 0.2  # Max 20% reduction in stiffness
        
        self.activity_induced_flexibility = activity_flexibility_factor
        return activity_flexibility_factor
    
    def calculate_flow_deformation_coupling(self, flow_velocity: np.ndarray, shear_rate: float = 0.0) -> float:
        """Calculate how flow forces affect sporozoite deformability"""
        flow_speed = np.linalg.norm(flow_velocity)
        
        if self.flow_regime == "flow_dominated":
            # In high Pe regime: flow forces cause passive deformation
            # Higher flow speed -> more deformation
            # Model: flow-induced deformation scales with flow velocity and shear
            flow_deformation_factor = min(0.7, flow_speed / 100.0)  # Scale with flow speed, max 70% flexibility
            shear_deformation_factor = min(0.5, abs(shear_rate) / 50.0)  # Scale with shear rate, max 50% flexibility
            total_flow_flexibility = flow_deformation_factor + shear_deformation_factor
        else:
            # In activity-dominated regime: minimal flow-induced deformation
            total_flow_flexibility = min(0.1, flow_speed / 200.0)  # Very small effect, max 10%
        
        self.flow_induced_flexibility = total_flow_flexibility
        return total_flow_flexibility
    
    def get_effective_spring_constant(self):
        """Calculate effective spring constant - EMERGENCY BOOST for blood flow"""
        base_spring = self.config.SPRING_CONSTANT_BASE
        
        # For blood flow simulations: MASSIVELY increase spring constant to prevent collapse
        if hasattr(self, 'in_blood_flow') and self.in_blood_flow:
            # Use 10x stronger springs to resist flow-induced deformation
            effective_spring = base_spring * 10.0
        else:
            # In tissue: use normal spring constant
            effective_spring = base_spring
        
        # Apply viability and stiffness factors
        effective_spring *= self.viability * self.stiffness
        
        return effective_spring
    
    def update_activity_level(self, flow_velocity: np.ndarray):
        """Update sporozoite activity level based on flow conditions"""
        flow_speed = np.linalg.norm(flow_velocity)
        
        if self.flow_regime == "flow_dominated":
            # In high Pe regime: sporozoite activity is overwhelmed by flow
            # Reduce effective activity as flow becomes dominant
            flow_suppression_factor = min(0.8, self.peclet_number / 10.0)  # Up to 80% suppression
            self.activity_level = self.motility * (1.0 - flow_suppression_factor)
        else:
            # In activity-dominated regime: sporozoite maintains full activity
            self.activity_level = self.motility
        
        # Ensure minimum activity level
        self.activity_level = max(0.1, self.activity_level)
    
    def apply_tissue_forces(self, tissue_field, dt: float):
        """Apply forces from implicit tissue field or blood flow field - ENHANCED WITH PÉCLET PHYSICS"""
        # Get forces at center of mass
        center_resistance = tissue_field.get_resistance_at_point(self.center_position)
        center_flow = tissue_field.get_flow_field_at_point(self.center_position)
        
        # Check if this is a blood flow field (has velocity methods)
        if hasattr(tissue_field, 'get_velocity_at_point'):
            # This is a blood flow field - apply Péclet number physics
            blood_velocity = tissue_field.get_velocity_at_point(self.center_position)
            
            # NEW: Update flow regime based on Péclet number
            self.update_flow_regime(blood_velocity, threshold_pe=1.0)
            
            # NEW: Update activity level based on flow conditions
            self.update_activity_level(blood_velocity)
            
            # NEW: Calculate activity-deformation coupling
            self.calculate_activity_deformation_coupling()
            
            # NEW: Calculate flow-deformation coupling
            shear_rate = 0.0
            if hasattr(tissue_field, 'get_shear_rate_at_point'):
                shear_rate = tissue_field.get_shear_rate_at_point(self.center_position)
                if isinstance(shear_rate, np.ndarray):
                    shear_rate = float(shear_rate.flatten()[0])
                else:
                    shear_rate = float(shear_rate)
            
            self.calculate_flow_deformation_coupling(blood_velocity, shear_rate)
            
            # Apply drag force based on flow regime
            if self.flow_regime == "flow_dominated":
                # High Pe regime: strong drag coupling, sporozoite is passively advected
                sporozoite_velocity = np.array([
                    np.cos(self.direction), 
                    np.sin(self.direction), 
                    0
                ]) * self.activity_level * 5.0  # Reduced effective velocity due to flow suppression
                
                relative_velocity = blood_velocity - sporozoite_velocity
                drag_coefficient = 0.8 * (1.0 + self.peclet_number / 10.0)  # Stronger drag at high Pe
                self.tissue_flow_force = relative_velocity * drag_coefficient
                
            else:
                # Activity-dominated regime: weaker drag coupling, sporozoite controls motion
                sporozoite_velocity = np.array([
                    np.cos(self.direction), 
                    np.sin(self.direction), 
                    0
                ]) * self.activity_level * 10.0  # Full effective velocity
                
                relative_velocity = blood_velocity - sporozoite_velocity
                drag_coefficient = 0.3 * self.motility  # Weaker drag, activity dominates
                self.tissue_flow_force = relative_velocity * drag_coefficient
            
            # Shear resistance based on regime
            if self.flow_regime == "flow_dominated":
                # Strong shear resistance in high Pe regime
                shear_resistance = np.array([-shear_rate * 0.2, 0, 0])
            else:
                # Weak shear resistance in activity-dominated regime
                shear_resistance = np.array([-shear_rate * 0.05, 0, 0])
                
            self.tissue_resistance_force = shear_resistance

            '''    
            # DEBUG: Print Péclet physics
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            if self._debug_counter % 50 == 0:  # Print occasionally
                blood_vel_mag = np.linalg.norm(blood_velocity)
                flow_force_mag = np.linalg.norm(self.tissue_flow_force)
                print(f"  Péclet Physics DEBUG - Sporozoite {self.id}:")
                print(f"    Blood velocity: {blood_velocity} (mag: {blood_vel_mag:.2f})")
                print(f"    Péclet number: {self.peclet_number:.2f}")
                print(f"    Flow regime: {self.flow_regime}")
                print(f"    Activity level: {self.activity_level:.3f} (base: {self.motility:.3f})")
                print(f"    Activity flexibility: {self.activity_induced_flexibility:.3f}")
                print(f"    Flow flexibility: {self.flow_induced_flexibility:.3f}")
                print(f"    Flow force: {self.tissue_flow_force} (mag: {flow_force_mag:.2f})")
            '''    
        else:
            # This is a tissue field - use original approach but with activity coupling
            self.flow_regime = "activity_dominated"  # No flow, so activity dominates
            self.activity_level = self.motility  # Full activity
            self.calculate_activity_deformation_coupling()
            self.flow_induced_flexibility = 0.0  # No flow-induced flexibility
            
            self.tissue_resistance_force = -center_resistance * 0.1
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
    
    def apply_deformation_forces(self, dt: float):
        """Apply spring-based deformation forces to vertices"""
        # Reset forces
        self.forces = np.zeros_like(self.vertices)
        
        # Spring forces between connected vertices
        spring_constant = self.config.SPRING_CONSTANT_BASE * self.stiffness
        
        for i, (v1, v2) in enumerate(self.edge_list):
            if i < len(self.rest_lengths):
                current_vec = self.vertices[v2] - self.vertices[v1]
                current_length = np.linalg.norm(current_vec)
                
                if current_length > 1e-8:  # Avoid division by zero
                    # Spring force (Hooke's law)
                    extension = current_length - self.rest_lengths[i]
                    force_magnitude = spring_constant * extension
                    force_direction = current_vec / current_length
                    
                    # Apply equal and opposite forces
                    force = force_magnitude * force_direction
                    self.forces[v1] += force
                    self.forces[v2] -= force
        
        # Add damping to prevent oscillations
        damping = self.config.DAMPING_FACTOR
        for i in range(len(self.vertices)):
            self.forces[i] -= damping * self.velocity[i]
    
    def apply_undulation_deformation(self, time: float):
        """Apply undulatory deformation forces to create swimming motion"""
        phase = self.undulation_phase + time * self.undulation_frequency * 2 * np.pi
        
        # Apply undulation forces that actually deform the mesh
        for i, vertex in enumerate(self.vertices):
            relative_pos = vertex - self.center_position
            
            # Calculate distance along sporozoite body (approximate)
            body_position = np.dot(relative_pos, [np.cos(self.direction), np.sin(self.direction), 0])
            normalized_position = body_position / (self.length/2) if self.length > 0 else 0
            
            # Undulation forces perpendicular to movement direction
            perpendicular = np.array([-np.sin(self.direction), np.cos(self.direction), 0])
            vertical = np.array([0, 0, 1])
            
            # Sinusoidal deformation along the body
            wave_amplitude = self.config.UNDULATION_AMPLITUDE * self.motility
            lateral_force = perpendicular * wave_amplitude * np.sin(phase + normalized_position * 2 * np.pi)
            vertical_force = vertical * wave_amplitude * np.cos(phase + normalized_position * 2 * np.pi) * 0.3
            
            undulation_force = (lateral_force + vertical_force) * self.config.UNDULATION_FORCE_SCALE
            self.forces[i] += undulation_force

    def calculate_vertex_propulsion_forces(self, propulsion_magnitude: float) -> np.ndarray:
        """
        Calculate vertex-based propulsion forces where each vertex moves counterclockwise
        along the sporozoite's curvature trajectory.
        
        Args:
            propulsion_magnitude: Constant magnitude of propulsion force for all vertices
            
        Returns:
            Array of propulsion forces for each vertex
        """
        vertex_propulsion_forces = np.zeros_like(self.vertices)
        n_vertices = len(self.vertices)
        
        # Get the current centerline points for reference
        centerline_points = self.get_current_centerline_points()
        n_centerline = len(centerline_points)
        
        # Map vertices to centerline segments for propulsion direction calculation
        n_segments = self.config.LONGITUDINAL_SEGMENTS
        n_radial = self.config.RADIAL_SEGMENTS
        
        # Process vertices organized by longitudinal segments (rings)
        for segment_idx in range(n_segments):
            # Get vertices in this ring
            ring_start = segment_idx * n_radial
            ring_end = min(ring_start + n_radial, n_vertices - 2)  # Exclude end caps
            
            if ring_end <= ring_start:
                continue
                
            # Calculate propulsion direction for this segment based on centerline curvature
            propulsion_direction = self._get_segment_propulsion_direction(segment_idx, centerline_points)
            
            # Apply propulsion to all vertices in this ring
            for vertex_idx in range(ring_start, ring_end):
                if vertex_idx < n_vertices:
                    vertex_propulsion_forces[vertex_idx] = propulsion_direction * propulsion_magnitude
        
        # Handle end cap vertices separately (head and tail nodes)
        if n_vertices >= 2:
            # Head node (first end cap): direction from 1st to 2nd centerline point
            head_vertex_idx = n_vertices - 2  # Second to last vertex (head cap center)
            if len(centerline_points) >= 2:
                head_direction = self._normalize_vector(centerline_points[1] - centerline_points[0])
                vertex_propulsion_forces[head_vertex_idx] = head_direction * propulsion_magnitude
            
            # Tail node (last end cap): direction from second-to-last to last centerline point  
            tail_vertex_idx = n_vertices - 1  # Last vertex (tail cap center)
            if len(centerline_points) >= 2:
                tail_direction = self._normalize_vector(centerline_points[-1] - centerline_points[-2])
                vertex_propulsion_forces[tail_vertex_idx] = tail_direction * propulsion_magnitude
        
        return vertex_propulsion_forces
    
    def _get_segment_propulsion_direction(self, segment_idx: int, centerline_points: np.ndarray) -> np.ndarray:
        """
        Calculate propulsion direction for a centerline segment based on adjacent segments.
        
        Args:
            segment_idx: Index of the current segment
            centerline_points: Array of centerline points
            
        Returns:
            Normalized propulsion direction vector
        """
        n_points = len(centerline_points)
        
        if n_points < 2:
            # Fallback: use sporozoite's overall direction
            return np.array([np.cos(self.direction), np.sin(self.direction), 0])
        
        # For general case: ith node propelled in direction identified by (i-1)th and (i+1)th nodes
        if segment_idx == 0:
            # Head segment: direction from 1st to 2nd point
            if n_points >= 2:
                direction = centerline_points[1] - centerline_points[0]
            else:
                direction = np.array([1, 0, 0])  # Default forward direction
        elif segment_idx == n_points - 1:
            # Tail segment: direction from second-to-last to last point
            if n_points >= 2:
                direction = centerline_points[-1] - centerline_points[-2]
            else:
                direction = np.array([1, 0, 0])  # Default forward direction
        else:
            # General case: direction from (i-1) to (i+1)
            if segment_idx - 1 >= 0 and segment_idx + 1 < n_points:
                direction = centerline_points[segment_idx + 1] - centerline_points[segment_idx - 1]
            else:
                # Fallback for edge cases
                direction = np.array([1, 0, 0])
        
        # Add counterclockwise rotation component based on sporozoite curvature
        direction = self._apply_counterclockwise_curvature(direction, segment_idx, centerline_points)
        
        return self._normalize_vector(direction)
    
    def _apply_counterclockwise_curvature(self, base_direction: np.ndarray, segment_idx: int, 
                                         centerline_points: np.ndarray) -> np.ndarray:
        """
        Apply counterclockwise curvature to the base propulsion direction.
        
        Args:
            base_direction: Base propulsion direction
            segment_idx: Current segment index
            centerline_points: Array of centerline points
            
        Returns:
            Direction vector with counterclockwise curvature applied
        """
        n_points = len(centerline_points)
        
        if n_points < 3 or segment_idx < 1 or segment_idx >= n_points - 1:
            # Not enough points to calculate curvature, return base direction
            return base_direction
        
        # Calculate local curvature using three consecutive points
        p_prev = centerline_points[segment_idx - 1]
        p_curr = centerline_points[segment_idx]
        p_next = centerline_points[segment_idx + 1]
        
        # Vectors between consecutive points
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        # Calculate curvature vector (points toward center of curvature)
        v1_norm = self._normalize_vector(v1)
        v2_norm = self._normalize_vector(v2)
        
        # Curvature direction (perpendicular to average tangent)
        tangent_avg = self._normalize_vector(v1_norm + v2_norm)
        
        # Calculate perpendicular vector for counterclockwise motion
        # Use cross product to get perpendicular in the plane
        if np.linalg.norm(tangent_avg) > 1e-6:
            # Create perpendicular vector in XY plane (counterclockwise)
            perp_vector = np.array([-tangent_avg[1], tangent_avg[0], 0])
            
            # Scale by curvature magnitude and motility
            curvature_magnitude = self._calculate_local_curvature(p_prev, p_curr, p_next)
            curvature_strength = 0.3 * self.motility  # Adjust this factor as needed
            
            # Blend base direction with curvature component
            curved_direction = base_direction + perp_vector * curvature_magnitude * curvature_strength
            
            return curved_direction
        else:
            return base_direction
    
    def _calculate_local_curvature(self, p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray) -> float:
        """
        Calculate local curvature magnitude at a point using three consecutive points.
        
        Args:
            p_prev: Previous point
            p_curr: Current point  
            p_next: Next point
            
        Returns:
            Curvature magnitude (1/radius)
        """
        # Vectors between consecutive points
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        v1_length = np.linalg.norm(v1)
        v2_length = np.linalg.norm(v2)
        
        if v1_length < 1e-8 or v2_length < 1e-8:
            return 0.0
        
        # Normalize vectors
        v1_norm = v1 / v1_length
        v2_norm = v2 / v2_length
        
        # Calculate angle between vectors
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        # Curvature = angle / average_segment_length
        avg_length = (v1_length + v2_length) / 2.0
        if avg_length > 1e-8:
            curvature = angle / avg_length
        else:
            curvature = 0.0
        
        return curvature
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector, handling zero vectors gracefully.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector or default direction if input is zero
        """
        magnitude = np.linalg.norm(vector)
        if magnitude > 1e-8:
            return vector / magnitude
        else:
            # Return default forward direction for zero vectors
            return np.array([1, 0, 0])
    
    def apply_vertex_propulsion(self, propulsion_strength: float, dt: float):
        """
        Apply vertex-based propulsion forces to the sporozoite mesh.
        
        Args:
            propulsion_strength: Overall strength of propulsion
            dt: Time step
        """
        # Calculate propulsion forces for each vertex
        vertex_propulsion_forces = self.calculate_vertex_propulsion_forces(propulsion_strength)
        
        # Apply forces to vertex velocities if they exist
        if hasattr(self, 'vertex_velocities') and len(self.vertex_velocities) == len(self.vertices):
            for i, force in enumerate(vertex_propulsion_forces):
                if i < len(self.vertex_velocities):
                    # Apply propulsion as acceleration
                    acceleration = force * dt / max(0.01, self.motility)  # Scale by motility
                    self.vertex_velocities[i] += acceleration
        else:
            # Initialize vertex velocities if they don't exist
            self.vertex_velocities = [np.zeros(3) for _ in range(len(self.vertices))]
            for i, force in enumerate(vertex_propulsion_forces):
                acceleration = force * dt / max(0.01, self.motility)
                self.vertex_velocities[i] = acceleration
        
        # Also update center position based on average vertex motion
        if len(vertex_propulsion_forces) > 0:
            avg_propulsion = np.mean(vertex_propulsion_forces, axis=0)
            center_acceleration = avg_propulsion * dt * self.motility
            
            # Update center velocity
            if not hasattr(self, 'center_velocity'):
                self.center_velocity = np.zeros(3)
            self.center_velocity += center_acceleration

    def update_motion(self, dt, forces, spring_constant=100.0, damping=0.5):
        """
        Update sporozoite motion with enhanced deformation capability.
        
        Args:
            dt: Time step
            forces: External forces dictionary
            spring_constant: Stiffness of the mesh (lower = more flexible)
            damping: Damping factor for stability (higher = more stable)
        """
        if not self.is_alive:
            return
            
        # Apply motility and viability effects - ENHANCED FOR DEFORMATION
        effective_spring = spring_constant * self.viability * 0.3  # Much weaker restoring forces
        effective_damping = damping * (1.5 - self.motility)  # Less damping for more movement
        
        # Use more substeps for very flexible meshes (low spring constant)
        if spring_constant < 20.0:
            num_substeps = max(3, int(dt / 0.008))  # More substeps for stability with very flexible meshes
        else:
            num_substeps = max(1, int(dt / 0.01))
        substep_dt = dt / num_substeps
        
        for _ in range(num_substeps):
            self._update_physics_substep_deformable(substep_dt, forces, effective_spring, effective_damping)
    
    def _update_physics_substep_deformable(self, dt, forces, spring_constant, damping):
        """FIXED physics substep with GENTLE force limits and proper mesh preservation"""
        
        # DEBUG: Track this specific issue
        if not hasattr(self, '_flatten_debug_counter'):
            self._flatten_debug_counter = 0
        self._flatten_debug_counter += 1
        
        # Check mesh integrity BEFORE physics update
        bounds = self.get_bounding_box()
        dimensions = bounds[1] - bounds[0]
        
        # Define LENIENT thresholds for mesh collapse (less strict)
        min_length_threshold = self.length * 0.2   # Must maintain at least 20% of original length
        min_diameter_threshold = self.diameter * 0.15  # Must maintain at least 15% of original diameter
        
        # Check for critical deformation
        is_critically_deformed = (
            dimensions[0] < min_length_threshold or
            dimensions[1] < min_diameter_threshold or
            dimensions[2] < min_diameter_threshold or
            np.any(dimensions < 0.02)  # Any dimension below 0.02 μm
        )
        
        if is_critically_deformed:
            print(f"      MESH EMERGENCY: Sporozoite {self.id} critically deformed!")
            print(f"        Dimensions: {dimensions}")
            print(f"        Thresholds - L: {min_length_threshold:.2f}, D: {min_diameter_threshold:.2f}")
            
            # COMPLETE restoration from original positions
            if hasattr(self, 'original_relative_positions'):
                for i, orig_rel_pos in enumerate(self.original_relative_positions):
                    if i < len(self.vertices):
                        self.vertices[i] = self.center_position + orig_rel_pos
                
                # Clear all velocities
                if hasattr(self, 'vertex_velocities'):
                    for i in range(len(self.vertex_velocities)):
                        self.vertex_velocities[i] = np.zeros(3)
                
                print(f"        EMERGENCY RESTORATION COMPLETE")
                return  # Skip physics update to prevent re-collapse
        
        # Initialize velocities if needed
        if not hasattr(self, 'vertex_velocities'):
            self.vertex_velocities = [np.zeros(3) for _ in range(len(self.vertices))]
        
        # GENTLE center motion with LIMITED force scaling
        old_center = self.center_position.copy()
        center_force = np.zeros(3)
        
        # Apply external forces with GENTLE scaling
        if 'directional' in forces:
            center_force += forces['directional'] * 0.8  # Only 20% reduction
        if 'random' in forces:
            center_force += forces['random'] * 0.9  # Only 10% reduction
        if 'boundary_repulsion' in forces:
            center_force += forces['boundary_repulsion'] * 0.5
        
        # Add flow forces with MODERATE limits
        if hasattr(self, 'tissue_flow_force'):
            flow_force_magnitude = np.linalg.norm(self.tissue_flow_force)
            max_flow_force = 25.0  # Increased limit (was 15.0)
            if flow_force_magnitude > max_flow_force:
                scaled_flow_force = self.tissue_flow_force * (max_flow_force / flow_force_magnitude)
            else:
                scaled_flow_force = self.tissue_flow_force
            center_force += scaled_flow_force * 0.6  # Increased flow influence (was 0.4)
            
        if hasattr(self, 'tissue_resistance_force'):
            center_force += self.tissue_resistance_force * 0.2
        
        # GENTLE center acceleration and movement
        center_acceleration = center_force * self.motility * 1.0  # Normal acceleration
        
        # Apply MODERATE damping (not extreme)
        center_velocity = getattr(self, 'center_velocity', np.zeros(3))
        center_velocity = center_velocity * (1.0 - damping * 3.0 * dt) + center_acceleration * dt
        
        # REASONABLE velocity limits (allow good movement)
        max_center_velocity = 20.0  # Increased from 15.0
        center_vel_magnitude = np.linalg.norm(center_velocity)
        if center_vel_magnitude > max_center_velocity:
            center_velocity = center_velocity * (max_center_velocity / center_vel_magnitude)
        
        # Update center position
        self.center_position += center_velocity * dt
        self.center_velocity = center_velocity
        center_move = self.center_position - old_center
        
        # GENTLE VERTEX PHYSICS with LIMITED spring forces
        vertex_forces = []
        max_spring_force = 0.0
        
        # Use MODERATE restoring forces (not extreme)
        if is_critically_deformed:
            # Emergency: Strong but not extreme springs
            emergency_spring_constant = spring_constant * 8.0  # Reduced from 20x
        else:
            # Normal: Gentle springs for shape maintenance
            emergency_spring_constant = spring_constant * 2.0  # Much gentler (was 5x)
        
        # Calculate GENTLE restoring forces
        for i, vertex in enumerate(self.vertices):
            force = np.zeros(3)
            
            # Gentle restoring force to original shape
            if i < len(self.original_relative_positions):
                target_position = self.center_position + self.original_relative_positions[i]
                displacement = target_position - vertex
                
                # LIMIT spring force magnitude to prevent explosion
                spring_force = displacement * emergency_spring_constant
                spring_force_mag = np.linalg.norm(spring_force)
                
                # CRITICAL: Cap maximum spring force per vertex
                max_allowed_spring_force = 50.0  # Much lower limit (was unlimited)
                if spring_force_mag > max_allowed_spring_force:
                    spring_force = spring_force * (max_allowed_spring_force / spring_force_mag)
                
                max_spring_force = max(max_spring_force, np.linalg.norm(spring_force))
                force += spring_force
            
            vertex_forces.append(force)
        
        # Apply GENTLE vertex updates
        for i, (vertex, force) in enumerate(zip(self.vertices, vertex_forces)):
            if i >= len(self.vertex_velocities):
                continue
            
            # GENTLE responsiveness to prevent oscillations
            responsiveness = 0.05  # Moderate responsiveness (was 0.1)
            acceleration = force * responsiveness
            
            # MODERATE acceleration limits
            acc_magnitude = np.linalg.norm(acceleration)
            max_acceleration = 20.0  # Reasonable limit
            if acc_magnitude > max_acceleration:
                acceleration = acceleration * (max_acceleration / acc_magnitude)
            
            # Update velocity with GENTLE damping
            velocity = self.vertex_velocities[i]
            damping_factor = damping * 8.0  # Moderate damping (not extreme)
            velocity = velocity * (1.0 - damping_factor * dt) + acceleration * dt
            
            # GENTLE velocity limits
            max_velocity = 8.0  # Reasonable velocity limit
            velocity_magnitude = np.linalg.norm(velocity)
            if velocity_magnitude > max_velocity:
                velocity = velocity * (max_velocity / velocity_magnitude)
            
            # Update position with CONTROLLED deviation limits
            new_position = vertex + velocity * dt
            
            # Keep vertices REASONABLY close to original positions
            max_deviation = self.length * 0.15  # Increased from 0.1 to allow more deformation
            relative_pos = new_position - self.center_position
            if i < len(self.original_relative_positions):
                deviation = relative_pos - self.original_relative_positions[i]
                deviation_magnitude = np.linalg.norm(deviation)
                if deviation_magnitude > max_deviation:
                    # Clamp to maximum allowed deviation
                    relative_pos = self.original_relative_positions[i] + deviation * (max_deviation / deviation_magnitude)
                    new_position = self.center_position + relative_pos
                    velocity *= 0.7  # Gentle damping when constrained
            
            self.vertices[i] = new_position
            self.vertex_velocities[i] = velocity
        
        # Debug output for monitoring
        if self._flatten_debug_counter <= 5 or is_critically_deformed:
            post_bounds = self.get_bounding_box()
            post_dims = post_bounds[1] - post_bounds[0]
            center_movement_magnitude = np.linalg.norm(center_move)
            print(f"      GENTLE PHYSICS - Post dims: {post_dims}")
            print(f"      Max spring force: {max_spring_force:.2f} (limit: 50.0)")
            print(f"      Center movement: {center_movement_magnitude:.6f} μm")
            
            if is_critically_deformed:
                restored = np.all(post_dims > [min_length_threshold*0.8, min_diameter_threshold*0.8, min_diameter_threshold*0.8])
                print(f"      MESH STATUS: {'RESTORED' if restored else 'STILL_CRITICAL'}")
            else:
                movement_ok = center_movement_magnitude > 0.0005  # Lower threshold for movement
                print(f"      MOVEMENT STATUS: {'OK' if movement_ok else 'SLOW'}")

    def _would_vertex_exit_vessel(self, position: np.ndarray, buffer_factor: float = 1.0) -> bool:
        """Check if a vertex position would be outside the blood vessel with adjustable buffer"""
        # Assume vessel is centered in domain - adjust these if vessel geometry is different
        vessel_center_y = 12.5  # Half of 25.0 μm domain height
        vessel_center_z = 7.5   # Half of 15.0 μm domain depth
        vessel_radius = 5.5 * buffer_factor  # Adjustable radius based on buffer factor
        
        # Calculate radial distance from vessel centerline (in Y-Z plane)
        radial_distance = np.sqrt((position[1] - vessel_center_y)**2 + (position[2] - vessel_center_z)**2)
        
        return radial_distance > vessel_radius
    
    def to_vtk_polydata(self) -> vtk.vtkPolyData:
        """Convert mesh to VTK PolyData for visualization - FIXED for ParaView compatibility"""
        polydata = vtk.vtkPolyData()
        
        # Add points
        points = vtk.vtkPoints()
        for vertex in self.vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
        polydata.SetPoints(points)
        
        # Add faces - FIXED: Proper error handling and validation
        polys = vtk.vtkCellArray()
        valid_faces = 0
        
        for face in self.faces:
            # Validate face indices
            valid_face = True
            for vertex_id in face:
                if vertex_id < 0 or vertex_id >= len(self.vertices):
                    valid_face = False
                    break
            
            if valid_face and len(face) >= 3:
                # Create polygon only for valid faces
                poly = vtk.vtkPolygon()
                poly.GetPointIds().SetNumberOfIds(len(face))
                for i, vertex_id in enumerate(face):
                    poly.GetPointIds().SetId(i, int(vertex_id))  # Ensure integer type
                polys.InsertNextCell(poly)
                valid_faces += 1
        
        polydata.SetPolys(polys)
        
        # Add vertex data arrays
        num_points = len(self.vertices)
        
        # Viability (use as scalars for default coloring)
        viability_array = vtk.vtkFloatArray()
        viability_array.SetName("Viability")
        viability_array.SetNumberOfTuples(num_points)
        for i in range(num_points):
            viability_array.SetValue(i, float(self.viability))
        polydata.GetPointData().SetScalars(viability_array)
        
        # Velocity magnitude
        velocity_array = vtk.vtkFloatArray()
        velocity_array.SetName("VelocityMagnitude")
        velocity_array.SetNumberOfTuples(num_points)
        for i in range(num_points):
            if i < len(self.velocity):
                vel_mag = np.linalg.norm(self.velocity[i])
            else:
                vel_mag = 0.0
            velocity_array.SetValue(i, float(vel_mag))
        polydata.GetPointData().AddArray(velocity_array)
        
        # Motility
        motility_array = vtk.vtkFloatArray()
        motility_array.SetName("Motility")
        motility_array.SetNumberOfTuples(num_points)
        for i in range(num_points):
            motility_array.SetValue(i, float(self.motility))
        polydata.GetPointData().AddArray(motility_array)
        
        # Sporozoite ID - FIXED: Proper array creation
        id_array = vtk.vtkIntArray()
        id_array.SetName("SporozoiteID")
        id_array.SetNumberOfTuples(num_points)
        for i in range(num_points):
            id_array.SetValue(i, int(self.id))
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
    
    @property
    def is_alive(self) -> bool:
        """Check if sporozoite is still alive"""
        return self.viability > 0.1