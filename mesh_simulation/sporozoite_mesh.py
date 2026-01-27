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
        
        # Store original relative positions BEFORE adding center position
        self.original_relative_positions = vertices.copy()
        
        # NOW add the center position to all vertices
        for i in range(len(vertices)):
            vertices[i] += self.center_position
        
        self.vertices = vertices
        self.faces = np.array(faces)
        self.original_vertices = self.vertices.copy()  # rest state
        
        # DEBUG: Print mesh creation info
        '''
        print(f"MESH CREATION DEBUG for sporozoite {self.id}:")
        print(f"  Created {len(vertices)} vertices, {len(faces)} faces")
        print(f"  Vertex range X: [{np.min(vertices[:, 0]):.3f}, {np.max(vertices[:, 0]):.3f}]")
        print(f"  Vertex range Y: [{np.min(vertices[:, 1]):.3f}, {np.max(vertices[:, 1]):.3f}]")
        print(f"  Vertex range Z: [{np.min(vertices[:, 2]):.3f}, {np.max(vertices[:, 2]):.3f}]")
        print(f"  First 3 vertices:")
        for i in range(min(3, len(vertices))):
            print(f"    Vertex {i}: {vertices[i]}")
        '''
        
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
    
    def get_effective_spring_constant(self) -> float:
        """Calculate effective spring constant based on activity and flow coupling"""
        # Base spring constant modified by both activity and flow effects
        activity_factor = 1.0 - self.activity_induced_flexibility  # Reduces stiffness
        flow_factor = 1.0 - self.flow_induced_flexibility          # Reduces stiffness
        
        # Combined effect: both activity and flow can make sporozoite more flexible
        combined_flexibility = self.activity_induced_flexibility + self.flow_induced_flexibility
        combined_flexibility = min(0.9, combined_flexibility)  # Cap at 90% flexibility increase
        
        effective_stiffness_factor = 1.0 - combined_flexibility
        effective_spring_constant = self.base_spring_constant * effective_stiffness_factor
        
        # Ensure minimum stiffness to prevent mesh collapse
        min_spring_constant = self.base_spring_constant * 0.1
        effective_spring_constant = max(min_spring_constant, effective_spring_constant)
        
        return effective_spring_constant
    
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
        """Enhanced physics substep with gentle vessel-aware constraints."""
        
        # Initialize velocities if not present
        if not hasattr(self, 'vertex_velocities'):
            self.vertex_velocities = [np.zeros(3) for _ in range(len(self.vertices))]
        
        # Calculate center position and movement
        old_center = self.center_position.copy()
        
        # Apply external forces to center
        center_force = np.zeros(3)
        if 'directional' in forces:
            center_force += forces['directional']
        if 'random' in forces:
            center_force += forces['random']
        if 'boundary_repulsion' in forces:
            center_force += forces['boundary_repulsion']
        
        # Add blood flow forces if available
        if hasattr(self, 'tissue_flow_force'):
            center_force += self.tissue_flow_force
        if hasattr(self, 'tissue_resistance_force'):
            center_force += self.tissue_resistance_force
        
        # Update center position with damping
        center_acceleration = center_force * self.motility
        center_velocity = getattr(self, 'center_velocity', np.zeros(3))
        center_velocity = center_velocity * (1.0 - damping * dt * 0.5) + center_acceleration * dt
        self.center_position += center_velocity * dt
        self.center_velocity = center_velocity
        
        # Calculate forces on each vertex with GENTLE vessel awareness
        vertex_forces = []
        for i, vertex in enumerate(self.vertices):
            force = np.zeros(3)
            
            # CONTROLLED restoring force to original shape
            if i < len(self.original_relative_positions):
                target_position = self.center_position + self.original_relative_positions[i]
                displacement = target_position - vertex
                
                # Make restoring force proportional to displacement but not too weak
                distance_from_rest = np.linalg.norm(displacement)
                
                # Gradual increase in restoring force based on distance
                if distance_from_rest > self.length * 0.4:  
                    # Strong restoration when very deformed (beyond 40% of length)
                    spring_force = displacement * spring_constant * 5.0
                elif distance_from_rest > self.length * 0.2:  
                    # Moderate restoration for significant deformation (20-40% of length)
                    spring_force = displacement * spring_constant * 1.0
                else:
                    # Weak restoration for small deformation (< 20% of length)
                    spring_force = displacement * spring_constant * 0.3
                
                # For very flexible meshes, reduce all restoring forces but don't eliminate them
                if spring_constant < 20.0:
                    spring_force *= 0.5  # Reduce but don't eliminate
                
                force += spring_force
            
            # GENTLE vessel-aware undulation forces
            if 'undulation' in forces and i < len(forces['undulation']):
                undulation_force = forces['undulation'][i]
                
                # Scale undulation force based on spring constant but limit maximum
                if spring_constant < 20.0:
                    undulation_scale = min(2.0, 50.0 / max(1.0, spring_constant))  # Cap at 2x scale
                else:
                    undulation_scale = 0.5
                
                undulation_force *= undulation_scale
                
                # GENTLE vessel constraint: only reduce outward forces if getting close to boundary
                proposed_position = vertex + undulation_force * dt * dt  # Rough estimate
                if self._would_vertex_exit_vessel(proposed_position):
                    # Only apply gentle reduction, don't eliminate forces completely
                    vessel_center_2d = np.array([12.5, 7.5])  # Assuming domain center
                    vertex_2d = proposed_position[1:3]  # Y, Z coordinates
                    radial_direction = vertex_2d - vessel_center_2d
                    radial_distance = np.linalg.norm(radial_direction)
                    
                    if radial_distance > 1e-6:
                        radial_unit = radial_direction / radial_distance
                        # Project undulation force and gently reduce radial component
                        force_2d = undulation_force[1:3]  # Y, Z components
                        radial_component = np.dot(force_2d, radial_unit)
                        if radial_component > 0:  # Outward force
                            # Gently reduce outward radial component instead of eliminating it
                            reduction_factor = 0.5  # Reduce by 50% instead of 100%
                            force_2d -= radial_component * radial_unit * reduction_factor
                            undulation_force[1:3] = force_2d
                
                # Limit undulation force magnitude to prevent instability
                force_magnitude = np.linalg.norm(undulation_force)
                max_undulation_force = self.length * 2.0  # Increased limit to allow more movement
                if force_magnitude > max_undulation_force:
                    undulation_force = undulation_force * (max_undulation_force / force_magnitude)
                
                force += undulation_force
            
            vertex_forces.append(force)
        
        # Apply neighbor-based spring forces for shape coherence with better control
        if hasattr(self, 'edge_list') and hasattr(self, 'rest_lengths'):
            for edge_idx, (v1, v2) in enumerate(self.edge_list):
                if edge_idx < len(self.rest_lengths) and v1 < len(vertex_forces) and v2 < len(vertex_forces):
                    current_vec = self.vertices[v2] - self.vertices[v1]
                    current_length = np.linalg.norm(current_vec)
                    
                    if current_length > 1e-8:
                        rest_length = self.rest_lengths[edge_idx]
                        
                        # Allow reasonable deformation but prevent extreme distortion
                        if spring_constant < 20.0:
                            # Flexible: allow 25% extension/compression (more than before)
                            max_extension = rest_length * 1.25
                            max_compression = rest_length * 0.75
                        else:
                            # Normal: allow 15% extension/compression
                            max_extension = rest_length * 1.15
                            max_compression = rest_length * 0.85
                        
                        # Calculate force based on deformation level
                        if current_length > max_extension:
                            excess = current_length - max_extension
                            force_magnitude = spring_constant * excess * 2.0  # Reduced force strength
                        elif current_length < max_compression:
                            excess = max_compression - current_length
                            force_magnitude = spring_constant * excess * 2.0  # Reduced force strength
                        else:
                            # Within allowed range: gentle forces to maintain shape
                            extension = current_length - rest_length
                            force_magnitude = spring_constant * extension * 0.1  # Very gentle internal forces
                        
                        # Limit maximum edge force to prevent instability
                        max_edge_force = spring_constant * rest_length * 0.5  # Same limit
                        if abs(force_magnitude) > max_edge_force:
                            force_magnitude = max_edge_force * (1 if force_magnitude > 0 else -1)
                        
                        force_direction = current_vec / current_length
                        edge_force = force_magnitude * force_direction
                        
                        # Distribute force to vertices
                        vertex_forces[v1] += edge_force * 0.5
                        vertex_forces[v2] -= edge_force * 0.5
        
        # Update vertex positions with GENTLE vessel awareness
        for i, (vertex, force) in enumerate(zip(self.vertices, vertex_forces)):
            if i >= len(self.vertex_velocities):
                continue
                
            # Apply force with controlled responsiveness
            mass_factor = 0.8 if spring_constant < 20.0 else 1.0
            acceleration = force * mass_factor
            
            # Limit acceleration to prevent explosions
            acc_magnitude = np.linalg.norm(acceleration)
            max_acceleration = 800.0  # Increased acceleration limit to allow more movement
            if acc_magnitude > max_acceleration:
                acceleration = acceleration * (max_acceleration / acc_magnitude)
            
            # Update velocity with controlled damping
            velocity = self.vertex_velocities[i]
            effective_damping_factor = damping * (0.8 if spring_constant < 20.0 else 1.0)
            velocity = velocity * (1.0 - effective_damping_factor * dt) + acceleration * dt
            
            # Limit velocity for stability
            max_velocity = 25.0 if spring_constant < 20.0 else 20.0  # Increased velocity limits
            velocity_magnitude = np.linalg.norm(velocity)
            if velocity_magnitude > max_velocity:
                velocity = velocity * (max_velocity / velocity_magnitude)
            
            # Calculate proposed new position
            proposed_position = vertex + velocity * dt
            
            # GENTLE vessel constraint: only apply if significantly outside vessel
            if self._would_vertex_exit_vessel(proposed_position, buffer_factor=1.1):
                # Only apply gentle correction if well outside vessel
                vessel_center_2d = np.array([12.5, 7.5])  # Assuming domain center
                vertex_2d = proposed_position[1:3]  # Y, Z coordinates
                radial_direction = vertex_2d - vessel_center_2d
                radial_distance = np.linalg.norm(radial_direction)
                
                if radial_distance > 1e-6:
                    radial_unit = radial_direction / radial_distance
                    # Only apply correction if significantly outside vessel
                    max_radius = 5.5  # Increased allowed radius for gentler constraint
                    if radial_distance > max_radius:
                        # Apply gentle correction toward vessel center
                        correction_strength = 0.1  # Very gentle correction
                        correction_vector = -radial_unit * (radial_distance - max_radius) * correction_strength
                        proposed_position[1:3] += correction_vector
                        # Apply gentle velocity damping
                        velocity_2d = velocity[1:3]
                        radial_velocity = np.dot(velocity_2d, radial_unit)
                        if radial_velocity > 0:  # Outward velocity
                            velocity_2d -= radial_velocity * radial_unit * 0.2  # Gentle damping
                            velocity[1:3] = velocity_2d
            
            # Keep vertices within reasonable distance from center (but allow more freedom)
            max_distance = self.length * (1.3 if spring_constant < 20.0 else 1.1)  # Increased allowed distance
            position_magnitude = np.linalg.norm(proposed_position - self.center_position)
            if position_magnitude > max_distance:
                direction = (proposed_position - self.center_position) / position_magnitude
                proposed_position = self.center_position + direction * max_distance
                # Gentle velocity reduction when constrained
                velocity *= 0.8  # Gentler velocity reduction
            
            self.vertices[i] = proposed_position
            self.vertex_velocities[i] = velocity
        
        # Allow center drift for all simulations to maintain mesh coherence
        if spring_constant < 15.0:
            # Allow center drift to follow deformation for mesh stability
            vertex_count = len(self.vertices)
            if vertex_count > 10:  # If we have end caps, exclude them
                actual_center = np.mean(self.vertices[:-2], axis=0)
            else:
                actual_center = np.mean(self.vertices, axis=0)
            
            center_drift = actual_center - self.center_position
            
            # Allow reasonable center drift to maintain mesh coherence
            max_drift = self.length * 0.05  # Slightly increased drift allowance
            drift_magnitude = np.linalg.norm(center_drift)
            if drift_magnitude > max_drift:
                center_drift = center_drift * (max_drift / drift_magnitude)
            
            self.center_position += center_drift * 0.03  # Gentle center adjustment
    
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
        #print(f"VTK DEBUG: Writing sporozoite ID {self.id} to polydata with {len(self.vertices)} vertices")
        
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
            #if vertex_idx < 3:
                #print(f"  VTK DEBUG: Vertex {vertex_idx} assigned ID {self.id}")
        polydata.GetPointData().AddArray(id_array)
        
        # DEBUG: Verify the ID array was created correctly
        retrieved_id_array = polydata.GetPointData().GetArray("SporozoiteID")
        if retrieved_id_array:
            #print(f"  VTK DEBUG: ID array created with {retrieved_id_array.GetNumberOfTuples()} values")
            if retrieved_id_array.GetNumberOfTuples() > 0:
                first_id = retrieved_id_array.GetValue(0)
                #print(f"  VTK DEBUG: First ID value in array: {first_id}")
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
    
    @property
    def is_alive(self) -> bool:
        """Check if sporozoite is still alive"""
        return self.viability > 0.1