"""
Salivary Gland Simulation with LJ Propulsion Forces and Centerline Arc Nodes

This simulation implements sporozoite movement using:
- Centerline representation with arc nodes
- Lennard-Jones propulsion forces
- Salivary gland environment constraints
"""

import numpy as np
import matplotlib.pyplot as plt
import vtk
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple, Dict

# Add parent directory to path to import simulation_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from simulation_config import CenterlineSimulationConfigs
except ImportError:
    # Fallback configuration if import fails
    class CenterlineConfig:
        def __init__(self):
            self.CENTERLINE_SEGMENTS = 10
            self.STRETCH_STIFFNESS = 200.0  # Reduced from 1000
            self.BENDING_STIFFNESS = 200.0  # Reduced from 1000
            self.PROPULSIVE_FORCE_AMPLITUDE = 150.0  # Increased from 50
            self.CENTERLINE_DAMPING = 0.3  # Reduced from 0.8
            self.NOISE_STRENGTH = 0.0
            self.BOUNDARY_REPULSION_STRENGTH = 5.0
            self.VIABILITY_DECAY_RATE = 0.01
            self.TIME_STEP = 0.008
            self.MAX_TIME = 5.0
            self.OUTPUT_INTERVAL = 0.08
            self.SPOROZOITE_LENGTH_RANGE = (12.0, 18.0)
            self.SPOROZOITE_DIAMETER_RANGE = (1.0, 2.0)
            self.MOTILITY_RANGE = (0.6, 1.0)
    
    class CenterlineSimulationConfigs:
        @staticmethod
        def salivary_gland_environment():
            return CenterlineConfig()

class CenterlineArcNode:
    """Individual node along the sporozoite centerline arc"""
    
    def __init__(self, position: np.ndarray, node_id: int):
        self.position = position.copy()
        self.velocity = np.zeros(3)
        self.force = np.zeros(3)
        self.node_id = node_id
        
        # LJ force parameters - STRENGTHENED for better repulsion
        self.sigma = 1.0  # Increased from 0.5 - LJ length scale (μm)
        self.epsilon = 20.0  # Increased from 1.0 - Much stronger energy scale
        self.cutoff_distance = 3.0  # Increased from 2.0 - Longer range interactions (μm)


class SimpleSporozoiteMeshGeometry:
    """Simple mesh geometry generator for visualization - NO PHYSICS"""
    
    def __init__(self, sporozoite_id: int, center_position: np.ndarray, length: float, diameter: float):
        self.id = sporozoite_id
        self.center_position = center_position.copy()
        self.length = length
        self.diameter = diameter
        
        # Mesh parameters (simplified)
        self.n_segments = 8  # Longitudinal segments
        self.n_radial = 6    # Radial segments
        
        # Create mesh geometry
        self.vertices = None
        self.faces = None
        self.original_relative_positions = None
        
        self._create_simple_mesh()
    
    def _create_simple_mesh(self):
        """Create simple curved rod mesh - GEOMETRY MATCHING THE CIRCULAR ARC"""
        vertices = []
        faces = []
        
        # Generate curved spine MATCHING the actual circular arc geometry used by nodes
        # Instead of a sine curve, use the same arc geometry as the centerline nodes
        t_values = np.linspace(0, 1, self.n_segments)
        spine_points = []
        
        # FIXED: Use the same circular arc geometry as the centerline nodes
        # Arc parameters (matching SporozoiteCenterlineArc)
        sporozoite_length = self.length  # Total length of sporozoite
        arc_angle = np.pi / 2  # 90 degrees
        radius_of_curvature = sporozoite_length / arc_angle  # Same as nodes
        
        for i, t in enumerate(t_values):
            # Parameter along the 90-degree arc (0 to 1)
            # Angle within the 90-degree arc (-45° to +45° relative to direction)
            local_angle = (t - 0.5) * arc_angle
            
            # Position on the circular arc (centered at origin)
            x = radius_of_curvature * np.cos(local_angle) - radius_of_curvature  # Offset to center
            y = radius_of_curvature * np.sin(local_angle)
            z = 0
            spine_points.append([x, y, z])
        
        spine_points = np.array(spine_points)
        
        # Create cross-sections
        for i, spine_point in enumerate(spine_points):
            # Calculate local coordinate system
            if i == 0:
                tangent = spine_points[i+1] - spine_points[i] if i < len(spine_points) - 1 else np.array([1, 0, 0])
            elif i == len(spine_points) - 1:
                tangent = spine_points[i] - spine_points[i-1]
            else:
                tangent = spine_points[i+1] - spine_points[i-1]
            
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-8:
                tangent = tangent / tangent_norm
            else:
                tangent = np.array([1, 0, 0])
            
            # Create perpendicular vectors
            if abs(tangent[2]) < 0.9:
                normal = np.cross(tangent, [0, 0, 1])
            else:
                normal = np.cross(tangent, [1, 0, 0])
            
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-8:
                normal = normal / normal_norm
            else:
                normal = np.array([0, 1, 0])
            
            binormal = np.cross(tangent, normal)
            binormal_norm = np.linalg.norm(binormal)
            if binormal_norm > 1e-8:
                binormal = binormal / binormal_norm
            else:
                binormal = np.array([0, 0, 1])
            
            # Radius varies along length (tapered ends)
            radius_factor = np.sin(np.pi * i / (self.n_segments - 1))
            radius_factor = max(0.1, radius_factor)
            radius = self.diameter/2 * radius_factor
            
            # Create circular cross-section
            for j in range(self.n_radial):
                angle = 2 * np.pi * j / self.n_radial
                local_point = (normal * np.cos(angle) + binormal * np.sin(angle)) * radius
                vertex = spine_point + local_point  # Relative to origin
                vertices.append(vertex)
        
        # Create faces
        for i in range(self.n_segments - 1):
            for j in range(self.n_radial):
                curr_ring = i * self.n_radial
                next_ring = (i + 1) * self.n_radial
                curr_j = j
                next_j = (j + 1) % self.n_radial
                
                v1 = curr_ring + curr_j
                v2 = curr_ring + next_j
                v3 = next_ring + next_j
                v4 = next_ring + curr_j
                
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        # Add end caps
        center_start = len(vertices)
        vertices.append(spine_points[0])
        center_end = len(vertices)
        vertices.append(spine_points[-1])
        
        # Start cap
        for j in range(self.n_radial):
            next_j = (j + 1) % self.n_radial
            faces.append([center_start, j, next_j])
        
        # End cap
        last_ring_start = (self.n_segments - 1) * self.n_radial
        for j in range(self.n_radial):
            next_j = (j + 1) % self.n_radial
            faces.append([center_end, last_ring_start + next_j, last_ring_start + j])
        
        # Store geometry
        vertices = np.array(vertices)
        self.original_relative_positions = vertices.copy()  # Relative to origin
        self.vertices = vertices + self.center_position  # Global positions
        self.faces = np.array(faces)
    
    def update_position(self, new_center: np.ndarray):
        """Update mesh position - PURE GEOMETRY, NO PHYSICS"""
        self.center_position = new_center.copy()
        
        # Simply translate all vertices to new center position
        for i in range(len(self.vertices)):
            if i < len(self.original_relative_positions):
                relative_pos = self.original_relative_positions[i]
                # Position relative to new center (pure translation, no rotation)
                self.vertices[i] = new_center + relative_pos

class SporozoiteCenterlineArc:
    """Sporozoite as a 90-degree arc that moves forward along a circular path WITH 3D mesh body"""
    
    def __init__(self, sporozoite_id: int, initial_center: np.ndarray, config):
        self.id = sporozoite_id
        self.config = config
        
        # Physical properties
        self.length = np.random.uniform(*config.SPOROZOITE_LENGTH_RANGE)
        self.diameter = np.random.uniform(*config.SPOROZOITE_DIAMETER_RANGE)
        self.viability = np.random.uniform(0.8, 1.0)
        self.motility = np.random.uniform(*config.MOTILITY_RANGE)
        self.propulsion_strength = config.PROPULSIVE_FORCE_AMPLITUDE * self.motility
        
        # Arc geometry - 90 degrees of a circle
        self.arc_angle = np.pi / 2  # 90 degrees
        self.radius_of_curvature = self.length / self.arc_angle  # Path radius equals curvature radius
        self.num_nodes = config.CENTERLINE_SEGMENTS + 1
        
        # Circle parameters for this rod's path (each rod has unique circle)
        self.circle_center = initial_center[:2].copy()  # XY center for circular path
        self.z_position = initial_center[2]  # Fixed Z
        
        # Current position along the circle (angle parameter)
        self.current_angle = np.random.uniform(0, 2 * np.pi)  # Random start position
        
        # Speed of movement along the circle
        self.speed = 1.2 * self.motility  # Different speeds per rod
        
        # NEW: Force-driven motion parameters
        self.mass = 1.0  # Unit mass for each sporozoite
        self.center_velocity = np.zeros(2)  # Velocity of circle center (XY only)
        self.arc_constraint_stiffness = 500.0  # Stiffness to maintain arc shape
        self.center_damping = 0.5  # Damping for center movement
        
        # Create centerline nodes (for physics)
        self.nodes = self._create_arc_nodes()
        
        # Create 3D mesh body around centerline
        self._create_mesh_body()
        
        # Time tracking
        self.time = 0.0
    
    def _create_arc_nodes(self) -> List[CenterlineArcNode]:
        """Create nodes positioned as a 90-degree arc (for physics calculations)"""
        nodes = []
        
        for i in range(self.num_nodes):
            # Parameter along the 90-degree arc (0 to 1)
            t = i / (self.num_nodes - 1)
            
            # Angle within the 90-degree arc (-45° to +45° relative to current direction)
            local_angle = (t - 0.5) * self.arc_angle
            
            # Position of this node on the circle
            node_angle = self.current_angle + local_angle
            
            # Calculate node position on the circular path
            x = self.circle_center[0] + self.radius_of_curvature * np.cos(node_angle)
            y = self.circle_center[1] + self.radius_of_curvature * np.sin(node_angle)
            z = self.z_position
            
            position = np.array([x, y, z])
            nodes.append(CenterlineArcNode(position, i))
        
        return nodes
    
    def _create_mesh_body(self):
        """Create 3D mesh body around the centerline arc - DIRECTLY USING CENTERLINE NODES"""
        try:
            # Instead of using SimpleSporozoiteMeshGeometry, build mesh directly around centerline nodes
            self._build_mesh_around_centerline()
            
        except Exception as e:
            print(f"  Warning: Could not create mesh for sporozoite {self.id}: {e}")
            # Fallback: no mesh, just centerline
            self.mesh_vertices = None
            self.mesh_faces = None
            self.original_relative_positions = None
            self.simple_mesh = None
    
    def _build_mesh_around_centerline(self):
        """Build mesh geometry directly around the actual centerline nodes"""
        if len(self.nodes) < 2:
            raise ValueError("Need at least 2 centerline nodes to create mesh")
        
        # Mesh parameters
        n_radial = self.num_nodes  # Vertices per cross-section should match centreline nodes
        
        vertices = []
        faces = []
        
        # Get centerline positions
        spine_positions = np.array([node.position for node in self.nodes])
        
        # Calculate tangent vectors
        tangents = []
        for i in range(len(spine_positions)):
            if i == 0:
                tangent = spine_positions[i+1] - spine_positions[i] if len(spine_positions) > 1 else np.array([1, 0, 0])
            elif i == len(spine_positions) - 1:
                tangent = spine_positions[i] - spine_positions[i-1]
            else:
                tangent = spine_positions[i+1] - spine_positions[i-1]
            
            tangent_length = np.linalg.norm(tangent)
            if tangent_length > 1e-8:
                tangent = tangent / tangent_length
            else:
                tangent = np.array([1, 0, 0])
            
            tangents.append(tangent)
        
        # Create cross-sections around each centerline node
        for i, (spine_point, tangent) in enumerate(zip(spine_positions, tangents)):
            # Local coordinate system
            if abs(tangent[2]) < 0.9:
                normal = np.cross(tangent, np.array([0, 0, 1]))
            else:
                normal = np.cross(tangent, np.array([1, 0, 0]))
            
            normal_length = np.linalg.norm(normal)
            if normal_length > 1e-8:
                normal = normal / normal_length
            else:
                normal = np.array([0, 1, 0])
            
            binormal = np.cross(tangent, normal)
            binormal_length = np.linalg.norm(binormal)
            if binormal_length > 1e-8:
                binormal = binormal / binormal_length
            else:
                binormal = np.array([0, 0, 1])
            
            # Radius varies along length (tapered ends)
            t = i / (len(spine_positions) - 1) if len(spine_positions) > 1 else 0.5
            radius_factor = np.sin(np.pi * t)
            radius_factor = max(0.1, radius_factor)
            radius = self.diameter/2 * radius_factor
            
            # Create circular cross-section
            for j in range(n_radial):
                angle = 2 * np.pi * j / n_radial
                local_offset = (normal * np.cos(angle) + binormal * np.sin(angle)) * radius
                vertex = spine_point + local_offset
                vertices.append(vertex)
        
        # Create faces connecting cross-sections
        for i in range(len(spine_positions) - 1):
            for j in range(n_radial):
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
        
        # Store results
        self.mesh_vertices = np.array(vertices)
        self.mesh_faces = np.array(faces)
        self.n_radial = n_radial  # Store for visualization
        
        #print(f"  Created mesh: {len(self.mesh_vertices)} vertices, {len(self.mesh_faces)} faces around {len(self.nodes)} centerline nodes")

    def _update_mesh_positions(self):
        """Update mesh to follow centerline nodes - REBUILD from scratch each time"""
        if self.mesh_vertices is None or len(self.nodes) == 0:
            return
        
        # Simply rebuild the entire mesh around current centerline positions
        self._build_mesh_around_centerline()
    
    def update_motion(self, dt: float, domain_size: Tuple[float, float, float]):
        """Move the arc forward along its circular path with LJ force modifications"""
        # Update time
        self.time += dt
        
        # NEW: Use force-driven approach with arc constraints
        self.update_motion_force_driven_with_constraints(dt)
        
        # Apply boundary conditions to circle center
        self._apply_boundary_conditions(domain_size)
        
        # Update viability
        self.viability -= self.config.VIABILITY_DECAY_RATE * dt
    
    def update_motion_force_driven_with_constraints(self, dt: float):
        """Force-driven motion that maintains circular gliding characteristics while allowing path adaptation"""
        # 1. Calculate collective force on the entire rod (for center movement)
        total_external_force = np.zeros(3)
        for node in self.nodes:
            total_external_force += node.force
        
        center_of_mass_force = total_external_force / len(self.nodes)
        
        # 2. Move the circle center based on collective forces (XY only)
        # This allows the entire circular path to translate dynamically
        center_acceleration = center_of_mass_force[:2] / self.mass
        
        # Apply damping to prevent oscillations
        self.center_velocity *= (1.0 - self.center_damping * dt)
        self.center_velocity += center_acceleration * dt
        
        # Update circle center position
        self.circle_center += self.center_velocity * dt
        
        # 3. Continue kinematic motion along the (possibly moved) circular path
        self.current_angle += self.speed * dt
        self.current_angle = self.current_angle % (2 * np.pi)
        
        # 4. Update each node with constraint forces to maintain arc shape
        for i, node in enumerate(self.nodes):
            # Calculate ideal position on the circular arc
            t = i / (self.num_nodes - 1)
            local_angle = (t - 0.5) * self.arc_angle
            node_angle = self.current_angle + local_angle
            
            ideal_position = np.array([
                self.circle_center[0] + self.radius_of_curvature * np.cos(node_angle),
                self.circle_center[1] + self.radius_of_curvature * np.sin(node_angle),
                self.z_position
            ])
            
            # Calculate constraint force to maintain arc shape
            position_error = node.position - ideal_position
            constraint_force = -self.arc_constraint_stiffness * position_error
            
            # Total force: external LJ forces + arc constraint forces
            total_node_force = node.force + constraint_force
            
            # Update node with force-based motion
            node_acceleration = total_node_force / self.mass
            node.velocity += node_acceleration * dt
            node.position += node.velocity * dt
            
            # Calculate ideal kinematic velocity for reference
            ideal_vx = -self.radius_of_curvature * self.speed * np.sin(node_angle) + self.center_velocity[0]
            ideal_vy = self.radius_of_curvature * self.speed * np.cos(node_angle) + self.center_velocity[1]
            ideal_vz = 0.0
            ideal_velocity = np.array([ideal_vx, ideal_vy, ideal_vz])
            
            # Blend actual velocity toward ideal kinematic velocity (optional damping)
            velocity_damping = 0.1  # Small damping to prevent excessive deviation
            node.velocity = (1.0 - velocity_damping) * node.velocity + velocity_damping * ideal_velocity
        
        # Update mesh positions to match new node positions
        self._update_mesh_positions()
    
    def _update_node_positions_with_forces(self, dt: float):
        """Update node positions with kinematic motion + LJ force perturbations"""
        # First, calculate ideal kinematic positions
        ideal_positions = []
        for i, node in enumerate(self.nodes):
            # Parameter along the 90-degree arc (0 to 1)
            t = i / (self.num_nodes - 1)
            
            # Angle within the 90-degree arc (-45° to +45° relative to current direction)
            local_angle = (t - 0.5) * self.arc_angle
            
            # Position of this node on the circle
            node_angle = self.current_angle + local_angle
            
            # Calculate ideal kinematic position
            x = self.circle_center[0] + self.radius_of_curvature * np.cos(node_angle)
            y = self.circle_center[1] + self.radius_of_curvature * np.sin(node_angle)
            z = self.z_position
            
            ideal_position = np.array([x, y, z])
            ideal_positions.append(ideal_position)
            
            # Calculate kinematic velocity (tangent to circle)
            vx = -self.radius_of_curvature * self.speed * np.sin(node_angle)
            vy = self.radius_of_curvature * self.speed * np.cos(node_angle)
            vz = 0.0
            
            kinematic_velocity = np.array([vx, vy, vz])
            
            # Apply LJ force perturbation to position
            # Use simple Euler integration: v = v_kinematic + F*dt/mass, x = x + v*dt
            mass = 1.0  # Unit mass
            force_acceleration = node.force / mass
            
            # Update velocity: combine kinematic motion with force-induced motion
            node.velocity = kinematic_velocity + force_acceleration * dt
            
            # Update position: ideal kinematic position + small force-based displacement
            force_displacement = force_acceleration * dt * dt  # Simple integration
            
            # Limit the force displacement to prevent instability
            max_displacement = self.diameter / 4  # Max displacement per timestep
            displacement_magnitude = np.linalg.norm(force_displacement)
            if displacement_magnitude > max_displacement:
                force_displacement = force_displacement * (max_displacement / displacement_magnitude)
            
            # Final position: kinematic + force perturbation
            node.position = ideal_position + force_displacement
        
        # Update mesh positions to match new node positions
        self._update_mesh_positions()
    
    def _update_node_positions(self):
        """Original kinematic-only node position update (kept for compatibility)"""
        for i, node in enumerate(self.nodes):
            # Parameter along the 90-degree arc (0 to 1)
            t = i / (self.num_nodes - 1)
            
            # Angle within the 90-degree arc (-45° to +45° relative to current direction)
            local_angle = (t - 0.5) * self.arc_angle
            
            # Position of this node on the circle
            node_angle = self.current_angle + local_angle
            
            # Calculate new node position
            x = self.circle_center[0] + self.radius_of_curvature * np.cos(node_angle)
            y = self.circle_center[1] + self.radius_of_curvature * np.sin(node_angle)
            z = self.z_position
            
            node.position = np.array([x, y, z])
            
            # Calculate velocity (tangent to circle)
            vx = -self.radius_of_curvature * self.speed * np.sin(node_angle)
            vy = self.radius_of_curvature * self.speed * np.cos(node_angle)
            vz = 0.0
            
            node.velocity = np.array([vx, vy, vz])
            node.force.fill(0.0)  # No forces in this simple model
        
        # Update mesh positions to match new node positions
        self._update_mesh_positions()
    
    def _apply_boundary_conditions(self, domain_size: Tuple[float, float, float]):
        """Keep circle center within domain bounds"""
        for dim in range(2):  # Only XY
            if self.circle_center[dim] < self.radius_of_curvature:
                self.circle_center[dim] = self.radius_of_curvature
            elif self.circle_center[dim] > domain_size[dim] - self.radius_of_curvature:
                self.circle_center[dim] = domain_size[dim] - self.radius_of_curvature
    
    def get_center_position(self) -> np.ndarray:
        """Get center position of the arc (head node position)"""
        return self.nodes[0].position.copy()
    
    def calculate_lj_forces(self, other_sporozoites: List['SporozoiteCenterlineArc']):
        """Calculate Lennard-Jones repulsion forces between this sporozoite's nodes and other sporozoites"""
        # Clear forces first
        for node in self.nodes:
            node.force.fill(0.0)
        
        # Calculate repulsion with other sporozoites
        for other in other_sporozoites:
            if other.id == self.id:
                continue  # Skip self
            
            # Node-to-node LJ repulsion
            for my_node in self.nodes:
                for other_node in other.nodes:
                    # Calculate distance between nodes
                    r_vec = my_node.position - other_node.position
                    r_distance = np.linalg.norm(r_vec)
                    
                    # Skip if too far (beyond cutoff)
                    if r_distance > my_node.cutoff_distance:
                        continue
                    
                    # Skip if too close (avoid singularity)
                    if r_distance < 0.01:
                        continue
                    
                    # Effective radius - use diameter as repulsion distance
                    # Strong repulsion when nodes are within combined radius
                    effective_radius = (self.diameter + other.diameter) / 2
                    
                    # Pure repulsion LJ potential (only repulsive part)
                    # F = 12 * epsilon * ((sigma/r)^13 - (sigma/r)^7) * (1/r)
                    # But we want strong repulsion at effective_radius, so use effective_radius as sigma
                    
                    sigma_eff = effective_radius
                    epsilon_eff = my_node.epsilon
                    
                    # Normalized distance
                    sigma_over_r = sigma_eff / r_distance
                    sigma_over_r_6 = sigma_over_r ** 6
                    sigma_over_r_12 = sigma_over_r_6 ** 2
                    
                    # LJ force magnitude (repulsive only)
                    # We only use the repulsive part: F = 12*eps*(sigma/r)^12/r - 6*eps*(sigma/r)^6/r
                    # But since we want pure repulsion, we can modify this to be stronger
                    force_magnitude = 12.0 * epsilon_eff * (sigma_over_r_12 - 0.5 * sigma_over_r_6) / r_distance
                    
                    # Direction (away from other node)
                    if r_distance > 1e-10:
                        force_direction = r_vec / r_distance
                    else:
                        # Random direction if nodes are exactly overlapping
                        force_direction = np.random.randn(3)
                        force_direction = force_direction / np.linalg.norm(force_direction)
                    
                    # Apply force to this node
                    repulsion_force = force_magnitude * force_direction
                    my_node.force += repulsion_force
    
    def calculate_internal_forces(self):
        """No internal forces - motion is purely kinematic"""
        pass
    
    def is_viable(self) -> bool:
        """Check if sporozoite is still viable"""
        return self.viability > 0.1

    def to_vtk_polydata(self) -> vtk.vtkPolyData:
        """Convert sporozoite to VTK PolyData - includes both centerline and 3D mesh body"""
        
        # If we have a 3D mesh body, create a mesh polydata
        if self.mesh_vertices is not None and self.mesh_faces is not None:
            return self._create_mesh_polydata()
        else:
            # Fallback to centerline representation
            return self._create_centerline_polydata()
    
    def _create_mesh_polydata(self) -> vtk.vtkPolyData:
        """Create VTK polydata for the 3D mesh body WITH proper centerline lines"""
        polydata = vtk.vtkPolyData()
        
        # Add ALL points: mesh vertices + actual centerline node positions
        points = vtk.vtkPoints()
        
        # First add mesh vertices
        for vertex in self.mesh_vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
        
        # Then add actual centerline node positions as separate points
        centerline_start_idx = len(self.mesh_vertices)
        for node in self.nodes:
            points.InsertNextPoint(node.position[0], node.position[1], node.position[2])
        
        polydata.SetPoints(points)
        
        # Add mesh faces as polygons
        polys = vtk.vtkCellArray()
        valid_faces = 0
        
        for face in self.mesh_faces:
            # Validate face indices
            valid_face = True
            for vertex_id in face:
                if vertex_id < 0 or vertex_id >= len(self.mesh_vertices):
                    valid_face = False
                    break
            
            if valid_face and len(face) >= 3:
                # Create polygon
                poly = vtk.vtkPolygon()
                poly.GetPointIds().SetNumberOfIds(len(face))
                for i, vertex_id in enumerate(face):
                    poly.GetPointIds().SetId(i, int(vertex_id))
                polys.InsertNextCell(poly)
                valid_faces += 1
        
        polydata.SetPolys(polys)
        
        # Add PROPER centerline lines using the actual centerline node points
        lines = vtk.vtkCellArray()
        for i in range(len(self.nodes) - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, centerline_start_idx + i)
            line.GetPointIds().SetId(1, centerline_start_idx + i + 1)
            lines.InsertNextCell(line)
        polydata.SetLines(lines)
        
        # Add scalar data arrays (uniform across ALL points: mesh + centerline)
        total_points = len(self.mesh_vertices) + len(self.nodes)
        
        # Viability (primary scalar for coloring)
        viability_array = vtk.vtkFloatArray()
        viability_array.SetName("Viability")
        viability_array.SetNumberOfTuples(total_points)
        for i in range(total_points):
            viability_array.SetValue(i, float(self.viability))
        polydata.GetPointData().SetScalars(viability_array)
        
        # Motility
        motility_array = vtk.vtkFloatArray()
        motility_array.SetName("Motility")
        motility_array.SetNumberOfTuples(total_points)
        for i in range(total_points):
            motility_array.SetValue(i, float(self.motility))
        polydata.GetPointData().AddArray(motility_array)
        
        # Sporozoite ID
        id_array = vtk.vtkIntArray()
        id_array.SetName("SporozoiteID")
        id_array.SetNumberOfTuples(total_points)
        for i in range(total_points):
            id_array.SetValue(i, int(self.id))
        polydata.GetPointData().AddArray(id_array)
        
        # Length and diameter
        length_array = vtk.vtkFloatArray()
        length_array.SetName("Length")
        length_array.SetNumberOfTuples(total_points)
        for i in range(total_points):
            length_array.SetValue(i, float(self.length))
        polydata.GetPointData().AddArray(length_array)
        
        diameter_array = vtk.vtkFloatArray()
        diameter_array.SetName("Diameter")
        diameter_array.SetNumberOfTuples(total_points)
        for i in range(total_points):
            diameter_array.SetValue(i, float(self.diameter))
        polydata.GetPointData().AddArray(diameter_array)
        
        return polydata
    
    def _create_centerline_polydata(self) -> vtk.vtkPolyData:
        """Create VTK polydata for centerline (fallback if mesh creation failed)"""
        polydata = vtk.vtkPolyData()
        
        # Create points from nodes
        points = vtk.vtkPoints()
        for node in self.nodes:
            points.InsertNextPoint(node.position[0], node.position[1], node.position[2])
        polydata.SetPoints(points)
        
        # Create lines connecting nodes (centerline)
        lines = vtk.vtkCellArray()
        for i in range(len(self.nodes) - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(line)
        polydata.SetLines(lines)
        
        # Add scalar data arrays
        # Viability
        viability_array = vtk.vtkFloatArray()
        viability_array.SetName("Viability")
        viability_array.SetNumberOfTuples(len(self.nodes))
        for i in range(len(self.nodes)):
            viability_array.SetValue(i, float(self.viability))
        polydata.GetPointData().SetScalars(viability_array)
        
        # Motility
        motility_array = vtk.vtkFloatArray()
        motility_array.SetName("Motility")
        motility_array.SetNumberOfTuples(len(self.nodes))
        for i in range(len(self.nodes)):
            motility_array.SetValue(i, float(self.motility))
        polydata.GetPointData().AddArray(motility_array)
        
        # Node velocities (vector)
        velocity_array = vtk.vtkFloatArray()
        velocity_array.SetName("Velocity")
        velocity_array.SetNumberOfComponents(3)
        velocity_array.SetNumberOfTuples(len(self.nodes))
        for i, node in enumerate(self.nodes):
            velocity_array.SetTuple3(i, 
                                   float(node.velocity[0]),
                                   float(node.velocity[1]), 
                                   float(node.velocity[2]))
        polydata.GetPointData().SetVectors(velocity_array)
        
        return polydata


class SalivaryGlandLJSimulation:
    """Salivary gland simulation with LJ forces and centerline arc nodes"""
    
    def __init__(self, num_sporozoites: int = 15, 
                 domain_size: Tuple[float, float, float] = (80.0, 80.0, 15.0),
                 config=None):
        
        self.num_sporozoites = num_sporozoites
        self.domain_size = domain_size
        self.config = config or CenterlineSimulationConfigs.salivary_gland_environment()
        
        # Initialize sporozoites
        print(f"Creating {num_sporozoites} sporozoites with centerline arc nodes...")
        self.sporozoites: List[SporozoiteCenterlineArc] = []
        self._initialize_sporozoites()
        
        # Calculate volume fraction for output directory naming
        self.volume_fraction = self._calculate_volume_fraction()
        
        # Simulation parameters
        self.time = 0.0
        self.dt = self.config.TIME_STEP
        self.max_time = self.config.MAX_TIME
        self.output_interval = self.config.OUTPUT_INTERVAL
        self.last_output_time = 0.0
        
        # Output setup with volume fraction in name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        volume_fraction_str = f"{self.volume_fraction:.3f}".replace('.', 'p')  # Replace . with p for filename
        self.output_dir = f"/Users/s2526994/Desktop/SporozoiteInfect/vtk_output/{volume_fraction_str}_circular_gliding_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Simulation initialized with {len(self.sporozoites)} sporozoites")
        print(f"Domain size: {domain_size}")
        print(f"Volume fraction: {self.volume_fraction:.6f}")
        print(f"Output directory: {self.output_dir}")
    
    def _calculate_volume_fraction(self) -> float:
        """Calculate the volume fraction of sporozoites in the domain"""
        # Calculate domain volume
        box_volume = self.domain_size[0] * self.domain_size[1] * self.domain_size[2]
        
        # Calculate total sporozoite volume
        sporozoite_volume = 0.0
        for sporozoite in self.sporozoites:
            if sporozoite.mesh_vertices is not None and sporozoite.mesh_faces is not None:
                # Simple approximation: cylinder volume = π * r² * h
                radius = sporozoite.diameter / 2
                length = sporozoite.length
                cylinder_volume = np.pi * radius**2 * length
                sporozoite_volume += cylinder_volume
            else:
                # Fallback calculation if mesh failed
                radius = sporozoite.diameter / 2
                length = sporozoite.length
                cylinder_volume = np.pi * radius**2 * length
                sporozoite_volume += cylinder_volume
        
        return sporozoite_volume / box_volume
    
    def _initialize_sporozoites(self):
        """Create initial sporozoites in salivary gland injection area"""
        # Injection zone (upper portion of domain)
        injection_center = np.array([
            self.domain_size[0] / 2,
            self.domain_size[1] / 2, 
            self.domain_size[2] * 0.8
        ])
        injection_radius = min(self.domain_size[:2]) / 4
        
        for i in range(self.num_sporozoites):
            # Random position in injection zone
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, injection_radius)
            
            offset = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                np.random.uniform(-2.0, 2.0)
            ])
            
            initial_position = injection_center + offset
            
            # Ensure within bounds
            for dim in range(3):
                initial_position[dim] = np.clip(
                    initial_position[dim], 
                    2.0, 
                    self.domain_size[dim] - 2.0
                )
            
            sporozoite = SporozoiteCenterlineArc(i, initial_position, self.config)
            self.sporozoites.append(sporozoite)
            
            print(f"  Sporozoite {i}: center at {initial_position}, "
                  f"length {sporozoite.length:.2f} μm, {sporozoite.num_nodes} nodes")
    
    def simulation_step(self):
        """Perform one simulation time step"""
        active_sporozoites = []
        
        # Calculate forces for each sporozoite
        for sporozoite in self.sporozoites:
            if sporozoite.is_viable():
                # Calculate LJ forces with other sporozoites
                sporozoite.calculate_lj_forces(self.sporozoites)
                
                # Calculate internal forces
                sporozoite.calculate_internal_forces()
                
                # Update motion
                sporozoite.update_motion(self.dt, self.domain_size)
                
                active_sporozoites.append(sporozoite)
        
        # Update sporozoite list
        self.sporozoites = active_sporozoites
    
    def write_output(self, step_number: int):
        """Write VTK files and state for current simulation step - WITH SEPARATE CENTERLINE TIME SERIES"""
        # Write main sporozoite data (with bounding box included)
        self._write_sporozoite_vtk(step_number)
        
        # Write separate centerline arc time series for comparison
        self._write_centerline_arcs_vtk(step_number)
        
        # Initialize time series collection files (only once at the beginning)
        if step_number == 0:
            self._write_time_series_collection_files()
        
        # Update both collections
        self._update_sporozoites_time_series_collection(step_number)
        self._update_centerline_arcs_time_series_collection(step_number)
        
        # Write ParaView state file (only at the end)
        if step_number == 0:
            self._write_paraview_state_file()

    def _write_sporozoite_vtk(self, step_number: int):
        """Write sporozoite centerline arcs to VTK with proper time information for animation"""
        multiblock = vtk.vtkMultiBlockDataSet()
        
        # Calculate total number of blocks: sporozoites + their components + bounding box
        total_blocks = len(self.sporozoites) * 3 + 1  # 3 components per sporozoite + bounding box
        multiblock.SetNumberOfBlocks(total_blocks)
        
        block_index = 0
        
        # Add bounding box as first block
        bbox_polydata = self._create_bounding_box_vtk()
        multiblock.SetBlock(block_index, bbox_polydata)
        multiblock.GetMetaData(block_index).Set(vtk.vtkCompositeDataSet.NAME(), "BoundingBox")
        block_index += 1
        
        for i, sporozoite in enumerate(self.sporozoites):
            # Create separate VTK components for this sporozoite
            
            # 1. Mesh body (full polydata)
            mesh_polydata = sporozoite.to_vtk_polydata()
            if mesh_polydata.GetNumberOfPoints() > 0:
                # Add time information
                time_array = vtk.vtkDoubleArray()
                time_array.SetName("TimeValue")
                time_array.SetNumberOfTuples(1)
                time_array.SetValue(0, self.time)
                mesh_polydata.GetFieldData().AddArray(time_array)
                
                multiblock.SetBlock(block_index, mesh_polydata)
                multiblock.GetMetaData(block_index).Set(vtk.vtkCompositeDataSet.NAME(), f"Sporozoite_{sporozoite.id}_Mesh")
                block_index += 1
            
            # 2. Center point
            center_polydata = self._create_center_point_vtk(sporozoite)
            multiblock.SetBlock(block_index, center_polydata)
            multiblock.GetMetaData(block_index).Set(vtk.vtkCompositeDataSet.NAME(), f"Sporozoite_{sporozoite.id}_Center")
            block_index += 1
            
            # 3. Linear representation (centerline arc)
            linear_polydata = self._create_centerline_arc_vtk(sporozoite)
            multiblock.SetBlock(block_index, linear_polydata)
            multiblock.GetMetaData(block_index).Set(vtk.vtkCompositeDataSet.NAME(), f"Sporozoite_{sporozoite.id}_Linear")
            block_index += 1
            
            print(f"    Added sporozoite {sporozoite.id} with 3 components to multiblock")
        
        # Write to file with proper format settings
        filename = os.path.join(self.output_dir, f"sporozoites_{step_number:04d}.vtm")
        
        writer = vtk.vtkXMLMultiBlockDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(multiblock)
        writer.SetDataModeToAscii()  # Use ASCII format for better compatibility
        writer.SetCompressorTypeToNone()  # Disable compression to avoid binary issues
        writer.Write()
        
        print(f"VTK DEBUG: Successfully wrote {filename} with {block_index} blocks")

    def _write_centerline_arcs_vtk(self, step_number: int):
        """Write separate VTK file containing ONLY the centerline arcs for easy comparison"""
        multiblock = vtk.vtkMultiBlockDataSet()
        multiblock.SetNumberOfBlocks(len(self.sporozoites))
        
        for i, sporozoite in enumerate(self.sporozoites):
            # Create pure centerline arc polydata
            centerline_polydata = vtk.vtkPolyData()
            
            # Add centerline node points
            points = vtk.vtkPoints()
            for node in sporozoite.nodes:
                points.InsertNextPoint(node.position[0], node.position[1], node.position[2])
            centerline_polydata.SetPoints(points)
            
            # Create lines connecting the nodes
            lines = vtk.vtkCellArray()
            for j in range(len(sporozoite.nodes) - 1):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, j)
                line.GetPointIds().SetId(1, j + 1)
                lines.InsertNextCell(line)
            centerline_polydata.SetLines(lines)
            
            # Add time information
            time_array = vtk.vtkDoubleArray()
            time_array.SetName("TimeValue")
            time_array.SetNumberOfTuples(1)
            time_array.SetValue(0, self.time)
            centerline_polydata.GetFieldData().AddArray(time_array)
            
            # Add sporozoite properties for all nodes
            num_nodes = len(sporozoite.nodes)
            
            # Viability (primary scalar for coloring)
            viability_array = vtk.vtkFloatArray()
            viability_array.SetName("Viability")
            viability_array.SetNumberOfTuples(num_nodes)
            for k in range(num_nodes):
                viability_array.SetValue(k, float(sporozoite.viability))
            centerline_polydata.GetPointData().SetScalars(viability_array)
            
            # Sporozoite ID
            id_array = vtk.vtkIntArray()
            id_array.SetName("SporozoiteID")
            id_array.SetNumberOfTuples(num_nodes)
            for k in range(num_nodes):
                id_array.SetValue(k, int(sporozoite.id))
            centerline_polydata.GetPointData().AddArray(id_array)
            
            # Node positions as individual components (for detailed analysis)
            x_pos_array = vtk.vtkFloatArray()
            x_pos_array.SetName("NodePositionX")
            x_pos_array.SetNumberOfTuples(num_nodes)
            y_pos_array = vtk.vtkFloatArray()
            y_pos_array.SetName("NodePositionY")
            y_pos_array.SetNumberOfTuples(num_nodes)
            z_pos_array = vtk.vtkFloatArray()
            z_pos_array.SetName("NodePositionZ")
            z_pos_array.SetNumberOfTuples(num_nodes)
            
            for k, node in enumerate(sporozoite.nodes):
                x_pos_array.SetValue(k, float(node.position[0]))
                y_pos_array.SetValue(k, float(node.position[1]))
                z_pos_array.SetValue(k, float(node.position[2]))
            
            centerline_polydata.GetPointData().AddArray(x_pos_array)
            centerline_polydata.GetPointData().AddArray(y_pos_array)
            centerline_polydata.GetPointData().AddArray(z_pos_array)
            
            # Node velocities as vectors
            velocity_array = vtk.vtkFloatArray()
            velocity_array.SetName("NodeVelocity")
            velocity_array.SetNumberOfComponents(3)
            velocity_array.SetNumberOfTuples(num_nodes)
            for k, node in enumerate(sporozoite.nodes):
                velocity_array.SetTuple3(k, 
                                       float(node.velocity[0]),
                                       float(node.velocity[1]), 
                                       float(node.velocity[2]))
            centerline_polydata.GetPointData().SetVectors(velocity_array)
            
            # Arc geometry parameters for verification
            arc_radius_array = vtk.vtkFloatArray()
            arc_radius_array.SetName("ArcRadius")
            arc_radius_array.SetNumberOfTuples(num_nodes)
            arc_angle_array = vtk.vtkFloatArray()
            arc_angle_array.SetName("CurrentAngle")
            arc_angle_array.SetNumberOfTuples(num_nodes)
            
            for k in range(num_nodes):
                arc_radius_array.SetValue(k, float(sporozoite.radius_of_curvature))
                arc_angle_array.SetValue(k, float(sporozoite.current_angle))
            
            centerline_polydata.GetPointData().AddArray(arc_radius_array)
            centerline_polydata.GetPointData().AddArray(arc_angle_array)
            
            # Add to multiblock
            multiblock.SetBlock(i, centerline_polydata)
            multiblock.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), f"CenterlineArc_{sporozoite.id}")
        
        # Write centerline arcs file
        filename = os.path.join(self.output_dir, f"centerline_arcs_{step_number:04d}.vtm")
        
        writer = vtk.vtkXMLMultiBlockDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(multiblock)
        writer.SetDataModeToAscii()
        writer.SetCompressorTypeToNone()
        writer.Write()
        
        print(f"VTK DEBUG: Successfully wrote centerline arcs {filename} with {len(self.sporozoites)} arcs")

    def _write_time_series_collection_files(self):
        """Write ParaView time series collection files for both sporozoites and centerline arcs"""
        # Create sporozoites time series collection file
        sporozoites_collection_file = os.path.join(self.output_dir, "sporozoites_timeseries.pvd")
        
        with open(sporozoites_collection_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            f.write('    <!-- Sporozoite time series data will be added here -->\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        # Create centerline arcs time series collection file
        centerline_collection_file = os.path.join(self.output_dir, "centerline_arcs_timeseries.pvd")
        
        with open(centerline_collection_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            f.write('    <!-- Centerline arc time series data will be added here -->\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        # Store the collection file paths for updates
        self.sporozoites_collection_file_path = sporozoites_collection_file
        self.centerline_collection_file_path = centerline_collection_file
        
        print(f"Created time series collection files:")
        print(f"  Sporozoites: {sporozoites_collection_file}")
        print(f"  Centerline Arcs: {centerline_collection_file}")

    def _update_sporozoites_time_series_collection(self, step_number: int):
        """Update the sporozoites time series collection file with new time step"""
        if not hasattr(self, 'sporozoites_collection_file_path'):
            return
            
        # Read existing content
        try:
            with open(self.sporozoites_collection_file_path, 'r') as f:
                lines = f.readlines()
        except:
            return
        
        # Find the insertion point (before </Collection>)
        insert_index = -1
        for i, line in enumerate(lines):
            if '</Collection>' in line:
                insert_index = i
                break
        
        if insert_index == -1:
            return
        
        # Create the new entry for sporozoites
        vtm_file = f"sporozoites_{step_number:04d}.vtm"
        new_entry = f'    <DataSet timestep="{self.time:.6f}" group="" part="0" file="{vtm_file}"/>\n'
        
        # Insert the new entry
        lines.insert(insert_index, new_entry)
        
        # Write back the updated file
        with open(self.sporozoites_collection_file_path, 'w') as f:
            for line in lines:
                if '<!-- Sporozoite time series data will be added here -->' not in line:
                    f.write(line)

    def _update_centerline_arcs_time_series_collection(self, step_number: int):
        """Update the centerline arcs time series collection file with new time step"""
        if not hasattr(self, 'centerline_collection_file_path'):
            return
            
        # Read existing content
        try:
            with open(self.centerline_collection_file_path, 'r') as f:
                lines = f.readlines()
        except:
            return
        
        # Find the insertion point (before </Collection>)
        insert_index = -1
        for i, line in enumerate(lines):
            if '</Collection>' in line:
                insert_index = i
                break
        
        if insert_index == -1:
            return
        
        # Create the new entry for centerline arcs
        vtm_file = f"centerline_arcs_{step_number:04d}.vtm"
        new_entry = f'    <DataSet timestep="{self.time:.6f}" group="" part="0" file="{vtm_file}"/>\n'
        
        # Insert the new entry
        lines.insert(insert_index, new_entry)
        
        # Write back the updated file
        with open(self.centerline_collection_file_path, 'w') as f:
            for line in lines:
                if '<!-- Centerline arc time series data will be added here -->' not in line:
                    f.write(line)

    def _write_paraview_state_file(self):
        """Write ParaView state file for automatic setup of salivary gland visualization"""
        state_file = os.path.join(self.output_dir, "salivary_gland_visualization_setup.pvsm")
        
        state_content = f'''<?xml version="1.0"?>
<ParaViewState version="5.9.0">
  <ServerManagerState>
    <!-- Sporozoite centerline reader -->
    <Proxy group="sources" type="XMLMultiBlockDataReader" id="100">
      <Property name="FileName">
        <Element index="0" value="{os.path.abspath(self.output_dir)}/sporozoites_*.vtm"/>
      </Property>
    </Proxy>
    
    <!-- Salivary gland environment reader -->
    <Proxy group="sources" type="XMLStructuredGridReader" id="200">
      <Property name="FileName">
        <Element index="0" value="{os.path.abspath(self.output_dir)}/salivary_gland_env_*.vts"/>
      </Property>
    </Proxy>
    
    <!-- Sporozoite representation (tubes for centerlines) -->
    <Proxy group="representations" type="GeometryRepresentation" id="101">
      <Property name="Input" proxy="100"/>
      <Property name="ColorArrayName">
        <Element index="0" value="Viability"/>
      </Property>
      <Property name="Representation">
        <Element index="0" value="Surface"/>
      </Property>
    </Proxy>
    
    <!-- Environment field representation -->
    <Proxy group="representations" type="UniformGridRepresentation" id="201">
      <Property name="Input" proxy="200"/>
      <Property name="ColorArrayName">
        <Element index="0" value="SalivaryGlandDensity"/>
      </Property>
      <Property name="Representation">
        <Element index="0" value="Volume"/>
      </Property>
    </Proxy>
    
    <!-- Animation settings -->
    <Proxy group="animation" type="AnimationScene" id="300">
      <Property name="PlayMode">
        <Element index="0" value="Sequence"/>
      </Property>
      <Property name="Duration">
        <Element index="0" value="40"/>
      </Property>
    </Proxy>
    
    <!-- Camera settings for salivary gland view -->
    <Proxy group="views" type="RenderView" id="400">
      <Property name="CameraPosition">
        <Element index="0" value="{self.domain_size[0] * 1.5}"/>
        <Element index="1" value="{self.domain_size[1] * 1.5}"/>
        <Element index="2" value="{self.domain_size[2] * 2}"/>
      </Property>
      <Property name="CameraFocalPoint">
        <Element index="0" value="{self.domain_size[0] / 2}"/>
        <Element index="1" value="{self.domain_size[1] / 2}"/>
        <Element index="2" value="{self.domain_size[2] / 2}"/>
      </Property>
    </Proxy>
  </ServerManagerState>
</ParaViewState>'''
        
        with open(state_file, 'w') as f:
            f.write(state_content)
        
        print(f"ParaView state file created: {state_file}")
    
    def _create_bounding_box_vtk(self) -> vtk.vtkPolyData:
        """Create bounding box for domain visualization"""
        polydata = vtk.vtkPolyData()
        
        # Create bounding box vertices
        points = vtk.vtkPoints()
        # Bottom face (z=0)
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(self.domain_size[0], 0, 0)
        points.InsertNextPoint(self.domain_size[0], self.domain_size[1], 0)
        points.InsertNextPoint(0, self.domain_size[1], 0)
        # Top face (z=domain_size[2])
        points.InsertNextPoint(0, 0, self.domain_size[2])
        points.InsertNextPoint(self.domain_size[0], 0, self.domain_size[2])
        points.InsertNextPoint(self.domain_size[0], self.domain_size[1], self.domain_size[2])
        points.InsertNextPoint(0, self.domain_size[1], self.domain_size[2])
        
        polydata.SetPoints(points)
        
        # Create wireframe lines for bounding box
        lines = vtk.vtkCellArray()
        # Bottom face edges
        box_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        for edge in box_edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)
        
        polydata.SetLines(lines)
        return polydata
    
    def _create_center_point_vtk(self, sporozoite: SporozoiteCenterlineArc) -> vtk.vtkPolyData:
        """Create center point representation for sporozoite"""
        polydata = vtk.vtkPolyData()
        
        # Create single point at sporozoite center
        points = vtk.vtkPoints()
        center_pos = sporozoite.get_center_position()
        points.InsertNextPoint(center_pos[0], center_pos[1], center_pos[2])
        polydata.SetPoints(points)
        
        # Create vertex cell
        vertices = vtk.vtkCellArray()
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, 0)
        vertices.InsertNextCell(vertex)
        polydata.SetVerts(vertices)
        
        # Add scalar data
        viability_array = vtk.vtkFloatArray()
        viability_array.SetName("Viability")
        viability_array.SetNumberOfTuples(1)
        viability_array.SetValue(0, float(sporozoite.viability))
        polydata.GetPointData().SetScalars(viability_array)
        
        return polydata
    
    def _create_centerline_arc_vtk(self, sporozoite: SporozoiteCenterlineArc) -> vtk.vtkPolyData:
        """Create centerline arc representation for sporozoite"""
        polydata = vtk.vtkPolyData()
        
        # Create points from nodes
        points = vtk.vtkPoints()
        for node in sporozoite.nodes:
            points.InsertNextPoint(node.position[0], node.position[1], node.position[2])
        polydata.SetPoints(points)
        
        # Create lines connecting nodes
        lines = vtk.vtkCellArray()
        for i in range(len(sporozoite.nodes) - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(line)
        polydata.SetLines(lines)
        
        # Add scalar data
        num_nodes = len(sporozoite.nodes)
        
        # Viability
        viability_array = vtk.vtkFloatArray()
        viability_array.SetName("Viability")
        viability_array.SetNumberOfTuples(num_nodes)
        for i in range(num_nodes):
            viability_array.SetValue(i, float(sporozoite.viability))
        polydata.GetPointData().SetScalars(viability_array)
        
        # Node velocities
        velocity_array = vtk.vtkFloatArray()
        velocity_array.SetName("NodeVelocity")
        velocity_array.SetNumberOfComponents(3)
        velocity_array.SetNumberOfTuples(num_nodes)
        for i, node in enumerate(sporozoite.nodes):
            velocity_array.SetTuple3(i, 
                                   float(node.velocity[0]),
                                   float(node.velocity[1]), 
                                   float(node.velocity[2]))
        polydata.GetPointData().SetVectors(velocity_array)
        
        return polydata

    def run_simulation(self):
        """Run the complete LJ-based salivary gland simulation"""
        print(f"\nStarting LJ-based salivary gland simulation...")
        print("-" * 60)
        
        step_count = 0
        output_count = 0
        
        # Initial output
        self.write_output(output_count)
        output_count += 1
        self.last_output_time = self.time
        
        start_time = time.time()
        
        while (self.time < self.max_time and 
               len(self.sporozoites) > 0 and 
               step_count < 10000):
            
            # Perform simulation step
            self.simulation_step()
            self.time += self.dt
            step_count += 1
            
            # Output at intervals
            if self.time - self.last_output_time >= self.output_interval:
                self.write_output(output_count)
                output_count += 1
                self.last_output_time = self.time
                
                # Progress report
                elapsed = time.time() - start_time
                print(f"Time: {self.time:6.2f}s | "
                      f"Active: {len(self.sporozoites):3d} | "
                      f"Elapsed: {elapsed:6.1f}s")
        
        # Final output
        self.write_output(output_count)
        
        # Summary
        total_time = time.time() - start_time
        print(f"\nSimulation completed!")
        print(f"Simulation time: {self.time:.2f}s")
        print(f"Computation time: {total_time:.1f}s")
        print(f"Final active sporozoites: {len(self.sporozoites)}")
        print(f"Results saved to: {self.output_dir}")
        
        return self.output_dir


def main():
    """Main function to run the salivary gland LJ simulation"""
    print("Salivary Gland Simulation with LJ Forces and Centerline Arc Nodes")
    print("=" * 70)
    
    # Create and run simulation
    sim = SalivaryGlandLJSimulation(
        num_sporozoites=10,
        domain_size=(50.0, 50.0, 10.0)
    )
    
    # Estimate volume fraction
    box_volume = sim.domain_size[0] * sim.domain_size[1] * sim.domain_size[2]
    sporozoite_volume = 0.0

    for sporozoite in sim.sporozoites:
        if sporozoite.mesh_vertices is not None and sporozoite.mesh_faces is not None:
            # Calculate volume of the sporozoite mesh using tetrahedrons
            vertices = sporozoite.mesh_vertices
            faces = sporozoite.mesh_faces
            for face in faces:
                if len(face) >= 3:
                    # Use the first vertex as the origin for the tetrahedron
                    v0 = vertices[face[0]]
                    v1 = vertices[face[1]]
                    v2 = vertices[face[2]]
                    tetra_volume = np.abs(np.dot(v0, np.cross(v1, v2))) / 6.0
                    sporozoite_volume += tetra_volume

    volume_fraction = sporozoite_volume / box_volume
    print(f"Estimated volume fraction: {volume_fraction:.6f}")
    output_dir = sim.run_simulation()
    
    print(f"\nSimulation complete! Check results in: {output_dir}")


if __name__ == "__main__":
    main()