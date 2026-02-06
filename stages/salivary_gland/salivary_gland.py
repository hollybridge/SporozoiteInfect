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
            self.STRETCH_STIFFNESS = 1000.0
            self.BENDING_STIFFNESS = 1000.0
            self.PROPULSIVE_FORCE_AMPLITUDE = 50.0
            self.CENTERLINE_DAMPING = 0.8
            self.NOISE_STRENGTH = 0.0
            self.BOUNDARY_REPULSION_STRENGTH = 5.0
            self.VIABILITY_DECAY_RATE = 0.01
            self.TIME_STEP = 0.008
            self.MAX_TIME = 12.0
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
        
        # LJ force parameters
        self.sigma = 0.5  # LJ length scale (μm)
        self.epsilon = 1.0  # LJ energy scale
        self.cutoff_distance = 2.0  # Cutoff for LJ interactions (μm)

class SporozoiteCenterlineArc:
    """Sporozoite represented as centerline with arc nodes"""
    
    def __init__(self, sporozoite_id: int, initial_center: np.ndarray, config):
        self.id = sporozoite_id
        self.config = config
        
        # Motion parameters (initialize BEFORE creating nodes)
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.undulation_phase = np.random.uniform(0, 2 * np.pi)
        
        # Physical properties
        self.length = np.random.uniform(*config.SPOROZOITE_LENGTH_RANGE)
        self.diameter = np.random.uniform(*config.SPOROZOITE_DIAMETER_RANGE)
        self.viability = np.random.uniform(0.8, 1.0)
        self.motility = np.random.uniform(*config.MOTILITY_RANGE)
        self.propulsion_strength = config.PROPULSIVE_FORCE_AMPLITUDE * self.motility
        
        # Initialize centerline arc nodes
        self.nodes = self._create_centerline_arc_nodes(initial_center)
        self.num_nodes = len(self.nodes)
        
        # Store rest curvature for bending forces
        self.rest_curvature = self._calculate_initial_curvature()
        
        # LJ force parameters
        self.lj_strength = 0.1  # 1 from C code
        self.interaction_width = 0.3  # WIDTH parameter from C code
    
    def _calculate_initial_curvature(self) -> List[np.ndarray]:
        """Calculate and store the initial curvature at each node for rest state"""
        rest_curvature = []
        
        for i in range(len(self.nodes)):
            if i == 0 or i == len(self.nodes) - 1:
                # End nodes have zero curvature
                rest_curvature.append(np.zeros(3))
            else:
                # Calculate initial curvature (second derivative)
                prev_pos = self.nodes[i - 1].position
                curr_pos = self.nodes[i].position
                next_pos = self.nodes[i + 1].position
                
                curvature_vec = prev_pos - 2 * curr_pos + next_pos
                rest_curvature.append(curvature_vec.copy())
        
        return rest_curvature
    
    def _create_centerline_arc_nodes(self, center: np.ndarray) -> List[CenterlineArcNode]:
        """Create nodes along centerline with 90-degree arc curvature to match circular propulsion"""
        nodes = []
        num_nodes = self.config.CENTERLINE_SEGMENTS + 1
        
        # Create 90-degree arc shape to match circular propulsion
        # The rod curves through 90 degrees (π/2 radians) from head to tail
        arc_angle = np.pi / 2  # 90 degrees in radians
        
        # Calculate radius of curvature based on sporozoite length
        # For a 90-degree arc: arc_length = radius * angle
        # So: radius = arc_length / angle = length / (π/2)
        radius_of_curvature = self.length / arc_angle
        
        for i in range(num_nodes):
            # Parameter along centerline (0 to 1)
            t = i / (num_nodes - 1)
            
            # Angle along the 90-degree arc (from -π/4 to +π/4 for symmetry)
            # This centers the arc so the middle node is at the "center" of the curve
            theta = (t - 0.5) * arc_angle  # Ranges from -π/4 to +π/4
            
            # Position along the 90-degree arc
            # Use parametric circle equations
            x_offset = radius_of_curvature * np.sin(theta)
            y_offset = radius_of_curvature * (1.0 - np.cos(theta))  # Offset so arc starts at origin
            z_offset = 0.0
            
            # Center the arc around the origin
            y_offset -= radius_of_curvature * (1.0 - np.cos(-arc_angle/2))  # Adjust for centering
            
            # Apply initial rotation around Z-axis
            cos_dir = np.cos(self.direction)
            sin_dir = np.sin(self.direction)
            
            rotated_x = x_offset * cos_dir - y_offset * sin_dir
            rotated_y = x_offset * sin_dir + y_offset * cos_dir
            
            position = center + np.array([rotated_x, rotated_y, z_offset])
            nodes.append(CenterlineArcNode(position, i))
        
        return nodes
    
    def get_center_position(self) -> np.ndarray:
        """Get center position of the sporozoite"""
        positions = np.array([node.position for node in self.nodes])
        return np.mean(positions, axis=0)
    
    def calculate_lj_forces(self, other_sporozoites: List['SporozoiteCenterlineArc']):
        """Calculate Lennard-Jones forces between nodes of different sporozoites"""
        # Reset forces
        for node in self.nodes:
            node.force.fill(0.0)
        
        # Calculate LJ forces with other sporozoites
        for other in other_sporozoites:
            if other.id == self.id:
                continue
                
            for my_node in self.nodes:
                for other_node in other.nodes:
                    distance_vec = my_node.position - other_node.position
                    distance = np.linalg.norm(distance_vec)
                    
                    if distance < my_node.cutoff_distance and distance > 1e-8:
                        # LJ force calculation (from C code)
                        sig2 = (2**(1/3) - (2*self.interaction_width)**2)
                        
                        # LJ force magnitude
                        lj_force_mag = self.lj_strength * distance * (
                            2 / (distance**2 + sig2)**7 - 
                            1 / (distance**2 + sig2)**4
                        )
                        
                        # Force direction (repulsive/attractive)
                        force_direction = distance_vec / distance
                        lj_force = lj_force_mag * force_direction
                        
                        my_node.force += lj_force
    
    def calculate_internal_forces(self):
        """Calculate internal forces: stretching, bending, and propulsion"""
        
        # Stretching forces (spring connections between adjacent nodes)
        rest_length = self.length / (self.num_nodes - 1)
        spring_constant = self.config.STRETCH_STIFFNESS
        
        for i in range(len(self.nodes) - 1):
            current_node = self.nodes[i]
            next_node = self.nodes[i + 1]
            
            # Spring vector and length
            spring_vec = next_node.position - current_node.position
            current_length = np.linalg.norm(spring_vec)
            
            if current_length > 1e-8:
                # Spring force
                force_magnitude = spring_constant * (current_length - rest_length)
                force_direction = spring_vec / current_length
                spring_force = force_magnitude * force_direction
                
                # Apply equal and opposite forces
                current_node.force += spring_force
                next_node.force -= spring_force
        
        # Bending forces (maintain rest curvature)
        bending_stiffness = self.config.BENDING_STIFFNESS
        
        for i in range(1, len(self.nodes) - 1):
            prev_node = self.nodes[i - 1]
            current_node = self.nodes[i]
            next_node = self.nodes[i + 1]
            
            # Current curvature
            current_curvature = (prev_node.position - 2 * current_node.position + next_node.position)
            
            # Desired rest curvature
            rest_curvature_vec = self.rest_curvature[i]
            
            # Bending force tries to restore rest curvature
            curvature_deviation = current_curvature - rest_curvature_vec
            bending_force = -bending_stiffness * curvature_deviation
            
            # Apply bending forces to the three nodes involved
            # The force distribution follows the second derivative pattern
            prev_node.force += bending_force
            current_node.force -= 2 * bending_force
            next_node.force += bending_force
        
        # Propulsion forces along centerline direction
        self._apply_propulsion_forces()
    
    def _apply_propulsion_forces(self):
        """Apply LJ-based propulsion forces along the centerline with circular rotation"""
        if len(self.nodes) < 2:
            return
        
        for i, node in enumerate(self.nodes):
            # Calculate local centerline direction
            if i == 0:
                # Head node - direction from current to next
                direction_vec = self.nodes[i + 1].position - node.position
            elif i == len(self.nodes) - 1:
                # Tail node - direction from previous to current
                direction_vec = node.position - self.nodes[i - 1].position
            else:
                # Middle nodes - average direction
                forward_vec = self.nodes[i + 1].position - node.position
                backward_vec = node.position - self.nodes[i - 1].position
                direction_vec = (forward_vec + backward_vec) / 2
            
            # Normalize direction
            if np.linalg.norm(direction_vec) > 1e-8:
                direction_vec = direction_vec / np.linalg.norm(direction_vec)
            
            # Propulsion strength varies along centerline (stronger at head)
            t = i / max(1, len(self.nodes) - 1)
            propulsion_factor = 1.0 - 0.5 * t  # Stronger at head (t=0)
            
            # Calculate linear propulsion force (existing)
            linear_propulsion_force = (self.propulsion_strength * propulsion_factor * 
                                     self.motility * direction_vec)
            
            # Apply 90-degree counterclockwise rotation matrix to create circular motion
            # Rotation matrix for 90 degrees CCW in 2D: [0 -1; 1 0]
            # For 3D: rotate in the XY plane, keep Z component unchanged
            circular_propulsion_force = np.zeros(3)
            circular_propulsion_force[0] = -linear_propulsion_force[1]  # -fy
            circular_propulsion_force[1] = linear_propulsion_force[0]   # fx
            circular_propulsion_force[2] = linear_propulsion_force[2]   # unchanged Z
            
            # Combine linear and circular components with adjustable weights
            linear_weight = 0.0      # 30% linear (forward) motion
            circular_weight = 0.7    # 70% circular (rotational) motion
            
            combined_propulsion_force = (linear_weight * linear_propulsion_force + 
                                       circular_weight * circular_propulsion_force)
            
            # Apply the combined propulsion force
            node.force += combined_propulsion_force
    
    def update_motion(self, dt: float, domain_size: Tuple[float, float, float]):
        """Update node positions using forces and physics"""
        damping = self.config.CENTERLINE_DAMPING
        
        for node in self.nodes:
            # Add random thermal forces
            thermal_force = np.random.normal(0, self.config.NOISE_STRENGTH, 3)
            node.force += thermal_force
            
            # Update velocity with damping
            node.velocity *= (1.0 - damping * dt)
            node.velocity += node.force * dt  # Assume unit mass
            
            # Update position
            node.position += node.velocity * dt
            
            # Apply boundary conditions
            self._apply_boundary_conditions(node, domain_size)
        
        # Update sporozoite properties
        self.viability -= self.config.VIABILITY_DECAY_RATE * dt
        self.undulation_phase += 2 * np.pi * dt
    
    def _apply_boundary_conditions(self, node: CenterlineArcNode, 
                                 domain_size: Tuple[float, float, float]):
        """Apply boundary conditions to keep nodes within domain"""
        boundary_stiffness = self.config.BOUNDARY_REPULSION_STRENGTH
        
        for dim in range(3):
            if node.position[dim] < 0:
                # Repulsive force from lower boundary
                penetration = -node.position[dim]
                node.force[dim] += boundary_stiffness * penetration
                node.position[dim] = max(0.1, node.position[dim])
            elif node.position[dim] > domain_size[dim]:
                # Repulsive force from upper boundary
                penetration = node.position[dim] - domain_size[dim]
                node.force[dim] -= boundary_stiffness * penetration
                node.position[dim] = min(domain_size[dim] - 0.1, node.position[dim])
    
    def is_viable(self) -> bool:
        """Check if sporozoite is still viable"""
        return self.viability > 0.1

    def to_vtk_polydata(self) -> vtk.vtkPolyData:
        """Convert sporozoite centerline arc to VTK PolyData for visualization"""
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
        
        # Node forces (vector)
        force_array = vtk.vtkFloatArray()
        force_array.SetName("Force")
        force_array.SetNumberOfComponents(3)
        force_array.SetNumberOfTuples(len(self.nodes))
        for i, node in enumerate(self.nodes):
            force_array.SetTuple3(i,
                                float(node.force[0]),
                                float(node.force[1]),
                                float(node.force[2]))
        polydata.GetPointData().AddArray(force_array)
        
        # Sporozoite properties (constant across all nodes)
        length_array = vtk.vtkFloatArray()
        length_array.SetName("Length")
        length_array.SetNumberOfTuples(len(self.nodes))
        for i in range(len(self.nodes)):
            length_array.SetValue(i, float(self.length))
        polydata.GetPointData().AddArray(length_array)
        
        diameter_array = vtk.vtkFloatArray()
        diameter_array.SetName("Diameter")
        diameter_array.SetNumberOfTuples(len(self.nodes))
        for i in range(len(self.nodes)):
            diameter_array.SetValue(i, float(self.diameter))
        polydata.GetPointData().AddArray(diameter_array)
        
        # Propulsion strength
        propulsion_array = vtk.vtkFloatArray()
        propulsion_array.SetName("PropulsionStrength")
        propulsion_array.SetNumberOfTuples(len(self.nodes))
        for i in range(len(self.nodes)):
            propulsion_array.SetValue(i, float(self.propulsion_strength))
        polydata.GetPointData().AddArray(propulsion_array)
        
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
        
        # Simulation parameters
        self.time = 0.0
        self.dt = self.config.TIME_STEP
        self.max_time = self.config.MAX_TIME
        self.output_interval = self.config.OUTPUT_INTERVAL
        self.last_output_time = 0.0
        
        # Output setup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"salivary_gland_lj_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Simulation initialized with {len(self.sporozoites)} sporozoites")
        print(f"Domain size: {domain_size}")
        print(f"Output directory: {self.output_dir}")
    
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
    
    def _write_sporozoite_vtk(self, step_number: int):
        """Write sporozoite centerline arcs to VTK with proper time information for animation"""
        multiblock = vtk.vtkMultiBlockDataSet()
        multiblock.SetNumberOfBlocks(len(self.sporozoites))
        
        for i, sporozoite in enumerate(self.sporozoites):
            # Convert sporozoite to VTK polydata
            polydata = sporozoite.to_vtk_polydata()
            
            # Validate the polydata
            if polydata.GetNumberOfPoints() == 0:
                print(f"    ERROR: Sporozoite {sporozoite.id} has no points!")
                continue
            if polydata.GetNumberOfLines() == 0:
                print(f"    WARNING: Sporozoite {sporozoite.id} has no lines!")
            
            # Add time information to each block
            time_array_block = vtk.vtkDoubleArray()
            time_array_block.SetName("TimeValue")
            time_array_block.SetNumberOfTuples(1)
            time_array_block.SetValue(0, self.time)
            polydata.GetFieldData().AddArray(time_array_block)
            
            multiblock.SetBlock(i, polydata)
            multiblock.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), f"Sporozoite_{sporozoite.id}")
        
        # Write to file with proper format settings
        filename = os.path.join(self.output_dir, f"sporozoites_{step_number:04d}.vtm")
        
        writer = vtk.vtkXMLMultiBlockDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(multiblock)
        writer.SetDataModeToAscii()  # Use ASCII format for better compatibility
        writer.SetCompressorTypeToNone()  # Disable compression to avoid binary issues
        writer.Write()
        
        print(f"VTK DEBUG: Successfully wrote {filename}")
    
    def _write_salivary_gland_environment_vtk(self, step_number: int):
        """Write salivary gland environment field data to VTK"""
        # Create a structured grid for the salivary gland environment
        grid_resolution = (20, 20, 8)  # Reasonable resolution for visualization
        
        # Create coordinate arrays
        x_coords = np.linspace(0, self.domain_size[0], grid_resolution[0])
        y_coords = np.linspace(0, self.domain_size[1], grid_resolution[1])
        z_coords = np.linspace(0, self.domain_size[2], grid_resolution[2])
        
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Create VTK structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(*grid_resolution)
        
        # Add time information
        time_array = vtk.vtkDoubleArray()
        time_array.SetName("TimeValue")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, self.time)
        grid.GetFieldData().AddArray(time_array)
        
        # Add points
        points = vtk.vtkPoints()
        for k in range(grid_resolution[2]):
            for j in range(grid_resolution[1]):
                for i in range(grid_resolution[0]):
                    points.InsertNextPoint(X[i,j,k], Y[i,j,k], Z[i,j,k])
        grid.SetPoints(points)
        
        # Create salivary gland environment fields
        # Density field (higher near injection site)
        density_array = vtk.vtkFloatArray()
        density_array.SetName("SalivaryGlandDensity")
        density_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        
        # Flow field (general downward flow in salivary gland)
        flow_array = vtk.vtkFloatArray()
        flow_array.SetName("FlowField")
        flow_array.SetNumberOfComponents(3)
        flow_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        
        # Resistance field (higher resistance in dense regions)
        resistance_array = vtk.vtkFloatArray()
        resistance_array.SetName("Resistance")
        resistance_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        
        point_index = 0
        for k in range(grid_resolution[2]):
            for j in range(grid_resolution[1]):
                for i in range(grid_resolution[0]):
                    x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                    
                    # Salivary gland density (higher near top - injection site)
                    injection_z = self.domain_size[2] * 0.8
                    z_factor = max(0, (z - injection_z/2) / (self.domain_size[2] - injection_z/2))
                    
                    # Add some radial variation
                    center_x, center_y = self.domain_size[0]/2, self.domain_size[1]/2
                    radial_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_radial = min(self.domain_size[:2]) / 2
                    radial_factor = 1.0 - (radial_dist / max_radial)**2
                    
                    density = z_factor * radial_factor * 0.8 + 0.2
                    density_array.SetValue(point_index, float(density))
                    
                    # Flow field (downward with some swirl)
                    flow_x = 0.1 * np.sin(2 * np.pi * y / self.domain_size[1])
                    flow_y = 0.1 * np.cos(2 * np.pi * x / self.domain_size[0])
                    flow_z = -2.0 * (1.0 - z / self.domain_size[2])  # Downward flow
                    flow_array.SetTuple3(point_index, float(flow_x), float(flow_y), float(flow_z))
                    
                    # Resistance (inversely related to flow, higher in dense regions)
                    resistance = density * 1.5 + 0.5
                    resistance_array.SetValue(point_index, float(resistance))
                    
                    point_index += 1
        
        # Add arrays to grid
        grid.GetPointData().SetScalars(density_array)
        grid.GetPointData().SetVectors(flow_array)
        grid.GetPointData().AddArray(resistance_array)
        
        # Write salivary gland environment field
        filename = os.path.join(self.output_dir, f"salivary_gland_env_{step_number:04d}.vts")
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.SetDataModeToAscii()
        writer.SetCompressorTypeToNone()
        writer.Write()
    
    def _write_simulation_state(self, step_number: int):
        """Write simulation state and statistics"""
        filename = os.path.join(self.output_dir, f"state_{step_number:04d}.txt")
        
        with open(filename, 'w') as f:
            f.write(f"Salivary Gland LJ Simulation State - Step {step_number}\n")
            f.write(f"Time: {self.time:.3f}s\n")
            f.write(f"Active sporozoites: {len(self.sporozoites)}\n")
            f.write(f"Domain size: {self.domain_size}\n")
            f.write(f"Environment type: Salivary Gland\n")
            f.write(f"Force model: Lennard-Jones with centerline arc nodes\n")
            f.write(f"Centerline segments: {self.config.CENTERLINE_SEGMENTS}\n")
            f.write(f"Propulsive force amplitude: {self.config.PROPULSIVE_FORCE_AMPLITUDE}\n")
            
            # Individual sporozoite data
            f.write("\nSporozoite Details:\n")
            for sporozoite in self.sporozoites:
                center_pos = sporozoite.get_center_position()
                x_pos = float(center_pos[0])
                y_pos = float(center_pos[1])
                z_pos = float(center_pos[2])
                viability = float(sporozoite.viability)
                motility = float(sporozoite.motility)
                sporozoite_id = int(sporozoite.id)
                
                f.write(f"ID {sporozoite_id}: center=[{x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f}], "
                       f"viability={viability:.3f}, motility={motility:.3f}, "
                       f"length={sporozoite.length:.2f}, nodes={sporozoite.num_nodes}\n")
    
    def _write_time_series_collection_files(self):
        """Write ParaView time series collection files for sporozoites and environment"""
        # Create sporozoites time series collection file
        sporozoites_collection_file = os.path.join(self.output_dir, "sporozoites_timeseries.pvd")
        
        with open(sporozoites_collection_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            f.write('    <!-- Sporozoite time series data will be added here -->\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        # Create salivary gland environment time series collection file
        environment_collection_file = os.path.join(self.output_dir, "salivary_gland_env_timeseries.pvd")
        
        with open(environment_collection_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            f.write('    <!-- Salivary gland environment time series data will be added here -->\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        # Store the collection file paths for updates
        self.sporozoites_collection_file_path = sporozoites_collection_file
        self.environment_collection_file_path = environment_collection_file
        
        print(f"Created time series collection files:")
        print(f"  Sporozoites: {sporozoites_collection_file}")
        print(f"  Environment: {environment_collection_file}")
    
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
    
    def _update_environment_time_series_collection(self, step_number: int):
        """Update the environment time series collection file with new time step"""
        if not hasattr(self, 'environment_collection_file_path'):
            return
            
        # Read existing content
        try:
            with open(self.environment_collection_file_path, 'r') as f:
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
        
        # Create the new entry for environment field
        vts_file = f"salivary_gland_env_{step_number:04d}.vts"
        new_entry = f'    <DataSet timestep="{self.time:.6f}" group="" part="0" file="{vts_file}"/>\n'
        
        # Insert the new entry
        lines.insert(insert_index, new_entry)
        
        # Write back the updated file
        with open(self.environment_collection_file_path, 'w') as f:
            for line in lines:
                if '<!-- Salivary gland environment time series data will be added here -->' not in line:
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
    
    def write_output(self, step_number: int):
        """Write VTK files and state for current simulation step"""
        # Write sporozoite centerlines
        self._write_sporozoite_vtk(step_number)
        
        # Write salivary gland environment field
        self._write_salivary_gland_environment_vtk(step_number)
        
        # Write simulation state info
        self._write_simulation_state(step_number)
        
        # Initialize time series collection files (only once at the beginning)
        if step_number == 0:
            self._write_time_series_collection_files()
        
        # Update time series collections
        self._update_sporozoites_time_series_collection(step_number)
        self._update_environment_time_series_collection(step_number)
        
        # Write ParaView state file (only at the end)
        if step_number == 0:
            self._write_paraview_state_file()
    

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
        num_sporozoites=8,
        domain_size=(50.0, 50.0, 10.0)
    )
    
    output_dir = sim.run_simulation()
    
    print(f"\nSimulation complete! Check results in: {output_dir}")


if __name__ == "__main__":
    main()