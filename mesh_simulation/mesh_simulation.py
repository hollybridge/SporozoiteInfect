#!/usr/bin/env python3
"""
High-Resolution Mesh-Based Sporozoite Simulation

This simulation models sporozoites as deformable polyhedral meshes moving through
an implicit dermal tissue environment. Outputs VTK files for ParaView visualization.

Author: Holly Evans
Date: January 2026
"""

import numpy as np
import vtk
import os
import sys
import argparse
import time
from datetime import datetime
from typing import List, Dict

# Add paths for local imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sporozoite_mesh import DeformableSporozoiteMesh
from dermal_field import ImplicitDermalTissue
from blood_flow_field import ImplicitBloodFlowField, FlowType
from simulation_config import SimulationConfig, PresetConfigs

class MeshBasedSporozoiteSimulation:
    """
    High-resolution simulation with deformable mesh sporozoites
    """
    
    def __init__(self, num_sporozoites: int = 5, domain_size: tuple = (100.0, 100.0, 50.0), 
                 config=None, use_blood_flow: bool = False, flow_type: str = "laminar",
                 inlet_velocity: float = 50.0, vessel_diameter: float = 20.0):
        self.num_sporozoites = num_sporozoites
        self.domain_size = domain_size
        self.config = config or SimulationConfig()
        self.use_blood_flow = use_blood_flow
        
        # Initialize environment field (either dermal tissue or blood flow)
        if use_blood_flow:
            print(f"Generating implicit blood flow field ({flow_type})...")
            flow_type_enum = FlowType(flow_type.lower())
            
            if flow_type_enum == FlowType.SIMPLE_SHEAR:
                self.environment_field = ImplicitBloodFlowField.create_simple_shear(
                    domain_size, shear_rate=inlet_velocity, vessel_diameter=vessel_diameter)
            elif flow_type_enum == FlowType.LAMINAR:
                self.environment_field = ImplicitBloodFlowField.create_laminar_flow(
                    domain_size, inlet_velocity=inlet_velocity, vessel_diameter=vessel_diameter)
            elif flow_type_enum == FlowType.TURBULENT:
                self.environment_field = ImplicitBloodFlowField.create_turbulent_flow(
                    domain_size, inlet_velocity=inlet_velocity, reynolds_number=2500.0)
            else:
                self.environment_field = ImplicitBloodFlowField(
                    domain_size, flow_type=flow_type_enum, inlet_velocity=inlet_velocity,
                    vessel_diameter=vessel_diameter)
        else:
            print("Generating implicit dermal tissue field...")
            self.environment_field = ImplicitDermalTissue(domain_size)
        
        # Keep reference for compatibility (some methods expect tissue_field)
        self.tissue_field = self.environment_field
        
        # Initialize sporozoites
        print(f"Creating {num_sporozoites} mesh-based sporozoites...")
        self.sporozoites: List[DeformableSporozoiteMesh] = []
        self._create_sporozoites()
        
        # Simulation parameters using config
        self.time = 0.0
        self.dt = self.config.TIME_STEP
        self.max_time = self.config.MAX_TIME
        self.output_interval = self.config.OUTPUT_INTERVAL
        self.last_output_time = 0.0
        
        # Output setup
        env_type = "bloodflow" if use_blood_flow else "tissue"
        self.output_dir = f"vtk_output/simulation_{env_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'active_sporozoites': num_sporozoites,
            'total_distance_traveled': 0.0,
            'average_motility': 0.0,
            'environment_type': env_type,
            'flow_type': flow_type if use_blood_flow else 'N/A'
        }
    
    def _create_sporozoites(self):
        """Create initial sporozoite meshes at injection site with proper boundary checking"""
        # Calculate safe injection area accounting for sporozoite size
        max_length = self.config.SPOROZOITE_LENGTH_RANGE[1]  # Maximum possible sporozoite length
        max_diameter = self.config.SPOROZOITE_DIAMETER_RANGE[1]  # Maximum possible diameter
        
        # Safe margins from domain boundaries (account for half the sporozoite extending in each direction)
        safety_margin = max(max_length, max_diameter) / 2 + 1.0  # Extra 1.0 for safety
        
        # Calculate safe injection volume within domain bounds
        safe_min = np.array([safety_margin, safety_margin, safety_margin])
        safe_max = np.array([
            self.domain_size[0] - safety_margin,
            self.domain_size[1] - safety_margin, 
            self.domain_size[2] - safety_margin
        ])
        
        # Ensure we have a valid injection area
        if np.any(safe_min >= safe_max):
            print("WARNING: Domain too small for safe sporozoite initialization!")
            print(f"Domain size: {self.domain_size}")
            print(f"Required safety margin: {safety_margin}")
            # Fall back to center of domain
            injection_center = np.array([d/2 for d in self.domain_size])
            injection_radius = min(self.domain_size) / 4  # Conservative radius
        else:
            if self.use_blood_flow:
                # For blood flow: initialize near the beginning (inlet) of the vessel
                # Blood flow typically flows in the X direction (0 -> domain_size[0])
                injection_center = np.array([
                    safe_min[0] + safety_margin,        # Near inlet (beginning of X domain)
                    self.domain_size[1] / 2,            # Center Y (vessel centerline)
                    self.domain_size[2] / 2             # Center Z (vessel centerline)
                ])
                
                # Create a smaller injection radius to keep sporozoites clustered near inlet
                # and ensure they stay within the vessel diameter
                if hasattr(self.environment_field, 'vessel_diameter'):
                    vessel_radius = self.environment_field.vessel_diameter / 2
                    # Use 60% of vessel radius to ensure sporozoites start well inside vessel
                    max_radial_spread = vessel_radius * 0.6
                else:
                    max_radial_spread = min(self.domain_size[1], self.domain_size[2]) / 4
                
                injection_radius = min(
                    safety_margin / 2,      # Small spread in X direction (along flow)
                    max_radial_spread,      # Constrained by vessel diameter in Y, Z
                    5.0                     # Maximum 5 μm spread for tight clustering
                )
                
                print(f"Blood flow injection: Placing sporozoites near vessel inlet")
                if hasattr(self.environment_field, 'vessel_diameter'):
                    print(f"  Vessel diameter: {self.environment_field.vessel_diameter:.1f} μm")
                    print(f"  Max radial spread: {max_radial_spread:.1f} μm")
            else:
                # For tissue simulations: use the original logic (near top of domain)
                injection_center = np.array([
                    (safe_min[0] + safe_max[0]) / 2,  # Center X
                    (safe_min[1] + safe_max[1]) / 2,  # Center Y  
                    safe_max[2] - safety_margin / 2   # Near top Z (salivary gland area)
                ])
                injection_radius = min(
                    (safe_max[0] - safe_min[0]) / 4,  # Quarter of safe X range
                    (safe_max[1] - safe_min[1]) / 4,  # Quarter of safe Y range
                    safety_margin / 2                 # Half the safety margin in Z
                )
        
        print(f"Injection center: {injection_center}")
        print(f"Injection radius: {injection_radius:.2f}")
        print(f"Safety margin: {safety_margin}")
        
        for i in range(self.num_sporozoites):
            # Generate safe random position within injection volume
            max_attempts = 50  # Prevent infinite loops
            attempts = 0
            
            while attempts < max_attempts:
                if self.use_blood_flow:
                    # For blood flow: create a more constrained distribution
                    # Small spread in X (flow direction), larger spread in Y,Z (radial)
                    x_offset = np.random.uniform(-injection_radius * 0.5, injection_radius * 0.5)
                    
                    # Radial offset in Y-Z plane (perpendicular to flow)
                    radial_distance = np.random.uniform(0, injection_radius)
                    radial_angle = np.random.uniform(0, 2 * np.pi)
                    y_offset = radial_distance * np.cos(radial_angle)
                    z_offset = radial_distance * np.sin(radial_angle)
                    
                    offset = np.array([x_offset, y_offset, z_offset])
                else:
                    # For tissue: uniform random distribution in all directions
                    offset = np.random.uniform(-injection_radius, injection_radius, 3)
                
                initial_pos = injection_center + offset
                
                # Verify position is safe (well within domain bounds)
                if (np.all(initial_pos >= safe_min) and np.all(initial_pos <= safe_max)):
                    # For blood flow, also check that sporozoite is within vessel
                    if self.use_blood_flow and hasattr(self.environment_field, 'vessel_diameter'):
                        vessel_radius = self.environment_field.vessel_diameter / 2
                        y_center = self.domain_size[1] / 2
                        z_center = self.domain_size[2] / 2
                        radial_distance = np.sqrt((initial_pos[1] - y_center)**2 + (initial_pos[2] - z_center)**2)
                        
                        if radial_distance <= vessel_radius * 0.8:  # 80% of vessel radius for safety
                            break
                    else:
                        break
                    
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"WARNING: Could not find safe position for sporozoite {i}, using center")
                initial_pos = injection_center.copy()
            
            # Create sporozoite mesh with config
            sporozoite = DeformableSporozoiteMesh(i, initial_pos, self.config)
            self.sporozoites.append(sporozoite)
            
            # Verify the created sporozoite is actually within bounds
            vertex_positions = np.array(sporozoite.vertices)
            min_coords = np.min(vertex_positions, axis=0)
            max_coords = np.max(vertex_positions, axis=0)
            
            # Check if any vertices are out of bounds
            out_of_bounds = (
                np.any(min_coords <= 0) or 
                np.any(max_coords >= np.array(self.domain_size))
            )
            
            print(f"  Created sporozoite {i}: {len(sporozoite.vertices)} vertices, "
                  f"{len(sporozoite.faces)} faces, ID={sporozoite.id}")
            print(f"    Center: {initial_pos}")
            print(f"    Bounds: [{min_coords[0]:.2f}, {min_coords[1]:.2f}, {min_coords[2]:.2f}] to "
                  f"[{max_coords[0]:.2f}, {max_coords[1]:.2f}, {max_coords[2]:.2f}]")
            
            # For blood flow, also report radial distance from vessel centerline
            if self.use_blood_flow and hasattr(self.environment_field, 'vessel_diameter'):
                y_center = self.domain_size[1] / 2
                z_center = self.domain_size[2] / 2
                radial_distance = np.sqrt((initial_pos[1] - y_center)**2 + (initial_pos[2] - z_center)**2)
                vessel_radius = self.environment_field.vessel_diameter / 2
                print(f"    Radial distance from vessel center: {radial_distance:.2f} μm "
                      f"(vessel radius: {vessel_radius:.1f} μm)")
            
            if out_of_bounds:
                print(f"      WARNING: Sporozoite {i} vertices extend outside domain bounds!")
            else:
                print(f"\n")
    
    def update_simulation_step(self):
        """Update one simulation time step"""
        active_sporozoites = []
        total_distance = 0.0
        total_motility = 0.0
        
        print(f"Step t={self.time:.3f}: Processing {len(self.sporozoites)} sporozoites")
        
        # DEBUG: Print current sporozoite IDs before processing
        current_ids = [s.id for s in self.sporozoites]
        #print(f"  Current sporozoite IDs: {current_ids}")
        
        for i, sporozoite in enumerate(self.sporozoites):
            # DEBUG: Verify sporozoite ID hasn't changed
            #print(f"  Processing sporozoite at index {i} with ID {sporozoite.id}")
            
            if sporozoite.is_viable():
                # Store previous position for distance calculation
                prev_position = sporozoite.center_position.copy()
                
                # Apply tissue forces
                sporozoite.apply_tissue_forces(self.tissue_field, self.dt)
                
                # Apply immune response only if enabled in config
                if self.config.ENABLE_IMMUNE_RESPONSE:
                    immune_damage = self.tissue_field.apply_immune_response(sporozoite.center_position)
                    sporozoite.reduce_viability(immune_damage * self.dt)
                
                # Calculate forces for this sporozoite
                forces = {}
                
                # Directional movement force
                direction_vec = np.array([np.cos(sporozoite.direction), np.sin(sporozoite.direction), 0])
                forces['directional'] = direction_vec * self.config.DIRECTIONAL_FORCE_STRENGTH
                
                # FIXED: Add blood flow/tissue forces to the directional forces
                if hasattr(sporozoite, 'tissue_flow_force'):
                    forces['directional'] += sporozoite.tissue_flow_force
                if hasattr(sporozoite, 'tissue_resistance_force'):
                    forces['directional'] += sporozoite.tissue_resistance_force
                
                # Random movement
                if np.random.random() < self.config.RANDOM_DIRECTION_PROBABILITY:
                    random_force = np.random.uniform(-1, 1, 3) * 2.0
                    forces['random'] = random_force
                else:
                    forces['random'] = np.zeros(3)
                
                # Boundary repulsion
                forces['boundary_repulsion'] = np.zeros(3)
                
                # Undulation forces (apply to each vertex)
                undulation_forces = []
                phase = sporozoite.undulation_phase + self.time * sporozoite.undulation_frequency * 2 * np.pi
                
                for i, vertex in enumerate(sporozoite.vertices):
                    relative_pos = vertex - sporozoite.center_position
                    body_position = np.dot(relative_pos, direction_vec)
                    normalized_position = body_position / (sporozoite.length/2) if sporozoite.length > 0 else 0
                    
                    # Perpendicular undulation - CONTROLLED FOR STABLE DEFORMATION
                    perpendicular = np.array([-np.sin(sporozoite.direction), np.cos(sporozoite.direction), 0])
                    vertical = np.array([0, 0, 1])
                    
                    # Controlled wave amplitude for visible but stable deformation
                    wave_amplitude = self.config.UNDULATION_AMPLITUDE * sporozoite.motility
                    
                    # Enhance forces for low spring constants but keep them reasonable
                    if self.config.SPRING_CONSTANT_BASE < 20.0:
                        wave_amplitude *= 1.5  # Moderate enhancement for flexible meshes
                    
                    # Create controlled undulation pattern
                    lateral_phase = phase + normalized_position * 2 * np.pi  # Fewer waves for stability
                    vertical_phase = phase + normalized_position * np.pi
                    
                    lateral_force = perpendicular * wave_amplitude * np.sin(lateral_phase)
                    vertical_force = vertical * wave_amplitude * np.cos(vertical_phase) * 0.3
                    
                    # Add minimal random variation for natural movement
                    random_component = np.random.normal(0, wave_amplitude * 0.05, 3)
                    
                    undulation_force = (lateral_force + vertical_force + random_component) * self.config.UNDULATION_FORCE_SCALE
                    undulation_forces.append(undulation_force)
                
                forces['undulation'] = undulation_forces
                
                # Update sporozoite motion and deformation with forces and spring constant
                # NEW: Use effective spring constant based on activity-flow coupling
                effective_spring_constant = sporozoite.get_effective_spring_constant()
                sporozoite.update_motion(self.dt, forces, 
                                       spring_constant=effective_spring_constant,
                                       damping=self.config.DAMPING_FACTOR)
                
                # Keep sporozoites within domain bounds
                self._apply_boundary_conditions(sporozoite)
                
                # Calculate distance traveled
                distance_moved = np.linalg.norm(sporozoite.center_position - prev_position)
                total_distance += distance_moved
                total_motility += sporozoite.motility
                
                #print(f"    Sporozoite ID {sporozoite.id}: moved {float(distance_moved):.4f} units (motility: {float(sporozoite.motility):.3f}, viability: {float(sporozoite.viability):.3f})")
                
                active_sporozoites.append(sporozoite)
            else:
                print(f"    Sporozoite ID {sporozoite.id} removed (not viable: {sporozoite.viability:.3f})")
        
        # DEBUG: Print surviving sporozoite IDs
        surviving_ids = [s.id for s in active_sporozoites]
        print(f"  Surviving sporozoite IDs: {surviving_ids}")
        
        # Update sporozoite list
        self.sporozoites = active_sporozoites
        
        # Update statistics
        self.stats['active_sporozoites'] = len(active_sporozoites)
        self.stats['total_distance_traveled'] += total_distance
        if len(active_sporozoites) > 0:
            self.stats['average_motility'] = total_motility / len(active_sporozoites)
    
    def _apply_boundary_conditions(self, sporozoite):
        """Apply boundary conditions for both domain and vessel boundaries"""
        center = sporozoite.center_position
        damping = self.config.BOUNDARY_DAMPING
        
        # First apply domain boundary conditions (existing code)
        self._apply_domain_boundary_conditions(sporozoite)
        
        # Then apply vessel boundary conditions for blood flow simulations
        if self.use_blood_flow:
            self._apply_vessel_boundary_conditions(sporozoite)
    
    def _apply_domain_boundary_conditions(self, sporozoite):
        """Apply domain boundary conditions using rigid body approach"""
        center = sporozoite.center_position
        damping = self.config.BOUNDARY_DAMPING
        
        # Calculate how much the center needs to be moved to keep all vertices in domain bounds
        center_correction = np.array([0.0, 0.0, 0.0])
        
        # Check if any vertices are out of domain bounds
        for vertex in sporozoite.vertices:
            # X boundaries - PERIODIC for blood flow
            if self.use_blood_flow:
                # Periodic boundary in X direction (flow direction)
                if vertex[0] <= 0:
                    correction_needed = self.domain_size[0] - 0.1 - vertex[0]
                    center_correction[0] = min(center_correction[0], correction_needed)
                elif vertex[0] >= self.domain_size[0]:
                    correction_needed = 0.1 - vertex[0]
                    center_correction[0] = max(center_correction[0], correction_needed)
            else:
                # Reflective boundaries for tissue simulations
                if vertex[0] <= 0:
                    correction_needed = 0.1 - vertex[0]
                    center_correction[0] = max(center_correction[0], correction_needed)
                elif vertex[0] >= self.domain_size[0]:
                    correction_needed = self.domain_size[0] - 0.1 - vertex[0]
                    center_correction[0] = min(center_correction[0], correction_needed)
            
            # Y and Z boundaries - always reflective
            if vertex[1] <= 0:
                correction_needed = 0.1 - vertex[1]
                center_correction[1] = max(center_correction[1], correction_needed)
            elif vertex[1] >= self.domain_size[1]:
                correction_needed = self.domain_size[1] - 0.1 - vertex[1]
                center_correction[1] = min(center_correction[1], correction_needed)
            
            if vertex[2] <= 0:
                correction_needed = 0.1 - vertex[2]
                center_correction[2] = max(center_correction[2], correction_needed)
            elif vertex[2] >= self.domain_size[2]:
                correction_needed = self.domain_size[2] - 0.1 - vertex[2]
                center_correction[2] = min(center_correction[2], correction_needed)
        
        # Apply center correction if needed (rigid body translation)
        if np.linalg.norm(center_correction) > 1e-8:
            sporozoite.center_position += center_correction
            
            # Update all vertices rigidly with the center correction
            for i in range(len(sporozoite.vertices)):
                sporozoite.vertices[i] += center_correction
            
            # Apply damping to velocities
            if hasattr(sporozoite, 'vertex_velocities'):
                for i in range(len(sporozoite.vertex_velocities)):
                    sporozoite.vertex_velocities[i] *= damping
    
    def _apply_vessel_boundary_conditions(self, sporozoite):
        """Apply gentle vessel boundary conditions to keep sporozoites within blood vessel"""
        if not hasattr(self.environment_field, 'vessel_diameter'):
            return
        
        vessel_diameter = self.environment_field.vessel_diameter
        vessel_radius = vessel_diameter / 2
        
        # Calculate vessel center coordinates
        y_center = self.domain_size[1] / 2
        z_center = self.domain_size[2] / 2
        
        # Check if sporozoite center is outside vessel
        center_y = sporozoite.center_position[1]
        center_z = sporozoite.center_position[2]
        
        # Distance from vessel centerline
        radial_distance = np.sqrt((center_y - y_center)**2 + (center_z - z_center)**2)
        
        # Allow more buffer space inside vessel (85% of radius instead of 90%)
        max_allowed_radius = vessel_radius * 0.85
        
        if radial_distance > max_allowed_radius:
            # Calculate gentle correction to move sporozoite back inside vessel
            if radial_distance > 1e-8:
                # Direction from vessel center to sporozoite
                radial_direction_y = (center_y - y_center) / radial_distance
                radial_direction_z = (center_z - z_center) / radial_distance
                
                # New position at max allowed radius
                new_center_y = y_center + radial_direction_y * max_allowed_radius
                new_center_z = z_center + radial_direction_z * max_allowed_radius
                
                # Calculate gentle correction needed
                correction_y = (new_center_y - center_y) * 0.1  # Only 10% correction per step
                correction_z = (new_center_z - center_z) * 0.1  # Very gentle
                
                # Apply correction to center and all vertices
                sporozoite.center_position[1] += correction_y
                sporozoite.center_position[2] += correction_z
                
                for i in range(len(sporozoite.vertices)):
                    sporozoite.vertices[i][1] += correction_y
                    sporozoite.vertices[i][2] += correction_z
                
                # Apply gentle damping when approaching vessel walls
                vessel_wall_damping = 0.9  # Much gentler damping
                if hasattr(sporozoite, 'vertex_velocities'):
                    for i in range(len(sporozoite.vertex_velocities)):
                        # Gently reduce radial velocity components
                        vel = sporozoite.vertex_velocities[i]
                        vel[1] *= vessel_wall_damping  # Y velocity
                        vel[2] *= vessel_wall_damping  # Z velocity
                        vel[0] *= 0.95  # X velocity (very gentle damping for flow direction)
                
                # Debug output for vessel boundary corrections
                print(f"  VESSEL BOUNDARY: Sporozoite {sporozoite.id} gently moved back inside vessel "
                      f"(was {radial_distance:.2f}, target {max_allowed_radius:.2f})")
        
        # Remove the aggressive vertex-by-vertex corrections that were causing instability
        # The gentler center-based correction above should be sufficient
    
    def _write_sporozoite_vtk(self, step_number: int):
        """Write sporozoite mesh data to VTK with proper time information for animation"""
        if not self.sporozoites:
            print(f"VTK WRITE DEBUG: No sporozoites to write at step {step_number}")
            return
        
        # Create multi-block dataset for all sporozoites
        multiblock = vtk.vtkMultiBlockDataSet()
        multiblock.SetNumberOfBlocks(len(self.sporozoites))
        
        # Add time information to the dataset
        time_array = vtk.vtkDoubleArray()
        time_array.SetName("TimeValue")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, self.time)
        multiblock.GetFieldData().AddArray(time_array)
        
        for i, sporozoite in enumerate(self.sporozoites):
            polydata = sporozoite.to_vtk_polydata()
            
            # Add time information to each block as well
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

    def _write_tissue_field_vtk(self, step_number: int):
        """Write tissue field data to VTK with proper time information for animation"""
        field_data = self.tissue_field.get_field_visualization_data()
        X, Y, Z = field_data['coordinates']
        
        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(*self.tissue_field.grid_resolution)
        
        # Add time information to the dataset
        time_array = vtk.vtkDoubleArray()
        time_array.SetName("TimeValue")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, self.time)
        grid.GetFieldData().AddArray(time_array)
        
        # Add points
        points = vtk.vtkPoints()
        for k in range(X.shape[2]):
            for j in range(X.shape[1]):
                for i in range(X.shape[0]):
                    points.InsertNextPoint(X[i,j,k], Y[i,j,k], Z[i,j,k])
        grid.SetPoints(points)
        
        # Add scalar fields - fix the array creation and ordering
        # Collagen density
        collagen_array = vtk.vtkFloatArray()
        collagen_array.SetName("CollagenDensity")
        collagen_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        collagen_data = field_data['collagen_density']
        point_index = 0
        for k in range(collagen_data.shape[2]):
            for j in range(collagen_data.shape[1]):
                for i in range(collagen_data.shape[0]):
                    collagen_array.SetValue(point_index, float(collagen_data[i,j,k]))
                    point_index += 1
        grid.GetPointData().SetScalars(collagen_array)
        
        # Immune cell density
        immune_array = vtk.vtkFloatArray()
        immune_array.SetName("ImmuneCellDensity")
        immune_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        immune_data = field_data['immune_cells']
        point_index = 0
        for k in range(immune_data.shape[2]):
            for j in range(immune_data.shape[1]):
                for i in range(immune_data.shape[0]):
                    immune_array.SetValue(point_index, float(immune_data[i,j,k]))
                    point_index += 1
        grid.GetPointData().AddArray(immune_array)
        
        # Pressure field
        pressure_array = vtk.vtkFloatArray()
        pressure_array.SetName("Pressure")
        pressure_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        pressure_data = field_data['pressure']
        point_index = 0
        for k in range(pressure_data.shape[2]):
            for j in range(pressure_data.shape[1]):
                for i in range(pressure_data.shape[0]):
                    pressure_array.SetValue(point_index, float(pressure_data[i,j,k]))
                    point_index += 1
        grid.GetPointData().AddArray(pressure_array)
        
        # Flow field (vector)
        flow_array = vtk.vtkFloatArray()
        flow_array.SetName("FlowField")
        flow_array.SetNumberOfComponents(3)
        flow_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        flow_data = field_data['flow_field']
        point_index = 0
        for k in range(flow_data.shape[2]):
            for j in range(flow_data.shape[1]):
                for i in range(flow_data.shape[0]):
                    flow_array.SetTuple3(point_index, 
                                        float(flow_data[i,j,k,0]), 
                                        float(flow_data[i,j,k,1]), 
                                        float(flow_data[i,j,k,2]))
                    point_index += 1
        grid.GetPointData().SetVectors(flow_array)
        
        # Write tissue field with proper format settings
        filename = os.path.join(self.output_dir, f"tissue_field_{step_number:04d}.vts")
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.SetDataModeToAscii()  # Use ASCII format for better compatibility
        writer.SetCompressorTypeToNone()  # Disable compression to avoid binary issues
        writer.Write()

    def _write_blood_flow_field_vtk(self, step_number: int):
        """Write blood flow field data to VTK with proper time information for animation"""
        field_data = self.environment_field.get_field_visualization_data()
        X, Y, Z = field_data['coordinates']
        
        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(*self.environment_field.grid_resolution)
        
        # Add time information to the dataset
        time_array = vtk.vtkDoubleArray()
        time_array.SetName("TimeValue")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, self.time)
        grid.GetFieldData().AddArray(time_array)
        
        # Add points
        points = vtk.vtkPoints()
        for k in range(X.shape[2]):
            for j in range(X.shape[1]):
                for i in range(X.shape[0]):
                    points.InsertNextPoint(X[i,j,k], Y[i,j,k], Z[i,j,k])
        grid.SetPoints(points)
        
        # Add blood flow specific fields
        # Velocity magnitude (scalar)
        velocity_magnitude_array = vtk.vtkFloatArray()
        velocity_magnitude_array.SetName("VelocityMagnitude")
        velocity_magnitude_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        velocity_data = field_data['velocity_field']
        point_index = 0
        for k in range(velocity_data.shape[2]):
            for j in range(velocity_data.shape[1]):
                for i in range(velocity_data.shape[0]):
                    vel_mag = np.linalg.norm(velocity_data[i,j,k,:])
                    velocity_magnitude_array.SetValue(point_index, float(vel_mag))
                    point_index += 1
        grid.GetPointData().SetScalars(velocity_magnitude_array)
        
        # Pressure field
        pressure_array = vtk.vtkFloatArray()
        pressure_array.SetName("Pressure")
        pressure_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        pressure_data = field_data['pressure']
        point_index = 0
        for k in range(pressure_data.shape[2]):
            for j in range(pressure_data.shape[1]):
                for i in range(pressure_data.shape[0]):
                    pressure_array.SetValue(point_index, float(pressure_data[i,j,k]))
                    point_index += 1
        grid.GetPointData().AddArray(pressure_array)
        
        # Shear rate field
        shear_rate_array = vtk.vtkFloatArray()
        shear_rate_array.SetName("ShearRate")
        shear_rate_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        shear_rate_data = field_data['shear_rate']
        point_index = 0
        for k in range(shear_rate_data.shape[2]):
            for j in range(shear_rate_data.shape[1]):
                for i in range(shear_rate_data.shape[0]):
                    shear_rate_array.SetValue(point_index, float(shear_rate_data[i,j,k]))
                    point_index += 1
        grid.GetPointData().AddArray(shear_rate_array)
        
        # Turbulence intensity field (if available)
        if 'turbulence_intensity' in field_data:
            turbulence_array = vtk.vtkFloatArray()
            turbulence_array.SetName("TurbulenceIntensity")
            turbulence_array.SetNumberOfTuples(grid.GetNumberOfPoints())
            turbulence_data = field_data['turbulence_intensity']
            point_index = 0
            for k in range(turbulence_data.shape[2]):
                for j in range(turbulence_data.shape[1]):
                    for i in range(turbulence_data.shape[0]):
                        turbulence_array.SetValue(point_index, float(turbulence_data[i,j,k]))
                        point_index += 1
            grid.GetPointData().AddArray(turbulence_array)
        
        # Velocity field (vector)
        velocity_vector_array = vtk.vtkFloatArray()
        velocity_vector_array.SetName("Velocity")
        velocity_vector_array.SetNumberOfComponents(3)
        velocity_vector_array.SetNumberOfTuples(grid.GetNumberOfPoints())
        point_index = 0
        for k in range(velocity_data.shape[2]):
            for j in range(velocity_data.shape[1]):
                for i in range(velocity_data.shape[0]):
                    velocity_vector_array.SetTuple3(point_index, 
                                                  float(velocity_data[i,j,k,0]), 
                                                  float(velocity_data[i,j,k,1]), 
                                                  float(velocity_data[i,j,k,2]))
                    point_index += 1
        grid.GetPointData().SetVectors(velocity_vector_array)
        
        # Add flow type and parameters as field data
        flow_type_array = vtk.vtkStringArray()
        flow_type_array.SetName("FlowType")
        flow_type_array.SetNumberOfTuples(1)
        flow_type_array.SetValue(0, field_data['flow_type'])
        grid.GetFieldData().AddArray(flow_type_array)
        
        # Write blood flow field with proper format settings
        filename = os.path.join(self.output_dir, f"blood_flow_field_{step_number:04d}.vts")
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.SetDataModeToAscii()  # Use ASCII format for better compatibility
        writer.SetCompressorTypeToNone()  # Disable compression to avoid binary issues
        writer.Write()

    def _write_presets_config_file(self):
        """Write comprehensive presets configuration file with all available options"""
        # Placeholder for writing presets configuration file
        pass

    def write_vtk_output(self, step_number: int):
        """Write VTK files for current simulation state"""
        # Write sporozoite meshes
        self._write_sporozoite_vtk(step_number)
        
        # Write environment field data (tissue or blood flow)
        if self.use_blood_flow:
            self._write_blood_flow_field_vtk(step_number)
        else:
            self._write_tissue_field_vtk(step_number)
        
        # Write simulation state info
        self._write_simulation_state(step_number)
        
        # Write presets configuration file (only once at the beginning)
        if step_number == 0:
            self._write_presets_config_file()
            self._write_time_series_collection_files()
        
        # Update time series collections
        self._update_sporozoites_time_series_collection(step_number)
        self._update_environment_time_series_collection(step_number)

    def run_simulation(self):
        """Run the complete mesh-based simulation"""
        print("\nStarting mesh-based sporozoite simulation...")
        print(f"Domain size: {self.domain_size}")
        print(f"Time step: {self.dt}")
        print(f"Max time: {self.max_time}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        step_count = 0
        output_count = 0
        
        # Initial output
        print(f"Writing initial state...")
        self.write_vtk_output(output_count)
        output_count += 1
        self.last_output_time = self.time
        
        start_time = time.time()
        
        while self.time < self.max_time and len(self.sporozoites) > 0:
            # Update simulation
            self.update_simulation_step()
            self.time += self.dt
            step_count += 1
            
            # Output VTK files at intervals
            if self.time - self.last_output_time >= self.output_interval:
                self.write_vtk_output(output_count)
                output_count += 1
                self.last_output_time = self.time
                
                # Progress report
                elapsed = time.time() - start_time
                print(f"Time: {self.time:6.2f} | "
                      f"Active: {len(self.sporozoites):2d} | "
                      f"Avg motility: {self.stats['average_motility']:.3f} | "
                      f"Elapsed: {elapsed:.1f}s")
            
            # Safety check for very long simulations
            if step_count > 100000:
                print("Maximum step count reached, ending simulation")
                break
        
        # Final output
        if output_count == 0 or self.time - self.last_output_time > 0.1:
            print("Writing final state...")
            self.write_vtk_output(output_count)
        
        total_time = time.time() - start_time
        
        print(f"\nSimulation completed!")
        print(f"Total simulation time: {self.time:.2f}")
        print(f"Total computation time: {total_time:.1f}s")
        print(f"Final active sporozoites: {len(self.sporozoites)}")
        print(f"VTK files written to: {self.output_dir}")
        print(f"Total VTK outputs: {output_count + 1}")
        
        # Write ParaView state file
        self._write_paraview_state_file()
        
    def _write_time_series_collection_files(self):
        """Write ParaView time series collection files for both sporozoites and environment field"""
        # Create sporozoites time series collection file
        sporozoites_collection_file = os.path.join(self.output_dir, "sporozoites_timeseries.pvd")
        
        with open(sporozoites_collection_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            f.write('    <!-- Sporozoite time series data will be added here -->\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        # Create environment field time series collection file
        if self.use_blood_flow:
            env_field_name = "blood_flow_field"
            env_description = "Blood flow field"
        else:
            env_field_name = "tissue_field" 
            env_description = "Dermal tissue field"
            
        environment_collection_file = os.path.join(self.output_dir, f"{env_field_name}_timeseries.pvd")
        
        with open(environment_collection_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            f.write(f'    <!-- {env_description} time series data will be added here -->\n')
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
            # Remove the comment line if it exists
            for line in lines:
                if '<!-- Sporozoite time series data will be added here -->' not in line:
                    f.write(line)

    def _update_environment_time_series_collection(self, step_number: int):
        """Update the environment field time series collection file with new time step"""
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
        if self.use_blood_flow:
            vts_file = f"blood_flow_field_{step_number:04d}.vts"
        else:
            vts_file = f"tissue_field_{step_number:04d}.vts"
            
        new_entry = f'    <DataSet timestep="{self.time:.6f}" group="" part="0" file="{vts_file}"/>\n'
        
        # Insert the new entry
        lines.insert(insert_index, new_entry)
        
        # Write back the updated file
        with open(self.environment_collection_file_path, 'w') as f:
            # Remove the comment line if it exists
            for line in lines:
                if ('<!-- Blood flow field time series data will be added here -->' not in line and
                    '<!-- Dermal tissue field time series data will be added here -->' not in line):
                    f.write(line)

    def _write_simulation_state(self, step_number: int):
        """Write simulation state and statistics"""
        filename = os.path.join(self.output_dir, f"state_{step_number:04d}.txt")
        
        with open(filename, 'w') as f:
            f.write(f"Simulation State - Step {step_number}\n")
            f.write(f"Time: {self.time:.3f}\n")
            f.write(f"Active sporozoites: {self.stats['active_sporozoites']}\n")
            f.write(f"Total distance traveled: {self.stats['total_distance_traveled']:.2f}\n")
            f.write(f"Average motility: {self.stats['average_motility']:.3f}\n")
            f.write(f"Domain size: {self.domain_size}\n")
            f.write(f"Environment type: {self.stats['environment_type']}\n")
            f.write(f"Flow type: {self.stats['flow_type']}\n")
            
            # Individual sporozoite data
            f.write("\nSporozoite Details:\n")
            for sporozoite in self.sporozoites:
                # Extract values first to avoid formatting issues
                x_pos = float(sporozoite.center_position[0])
                y_pos = float(sporozoite.center_position[1])
                z_pos = float(sporozoite.center_position[2])
                viability = float(sporozoite.viability)
                motility = float(sporozoite.motility)
                sporozoite_id = int(sporozoite.id)
                
                f.write(f"ID {sporozoite_id}: pos=[{x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f}], "
                       f"viability={viability:.3f}, "
                       f"motility={motility:.3f}\n")

    def _write_paraview_state_file(self):
        """Write ParaView state file for automatic setup"""
        state_file = os.path.join(self.output_dir, "simulation_setup.pvsm")
        
        # Dynamic file path based on environment type
        if self.use_blood_flow:
            env_field_pattern = "blood_flow_field_*.vts"
            env_color_field = "VelocityMagnitude"
        else:
            env_field_pattern = "tissue_field_*.vts"
            env_color_field = "CollagenDensity"
        
        # This creates a ParaView state file that automatically loads and configures the visualization
        state_content = f'''<?xml version="1.0"?>
<ParaViewState version="5.9.0">
  <ServerManagerState>
    <!-- Sporozoite reader -->
    <Proxy group="sources" type="XMLMultiBlockDataReader" id="100">
      <Property name="FileName">
        <Element index="0" value="{os.path.abspath(self.output_dir)}/sporozoites_*.vtm"/>
      </Property>
    </Proxy>
    
    <!-- Environment field reader -->
    <Proxy group="sources" type="XMLStructuredGridReader" id="200">
      <Property name="FileName">
        <Element index="0" value="{os.path.abspath(self.output_dir)}/{env_field_pattern}"/>
      </Property>
    </Proxy>
    
    <!-- Sporozoite representation -->
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
        <Element index="0" value="{env_color_field}"/>
      </Property>
      <Property name="Representation">
        <Element index="0" value="Outline"/>
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
    
    <!-- Camera settings -->
    <Proxy group="views" type="RenderView" id="400">
      <Property name="CameraPosition">
        <Element index="0" value="150"/>
        <Element index="1" value="150"/>
        <Element index="2" value="100"/>
      </Property>
      <Property name="CameraFocalPoint">
        <Element index="0" value="50"/>
        <Element index="1" value="50"/>
        <Element index="2" value="25"/>
      </Property>
    </Proxy>
  </ServerManagerState>
</ParaViewState>'''
        
        with open(state_file, 'w') as f:
            f.write(state_content)
        
        print(f"ParaView state file created: {state_file}")

    def _write_presets_config_file(self):
        """Write comprehensive presets configuration file with all available options"""
        presets_file = os.path.join(self.output_dir, "simulation_presets.cfg")
        
        with open(presets_file, 'w') as f:
            f.write("# Sporozoite Simulation Presets Configuration\n")
            f.write("# Generated automatically during simulation\n")
            f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Use these presets to reproduce or modify simulation behavior\n")
            f.write("\n")
            
            # Current simulation configuration
            f.write("="*60 + "\n")
            f.write("CURRENT SIMULATION CONFIGURATION\n")
            f.write("="*60 + "\n")
            f.write(f"ENVIRONMENT_TYPE = {self.stats['environment_type']}\n")
            f.write(f"FLOW_TYPE = {self.stats['flow_type']}\n")
            f.write(f"TIME_STEP = {self.config.TIME_STEP}\n")
            f.write(f"MAX_TIME = {self.config.MAX_TIME}\n")
            f.write(f"OUTPUT_INTERVAL = {self.config.OUTPUT_INTERVAL}\n")
            f.write(f"DIRECTIONAL_FORCE_STRENGTH = {self.config.DIRECTIONAL_FORCE_STRENGTH}\n")
            f.write(f"UNDULATION_AMPLITUDE = {self.config.UNDULATION_AMPLITUDE}\n")
            f.write(f"SPRING_CONSTANT_BASE = {self.config.SPRING_CONSTANT_BASE}\n")
            f.write(f"DAMPING_FACTOR = {self.config.DAMPING_FACTOR}\n")
            f.write("\n")
            
            # Time series files info
            f.write("="*60 + "\n")
            f.write("TIME SERIES FILES FOR PARAVIEW\n")
            f.write("="*60 + "\n")
            f.write("Load these files in ParaView for time series animation:\n")
            f.write("1. sporozoites_timeseries.pvd - for sporozoite animation\n")
            if self.use_blood_flow:
                f.write("2. blood_flow_field_timeseries.pvd - for blood flow field animation\n")
            else:
                f.write("2. tissue_field_timeseries.pvd - for tissue field animation\n")
            f.write("\nBoth files will automatically load the complete time series.\n")
            f.write("Use the Play button in ParaView to animate through time.\n")
            f.write("\n")
            
            # Current simulation stats
            f.write("="*60 + "\n")
            f.write("CURRENT SIMULATION STATUS\n")
            f.write("="*60 + "\n")
            f.write(f"Current time: {self.time:.3f}\n")
            f.write(f"Active sporozoites: {len(self.sporozoites)}\n")
            f.write(f"Total distance traveled: {self.stats['total_distance_traveled']:.2f}\n")
            f.write(f"Average motility: {self.stats['average_motility']:.3f}\n")
            f.write(f"Domain size: {self.domain_size}\n")
            f.write(f"Output directory: {self.output_dir}\n")

def main():
    """Main function with command line arguments and movement presets"""
    parser = argparse.ArgumentParser(description='Mesh-Based Sporozoite Simulation with Movement Presets')
    parser.add_argument('--sporozoites', '-s', type=int, default=5,
                       help='Number of sporozoites (default: 5, more is computationally expensive)')
    parser.add_argument('--time', '-t', type=float, default=20.0,
                       help='Maximum simulation time (default: 20.0)')
    parser.add_argument('--domain-size', nargs=3, type=float, 
                       default=[100.0, 100.0, 50.0],
                       help='Domain size [x y z] in micrometers (default: 100 100 50)')
    parser.add_argument('--output-interval', type=float, default=0.5,
                       help='VTK output interval (default: 0.5)')
    
    # Add blood flow simulation options
    parser.add_argument('--use-blood-flow', action='store_true',
                       help='Use blood flow field instead of dermal tissue field')
    parser.add_argument('--flow-type', choices=['simple_shear', 'laminar', 'turbulent'],
                       default='laminar',
                       help='Type of blood flow (default: laminar)')
    parser.add_argument('--inlet-velocity', type=float, default=50.0,
                       help='Inlet velocity in μm/s (default: 50.0)')
    parser.add_argument('--vessel-diameter', type=float, default=20.0,
                       help='Blood vessel diameter in μm (default: 20.0)')
    
    # Add movement preset options
    parser.add_argument('--movement-preset', choices=['default', 'minimal-deformation', 'realistic', 'fast', 'gentle-flexible'],
                       default='minimal-deformation',
                       help='Movement behavior preset (default: minimal-deformation)')
    
    # Add individual parameter overrides
    parser.add_argument('--time-step', type=float, 
                       help='Override time step (default depends on preset)')
    parser.add_argument('--directional-force', type=float,
                       help='Override directional force strength')
    parser.add_argument('--undulation-amplitude', type=float,
                       help='Override undulation amplitude')
    parser.add_argument('--damping', type=float,
                       help='Override damping factor')
    parser.add_argument('--spring-constant', type=float,
                       help='Override spring constant (lower = more deformation)')
    
    args = parser.parse_args()
    
    # Select configuration based on preset
    if args.movement_preset == 'minimal-deformation':
        config = PresetConfigs.minimal_deformation()
        print("Using MINIMAL DEFORMATION preset - RECOMMENDED for shape preservation")
    elif args.movement_preset == 'realistic':
        config = PresetConfigs.realistic_movement()
        print("Using REALISTIC MOVEMENT preset - balanced movement")
    elif args.movement_preset == 'fast':
        config = PresetConfigs.fast_movement()
        print("Using FAST MOVEMENT preset - quick movement with shape control")
    elif args.movement_preset == 'gentle-flexible':
        config = PresetConfigs.gentle_flexible()
        print("Using GENTLE FLEXIBLE preset - slight flexibility allowed")
    else:
        config = SimulationConfig()
        print("Using DEFAULT configuration")
    
    # Apply command line overrides
    if args.time_step is not None:
        config.TIME_STEP = args.time_step
        print(f"Overriding time step to: {args.time_step}")
    
    if args.directional_force is not None:
        config.DIRECTIONAL_FORCE_STRENGTH = args.directional_force
        print(f"Overriding directional force to: {args.directional_force}")
    
    if args.undulation_amplitude is not None:
        config.UNDULATION_AMPLITUDE = args.undulation_amplitude
        print(f"Overriding undulation amplitude to: {args.undulation_amplitude}")
    
    if args.damping is not None:
        config.DAMPING_FACTOR = args.damping
        print(f"Overriding damping factor to: {args.damping}")
    
    if args.spring_constant is not None:
        config.SPRING_CONSTANT_BASE = args.spring_constant
        print(f"Overriding spring constant to: {args.spring_constant}")
    
    # Override time and output interval from args
    config.MAX_TIME = args.time
    config.OUTPUT_INTERVAL = args.output_interval
    
    # Print configuration summary
    print(f"\nConfiguration Summary:")
    print(f"  Time step: {config.TIME_STEP}")
    print(f"  Directional force: {config.DIRECTIONAL_FORCE_STRENGTH}")
    print(f"  Undulation amplitude: {config.UNDULATION_AMPLITUDE}")
    print(f"  Spring constant: {config.SPRING_CONSTANT_BASE}")
    print(f"  Damping factor: {config.DAMPING_FACTOR}")
    print(f"  Max speed range: {config.MAX_SPEED_RANGE}")
    
    # Print blood flow configuration if enabled
    if args.use_blood_flow:
        print(f"\nBlood Flow Configuration:")
        print(f"  Flow type: {args.flow_type}")
        print(f"  Inlet velocity: {args.inlet_velocity} μm/s")
        print(f"  Vessel diameter: {args.vessel_diameter} μm")
        print(f"  Reynolds number: ~{(args.inlet_velocity * args.vessel_diameter / 1000):.0f} (estimated)")
    
    # Validate arguments
    if args.sporozoites > 10:
        print("Warning: More than 10 sporozoites may be computationally expensive")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Create and run simulation
    simulation = MeshBasedSporozoiteSimulation(
        num_sporozoites=args.sporozoites,
        domain_size=tuple(args.domain_size),
        config=config,
        use_blood_flow=args.use_blood_flow,
        flow_type=args.flow_type,
        inlet_velocity=args.inlet_velocity,
        vessel_diameter=args.vessel_diameter
    )
    
    simulation.run_simulation()
    
    # Print final tips
    print(f"\n" + "="*60)
    print("TROUBLESHOOTING TIPS:")
    print("="*60)
    print("If sporozoites are:")
    print("• Not moving enough: try --movement-preset fast or --directional-force 25")
    print("• Too deformed: try --undulation-amplitude 1.0 or --movement-preset realistic") 
    print("• Too stiff: try --damping 0.2")
    print("• Moving too fast: try --movement-preset slow-deformable or --time-step 0.05")
    print("\nExample commands:")
    print("DERMAL TISSUE SIMULATION:")
    print("  python mesh_simulation.py --movement-preset fast --sporozoites 3")
    print("  python mesh_simulation.py --directional-force 20 --undulation-amplitude 1.5")
    print("  python mesh_simulation.py --movement-preset realistic --time 10")
    print("\nBLOOD FLOW SIMULATION:")
    print("  python mesh_simulation.py --use-blood-flow --flow-type laminar --sporozoites 3")
    print("  python mesh_simulation.py --use-blood-flow --flow-type turbulent --inlet-velocity 100")
    print("  python mesh_simulation.py --use-blood-flow --flow-type simple_shear --vessel-diameter 30")
    print("  python mesh_simulation.py --use-blood-flow --flow-type laminar --movement-preset fast")

if __name__ == "__main__":
    main()