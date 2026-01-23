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
            # Use safe area near top of domain (salivary gland area)
            injection_center = np.array([
                (safe_min[0] + safe_max[0]) / 2,  # Center X
                (safe_min[1] + safe_max[1]) / 2,  # Center Y  
                safe_max[2] - safety_margin / 2   # Near top Z
            ])
            injection_radius = min(
                (safe_max[0] - safe_min[0]) / 4,  # Quarter of safe X range
                (safe_max[1] - safe_min[1]) / 4,  # Quarter of safe Y range
                safety_margin / 2                 # Half the safety margin in Z
            )
        
        print(f"Safe injection center: {injection_center}")
        print(f"Safe injection radius: {injection_radius}")
        print(f"Safety margin: {safety_margin}")
        
        for i in range(self.num_sporozoites):
            # Generate safe random position within injection volume
            max_attempts = 50  # Prevent infinite loops
            attempts = 0
            
            while attempts < max_attempts:
                # Random position within injection volume
                offset = np.random.uniform(-injection_radius, injection_radius, 3)
                initial_pos = injection_center + offset
                
                # Verify position is safe (well within domain bounds)
                if (np.all(initial_pos >= safe_min) and np.all(initial_pos <= safe_max)):
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
                sporozoite.update_motion(self.dt, forces, 
                                       spring_constant=self.config.SPRING_CONSTANT_BASE,
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
        """Apply boundary conditions using rigid body approach - FIXED VERSION WITH DEBUGGING"""
        # Check if center of mass is near boundaries and apply corrections to center only
        center = sporozoite.center_position
        damping = self.config.BOUNDARY_DAMPING
        
        # DEBUG: Check if any vertices are out of bounds BEFORE correction
        
        vertices_out_of_bounds = []
        for i, vertex in enumerate(sporozoite.vertices):
            if (vertex[0] <= 0 or vertex[0] >= self.domain_size[0] or
                vertex[1] <= 0 or vertex[1] >= self.domain_size[1] or
                vertex[2] <= 0 or vertex[2] >= self.domain_size[2]):
                vertices_out_of_bounds.append((i, vertex.copy()))
        
        if vertices_out_of_bounds:
            #print(f"BOUNDARY WARNING: Sporozoite {sporozoite.id} has {len(vertices_out_of_bounds)} vertices out of bounds")
            for i, vertex in vertices_out_of_bounds:
                #print(f"  Vertex {i}: {vertex}")
                pass
        
        # Calculate how much the center needs to be moved to keep all vertices in bounds
        center_correction = np.array([0.0, 0.0, 0.0])
        
        # Check if any vertices are out of bounds and calculate center correction
        for vertex in sporozoite.vertices:
            # X boundaries
            if vertex[0] <= 0:
                correction_needed = 0.1 - vertex[0]
                center_correction[0] = max(center_correction[0], correction_needed)
            elif vertex[0] >= self.domain_size[0]:
                correction_needed = self.domain_size[0] - 0.1 - vertex[0]
                center_correction[0] = min(center_correction[0], correction_needed)
            
            # Y boundaries
            if vertex[1] <= 0:
                correction_needed = 0.1 - vertex[1]
                center_correction[1] = max(center_correction[1], correction_needed)
            elif vertex[1] >= self.domain_size[1]:
                correction_needed = self.domain_size[1] - 0.1 - vertex[1]
                center_correction[1] = min(center_correction[1], correction_needed)
            
            # Z boundaries
            if vertex[2] <= 0:
                correction_needed = 0.1 - vertex[2]
                center_correction[2] = max(center_correction[2], correction_needed)
            elif vertex[2] >= self.domain_size[2]:
                correction_needed = self.domain_size[2] - 0.1 - vertex[2]
                center_correction[2] = min(center_correction[2], correction_needed)
        
        # Apply center correction if needed (rigid body translation)
        if np.linalg.norm(center_correction) > 1e-8:
            #print(f"BOUNDARY CORRECTION: Sporozoite {sporozoite.id} center moved by {center_correction}")
            #print(f"  Center before: {sporozoite.center_position}")
            
            sporozoite.center_position += center_correction
            
            # Update all vertices rigidly with the center correction
            for i in range(len(sporozoite.vertices)):
                sporozoite.vertices[i] += center_correction
            
            #print(f"  Center after: {sporozoite.center_position}")
            
            # Apply damping to all vertex velocities uniformly (rigid body)
            for i in range(len(sporozoite.velocity)):
                sporozoite.velocity[i] *= damping
            
            #print(f"  Applied damping factor: {damping}")
        
        # DEBUG: Final check if correction worked
        vertices_still_out = 0
        for vertex in sporozoite.vertices:
            if (vertex[0] <= 0 or vertex[0] >= self.domain_size[0] or
                vertex[1] <= 0 or vertex[1] >= self.domain_size[1] or
                vertex[2] <= 0 or vertex[2] >= self.domain_size[2]):
                vertices_still_out += 1
        
        if vertices_still_out > 0:
            print(f"BOUNDARY ERROR: {vertices_still_out} vertices still out of bounds after correction!")
    
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
            f.write(f"TIME_STEP = {self.config.TIME_STEP}\n")
            f.write(f"MAX_TIME = {self.config.MAX_TIME}\n")
            f.write(f"OUTPUT_INTERVAL = {self.config.OUTPUT_INTERVAL}\n")
            f.write(f"DIRECTIONAL_FORCE_STRENGTH = {self.config.DIRECTIONAL_FORCE_STRENGTH}\n")
            f.write(f"MOTILITY_RANGE = {self.config.MOTILITY_RANGE}\n")
            f.write(f"MAX_SPEED_RANGE = {self.config.MAX_SPEED_RANGE}\n")
            f.write(f"UNDULATION_AMPLITUDE = {self.config.UNDULATION_AMPLITUDE}\n")
            f.write(f"UNDULATION_FREQUENCY_RANGE = {self.config.UNDULATION_FREQUENCY_RANGE}\n")
            f.write(f"UNDULATION_FORCE_SCALE = {self.config.UNDULATION_FORCE_SCALE}\n")
            f.write(f"SPRING_CONSTANT_BASE = {self.config.SPRING_CONSTANT_BASE}\n")
            f.write(f"STIFFNESS_RANGE = {self.config.STIFFNESS_RANGE}\n")
            f.write(f"DAMPING_FACTOR = {self.config.DAMPING_FACTOR}\n")
            f.write(f"MASS = {self.config.MASS}\n")
            f.write(f"TISSUE_RESISTANCE_SCALE = {self.config.TISSUE_RESISTANCE_SCALE}\n")
            f.write(f"TISSUE_FLOW_SCALE = {self.config.TISSUE_FLOW_SCALE}\n")
            f.write(f"RANDOM_DIRECTION_PROBABILITY = {self.config.RANDOM_DIRECTION_PROBABILITY}\n")
            f.write(f"MAX_DIRECTION_CHANGE = {self.config.MAX_DIRECTION_CHANGE}\n")
            f.write(f"BOUNDARY_DAMPING = {self.config.BOUNDARY_DAMPING}\n")
            f.write(f"SPOROZOITE_LENGTH_RANGE = {self.config.SPOROZOITE_LENGTH_RANGE}\n")
            f.write(f"SPOROZOITE_DIAMETER_RANGE = {self.config.SPOROZOITE_DIAMETER_RANGE}\n")
            f.write(f"LONGITUDINAL_SEGMENTS = {self.config.LONGITUDINAL_SEGMENTS}\n")
            f.write(f"RADIAL_SEGMENTS = {self.config.RADIAL_SEGMENTS}\n")
            f.write("\n")
            
            # Preset configurations
            f.write("="*60 + "\n")
            f.write("AVAILABLE PRESETS\n")
            f.write("="*60 + "\n")
            f.write("\n")
            
            # Fast movement preset
            fast_config = PresetConfigs.fast_movement()
            f.write("# FAST MOVEMENT PRESET\n")
            f.write("# Command: --movement-preset fast\n")
            f.write("# Description: Quick-moving sporozoites with minimal deformation\n")
            f.write(f"FAST_TIME_STEP = {fast_config.TIME_STEP}\n")
            f.write(f"FAST_DIRECTIONAL_FORCE_STRENGTH = {fast_config.DIRECTIONAL_FORCE_STRENGTH}\n")
            f.write(f"FAST_MAX_SPEED_RANGE = {fast_config.MAX_SPEED_RANGE}\n")
            f.write(f"FAST_UNDULATION_AMPLITUDE = {fast_config.UNDULATION_AMPLITUDE}\n")
            f.write(f"FAST_DAMPING_FACTOR = {fast_config.DAMPING_FACTOR}\n")
            f.write(f"FAST_SPRING_CONSTANT_BASE = {fast_config.SPRING_CONSTANT_BASE}\n")
            f.write("\n")
            
            # Realistic movement preset
            realistic_config = PresetConfigs.realistic_movement()
            f.write("# REALISTIC MOVEMENT PRESET (DEFAULT)\n")
            f.write("# Command: --movement-preset realistic\n")
            f.write("# Description: Balanced speed and deformation, biologically plausible\n")
            f.write(f"REALISTIC_TIME_STEP = {realistic_config.TIME_STEP}\n")
            f.write(f"REALISTIC_DIRECTIONAL_FORCE_STRENGTH = {realistic_config.DIRECTIONAL_FORCE_STRENGTH}\n")
            f.write(f"REALISTIC_MAX_SPEED_RANGE = {realistic_config.MAX_SPEED_RANGE}\n")
            f.write(f"REALISTIC_UNDULATION_AMPLITUDE = {realistic_config.UNDULATION_AMPLITUDE}\n")
            f.write(f"REALISTIC_UNDULATION_FORCE_SCALE = {realistic_config.UNDULATION_FORCE_SCALE}\n")
            f.write(f"REALISTIC_DAMPING_FACTOR = {realistic_config.DAMPING_FACTOR}\n")
            f.write(f"REALISTIC_SPRING_CONSTANT_BASE = {realistic_config.SPRING_CONSTANT_BASE}\n")
            f.write("\n")

            
            # Command examples
            f.write("="*60 + "\n")
            f.write("COMMAND LINE EXAMPLES\n")
            f.write("="*60 + "\n")
            f.write("# Use preset configurations:\n")
            f.write("python mesh_simulation.py --movement-preset fast --sporozoites 3\n")
            f.write("python mesh_simulation.py --movement-preset realistic --sporozoites 5\n")
            f.write("python mesh_simulation.py --movement-preset slow-deformable --sporozoites 2\n")
            f.write("\n")
            f.write("# Custom parameter overrides:\n")
            f.write("python mesh_simulation.py --directional-force 20.0 --undulation-amplitude 1.5\n")
            f.write("python mesh_simulation.py --time-step 0.05 --damping 0.2 --sporozoites 3\n")
            f.write("python mesh_simulation.py --movement-preset realistic --directional-force 25.0\n")
            f.write("\n")
            f.write("# Troubleshooting combinations:\n")
            f.write("# Not moving enough:\n")
            f.write("python mesh_simulation.py --directional-force 25.0 --time-step 0.15\n")
            f.write("# Too deformed/wrinkled:\n")
            f.write("python mesh_simulation.py --undulation-amplitude 1.0 --spring-constant 30.0\n")
            f.write("# Too stiff:\n")
            f.write("python mesh_simulation.py --damping 0.2 --spring-constant 10.0\n")
            f.write("# Moving too fast:\n")
            f.write("python mesh_simulation.py --time-step 0.05 --directional-force 8.0\n")
            f.write("\n")
            
            # Parameter explanations
            f.write("="*60 + "\n")
            f.write("PARAMETER DESCRIPTIONS\n")
            f.write("="*60 + "\n")
            f.write("TIME_STEP: Simulation time increment (smaller = more stable, larger = faster)\n")
            f.write("DIRECTIONAL_FORCE_STRENGTH: Forward propulsion strength (higher = faster movement)\n")
            f.write("UNDULATION_AMPLITUDE: Swimming motion amplitude (higher = more undulation)\n")
            f.write("UNDULATION_FORCE_SCALE: Strength of undulation forces\n")
            f.write("SPRING_CONSTANT_BASE: Mesh stiffness (higher = less deformation)\n")
            f.write("DAMPING_FACTOR: Velocity decay (higher = more damping)\n")
            f.write("MASS: Sporozoite mass (lower = more responsive to forces)\n")
            f.write("MAX_SPEED_RANGE: Range of maximum speeds for random sporozoites\n")
            f.write("MOTILITY_RANGE: Range of motility factors (0-1)\n")
            f.write("TISSUE_RESISTANCE_SCALE: How much tissue resists movement\n")
            f.write("TISSUE_FLOW_SCALE: How much tissue flow affects sporozoites\n")
            f.write("BOUNDARY_DAMPING: Velocity reduction at domain boundaries\n")
            f.write("RANDOM_DIRECTION_PROBABILITY: Chance of changing direction per step\n")
            f.write("MAX_DIRECTION_CHANGE: Maximum angle change in radians\n")
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
        
        # Update the time series collection file
        self._update_time_series_collection(step_number)
    
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
    
    <!-- Tissue field reader -->
    <Proxy group="sources" type="XMLStructuredGridReader" id="200">
      <Property name="FileName">
        <Element index="0" value="{os.path.abspath(self.output_dir)}/tissue_field_*.vts"/>
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
    
    <!-- Tissue field representation -->
    <Proxy group="representations" type="UniformGridRepresentation" id="201">
      <Property name="Input" proxy="200"/>
      <Property name="ColorArrayName">
        <Element index="0" value="CollagenDensity"/>
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
    
    def write_vtk_output(self, step_number: int):
        """Write VTK files for current simulation state"""
        # Write sporozoite meshes
        self._write_sporozoite_vtk(step_number)
        
        # Write tissue field data
        self._write_tissue_field_vtk(step_number)
        
        # Write simulation state info
        self._write_simulation_state(step_number)
        
        # Write presets configuration file (only once at the beginning)
        if step_number == 0:
            self._write_presets_config_file()
            self._write_time_series_collection_file()
    
    def _write_time_series_collection_file(self):
        """Write a ParaView time series collection file for proper animation"""
        # This creates a .pvd file that explicitly tells ParaView about the time series
        collection_file = os.path.join(self.output_dir, "sporozoites_timeseries.pvd")
        
        with open(collection_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            
            # We'll need to update this file each time we write output
            # For now, just create the header
            f.write('    <!-- Time series data will be added here -->\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        # Store the collection file path for updates
        self.collection_file_path = collection_file
    
    def _update_time_series_collection(self, step_number: int):
        """Update the time series collection file with new time step"""
        if not hasattr(self, 'collection_file_path'):
            return
            
        # Read existing content
        try:
            with open(self.collection_file_path, 'r') as f:
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
        
        # Create the new entry
        vtm_file = f"sporozoites_{step_number:04d}.vtm"
        new_entry = f'    <DataSet timestep="{self.time:.6f}" group="" part="0" file="{vtm_file}"/>\n'
        
        # Insert the new entry
        lines.insert(insert_index, new_entry)
        
        # Write back the updated file
        with open(self.collection_file_path, 'w') as f:
            # Remove the comment line if it exists
            for line in lines:
                if '<!-- Time series data will be added here -->' not in line:
                    f.write(line)

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