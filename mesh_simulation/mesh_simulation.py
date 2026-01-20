#!/usr/bin/env python3
"""
High-Resolution Mesh-Based Sporozoite Simulation

This simulation models sporozoites as deformable polyhedral meshes moving through
an implicit dermal tissue environment. Outputs VTK files for ParaView visualization.

Author: GitHub Copilot
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
from simulation_config import SimulationConfig, PresetConfigs

class MeshBasedSporozoiteSimulation:
    """
    High-resolution simulation with deformable mesh sporozoites
    """
    
    def __init__(self, num_sporozoites: int = 5, domain_size: tuple = (100.0, 100.0, 50.0), 
                 config=None):
        self.num_sporozoites = num_sporozoites
        self.domain_size = domain_size
        self.config = config or SimulationConfig()
        
        # Initialize dermal tissue field
        print("Generating implicit dermal tissue field...")
        self.tissue_field = ImplicitDermalTissue(domain_size)
        
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
        self.output_dir = f"vtk_output/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'active_sporozoites': num_sporozoites,
            'total_distance_traveled': 0.0,
            'average_motility': 0.0
        }
    
    def _create_sporozoites(self):
        """Create initial sporozoite meshes at injection site"""
        # Injection occurs in a small volume (salivary gland area)
        injection_center = np.array([0.0, 0.0, 20.0])  # near top of domain
        injection_radius = 5.0
        
        for i in range(self.num_sporozoites):
            # Random position within injection volume
            offset = np.random.uniform(-injection_radius, injection_radius, 3)
            initial_pos = injection_center + offset
            
            # Create sporozoite mesh with config
            sporozoite = DeformableSporozoiteMesh(i, initial_pos, self.config)
            self.sporozoites.append(sporozoite)
            
            print(f"  Created sporozoite {i}: {len(sporozoite.vertices)} vertices, "
                  f"{len(sporozoite.faces)} faces")
    
    def update_simulation_step(self):
        """Update one simulation time step"""
        active_sporozoites = []
        total_distance = 0.0
        total_motility = 0.0
        
        for sporozoite in self.sporozoites:
            if sporozoite.is_viable():
                # Store previous position for distance calculation
                prev_position = sporozoite.center_position.copy()
                
                # Apply tissue forces
                sporozoite.apply_tissue_forces(self.tissue_field, self.dt)
                
                # Apply immune response
                immune_damage = self.tissue_field.apply_immune_response(sporozoite.center_position)
                sporozoite.reduce_viability(immune_damage * self.dt)
                
                # Update sporozoite motion and deformation
                sporozoite.update_motion(self.dt, self.time)
                
                # Keep sporozoites within domain bounds
                self._apply_boundary_conditions(sporozoite)
                
                # Calculate distance traveled
                distance_moved = np.linalg.norm(sporozoite.center_position - prev_position)
                total_distance += distance_moved
                total_motility += sporozoite.motility
                
                active_sporozoites.append(sporozoite)
        
        # Update sporozoite list
        self.sporozoites = active_sporozoites
        
        # Update statistics
        self.stats['active_sporozoites'] = len(active_sporozoites)
        self.stats['total_distance_traveled'] += total_distance
        if len(active_sporozoites) > 0:
            self.stats['average_motility'] = total_motility / len(active_sporozoites)
    
    def _apply_boundary_conditions(self, sporozoite):
        """Apply boundary conditions to keep sporozoites within domain using config damping"""
        damping = self.config.BOUNDARY_DAMPING
        
        for i, vertex in enumerate(sporozoite.vertices):
            # X boundaries
            if vertex[0] <= 0:
                sporozoite.vertices[i, 0] = 0.1
                if sporozoite.velocity[i, 0] < 0:
                    sporozoite.velocity[i, 0] *= -damping
            elif vertex[0] >= self.domain_size[0]:
                sporozoite.vertices[i, 0] = self.domain_size[0] - 0.1
                if sporozoite.velocity[i, 0] > 0:
                    sporozoite.velocity[i, 0] *= -damping
            
            # Y boundaries
            if vertex[1] <= 0:
                sporozoite.vertices[i, 1] = 0.1
                if sporozoite.velocity[i, 1] < 0:
                    sporozoite.velocity[i, 1] *= -damping
            elif vertex[1] >= self.domain_size[1]:
                sporozoite.vertices[i, 1] = self.domain_size[1] - 0.1
                if sporozoite.velocity[i, 1] > 0:
                    sporozoite.velocity[i, 1] *= -damping
            
            # Z boundaries
            if vertex[2] <= 0:
                sporozoite.vertices[i, 2] = 0.1
                if sporozoite.velocity[i, 2] < 0:
                    sporozoite.velocity[i, 2] *= -damping
            elif vertex[2] >= self.domain_size[2]:
                sporozoite.vertices[i, 2] = self.domain_size[2] - 0.1
                if sporozoite.velocity[i, 2] > 0:
                    sporozoite.velocity[i, 2] *= -damping

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
            
            # Slow deformable preset
            slow_config = PresetConfigs.slow_deformable()
            f.write("# SLOW DEFORMABLE PRESET\n")
            f.write("# Command: --movement-preset slow-deformable\n")
            f.write("# Description: Highly flexible sporozoites, slow but deformable\n")
            f.write(f"SLOW_TIME_STEP = {slow_config.TIME_STEP}\n")
            f.write(f"SLOW_DIRECTIONAL_FORCE_STRENGTH = {slow_config.DIRECTIONAL_FORCE_STRENGTH}\n")
            f.write(f"SLOW_SPRING_CONSTANT_BASE = {slow_config.SPRING_CONSTANT_BASE}\n")
            f.write(f"SLOW_STIFFNESS_RANGE = {slow_config.STIFFNESS_RANGE}\n")
            f.write(f"SLOW_UNDULATION_AMPLITUDE = {slow_config.UNDULATION_AMPLITUDE}\n")
            f.write(f"SLOW_DAMPING_FACTOR = {slow_config.DAMPING_FACTOR}\n")
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
        self._write_paraview_instructions()
    
    def _write_sporozoite_vtk(self, step_number: int):
        """Write sporozoite mesh data to VTK"""
        if not self.sporozoites:
            return
        
        # Create multi-block dataset for all sporozoites
        multiblock = vtk.vtkMultiBlockDataSet()
        multiblock.SetNumberOfBlocks(len(self.sporozoites))
        
        for i, sporozoite in enumerate(self.sporozoites):
            polydata = sporozoite.to_vtk_polydata()
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
        """Write tissue field data to VTK"""
        field_data = self.tissue_field.get_field_visualization_data()
        X, Y, Z = field_data['coordinates']
        
        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(*self.tissue_field.grid_resolution)
        
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
    
    def _write_paraview_instructions(self):
        """Write instructions for loading data in ParaView"""
        instructions_file = os.path.join(self.output_dir, "ParaView_Instructions.txt")
        
        with open(instructions_file, 'w') as f:
            f.write("ParaView Visualization Instructions\n")
            f.write("====================================\n\n")
            f.write("To visualize the mesh-based sporozoite simulation:\n\n")
            f.write("1. Open ParaView\n")
            f.write("2. Load sporozoite data:\n")
            f.write("   - File > Open > sporozoites_*.vtm (select all)\n")
            f.write("   - Click 'Apply' in Properties panel\n")
            f.write("   - In toolbar, click the 'Play' button to animate\n\n")
            f.write("3. Load tissue field data:\n")
            f.write("   - File > Open > tissue_field_*.vts (select all)\n")
            f.write("   - Click 'Apply'\n")
            f.write("   - Change representation to 'Volume' for 3D field visualization\n")
            f.write("   - Or use 'Slice' filter to see cross-sections\n\n")
            f.write("4. Visualization tips:\n")
            f.write("   - Use 'Glyph' filter on flow field vectors\n")
            f.write("   - Color sporozoites by 'Viability' or 'VelocityMagnitude'\n")
            f.write("   - Add 'Streamlines' to show flow patterns\n")
            f.write("   - Use 'Clip' filter to see internal structure\n\n")
            f.write("5. Animation:\n")
            f.write("   - Set animation mode to 'Real Time'\n")
            f.write("   - Adjust time step size for smooth playback\n\n")
            f.write("Data fields available:\n")
            f.write("- Sporozoites: Viability, VelocityMagnitude, Motility, SporozoiteID\n")
            f.write("- Tissue: CollagenDensity, ImmuneCellDensity, Pressure, FlowField\n")

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
    
    # Add movement preset options
    parser.add_argument('--movement-preset', choices=['default', 'fast', 'realistic', 'slow-deformable'],
                       default='realistic',
                       help='Movement behavior preset (default: realistic)')
    
    # Add individual parameter overrides
    parser.add_argument('--time-step', type=float, 
                       help='Override time step (default depends on preset)')
    parser.add_argument('--directional-force', type=float,
                       help='Override directional force strength')
    parser.add_argument('--undulation-amplitude', type=float,
                       help='Override undulation amplitude')
    parser.add_argument('--damping', type=float,
                       help='Override damping factor')
    
    args = parser.parse_args()
    
    # Select configuration based on preset
    if args.movement_preset == 'fast':
        config = PresetConfigs.fast_movement()
        print("Using FAST MOVEMENT preset - sporozoites will move quickly")
    elif args.movement_preset == 'realistic':
        config = PresetConfigs.realistic_movement()
        print("Using REALISTIC MOVEMENT preset - balanced speed and deformation")
    elif args.movement_preset == 'slow-deformable':
        config = PresetConfigs.slow_deformable()
        print("Using SLOW DEFORMABLE preset - highly deformable, slow movement")
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
        config=config
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
    print("  python mesh_simulation.py --movement-preset fast --sporozoites 3")
    print("  python mesh_simulation.py --directional-force 20 --undulation-amplitude 1.5")
    print("  python mesh_simulation.py --movement-preset realistic --time 10")

if __name__ == "__main__":
    main()