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
from sporozoite_mesh import DeformableSporozoiteMesh
from dermal_field import ImplicitDermalTissue

class MeshBasedSporozoiteSimulation:
    """
    High-resolution simulation with deformable mesh sporozoites
    """
    
    def __init__(self, num_sporozoites: int = 5, domain_size: tuple = (100.0, 100.0, 50.0)):
        self.num_sporozoites = num_sporozoites
        self.domain_size = domain_size
        
        # Initialize dermal tissue field
        print("Generating implicit dermal tissue field...")
        self.tissue_field = ImplicitDermalTissue(domain_size)
        
        # Initialize sporozoites
        print(f"Creating {num_sporozoites} mesh-based sporozoites...")
        self.sporozoites: List[DeformableSporozoiteMesh] = []
        self._create_sporozoites()
        
        # Simulation parameters
        self.time = 0.0
        self.dt = 0.01  # smaller time step for stability
        self.max_time = 20.0  # simulation time
        self.output_interval = 0.5  # VTK output every 0.5 time units
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
            
            # Create sporozoite mesh
            sporozoite = DeformableSporozoiteMesh(i, initial_pos)
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
        """Apply boundary conditions to keep sporozoites within domain"""
        for i, vertex in enumerate(sporozoite.vertices):
            # X boundaries
            if vertex[0] <= 0:
                sporozoite.vertices[i, 0] = 0.1
                if sporozoite.velocity[i, 0] < 0:
                    sporozoite.velocity[i, 0] = 0
            elif vertex[0] >= self.domain_size[0]:
                sporozoite.vertices[i, 0] = self.domain_size[0] - 0.1
                if sporozoite.velocity[i, 0] > 0:
                    sporozoite.velocity[i, 0] = 0
            
            # Y boundaries
            if vertex[1] <= 0:
                sporozoite.vertices[i, 1] = 0.1
                if sporozoite.velocity[i, 1] < 0:
                    sporozoite.velocity[i, 1] = 0
            elif vertex[1] >= self.domain_size[1]:
                sporozoite.vertices[i, 1] = self.domain_size[1] - 0.1
                if sporozoite.velocity[i, 1] > 0:
                    sporozoite.velocity[i, 1] = 0
            
            # Z boundaries
            if vertex[2] <= 0:
                sporozoite.vertices[i, 2] = 0.1
                if sporozoite.velocity[i, 2] < 0:
                    sporozoite.velocity[i, 2] = 0
            elif vertex[2] >= self.domain_size[2]:
                sporozoite.vertices[i, 2] = self.domain_size[2] - 0.1
                if sporozoite.velocity[i, 2] > 0:
                    sporozoite.velocity[i, 2] = 0
    
    def write_vtk_output(self, step_number: int):
        """Write VTK files for current simulation state"""
        # Write sporozoite meshes
        self._write_sporozoite_vtk(step_number)
        
        # Write tissue field data
        self._write_tissue_field_vtk(step_number)
        
        # Write simulation state info
        self._write_simulation_state(step_number)
    
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
        
        # Write to file
        filename = os.path.join(self.output_dir, f"sporozoites_{step_number:04d}.vtm")
        writer = vtk.vtkXMLMultiBlockDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(multiblock)
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
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    points.InsertNextPoint(X[i,j,k], Y[i,j,k], Z[i,j,k])
        grid.SetPoints(points)
        
        # Add scalar fields
        # Collagen density
        collagen_array = vtk.vtkFloatArray()
        collagen_array.SetName("CollagenDensity")
        collagen_data = field_data['collagen_density'].ravel()
        for val in collagen_data:
            collagen_array.InsertNextValue(val)
        grid.GetPointData().SetScalars(collagen_array)
        
        # Immune cell density
        immune_array = vtk.vtkFloatArray()
        immune_array.SetName("ImmuneCellDensity")
        immune_data = field_data['immune_cells'].ravel()
        for val in immune_data:
            immune_array.InsertNextValue(val)
        grid.GetPointData().AddArray(immune_array)
        
        # Pressure field
        pressure_array = vtk.vtkFloatArray()
        pressure_array.SetName("Pressure")
        pressure_data = field_data['pressure'].ravel()
        for val in pressure_data:
            pressure_array.InsertNextValue(val)
        grid.GetPointData().AddArray(pressure_array)
        
        # Flow field (vector)
        flow_array = vtk.vtkFloatArray()
        flow_array.SetName("FlowField")
        flow_array.SetNumberOfComponents(3)
        flow_data = field_data['flow_field']
        for i in range(flow_data.shape[0]):
            for j in range(flow_data.shape[1]):
                for k in range(flow_data.shape[2]):
                    flow_array.InsertNextTuple3(flow_data[i,j,k,0], 
                                              flow_data[i,j,k,1], 
                                              flow_data[i,j,k,2])
        grid.GetPointData().SetVectors(flow_array)
        
        # Write tissue field
        filename = os.path.join(self.output_dir, f"tissue_field_{step_number:04d}.vts")
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
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
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Mesh-Based Sporozoite Simulation')
    parser.add_argument('--sporozoites', '-s', type=int, default=5,
                       help='Number of sporozoites (default: 5, more is computationally expensive)')
    parser.add_argument('--time', '-t', type=float, default=20.0,
                       help='Maximum simulation time (default: 20.0)')
    parser.add_argument('--domain-size', nargs=3, type=float, 
                       default=[100.0, 100.0, 50.0],
                       help='Domain size [x y z] in micrometers (default: 100 100 50)')
    parser.add_argument('--output-interval', type=float, default=0.5,
                       help='VTK output interval (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.sporozoites > 10:
        print("Warning: More than 10 sporozoites may be computationally expensive")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Create and run simulation
    simulation = MeshBasedSporozoiteSimulation(
        num_sporozoites=args.sporozoites,
        domain_size=tuple(args.domain_size)
    )
    
    simulation.max_time = args.time
    simulation.output_interval = args.output_interval
    
    simulation.run_simulation()

if __name__ == "__main__":
    main()