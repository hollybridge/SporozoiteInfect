#!/usr/bin/env python3
"""
Plasmodium Sporozoite Infection Simulation

This simulation models the journey of Plasmodium sporozoites from mosquito
salivary glands through dermal tissue to blood vessels, representing the
early stages of malaria infection.

Author: GitHub Copilot
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import sys
import os
import time
import argparse
from datetime import datetime

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'stages', 'dermal'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'stages', 'vascular'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from dermal_stage import DermalStageSimulation
from vascular_stage import VascularStageSimulation
from sporozoite import Sporozoite, SporozoiteState

class SporozoiteInfectionSimulation:
    """Main simulation class coordinating both dermal and vascular stages"""
    
    def __init__(self, num_sporozoites: int = 30, enable_visualization: bool = True, save_gif: bool = False):
        self.num_sporozoites = num_sporozoites
        self.enable_visualization = enable_visualization
        self.save_gif = save_gif
        
        # Initialize stage simulations
        self.dermal_sim = DermalStageSimulation(num_sporozoites)
        self.vascular_sim = VascularStageSimulation()
        
        # Simulation parameters
        self.total_time = 0.0
        self.max_simulation_time = 50.0  # simulation time units
        self.time_step = 0.1
        
        # Statistics tracking
        self.stats = {
            'total_extruded': 0,
            'total_reached_vessels': 0,
            'total_entered_vessels': 0,
            'dermal_survival_rate': [],
            'vascular_success_rate': []
        }
        
        # GIF creation setup
        if save_gif:
            self.frames_dir = f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.frames_dir, exist_ok=True)
            self.frame_count = 0
            self.frame_interval = 2.0  # Save every 2 time units for efficiency
            self.last_frame_time = 0.0
        
        # Visualization setup
        if enable_visualization:
            self.setup_visualization()
            self.snapshots = []
    
    def setup_visualization(self):
        """Setup matplotlib visualization"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Dermal stage plot
        self.ax1.set_xlim(0, 200)
        self.ax1.set_ylim(0, 200)
        self.ax1.set_title('Dermal Stage: Salivary Gland Extrusion')
        self.ax1.set_xlabel('Position (μm)')
        self.ax1.set_ylabel('Position (μm)')
        self.ax1.grid(True, alpha=0.3)
        
        # Add salivary gland representation
        salivary_gland = Circle((50, 50), 15, color='red', alpha=0.5, 
                               label='Salivary Gland')
        self.ax1.add_patch(salivary_gland)
        
        # Vascular stage plot
        self.ax2.set_xlim(0, 200)
        self.ax2.set_ylim(0, 200)
        self.ax2.set_title('Vascular Stage: Migration to Blood Vessels')
        self.ax2.set_xlabel('Position (μm)')
        self.ax2.set_ylabel('Position (μm)')
        self.ax2.grid(True, alpha=0.3)
        
        # Draw blood vessels
        self.draw_blood_vessels()
        
        # Initialize scatter plots for sporozoites
        self.dermal_scatter = self.ax1.scatter([], [], c=[], s=[], alpha=0.7)
        self.vascular_scatter = self.ax2.scatter([], [], c=[], s=[], alpha=0.7)
        
        # Legend
        self.ax1.legend()
        
        plt.tight_layout()
    
    def draw_blood_vessels(self):
        """Draw blood vessels on the vascular plot"""
        vessel_info = self.vascular_sim.get_vessel_info()
        
        for vessel in vessel_info:
            start = vessel['start_pos']
            end = vessel['end_pos']
            diameter = vessel['diameter']
            
            # Draw vessel as a line with width proportional to diameter
            self.ax2.plot([start[0], end[0]], [start[1], end[1]], 
                         'b-', linewidth=diameter*0.5, alpha=0.6)
        
        # Add vessel legend
        self.ax2.plot([], [], 'b-', linewidth=3, alpha=0.6, label='Blood Vessels')
        self.ax2.legend()
    
    def transfer_sporozoites_to_vascular(self):
        """Transfer viable sporozoites from dermal to vascular stage"""
        transferred = 0
        
        # Check sporozoites in dermal tissue that are near the boundary
        # or have migrated sufficiently to reach vascular areas
        for sporozoite in self.dermal_sim.dermal_tissue.sporozoites[:]:
            # Transfer conditions: reached edge of dermal area or random migration success
            x, y = sporozoite.position
            near_edge = (x > 180 or y > 180 or x < 20 or y < 20)
            random_success = np.random.random() < 0.02  # Small probability per time step
            
            if (near_edge or random_success) and sporozoite.is_viable():
                # Remove from dermal simulation
                self.dermal_sim.dermal_tissue.sporozoites.remove(sporozoite)
                
                # Add to vascular simulation
                self.vascular_sim.add_sporozoite_from_dermis(sporozoite)
                transferred += 1
                self.stats['total_reached_vessels'] += 1
        
        return transferred
    
    def run_simulation_step(self):
        """Run one step of the complete simulation"""
        # Run dermal stage
        dermal_stats = self.dermal_sim.run_simulation_step()
        
        # Transfer sporozoites to vascular stage
        transferred = self.transfer_sporozoites_to_vascular()
        
        # Run vascular stage
        vascular_stats = self.vascular_sim.run_simulation_step()
        
        # Update statistics
        self.stats['total_extruded'] += dermal_stats['extruded_this_step']
        self.stats['total_entered_vessels'] += vascular_stats['newly_entered']
        
        # Calculate survival rates
        if self.stats['total_extruded'] > 0:
            survival_rate = self.stats['total_reached_vessels'] / self.stats['total_extruded']
            self.stats['dermal_survival_rate'].append(survival_rate)
        
        if self.stats['total_reached_vessels'] > 0:
            success_rate = self.stats['total_entered_vessels'] / self.stats['total_reached_vessels']
            self.stats['vascular_success_rate'].append(success_rate)
        
        return {
            'dermal': dermal_stats,
            'vascular': vascular_stats,
            'transferred': transferred,
            'time': self.total_time
        }
    
    def update_visualization(self):
        """Update the visualization with current sporozoite positions"""
        if not self.enable_visualization:
            return
        
        # Get sporozoite positions
        dermal_positions = self.dermal_sim.get_sporozoite_positions()
        vascular_positions = self.vascular_sim.get_sporozoite_positions()
        
        # Update dermal stage plot
        if dermal_positions:
            dermal_x = [pos['position'][0] for pos in dermal_positions]
            dermal_y = [pos['position'][1] for pos in dermal_positions]
            
            # Color code by state
            colors = []
            sizes = []
            for pos in dermal_positions:
                if pos['state'] == 'salivary_gland':
                    colors.append('red')
                    sizes.append(20)
                elif pos['state'] == 'dermal_tissue':
                    colors.append('orange')
                    sizes.append(15)
                else:
                    colors.append('yellow')
                    sizes.append(10)
            
            self.dermal_scatter.set_offsets(np.column_stack([dermal_x, dermal_y]))
            self.dermal_scatter.set_color(colors)
            self.dermal_scatter.set_sizes(sizes)
        else:
            self.dermal_scatter.set_offsets(np.empty((0, 2)))
        
        # Update vascular stage plot
        if vascular_positions:
            vascular_x = [pos['position'][0] for pos in vascular_positions]
            vascular_y = [pos['position'][1] for pos in vascular_positions]
            
            # Color code by state
            colors = []
            sizes = []
            for pos in vascular_positions:
                if pos['state'] == 'migrating':
                    colors.append('green')
                    sizes.append(12)
                elif pos['state'] == 'blood_vessel':
                    colors.append('blue')
                    sizes.append(8)
                else:
                    colors.append('purple')
                    sizes.append(6)
            
            self.vascular_scatter.set_offsets(np.column_stack([vascular_x, vascular_y]))
            self.vascular_scatter.set_color(colors)
            self.vascular_scatter.set_sizes(sizes)
        else:
            self.vascular_scatter.set_offsets(np.empty((0, 2)))
        
        # Update titles with current counts
        dermal_count = len(dermal_positions)
        vascular_count = len(vascular_positions)
        
        self.ax1.set_title(f'Dermal Stage (Count: {dermal_count}, Time: {self.total_time:.1f})')
        self.ax2.set_title(f'Vascular Stage (Count: {vascular_count}, Entered: {self.stats["total_entered_vessels"]})')
        
        # Save frame for GIF if enabled and at interval
        if self.save_gif and (self.total_time - self.last_frame_time) >= self.frame_interval:
            frame_filename = os.path.join(self.frames_dir, f"frame_{self.frame_count:04d}.png")
            self.fig.savefig(frame_filename, dpi=80, bbox_inches='tight')
            self.frame_count += 1
            self.last_frame_time = self.total_time
    
    def create_gif_from_frames(self, gif_filename: str = "simulation.gif"):
        """Create GIF from saved frame images"""
        if not self.save_gif or self.frame_count == 0:
            return
            
        print(f"Creating GIF from {self.frame_count} frames...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = f"sporozoite_simulation_{timestamp}.gif"
        
        try:
            # Import imageio with v2 API to avoid deprecation warnings
            import imageio.v2 as imageio
            images = []
            
            for i in range(self.frame_count):
                frame_path = os.path.join(self.frames_dir, f"frame_{i:04d}.png")
                if os.path.exists(frame_path):
                    images.append(imageio.imread(frame_path))
            
            if images:
                # Save as GIF with optimized settings for file size
                imageio.mimsave(gif_path, images, fps=5, loop=0)
                print(f"GIF saved as {gif_path}")
                
                # Clean up frame files to save space
                import shutil
                shutil.rmtree(self.frames_dir)
                print(f"Cleaned up temporary frames directory: {self.frames_dir}")
            
        except ImportError as e:
            print(f"imageio not available: {e}")
            # Fallback: create simple script for manual GIF creation
            script_path = "create_gif.sh"
            with open(script_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"# Create GIF from frames in {self.frames_dir}\n")
                f.write("# Using Python3 with imageio (preferred method)\n")
                f.write(f"python3 -c \"\n")
                f.write(f"import imageio.v2 as imageio\n")
                f.write(f"import glob\n")
                f.write(f"frames = sorted(glob.glob('{self.frames_dir}/frame_*.png'))\n")
                f.write(f"images = [imageio.imread(f) for f in frames]\n")
                f.write(f"imageio.mimsave('{gif_path}', images, fps=5, loop=0)\n")
                f.write(f"print('GIF created: {gif_path}')\n")
                f.write(f'"\n')
                f.write("\n# Alternative ffmpeg method (requires ffmpeg installation):\n")
                f.write(f"# ffmpeg -r 5 -i {self.frames_dir}/frame_%04d.png -vf palettegen palette.png\n")
                f.write(f"# ffmpeg -r 5 -i {self.frames_dir}/frame_%04d.png -i palette.png -lavfi paletteuse {gif_path}\n")
                f.write("# rm palette.png\n")
            
            print(f"imageio not available. Created script '{script_path}' with Python3 method")
            print("Run: chmod +x create_gif.sh && ./create_gif.sh")
    
    def run_complete_simulation(self, display_interval: float = 1.0):
        """Run the complete simulation"""
        print("Starting Plasmodium Sporozoite Infection Simulation...")
        print(f"Initial sporozoites: {self.num_sporozoites}")
        if self.save_gif:
            print(f"GIF frames will be saved every {self.frame_interval} time units")
        print("-" * 50)
        
        step_count = 0
        last_display_time = 0
        
        while self.total_time < self.max_simulation_time:
            # Run simulation step
            step_stats = self.run_simulation_step()
            self.total_time += self.time_step
            step_count += 1
            
            # Update visualization
            if self.enable_visualization and (self.total_time - last_display_time) >= display_interval:
                self.update_visualization()
                plt.pause(0.01)
                last_display_time = self.total_time
            
            # Print progress periodically
            if step_count % 100 == 0:
                self.print_progress_report()
            
            # Check termination conditions
            total_active = (step_stats['dermal']['dermal_tissue_count'] + 
                          step_stats['dermal']['salivary_gland_count'] +
                          step_stats['vascular']['migrating_count'] + 
                          step_stats['vascular']['in_vessels_count'])
            
            if total_active == 0:
                print("All sporozoites eliminated or completed infection cycle.")
                break
        
        print("\nSimulation completed!")
        self.print_final_report()
        
        # Create GIF if enabled
        if self.save_gif:
            self.create_gif_from_frames()
        
        if self.enable_visualization:
            plt.show()
    
    def print_progress_report(self):
        """Print current simulation progress"""
        print(f"Time: {self.total_time:6.1f} | "
              f"Extruded: {self.stats['total_extruded']:3d} | "
              f"Reached vessels: {self.stats['total_reached_vessels']:3d} | "
              f"Entered vessels: {self.stats['total_entered_vessels']:3d}")
    
    def print_final_report(self):
        """Print final simulation statistics"""
        print("\n" + "="*60)
        print("FINAL SIMULATION REPORT")
        print("="*60)
        print(f"Total simulation time: {self.total_time:.1f} time units")
        print(f"Initial sporozoites: {self.num_sporozoites}")
        print(f"Total extruded from salivary gland: {self.stats['total_extruded']}")
        print(f"Total reached vascular areas: {self.stats['total_reached_vessels']}")
        print(f"Total entered blood vessels: {self.stats['total_entered_vessels']}")
        
        # Calculate final rates
        if self.stats['total_extruded'] > 0:
            extrusion_rate = (self.stats['total_extruded'] / self.num_sporozoites) * 100
            print(f"Extrusion success rate: {extrusion_rate:.1f}%")
            
            if self.stats['total_reached_vessels'] > 0:
                dermal_survival = (self.stats['total_reached_vessels'] / self.stats['total_extruded']) * 100
                print(f"Dermal survival rate: {dermal_survival:.1f}%")
                
                if self.stats['total_entered_vessels'] > 0:
                    vessel_entry = (self.stats['total_entered_vessels'] / self.stats['total_reached_vessels']) * 100
                    print(f"Vessel entry success rate: {vessel_entry:.1f}%")
                    
                    overall_success = (self.stats['total_entered_vessels'] / self.num_sporozoites) * 100
                    print(f"Overall infection success rate: {overall_success:.1f}%")
        
        print("="*60)

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Plasmodium Sporozoite Infection Simulation')
    parser.add_argument('--sporozoites', '-s', type=int, default=30,
                       help='Number of initial sporozoites (default: 30)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization (run headless)')
    parser.add_argument('--time', '-t', type=float, default=50.0,
                       help='Maximum simulation time (default: 50.0)')
    parser.add_argument('--gif', action='store_true',
                       help='Save simulation as animated GIF')
    
    args = parser.parse_args()
    
    # Create and run simulation
    simulation = SporozoiteInfectionSimulation(
        num_sporozoites=args.sporozoites,
        enable_visualization=not args.no_viz,
        save_gif=args.gif
    )
    
    if args.time != 50.0:
        simulation.max_simulation_time = args.time
    
    simulation.run_complete_simulation()

if __name__ == "__main__":
    main()