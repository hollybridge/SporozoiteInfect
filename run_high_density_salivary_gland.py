#!/usr/bin/env python3
"""
High-Density Salivary Gland Simulation Runner

This script runs a high-density sporozoite interaction simulation specifically
focused on the salivary gland environment with Lennard-Jones potential interactions.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add paths for local imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'stages', 'salivary_gland'))

from stages.salivary_gland.salivary_gland_simulation import SporozoiteInteractionSimulation, SporozoiteInteractionConfigs

def run_high_density_simulation():
    """Run a high-density sporozoite interaction simulation"""
    
    print("=" * 70)
    print("HIGH-DENSITY SALIVARY GLAND SPOROZOITE SIMULATION")
    print("=" * 70)
    print("This simulation models sporozoite-sporozoite interactions in a")
    print("high-density salivary gland environment using Lennard-Jones potential.")
    print()
    
    # High-density simulation parameters
    num_sporozoites = 200  # High density
    domain_size = (80.0, 80.0, 10.0)  # Smaller domain for higher density
    simulation_time = 15.0  # Shorter simulation for high density
    
    # Calculate expected density
    domain_volume = np.prod(domain_size)
    typical_sporozoite_volume = (4.0/3.0) * np.pi * (7.5) * (0.75)**2  # ~11.8 μm³
    total_sporozoite_volume = num_sporozoites * typical_sporozoite_volume
    volume_fraction = total_sporozoite_volume / domain_volume
    
    print(f"SIMULATION PARAMETERS:")
    print(f"  Number of sporozoites: {num_sporozoites}")
    print(f"  Domain size: {domain_size[0]} × {domain_size[1]} × {domain_size[2]} μm")
    print(f"  Domain volume: {domain_volume:.1f} μm³")
    print(f"  Expected volume fraction: {volume_fraction*100:.2f}%")
    print(f"  Density regime: {'HIGH DENSITY' if volume_fraction > 0.05 else 'MEDIUM DENSITY'}")
    print(f"  Simulation time: {simulation_time} seconds")
    print()
    
    # Use high-density configuration with smaller timestep for circular motion
    config = SporozoiteInteractionConfigs.high_density()
    config.TIME_STEP = 0.01  # Much smaller timestep (10 milliseconds) for smoother circular motion
    config.MAX_TIME = simulation_time
    config.OUTPUT_INTERVAL = 0.1  # More frequent output (every 0.1 seconds) to capture circular motion
    
    print(f"PHYSICS CONFIGURATION:")
    print(f"  Time step: {config.TIME_STEP} s")
    print(f"  Directional force: {config.DIRECTIONAL_FORCE_STRENGTH}")
    print(f"  Spring constant: {config.SPRING_CONSTANT_BASE}")
    print(f"  Damping factor: {config.DAMPING_FACTOR}")
    print(f"  Undulation amplitude: {config.UNDULATION_AMPLITUDE}")
    print()
    
    # Create and run simulation
    print("Creating simulation...")
    simulation = SporozoiteInteractionSimulation(
        num_sporozoites=num_sporozoites,
        domain_size=domain_size,
        config=config
    )
    
    print()
    print("Starting simulation...")
    print("This may take several minutes due to high sporozoite density...")
    print("=" * 70)
    
    # Run the simulation
    output_dir = simulation.run_simulation()
    
    print("=" * 70)
    print("SIMULATION COMPLETED!")
    print("=" * 70)
    
    # Get final statistics
    final_stats = simulation.get_simulation_statistics()
    
    print("FINAL RESULTS:")
    print(f"  Final time: {final_stats.get('time', 0):.2f} seconds")
    print(f"  Surviving sporozoites: {final_stats.get('alive_sporozoites', 0)}")
    print(f"  Final volume fraction: {final_stats.get('volume_fraction_percent', 0):.3f}%")
    print(f"  Density regime: {final_stats.get('density_regime', 'unknown')}")
    print(f"  Number of clusters: {final_stats.get('num_clusters', 0)}")
    print(f"  Largest cluster size: {final_stats.get('largest_cluster', 0)}")
    print()
    
    print("LENNARD-JONES INTERACTION RESULTS:")
    print(f"  Total LJ energy: {final_stats.get('total_lj_energy', 0):.2f}")
    print(f"  LJ interactions: {final_stats.get('num_lj_interactions', 0)}")
    print(f"  Maximum LJ force: {final_stats.get('max_lj_force', 0):.3f}")
    print(f"  Minimum distance: {final_stats.get('min_distance', 0):.3f} μm")
    print()
    
    print("OUTPUT FILES:")
    print(f"  VTK output directory: {output_dir}")
    print(f"  ParaView time series: sporozoites_timeseries.pvd")
    print(f"  Statistics files: statistics_XXXX.txt")
    print()
    
    print("VISUALIZATION INSTRUCTIONS:")
    print("1. Open ParaView")
    print("2. Load 'sporozoites_timeseries.pvd' for animation")
    print("3. Color by 'SporozoiteID' to track individual sporozoites")
    print("4. Color by 'Viability' to see health changes")
    print("5. Use the Play button to animate through time")
    print()
    
    return output_dir

if __name__ == "__main__":
    try:
        output_dir = run_high_density_simulation()
        print(f"SUCCESS: Simulation completed. Output in: {output_dir}")
    except Exception as e:
        print(f"ERROR: Simulation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)