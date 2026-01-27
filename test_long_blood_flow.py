#!/usr/bin/env python3
"""
Extended Blood Flow Test - Long Channel Simulation

Test script for demonstrating sporozoite movement through a much longer blood vessel
with extended simulation time to observe transport dynamics.
"""

import sys
import os
import numpy as np

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'mesh_simulation'))
from mesh_simulation import MeshBasedSporozoiteSimulation
from simulation_config import PresetConfigs

def run_long_channel_test():
    """Run extended blood flow test with long channel"""
    
    print("=" * 70)
    print("EXTENDED BLOOD FLOW TEST - LONG CHANNEL")
    print("=" * 70)
    
    # Configuration for long channel test
    print("\nTest Configuration:")
    print("- Much longer blood vessel channel (300 x 30 x 20 μm)")
    print("- Extended simulation time (30 seconds)")
    print("- Multiple sporozoites to observe transport")
    print("- Laminar flow with realistic blood velocities")
    print("- Frequent output for detailed tracking")
    
    # Use optimized configuration for blood flow
    config = PresetConfigs.realistic_movement()
    config.MAX_TIME = 30.0  # Extended simulation time
    config.OUTPUT_INTERVAL = 2.0  # Output every 2 seconds
    config.DIRECTIONAL_FORCE_STRENGTH = 15.0  # Moderate directional force
    config.TIME_STEP = 0.1  # Slightly smaller time step for stability
    
    print(f"\nSimulation Parameters:")
    print(f"- Time step: {config.TIME_STEP}s")
    print(f"- Total time: {config.MAX_TIME}s")
    print(f"- Output interval: {config.OUTPUT_INTERVAL}s")
    print(f"- Directional force: {config.DIRECTIONAL_FORCE_STRENGTH}")
    
    # Long channel dimensions - much longer in X direction
    domain_size = (300.0, 30.0, 20.0)  # 300 μm long channel
    vessel_diameter = 15.0  # 15 μm diameter vessel
    inlet_velocity = 60.0  # 60 μm/s blood velocity
    
    print(f"\nBlood Flow Parameters:")
    print(f"- Channel length: {domain_size[0]} μm")
    print(f"- Channel cross-section: {domain_size[1]} x {domain_size[2]} μm")
    print(f"- Vessel diameter: {vessel_diameter} μm")
    print(f"- Inlet velocity: {inlet_velocity} μm/s")
    print(f"- Estimated transit time: ~{domain_size[0]/inlet_velocity:.1f} seconds")
    
    # Create simulation with long channel
    simulation = MeshBasedSporozoiteSimulation(
        num_sporozoites=3,  # Multiple sporozoites for better visualization
        domain_size=domain_size,
        config=config,
        use_blood_flow=True,
        flow_type='laminar',
        inlet_velocity=inlet_velocity,
        vessel_diameter=vessel_diameter
    )
    
    print(f"\nStarting extended blood flow simulation...")
    print(f"Expected outputs: ~{int(config.MAX_TIME / config.OUTPUT_INTERVAL)} VTK files")
    print(f"This will demonstrate:")
    print(f"- Sporozoite transport along the long vessel")
    print(f"- Blood flow effects on movement")
    print(f"- Periodic boundary conditions")
    print(f"- Velocity-dependent drag forces")
    
    # Run the simulation
    simulation.run_simulation()
    
    print(f"\n" + "=" * 70)
    print("EXTENDED SIMULATION COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {simulation.output_dir}")
    print(f"\nTo visualize in ParaView:")
    print(f"1. Load: {simulation.output_dir}/sporozoites_timeseries.pvd")
    print(f"2. Load: {simulation.output_dir}/blood_flow_field_timeseries.pvd")
    print(f"3. Use the Play button to see sporozoite transport through the long vessel")
    print(f"\nKey things to observe:")
    print(f"- Sporozoites moving with blood flow in X direction")
    print(f"- Velocity profiles in the vessel cross-section")
    print(f"- Periodic wrapping when sporozoites exit and re-enter")
    print(f"- Realistic blood flow transport dynamics")

def run_comparison_tests():
    """Run comparison tests with different flow types and velocities"""
    
    print("\n" + "=" * 70)
    print("BLOOD FLOW COMPARISON TESTS")
    print("=" * 70)
    
    # Test parameters
    base_config = PresetConfigs.realistic_movement()
    base_config.MAX_TIME = 20.0
    base_config.OUTPUT_INTERVAL = 2.0
    domain_size = (200.0, 25.0, 15.0)  # Moderate length channel
    
    test_cases = [
        {
            'name': 'Low Velocity Laminar',
            'flow_type': 'laminar',
            'inlet_velocity': 30.0,
            'vessel_diameter': 12.0
        },
        {
            'name': 'High Velocity Laminar', 
            'flow_type': 'laminar',
            'inlet_velocity': 80.0,
            'vessel_diameter': 12.0
        },
        {
            'name': 'Turbulent Flow',
            'flow_type': 'turbulent',
            'inlet_velocity': 120.0,
            'vessel_diameter': 12.0
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nRunning Test {i+1}: {test_case['name']}")
        print(f"  Flow type: {test_case['flow_type']}")
        print(f"  Inlet velocity: {test_case['inlet_velocity']} μm/s")
        print(f"  Vessel diameter: {test_case['vessel_diameter']} μm")
        
        simulation = MeshBasedSporozoiteSimulation(
            num_sporozoites=2,
            domain_size=domain_size,
            config=base_config,
            use_blood_flow=True,
            flow_type=test_case['flow_type'],
            inlet_velocity=test_case['inlet_velocity'],
            vessel_diameter=test_case['vessel_diameter']
        )
        
        simulation.run_simulation()
        print(f"  Results: {simulation.output_dir}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'comparison':
        run_comparison_tests()
    else:
        run_long_channel_test()
    
    print(f"\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)