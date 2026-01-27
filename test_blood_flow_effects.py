#!/usr/bin/env python3
"""
Test Blood Flow Effects on Sporozoites

This script creates a controlled test where sporozoites are forced to start at (0,0,0)
to clearly observe blood flow effects without random positioning.

Author: Holly Evans
Date: January 2026
"""

import sys
import os
import numpy as np

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'mesh_simulation'))

# Direct imports from the mesh_simulation directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mesh_simulation'))
from mesh_simulation import MeshBasedSporozoiteSimulation
from simulation_config import SimulationConfig, PresetConfigs

class BloodFlowTestSimulation(MeshBasedSporozoiteSimulation):
    """
    Modified simulation that forces sporozoites to start at origin for testing blood flow effects
    """
    
    def _create_sporozoites(self):
        """Override to create sporozoites at controlled positions for testing"""
        print("\n=== BLOOD FLOW TEST MODE ===")
        print("Forcing sporozoites to start at controlled positions")
        
        # Import sporozoite mesh here to avoid circular imports
        from sporozoite_mesh import DeformableSporozoiteMesh
        
        for i in range(self.num_sporozoites):
            # Force sporozoites to start at different controlled positions for comparison
            if i == 0:
                # First sporozoite at origin
                initial_pos = np.array([0.1, 0.1, 0.1])  # Slightly offset from exact 0,0,0 for numerical stability
                print(f"  Sporozoite {i}: ORIGIN TEST at {initial_pos}")
            elif i == 1 and self.num_sporozoites > 1:
                # Second sporozoite at vessel center
                center_y = self.domain_size[1] / 2
                center_z = self.domain_size[2] / 2
                initial_pos = np.array([5.0, center_y, center_z])
                print(f"  Sporozoite {i}: VESSEL CENTER TEST at {initial_pos}")
            elif i == 2 and self.num_sporozoites > 2:
                # Third sporozoite near vessel wall
                center_y = self.domain_size[1] / 2
                center_z = self.domain_size[2] / 2
                wall_offset = 8.0  # Near wall but not at wall
                initial_pos = np.array([10.0, center_y + wall_offset, center_z])
                print(f"  Sporozoite {i}: NEAR WALL TEST at {initial_pos}")
            else:
                # Additional sporozoites at various test positions
                test_positions = [
                    [15.0, 10.0, 10.0],
                    [20.0, 15.0, 15.0],
                    [25.0, 20.0, 20.0]
                ]
                pos_idx = (i - 3) % len(test_positions)
                initial_pos = np.array(test_positions[pos_idx])
                print(f"  Sporozoite {i}: TEST POSITION {pos_idx+1} at {initial_pos}")
            
            # Create sporozoite mesh with config
            sporozoite = DeformableSporozoiteMesh(i, initial_pos, self.config)
            self.sporozoites.append(sporozoite)
            
            # Verify position and check if in vessel
            in_vessel = "UNKNOWN"
            try:
                if hasattr(self.environment_field, 'is_point_in_vessel'):
                    in_vessel = "YES" if self.environment_field.is_point_in_vessel(initial_pos) else "NO"
                    
                # Get flow velocity at this position
                flow_velocity = self.environment_field.get_velocity_at_point(initial_pos)
                flow_speed = np.linalg.norm(flow_velocity)
                
                print(f"    In vessel: {in_vessel}")
                print(f"    Flow velocity: [{flow_velocity[0]:.2f}, {flow_velocity[1]:.2f}, {flow_velocity[2]:.2f}] μm/s")
                print(f"    Flow speed: {flow_speed:.2f} μm/s")
                
            except Exception as e:
                print(f"    Could not check vessel status: {e}")
            
            print()

def run_blood_flow_test():
    """Run the blood flow test simulation"""
    
    print("Blood Flow Effect Test Simulation")
    print("="*50)
    
    # Test different flow types
    flow_types = ["laminar", "turbulent", "simple_shear"]
    
    for flow_type in flow_types:
        print(f"\n{'='*60}")
        print(f"TESTING {flow_type.upper()} FLOW")
        print(f"{'='*60}")
        
        # Create test configuration
        config = PresetConfigs.fast_movement()  # Use fast preset for quick visible effects
        config.TIME_STEP = 0.05  # Smaller time step for stability
        config.OUTPUT_INTERVAL = 0.1  # More frequent outputs
        config.MAX_TIME = 5.0  # Short test simulation
        
        # Enhanced forces to make blood flow effects more visible
        config.TISSUE_FLOW_SCALE = 5.0  # Increase flow effect
        config.DIRECTIONAL_FORCE_STRENGTH = 5.0  # Reduce self-propulsion to see flow effects
        
        print(f"Configuration:")
        print(f"  Flow scale: {config.TISSUE_FLOW_SCALE}")
        print(f"  Directional force: {config.DIRECTIONAL_FORCE_STRENGTH}")
        print(f"  Time step: {config.TIME_STEP}")
        
        # Create simulation with controlled domain size
        domain_size = (50.0, 30.0, 20.0)  # Smaller domain for testing
        vessel_diameter = 15.0
        inlet_velocity = 60.0
        
        print(f"  Domain: {domain_size}")
        print(f"  Vessel diameter: {vessel_diameter} μm")
        print(f"  Inlet velocity: {inlet_velocity} μm/s")
        
        # Run simulation
        simulation = BloodFlowTestSimulation(
            num_sporozoites=3,  # Test with 3 sporozoites at different positions
            domain_size=domain_size,
            config=config,
            use_blood_flow=True,
            flow_type=flow_type,
            inlet_velocity=inlet_velocity,
            vessel_diameter=vessel_diameter
        )
        
        # Print initial analysis
        print(f"\nInitial Analysis:")
        print(f"  Created {len(simulation.sporozoites)} test sporozoites")
        print(f"  Environment: {simulation.stats['environment_type']}")
        print(f"  Flow type: {simulation.stats['flow_type']}")
        
        # Run short simulation
        print(f"\nRunning {config.MAX_TIME}s simulation...")
        simulation.run_simulation()
        
        # Print results summary
        print(f"\nTest Results Summary:")
        print(f"  Final active sporozoites: {len(simulation.sporozoites)}")
        print(f"  Total distance traveled: {simulation.stats['total_distance_traveled']:.2f}")
        print(f"  Average motility: {simulation.stats['average_motility']:.3f}")
        print(f"  Output directory: {simulation.output_dir}")
        
        # Analyze final positions
        print(f"\nFinal Sporozoite Positions:")
        for sporozoite in simulation.sporozoites:
            initial_pos = "UNKNOWN"
            if sporozoite.id == 0:
                initial_pos = "ORIGIN"
            elif sporozoite.id == 1:
                initial_pos = "VESSEL_CENTER"  
            elif sporozoite.id == 2:
                initial_pos = "NEAR_WALL"
            
            final_pos = sporozoite.center_position
            print(f"  ID {sporozoite.id} ({initial_pos}): [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]")
        
        print(f"\n   VTK files saved to: {simulation.output_dir}")
        # Wait for user input before next test (if running interactively)
        try:
            input(f"\nPress Enter to continue to next flow type test, or Ctrl+C to exit...")
        except KeyboardInterrupt:
            print(f"\nTest interrupted by user")
            break
        except:
            # Non-interactive mode, continue automatically
            pass

def analyze_flow_at_positions():
    """Analyze flow characteristics at different test positions"""
    
    print(f"\n{'='*60}")
    print("FLOW FIELD ANALYSIS AT TEST POSITIONS")
    print(f"{'='*60}")
    
    # Import blood flow field
    from blood_flow_field import ImplicitBloodFlowField, FlowType
    
    domain_size = (50.0, 30.0, 20.0)
    vessel_diameter = 15.0
    inlet_velocity = 60.0
    
    for flow_type in [FlowType.LAMINAR, FlowType.TURBULENT, FlowType.SIMPLE_SHEAR]:
        print(f"\n{flow_type.value.upper()} FLOW ANALYSIS:")
        print("-" * 40)
        
        # Create blood flow field
        blood_flow = ImplicitBloodFlowField(
            domain_size=domain_size,
            flow_type=flow_type,
            inlet_velocity=inlet_velocity,
            vessel_diameter=vessel_diameter
        )
        
        # Test positions
        test_positions = [
            ("Origin", np.array([0.1, 0.1, 0.1])),
            ("Vessel Center", np.array([5.0, domain_size[1]/2, domain_size[2]/2])),
            ("Near Wall", np.array([10.0, domain_size[1]/2 + 8.0, domain_size[2]/2])),
            ("Outside Vessel", np.array([15.0, 5.0, 5.0])),
        ]
        
        for name, pos in test_positions:
            velocity = blood_flow.get_velocity_at_point(pos)
            speed = np.linalg.norm(velocity)
            pressure = blood_flow.get_pressure_at_point(pos)
            shear_rate = blood_flow.get_shear_rate_at_point(pos)
            in_vessel = blood_flow.is_point_in_vessel(pos)
            
            # Convert numpy arrays to float for formatting
            pressure_val = float(pressure) if hasattr(pressure, '__float__') else pressure
            shear_val = float(shear_rate) if hasattr(shear_rate, '__float__') else shear_rate
            
            print(f"  {name:12s}: In vessel={str(in_vessel):5s}, Speed={speed:6.2f} μm/s, "
                  f"Pressure={pressure_val:6.2f} Pa, Shear={shear_val:5.2f} s⁻¹")

if __name__ == "__main__":
    print("Blood Flow Effects Test Suite")
    print("="*60)
    
    try:
        # First analyze the flow field at test positions
        analyze_flow_at_positions()
        
        # Then run the actual simulation tests
        run_blood_flow_test()
        
        print(f"\n{'='*60}")
        print("BLOOD FLOW TESTS COMPLETED!")
        print("="*60)
        print("Key files to examine:")
        print("1. VTK outputs in simulation_bloodflow_* directories")
        print("2. Load sporozoites_timeseries.pvd in ParaView")
        print("3. Load blood_flow_field_timeseries.pvd for flow visualization")
        print("4. Compare sporozoite trajectories from different starting positions")
        print("\nExpected observations:")
        print("• Sporozoite at origin should be affected by flow if inside vessel")
        print("• Sporozoite at vessel center should follow main flow direction")
        print("• Sporozoite near wall should experience different flow conditions")
        print("• Turbulent flow should show more complex motion patterns")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()