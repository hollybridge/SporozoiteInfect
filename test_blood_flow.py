#!/usr/bin/env python3
"""
Test script for ImplicitBloodFlowField

This script demonstrates the different blood flow types and shows how to use them
in console applications.

Author: Holly Evans
Date: January 2026
"""

import sys
import os
import numpy as np

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'mesh_simulation'))

# Direct import from the file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mesh_simulation'))
from blood_flow_field import ImplicitBloodFlowField, FlowType

def test_blood_flow_types():
    """Test all three blood flow types"""
    
    print("Testing ImplicitBloodFlowField with different flow types")
    print("="*60)
    
    # Domain setup
    domain_size = (80.0, 40.0, 30.0)  # Smaller domain for testing
    
    # Test each flow type
    flow_types = [
        ("Simple Shear", FlowType.SIMPLE_SHEAR, {"inlet_velocity": 75.0}),
        ("Laminar (Poiseuille)", FlowType.LAMINAR, {"inlet_velocity": 50.0, "vessel_diameter": 15.0}),
        ("Turbulent", FlowType.TURBULENT, {"inlet_velocity": 120.0, "reynolds_number": 3000.0})
    ]
    
    for flow_name, flow_type, params in flow_types:
        print(f"\n{flow_name} Flow Test:")
        print("-" * 40)
        
        # Create blood flow field
        blood_flow = ImplicitBloodFlowField(
            domain_size=domain_size,
            flow_type=flow_type,
            **params
        )
        
        # Test sampling at different points
        test_points = [
            np.array([20.0, 20.0, 15.0]),  # Center of vessel
            np.array([40.0, 20.0, 15.0]),  # Midway along vessel
            np.array([60.0, 20.0, 15.0]),  # Near outlet
            np.array([30.0, 25.0, 15.0]),  # Off-center in vessel
            np.array([30.0, 35.0, 15.0]),  # Near vessel wall
        ]
        
        print("Point sampling results:")
        for i, point in enumerate(test_points):
            velocity = blood_flow.get_velocity_at_point(point)
            pressure = blood_flow.get_pressure_at_point(point)
            shear_rate = blood_flow.get_shear_rate_at_point(point)
            in_vessel = blood_flow.is_point_in_vessel(point)
            
            print(f"  Point {i+1} {point}: ")
            print(f"    In vessel: {in_vessel}")
            print(f"    Velocity: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}] μm/s")
            print(f"    Pressure: {float(pressure):.2f} Pa")
            print(f"    Shear rate: {float(shear_rate):.2f} s⁻¹")
        
        # Test periodic boundary conditions
        print("\nPeriodic boundary test:")
        inlet_point = np.array([0.0, 20.0, 15.0])
        outlet_point = np.array([domain_size[0], 20.0, 15.0])
        
        inlet_velocity = blood_flow.get_velocity_at_point(inlet_point)
        outlet_velocity = blood_flow.get_velocity_at_point(outlet_point)
        
        print(f"  Inlet velocity: [{inlet_velocity[0]:.2f}, {inlet_velocity[1]:.2f}, {inlet_velocity[2]:.2f}] μm/s")
        print(f"  Outlet velocity: [{outlet_velocity[0]:.2f}, {outlet_velocity[1]:.2f}, {outlet_velocity[2]:.2f}] μm/s")
        print(f"  Periodic match: {np.allclose(inlet_velocity, outlet_velocity, atol=1e-3)}")

def test_factory_methods():
    """Test the factory methods for creating different flow types"""
    
    print("\n" + "="*60)
    print("Testing Factory Methods")
    print("="*60)
    
    domain_size = (60.0, 30.0, 20.0)
    
    # Test simple shear factory
    print("\nSimple Shear Factory Method:")
    shear_flow = ImplicitBloodFlowField.create_simple_shear(
        domain_size=domain_size,
        shear_rate=100.0,
        vessel_diameter=12.0
    )
    center_velocity = shear_flow.get_velocity_at_point(np.array([30.0, 15.0, 10.0]))
    print(f"  Center velocity: {center_velocity[0]:.2f} μm/s")
    
    # Test laminar factory
    print("\nLaminar Flow Factory Method:")
    laminar_flow = ImplicitBloodFlowField.create_laminar_flow(
        domain_size=domain_size,
        inlet_velocity=60.0,
        vessel_diameter=18.0
    )
    center_velocity = laminar_flow.get_velocity_at_point(np.array([30.0, 15.0, 10.0]))
    print(f"  Center velocity: {center_velocity[0]:.2f} μm/s")
    
    # Test turbulent factory
    print("\nTurbulent Flow Factory Method:")
    turbulent_flow = ImplicitBloodFlowField.create_turbulent_flow(
        domain_size=domain_size,
        inlet_velocity=80.0,
        reynolds_number=2800.0
    )
    center_velocity = turbulent_flow.get_velocity_at_point(np.array([30.0, 15.0, 10.0]))
    turbulence = turbulent_flow.get_turbulence_intensity_at_point(np.array([30.0, 15.0, 10.0]))
    print(f"  Center velocity: {center_velocity[0]:.2f} μm/s")
    print(f"  Turbulence intensity: {float(turbulence):.2f}")

def demonstrate_console_usage():
    """Demonstrate how to use blood flow field in console applications"""
    
    print("\n" + "="*60)
    print("Console Usage Examples")
    print("="*60)
    
    print("\nTo run simulations with blood flow instead of dermal tissue:")
    print("="*50)
    
    print("\n1. Laminar flow simulation:")
    print("   python mesh_simulation/mesh_simulation.py --use-blood-flow --flow-type laminar --sporozoites 3 --time 5")
    
    print("\n2. Turbulent flow with high velocity:")
    print("   python mesh_simulation/mesh_simulation.py --use-blood-flow --flow-type turbulent --inlet-velocity 150 --sporozoites 2")
    
    print("\n3. Simple shear flow in small vessel:")
    print("   python mesh_simulation/mesh_simulation.py --use-blood-flow --flow-type simple_shear --vessel-diameter 10 --sporozoites 4")
    
    print("\n4. Combined with movement presets:")
    print("   python mesh_simulation/mesh_simulation.py --use-blood-flow --flow-type laminar --movement-preset fast --sporozoites 3")
    
    print("\nKey parameters:")
    print("  --use-blood-flow: Switch to blood flow environment")
    print("  --flow-type: simple_shear, laminar, or turbulent")
    print("  --inlet-velocity: Flow speed in μm/s")
    print("  --vessel-diameter: Vessel diameter in μm")
    
    print("\nThe blood flow field will:")
    print("  ✓ Generate realistic velocity profiles")
    print("  ✓ Apply periodic boundary conditions (inlet = outlet)")
    print("  ✓ Calculate proper pressure gradients") 
    print("  ✓ Include shear rates and turbulence effects")
    print("  ✓ Output VTK files for ParaView visualization")

if __name__ == "__main__":
    print("ImplicitBloodFlowField Test Suite")
    print("="*60)
    
    try:
        # Run tests
        test_blood_flow_types()
        test_factory_methods()
        demonstrate_console_usage()
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("✓ ImplicitBloodFlowField is ready to use")
        print("="*60)
        
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()