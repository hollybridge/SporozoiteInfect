#!/usr/bin/env python3
"""
Deformation Debug Script - Simplified Version

This script runs the mesh simulation with gentle flexible preset
and monitors deformation through the existing VTK outputs.

Author: Holly Evans
Date: January 2026
"""

import os
import sys
import subprocess
import json
import numpy as np

def run_gentle_flexible_debug():
    """Run mesh simulation with gentle flexible preset for deformation debugging"""
    
    print(" SPOROZOITE DEFORMATION DEBUG SESSION")
    print("="*60)
    
    # Change to mesh_simulation directory
    mesh_sim_dir = os.path.join(os.path.dirname(__file__), 'mesh_simulation')
    
    # Build the command to run mesh simulation with gentle flexible preset
    cmd = [
        'python3', 'mesh_simulation.py',
        '--movement-preset', 'gentle-flexible',
        '--sporozoites', '3',  # Fewer sporozoites for detailed analysis
        '--time', '3.0',       # Shorter simulation for debugging
        '--output-interval', '0.1',  # More frequent outputs
        '--domain-size', '40', '40', '25'  # Smaller domain
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {mesh_sim_dir}")
    print(f"Configuration: Gentle Flexible Preset")
    print("-" * 60)
    
    # Run the simulation
    try:
        # Change to mesh_simulation directory and run
        original_dir = os.getcwd()
        os.chdir(mesh_sim_dir)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Change back to original directory
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print(" Simulation completed successfully!")
            print("\nSimulation Output:")
            print(result.stdout)
            
            # Analyze the output
            analyze_simulation_output(result.stdout)
            
        else:
            print("Simulation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("Simulation timed out after 5 minutes")
    except Exception as e:
        print(f"Error running simulation: {e}")

def analyze_simulation_output(output_text):
    """Analyze the simulation output for deformation issues"""
    
    print("\n" + "="*60)
    print("DEFORMATION ANALYSIS")
    print("="*60)
    
    lines = output_text.split('\n')
    
    # Look for boundary warnings (indicates excessive deformation)
    boundary_warnings = []
    boundary_corrections = []
    sporozoite_movements = []
    
    for line in lines:
        if "BOUNDARY WARNING" in line:
            boundary_warnings.append(line.strip())
        elif "BOUNDARY CORRECTION" in line:
            boundary_corrections.append(line.strip())
        elif "moved" in line and "units" in line:
            sporozoite_movements.append(line.strip())
    
    # Analysis
    print(f"DEFORMATION INDICATORS:")
    print(f"   Boundary warnings: {len(boundary_warnings)}")
    print(f"   Boundary corrections: {len(boundary_corrections)}")
    
    if boundary_warnings:
        print(f"\n  BOUNDARY WARNINGS DETECTED:")
        for warning in boundary_warnings[:5]:  # Show first 5
            print(f"   {warning}")
        if len(boundary_warnings) > 5:
            print(f"   ... and {len(boundary_warnings) - 5} more warnings")
    
    if boundary_corrections:
        print(f"\nBOUNDARY CORRECTIONS APPLIED:")
        for correction in boundary_corrections[:5]:  # Show first 5
            print(f"   {correction}")
        if len(boundary_corrections) > 5:
            print(f"   ... and {len(boundary_corrections) - 5} more corrections")
    
    # Movement analysis
    if sporozoite_movements:
        print(f"\n MOVEMENT SUMMARY (last few steps):")
        for movement in sporozoite_movements[-6:]:  # Show last 6 movements
            print(f"   {movement}")
    
    # Recommendations
    print(f"\n ANALYSIS & RECOMMENDATIONS:")
    
    if len(boundary_warnings) > 10:
        print("   HIGH DEFORMATION DETECTED!")
        print("      - Many vertices are going out of bounds")
        print("      - This suggests excessive mesh deformation")
        print("      - Consider:")
        print("        • Increasing SPRING_CONSTANT_BASE")
        print("        • Decreasing UNDULATION_AMPLITUDE") 
        print("        • Increasing STIFFNESS_RANGE values")
        print("        • Decreasing TIME_STEP for more stability")
    elif len(boundary_warnings) > 0:
        print("     MODERATE DEFORMATION DETECTED")
        print("      - Some deformation is occurring but manageable")
        print("      - Monitor for patterns in specific sporozoites")
    else:
        print("    NO SIGNIFICANT DEFORMATION ISSUES")
        print("      - Sporozoites staying within bounds")
        print("      - Mesh appears stable")
    
    # Check for VTK output directory
    vtk_dirs = [d for d in os.listdir('.') if d.startswith('vtk_output')]
    if vtk_dirs:
        latest_sim = sorted(vtk_dirs)[-1]
        vtk_path = os.path.join('vtk_output', latest_sim)
        if os.path.exists(vtk_path):
            print(f"\n VTK OUTPUT READY:")
            print(f"   Directory: {vtk_path}")
            print(f"   Files: {len([f for f in os.listdir(vtk_path) if f.endswith('.vtm')])} VTM files")
            print(f"   Use ParaView to visualize detailed deformation")

def check_current_config():
    """Check the current gentle flexible configuration"""
    
    print("\nGENTLE FLEXIBLE CONFIGURATION:")
    print("-" * 40)
    
    try:
        # Import config to check current values
        sys.path.append('.')
        from simulation_config import PresetConfigs
        
        config = PresetConfigs.gentle_flexible()
        
        print(f"Time step: {config.TIME_STEP}")
        print(f"Directional force: {config.DIRECTIONAL_FORCE_STRENGTH}")
        print(f"Undulation amplitude: {config.UNDULATION_AMPLITUDE}")
        print(f"Undulation force scale: {config.UNDULATION_FORCE_SCALE}")
        print(f"Spring constant: {config.SPRING_CONSTANT_BASE}")
        print(f"Stiffness range: {config.STIFFNESS_RANGE}")
        print(f"Damping factor: {config.DAMPING_FACTOR}")
        print(f"Mass: {config.MASS}")
        
        # Deformation risk assessment
        print(f"\n DEFORMATION RISK ASSESSMENT:")
        
        risk_factors = []
        if config.UNDULATION_AMPLITUDE > 0.3:
            risk_factors.append(f"High undulation amplitude ({config.UNDULATION_AMPLITUDE})")
        if config.SPRING_CONSTANT_BASE < 50.0:
            risk_factors.append(f"Low spring constant ({config.SPRING_CONSTANT_BASE})")
        if config.STIFFNESS_RANGE[0] < 0.7:
            risk_factors.append(f"Low minimum stiffness ({config.STIFFNESS_RANGE[0]})")
        if config.TIME_STEP > 0.1:
            risk_factors.append(f"Large time step ({config.TIME_STEP})")
        
        if risk_factors:
            print("     Potential deformation risks:")
            for risk in risk_factors:
                print(f"      • {risk}")
        else:
            print("    Configuration appears stable for deformation")
            
    except Exception as e:
        print(f"    Could not load configuration: {e}")

def main():
    """Main debug function"""
    
    # Check current configuration
    check_current_config()
    
    # Run the debug simulation
    run_gentle_flexible_debug()
    
    print(f"\n" + "="*60)
    print(" DEFORMATION DEBUG COMPLETE")
    print("="*60)
    print("Next steps:")
    print("1. Review the boundary warnings above")
    print("2. Open the VTK files in ParaView to visualize deformation")
    print("3. If excessive deformation found, adjust parameters:")
    print("   • Increase SPRING_CONSTANT_BASE")
    print("   • Decrease UNDULATION_AMPLITUDE")
    print("   • Increase STIFFNESS_RANGE values")
    print("   • Decrease TIME_STEP")

if __name__ == "__main__":
    main()