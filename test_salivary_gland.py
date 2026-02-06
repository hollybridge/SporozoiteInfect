"""
Test Sporozoite Interaction Simulation - High-Density 3D Environment

This script demonstrates sporozoite interactions without any tissue field.
Pure focus on the equation of motion: Propulsion + Interaction + Internal + Stochastic Noise
"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'stages', 'salivary_gland'))

from stages.salivary_gland.salivary_gland_simulation import SporozoiteInteractionSimulation, SporozoiteInteractionConfigs
from simulation_config import PresetConfigs

def test_high_density_interactions():
    """Test high-density sporozoite interactions"""
    print("="*60)
    print("SPOROZOITE INTERACTIONS - HIGH DENSITY TEST")
    print("="*60)
    
    # Configuration for high-density environment
    config = SporozoiteInteractionConfigs.high_density()
    config.MAX_TIME = 10.0  # Shorter simulation for testing
    config.OUTPUT_INTERVAL = 0.5  # Frequent output
    
    # Create simulation with high density
    simulation = SporozoiteInteractionSimulation(
        num_sporozoites=80,  # High density
        domain_size=(120.0, 120.0, 80.0),  # Smaller domain for higher density
        config=config
    )
    
    # Run simulation
    output_dir = simulation.run_simulation()
    
    print(f"\nHigh-density interaction simulation completed!")
    print(f"Output saved to: {output_dir}")
    
    return output_dir

def test_medium_density_interactions():
    """Test medium-density sporozoite interactions"""
    print("\n" + "="*60)
    print("SPOROZOITE INTERACTIONS - MEDIUM DENSITY TEST")
    print("="*60)
    
    # Configuration for medium-density environment
    config = SporozoiteInteractionConfigs.medium_density()
    config.MAX_TIME = 8.0
    config.OUTPUT_INTERVAL = 0.4
    
    # Create simulation with medium density
    simulation = SporozoiteInteractionSimulation(
        num_sporozoites=50,
        domain_size=(150.0, 150.0, 100.0),
        config=config
    )
    
    # Run simulation
    output_dir = simulation.run_simulation()
    
    print(f"\nMedium-density interaction simulation completed!")
    print(f"Output saved to: {output_dir}")
    
    return output_dir

def test_clustering_dynamics():
    """Test sporozoite clustering dynamics"""
    print("\n" + "="*60)
    print("SPOROZOITE INTERACTIONS - CLUSTERING DYNAMICS")
    print("="*60)
    
    # Configuration optimized for clustering observation
    config = SporozoiteInteractionConfigs.medium_density()
    config.MAX_TIME = 15.0  # Longer time to see clustering
    config.OUTPUT_INTERVAL = 0.3
    config.DIRECTIONAL_FORCE_STRENGTH = 5.0  # Reduced movement to encourage clustering
    
    # Create simulation
    simulation = SporozoiteInteractionSimulation(
        num_sporozoites=60,
        domain_size=(140.0, 140.0, 90.0),
        config=config
    )
    
    # Modify interaction parameters to enhance clustering
    simulation.interaction_manager.cohesion_strength = 1.5  # Stronger cohesion
    simulation.interaction_manager.alignment_strength = 2.0  # Stronger alignment
    simulation.noise_strength = 1.0  # Reduced noise for clearer patterns
    
    # Run simulation
    output_dir = simulation.run_simulation()
    
    print(f"\nClustering dynamics simulation completed!")
    print(f"Output saved to: {output_dir}")
    
    return output_dir

def compare_equation_of_motion_components():
    """Compare different components of the equation of motion"""
    print("\n" + "="*60)
    print("EQUATION OF MOTION COMPONENT COMPARISON")
    print("="*60)
    
    # Base configuration
    config = SporozoiteInteractionConfigs.medium_density()
    config.MAX_TIME = 6.0
    config.OUTPUT_INTERVAL = 0.5
    
    results = {}
    
    # Test 1: Propulsion-dominated (reduce other forces)
    print("\nTest 1: Propulsion-dominated motion")
    sim1 = SporozoiteInteractionSimulation(
        num_sporozoites=30,
        domain_size=(100.0, 100.0, 70.0),
        config=config
    )
    # Reduce interaction and noise
    sim1.interaction_manager.repulsion_strength = 1.0
    sim1.interaction_manager.cohesion_strength = 0.1
    sim1.noise_strength = 0.5
    results['propulsion'] = sim1.run_simulation()
    
    # Test 2: Interaction-dominated (high interaction forces)
    print("\nTest 2: Interaction-dominated motion")
    sim2 = SporozoiteInteractionSimulation(
        num_sporozoites=30,
        domain_size=(100.0, 100.0, 70.0),
        config=config
    )
    # Increase interactions, reduce propulsion
    sim2.propulsion_strength = 3.0
    sim2.interaction_manager.repulsion_strength = 8.0
    sim2.interaction_manager.alignment_strength = 3.0
    sim2.interaction_manager.cohesion_strength = 2.0
    sim2.noise_strength = 0.5
    results['interaction'] = sim2.run_simulation()
    
    # Test 3: Noise-dominated (high stochastic forces)
    print("\nTest 3: Noise-dominated motion")
    sim3 = SporozoiteInteractionSimulation(
        num_sporozoites=30,
        domain_size=(100.0, 100.0, 70.0),
        config=config
    )
    # High noise, reduced other forces
    sim3.propulsion_strength = 2.0
    sim3.interaction_manager.repulsion_strength = 2.0
    sim3.interaction_manager.alignment_strength = 0.5
    sim3.interaction_manager.cohesion_strength = 0.2
    sim3.noise_strength = 5.0
    results['noise'] = sim3.run_simulation()
    
    print("\n" + "="*60)
    print("EQUATION OF MOTION COMPARISON COMPLETE")
    print("="*60)
    for test_name, output_dir in results.items():
        print(f"{test_name.upper()}-dominated: {output_dir}")
    
    return results

def test_volume_exclusion_effects():
    """Test volume exclusion effects with very high density"""
    print("\n" + "="*60)
    print("VOLUME EXCLUSION EFFECTS - VERY HIGH DENSITY")
    print("="*60)
    
    # Configuration for very high density to test volume exclusion
    config = SporozoiteInteractionConfigs.high_density()
    config.MAX_TIME = 5.0
    config.OUTPUT_INTERVAL = 0.2
    
    # Create simulation with very high density
    simulation = SporozoiteInteractionSimulation(
        num_sporozoites=100,  # Very high density
        domain_size=(80.0, 80.0, 60.0),  # Very small domain
        config=config
    )
    
    # Strengthen volume exclusion effects
    simulation.interaction_manager.volume_exclusion_radius = 3.0  # Larger exclusion radius
    simulation.interaction_manager.repulsion_strength = 10.0  # Stronger repulsion
    
    # Run simulation
    output_dir = simulation.run_simulation()
    
    print(f"\nVolume exclusion simulation completed!")
    print(f"Output saved to: {output_dir}")
    
    return output_dir

def test_lennard_jones_potential():
    """Test the shifted Lennard-Jones potential interactions"""
    print("\n" + "="*60)
    print("LENNARD-JONES POTENTIAL TEST")
    print("="*60)
    
    # Configuration for LJ testing
    config = SporozoiteInteractionConfigs.medium_density()
    config.MAX_TIME = 8.0
    config.OUTPUT_INTERVAL = 0.3
    
    # Create simulation with custom LJ parameters
    simulation = SporozoiteInteractionSimulation(
        num_sporozoites=40,
        domain_size=(100.0, 100.0, 80.0),
        config=config
    )
    
    # Customize LJ parameters for testing
    simulation.interaction_manager.lj_epsilon = 8.0  # Stronger interactions
    simulation.interaction_manager.lj_sigma = 1.8   # Slightly smaller contact distance
    simulation.interaction_manager.lj_cutoff = 5.0  # Shorter cutoff
    simulation.interaction_manager.lj_shift = simulation.interaction_manager._calculate_lj_shift()
    
    print(f"Testing with custom LJ parameters:")
    print(f"  ε = {simulation.interaction_manager.lj_epsilon}")
    print(f"  σ = {simulation.interaction_manager.lj_sigma} μm")
    print(f"  cutoff = {simulation.interaction_manager.lj_cutoff} μm")
    
    # Run simulation
    output_dir = simulation.run_simulation()
    
    print(f"\nLennard-Jones potential test completed!")
    print(f"Output saved to: {output_dir}")
    
    return output_dir

def test_centerline_vs_vertex_lj():
    """Compare centerline-based vs vertex-based LJ calculations"""
    print("\n" + "="*60)
    print("CENTERLINE VS VERTEX LJ COMPARISON")
    print("="*60)
    
    # Base configuration
    config = SporozoiteInteractionConfigs.medium_density()
    config.MAX_TIME = 5.0
    config.OUTPUT_INTERVAL = 0.5
    
    results = {}
    
    # Test 1: Centerline-based LJ (efficient)
    print("\nTest 1: Centerline-based LJ interactions")
    sim1 = SporozoiteInteractionSimulation(
        num_sporozoites=25,
        domain_size=(80.0, 80.0, 60.0),
        config=config
    )
    sim1.interaction_manager.use_centerline_lj = True
    results['centerline_lj'] = sim1.run_simulation()
    
    # Test 2: Vertex-based LJ (more accurate but expensive)
    print("\nTest 2: Vertex-based LJ interactions")
    sim2 = SporozoiteInteractionSimulation(
        num_sporozoites=25,
        domain_size=(80.0, 80.0, 60.0),
        config=config
    )
    sim2.interaction_manager.use_centerline_lj = False
    results['vertex_lj'] = sim2.run_simulation()
    
    print("\n" + "="*60)
    print("CENTERLINE VS VERTEX LJ COMPARISON COMPLETE")
    print("="*60)
    for test_name, output_dir in results.items():
        print(f"{test_name.replace('_', ' ').upper()}: {output_dir}")
    
    return results

def test_lj_parameter_sensitivity():
    """Test sensitivity to different LJ parameters"""
    print("\n" + "="*60)
    print("LJ PARAMETER SENSITIVITY TEST")
    print("="*60)
    
    # Base configuration
    config = SporozoiteInteractionConfigs.medium_density()
    config.MAX_TIME = 6.0
    config.OUTPUT_INTERVAL = 0.4
    
    results = {}
    
    # Test different epsilon values (energy depth)
    epsilon_values = [2.0, 5.0, 10.0]
    for eps in epsilon_values:
        print(f"\nTesting with ε = {eps}")
        sim = SporozoiteInteractionSimulation(
            num_sporozoites=30,
            domain_size=(90.0, 90.0, 70.0),
            config=config
        )
        sim.interaction_manager.lj_epsilon = eps
        sim.interaction_manager.lj_shift = sim.interaction_manager._calculate_lj_shift()
        results[f'epsilon_{eps}'] = sim.run_simulation()
    
    # Test different sigma values (length scale)
    sigma_values = [1.0, 2.0, 3.0]
    for sig in sigma_values:
        print(f"\nTesting with σ = {sig}")
        sim = SporozoiteInteractionSimulation(
            num_sporozoites=30,
            domain_size=(90.0, 90.0, 70.0),
            config=config
        )
        sim.interaction_manager.lj_sigma = sig
        sim.interaction_manager.lj_shift = sim.interaction_manager._calculate_lj_shift()
        results[f'sigma_{sig}'] = sim.run_simulation()
    
    print("\n" + "="*60)
    print("LJ PARAMETER SENSITIVITY TESTS COMPLETE")
    print("="*60)
    for test_name, output_dir in results.items():
        print(f"{test_name.replace('_', ' ').upper()}: {output_dir}")
    
    return results

def test_ultra_high_density_400_sporozoites():
    """Test ultra-high density with 400 sporozoites at ~50% volume fraction"""
    print("="*60)
    print("ULTRA-HIGH DENSITY - 400 SPOROZOITES AT 50% VOLUME FRACTION")
    print("="*60)
    
    # Configuration for ultra-high density environment
    config = SporozoiteInteractionConfigs.high_density()
    config.MAX_TIME = 6.0  # Shorter time due to computational intensity
    config.OUTPUT_INTERVAL = 0.3  # More frequent output to track dynamics
    config.TIME_STEP = 0.02  # Smaller time step for stability at high density
    config.SPRING_CONSTANT_BASE = 120.0  # Stronger springs for shape maintenance
    config.DAMPING_FACTOR = 0.8  # More damping for stability
    
    # Domain size calculation for 50% volume fraction:
    # Average sporozoite volume ≈ 7.4 μm³ (prolate spheroid: length=13.5μm, diameter=1.5μm)  
    # 400 sporozoites = 2,960 μm³ total volume
    # For 50% volume fraction: domain volume = 5,920 μm³
    # Cube root ≈ 18.1 μm per side
    # Using rectangular domain for better packing: 25.4 × 18.1 × 12.9 μm
    
    print(f"Domain size optimized for 50% volume fraction:")
    print(f"  Sporozoites: 400")
    print(f"  Domain: 25.4 × 18.1 × 12.9 μm = 5,920 μm³")
    print(f"  Target density: 50% volume fraction")
    
    # Create ultra-high density simulation
    simulation = SporozoiteInteractionSimulation(
        num_sporozoites=400,
        domain_size=(25.4, 18.1, 12.9),  # Optimized for 50% volume fraction
        config=config
    )
    
    # Adjust interaction parameters for extreme density
    simulation.interaction_manager.lj_epsilon = 8.0  # Strong repulsion needed
    simulation.interaction_manager.lj_sigma = 1.8   # Smaller contact distance
    simulation.interaction_manager.lj_cutoff = 4.0  # Shorter cutoff for efficiency
    simulation.interaction_manager.repulsion_strength = 12.0  # Very strong repulsion
    simulation.propulsion_strength = 4.0  # Reduced propulsion due to crowding
    simulation.noise_strength = 1.0  # Reduced noise to prevent instability
    
    print(f"Interaction parameters adjusted for ultra-high density:")
    print(f"  LJ epsilon: {simulation.interaction_manager.lj_epsilon}")
    print(f"  Repulsion strength: {simulation.interaction_manager.repulsion_strength}")
    print(f"  Propulsion strength: {simulation.propulsion_strength}")
    
    # Run simulation
    output_dir = simulation.run_simulation()
    
    print(f"\nUltra-high density simulation completed!")
    print(f"Output saved to: {output_dir}")
    
    return output_dir

def test_high_density_comparison():
    """Compare different high-density scenarios: 30%, 40%, and 50% volume fraction"""
    print("\n" + "="*60)
    print("HIGH DENSITY COMPARISON - 30%, 40%, 50% VOLUME FRACTION")
    print("="*60)
    
    # Base configuration for all high-density tests
    config = SporozoiteInteractionConfigs.high_density()
    config.MAX_TIME = 5.0
    config.OUTPUT_INTERVAL = 0.25
    config.TIME_STEP = 0.03
    
    results = {}
    
    # Test 1: 30% volume fraction with 300 sporozoites
    print("\nTest 1: 30% volume fraction (300 sporozoites)")
    # Domain volume = 300 * 7.4 μm³ / 0.3 = 7,400 μm³
    # Cube root ≈ 19.5 μm, using 28 × 19 × 14 μm
    sim1 = SporozoiteInteractionSimulation(
        num_sporozoites=300,
        domain_size=(28.0, 19.0, 14.0),  # 7,448 μm³
        config=config
    )
    sim1.interaction_manager.lj_epsilon = 6.0
    sim1.interaction_manager.repulsion_strength = 8.0
    sim1.propulsion_strength = 5.0
    results['density_30_percent'] = sim1.run_simulation()
    
    # Test 2: 40% volume fraction with 350 sporozoites  
    print("\nTest 2: 40% volume fraction (350 sporozoites)")
    # Domain volume = 350 * 7.4 μm³ / 0.4 = 6,475 μm³
    # Using 26 × 18 × 13.8 μm
    sim2 = SporozoiteInteractionSimulation(
        num_sporozoites=350,
        domain_size=(26.0, 18.0, 13.8),  # 6,458 μm³
        config=config
    )
    sim2.interaction_manager.lj_epsilon = 7.0
    sim2.interaction_manager.repulsion_strength = 10.0
    sim2.propulsion_strength = 4.5
    results['density_40_percent'] = sim2.run_simulation()
    
    # Test 3: 50% volume fraction with 400 sporozoites
    print("\nTest 3: 50% volume fraction (400 sporozoites)")
    # Domain volume = 400 * 7.4 μm³ / 0.5 = 5,920 μm³
    sim3 = SporozoiteInteractionSimulation(
        num_sporozoites=400,
        domain_size=(25.4, 18.1, 12.9),  # 5,920 μm³
        config=config
    )
    sim3.interaction_manager.lj_epsilon = 8.0
    sim3.interaction_manager.repulsion_strength = 12.0
    sim3.propulsion_strength = 4.0
    sim3.noise_strength = 0.8
    results['density_50_percent'] = sim3.run_simulation()
    
    print("\n" + "="*60)
    print("HIGH DENSITY COMPARISON COMPLETE")
    print("="*60)
    for test_name, output_dir in results.items():
        density = test_name.split('_')[1]
        print(f"{density}% VOLUME FRACTION: {output_dir}")
    
    return results

def run_all_tests():
    """Run all sporozoite interaction simulation tests"""
    print("Starting comprehensive sporozoite interaction simulation tests...")
    
    results = {}
    
    # Test different density levels
    results['high_density'] = test_high_density_interactions()
    results['medium_density'] = test_medium_density_interactions()
    
    # Test clustering dynamics
    results['clustering'] = test_clustering_dynamics()
    
    # Test equation of motion components
    eom_results = compare_equation_of_motion_components()
    results.update(eom_results)
    
    # Test volume exclusion
    results['volume_exclusion'] = test_volume_exclusion_effects()
    
    # NEW: Test Lennard-Jones potential functionality
    results['lj_potential'] = test_lennard_jones_potential()
    results['centerline_vs_vertex'] = test_centerline_vs_vertex_lj()
    lj_param_results = test_lj_parameter_sensitivity()
    results.update(lj_param_results)
    
    # NEW: Test ultra-high density and comparisons
    results['ultra_high_density'] = test_ultra_high_density_400_sporozoites()
    high_density_comparison_results = test_high_density_comparison()
    results.update(high_density_comparison_results)
    
    print("\n" + "="*80)
    print("ALL SPOROZOITE INTERACTION SIMULATION TESTS COMPLETED")
    print("="*80)
    print("\nSUMMARY OF RESULTS:")
    print("-" * 40)
    for test_name, output_dir in results.items():
        if isinstance(output_dir, dict):
            print(f"{test_name.replace('_', ' ').title()}:")
            for sub_test, sub_dir in output_dir.items():
                print(f"  {sub_test}: {sub_dir}")
        else:
            print(f"{test_name.replace('_', ' ').title()}: {output_dir}")
    
    print("\nKey Features Demonstrated:")
    print("• Pure sporozoite-sporozoite interactions")
    print("• Shifted Lennard-Jones potential (centerline & vertex-based)")
    print("• Volume exclusion effects (no overlap)")
    print("• Neighbor-based repulsion and alignment")
    print("• Stochastic noise and Brownian motion")
    print("• Internal deformation forces")
    print("• Clustering dynamics")
    print("• High-density crowding effects")
    print("• LJ parameter sensitivity analysis")
    print("• Equation of motion: Propulsion + LJ Interaction + Internal + Noise")
    print("• NO tissue field or flow effects")
    
    return results

if __name__ == "__main__":
    # Run a quick test or full suite
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        print("Running quick test...")
        test_medium_density_interactions()
    elif len(sys.argv) > 1 and sys.argv[1] == "clustering":
        print("Running clustering test...")
        test_clustering_dynamics()
    elif len(sys.argv) > 1 and sys.argv[1] == "eom":
        print("Running equation of motion comparison...")
        compare_equation_of_motion_components()
    elif len(sys.argv) > 1 and sys.argv[1] == "volume":
        print("Running volume exclusion test...")
        test_volume_exclusion_effects()
    elif len(sys.argv) > 1 and sys.argv[1] == "lj":
        print("Running Lennard-Jones tests...")
        test_lennard_jones_potential()
        test_centerline_vs_vertex_lj()
    elif len(sys.argv) > 1 and sys.argv[1] == "ultra_high_density":
        print("Running ultra-high density test...")
        test_ultra_high_density_400_sporozoites()
    elif len(sys.argv) > 1 and sys.argv[1] == "high_density_comparison":
        print("Running high density comparison test...")
        test_high_density_comparison()
    else:
        print("Running full test suite...")
        run_all_tests()