"""
Simulation Configuration for Sporozoite Models

This module provides configuration classes and preset configurations
for various sporozoite simulation scenarios.
"""

import numpy as np
from typing import Tuple

class SimulationConfig:
    """Base configuration class for sporozoite simulations"""
    
    def __init__(self):
        # Time parameters
        self.TIME_STEP = 0.01  # seconds
        self.MAX_TIME = 10.0   # seconds
        self.OUTPUT_INTERVAL = 0.1  # seconds
        
        # Sporozoite geometry parameters
        self.SPOROZOITE_LENGTH_RANGE = (12.0, 18.0)  # μm
        self.SPOROZOITE_DIAMETER_RANGE = (1.0, 2.0)  # μm
        self.LONGITUDINAL_SEGMENTS = 8
        self.RADIAL_SEGMENTS = 6
        
        # Biological parameters
        self.MOTILITY_RANGE = (0.6, 1.0)
        self.STIFFNESS_RANGE = (0.7, 1.0)
        self.VIABILITY_DECAY_RATE = 0.01  # per second
        
        # Movement parameters
        self.MAX_SPEED_RANGE = (8.0, 15.0)  # μm/s
        self.UNDULATION_FREQUENCY_RANGE = (1.0, 3.0)  # Hz
        self.UNDULATION_AMPLITUDE = 0.5  # μm
        self.UNDULATION_FORCE_SCALE = 2.0
        
        # Physics parameters
        self.DIRECTIONAL_FORCE_STRENGTH = 8.0
        self.SPRING_CONSTANT_BASE = 100.0
        self.DAMPING_FACTOR = 0.5
        
        # Noise parameters
        self.NOISE_STRENGTH = 1.0
        self.BOUNDARY_REPULSION_STRENGTH = 5.0

class PresetConfigs:
    """Preset configuration factory"""
    
    @staticmethod
    def realistic_movement():
        """Standard realistic movement configuration"""
        config = SimulationConfig()
        config.TIME_STEP = 0.02
        config.DIRECTIONAL_FORCE_STRENGTH = 8.0
        config.UNDULATION_AMPLITUDE = 0.3
        config.SPRING_CONSTANT_BASE = 75.0
        return config
    
    @staticmethod
    def high_resolution():
        """High resolution with small time steps"""
        config = SimulationConfig()
        config.TIME_STEP = 0.005
        config.OUTPUT_INTERVAL = 0.05
        config.LONGITUDINAL_SEGMENTS = 12
        config.RADIAL_SEGMENTS = 8
        return config
    
    @staticmethod
    def debug_mode():
        """Debug configuration with verbose output"""
        config = SimulationConfig()
        config.TIME_STEP = 0.001
        config.MAX_TIME = 2.0
        config.OUTPUT_INTERVAL = 0.01
        config.LONGITUDINAL_SEGMENTS = 6
        config.RADIAL_SEGMENTS = 4
        return config

class CenterlineConfig(SimulationConfig):
    """Configuration class specifically for centerline-based sporozoite simulations"""
    
    def __init__(self):
        super().__init__()
        
        # Centerline-specific parameters
        self.CENTERLINE_SEGMENTS = 10  # Number of centerline control points
        self.CENTERLINE_SMOOTHING = 0.1  # Smoothing factor for centerline
        self.CENTERLINE_TENSION = 1.0  # Tension along the centerline
        
        # Centerline physics
        self.BENDING_STIFFNESS = 50.0  # Resistance to bending
        self.TORSIONAL_STIFFNESS = 25.0  # Resistance to twisting
        self.STRETCH_STIFFNESS = 100.0  # Resistance to stretching/compression
        
        # Centerline forces
        self.PROPULSIVE_FORCE_AMPLITUDE = 10.0  # Amplitude of propulsive waves
        self.WAVE_FREQUENCY = 2.0  # Frequency of propulsive waves (Hz)
        self.WAVE_LENGTH_FACTOR = 1.5  # Wave length as factor of sporozoite length
        
        # Centerline undulation parameters
        self.LATERAL_UNDULATION_AMPLITUDE = 0.8  # μm
        self.UNDULATION_PHASE_SHIFT = 0.0  # Phase shift between segments
        self.UNDULATION_DECAY = 0.1  # Decay factor along centerline
        
        # Integration and stability
        self.CENTERLINE_DAMPING = 0.8  # Higher damping for stability
        self.ADAPTIVE_TIME_STEP = True  # Use adaptive time stepping
        self.MIN_TIME_STEP = 0.001  # Minimum time step
        self.MAX_TIME_STEP = 0.01   # Maximum time step

class CenterlineSimulationConfigs:
    """Factory class for centerline simulation configurations"""
    
    @staticmethod
    def debug_forces():
        """Debug configuration with enhanced force visualization"""
        config = CenterlineConfig()
        
        # Reduce complexity for debugging
        config.TIME_STEP = 0.005
        config.MAX_TIME = 5.0
        config.OUTPUT_INTERVAL = 0.05
        
        # Simplified geometry
        config.CENTERLINE_SEGMENTS = 8
        config.LONGITUDINAL_SEGMENTS = 6
        config.RADIAL_SEGMENTS = 4
        
        # Enhanced force visibility
        config.DIRECTIONAL_FORCE_STRENGTH = 12.0
        config.PROPULSIVE_FORCE_AMPLITUDE = 8.0
        config.LATERAL_UNDULATION_AMPLITUDE = 1.2
        
        # Moderate physics for stability
        config.BENDING_STIFFNESS = 30.0
        config.CENTERLINE_DAMPING = 1.0
        config.SPRING_CONSTANT_BASE = 60.0
        
        # Reduced noise for clearer force patterns
        config.NOISE_STRENGTH = 0.5
        
        return config
    
    @staticmethod
    def realistic_swimming():
        """Realistic swimming behavior configuration"""
        config = CenterlineConfig()
        
        # Standard time parameters
        config.TIME_STEP = 0.01
        config.MAX_TIME = 15.0
        config.OUTPUT_INTERVAL = 0.1
        
        # Realistic geometry
        config.CENTERLINE_SEGMENTS = 12
        config.LONGITUDINAL_SEGMENTS = 10
        config.RADIAL_SEGMENTS = 8
        
        # Biologically realistic forces
        config.DIRECTIONAL_FORCE_STRENGTH = 6.0
        config.PROPULSIVE_FORCE_AMPLITUDE = 5.0
        config.WAVE_FREQUENCY = 2.5
        config.LATERAL_UNDULATION_AMPLITUDE = 0.6
        
        # Realistic material properties
        config.BENDING_STIFFNESS = 40.0
        config.TORSIONAL_STIFFNESS = 20.0
        config.STRETCH_STIFFNESS = 80.0
        
        # Natural noise and damping
        config.NOISE_STRENGTH = 1.2
        config.CENTERLINE_DAMPING = 0.6
        
        return config
    
    @staticmethod
    def high_performance():
        """High-performance configuration for detailed analysis"""
        config = CenterlineConfig()
        
        # High resolution time stepping
        config.TIME_STEP = 0.002
        config.MAX_TIME = 20.0
        config.OUTPUT_INTERVAL = 0.02
        config.ADAPTIVE_TIME_STEP = True
        
        # High resolution geometry
        config.CENTERLINE_SEGMENTS = 16
        config.LONGITUDINAL_SEGMENTS = 12
        config.RADIAL_SEGMENTS = 10
        
        # Precise force parameters
        config.DIRECTIONAL_FORCE_STRENGTH = 8.0
        config.PROPULSIVE_FORCE_AMPLITUDE = 6.5
        config.WAVE_FREQUENCY = 2.8
        config.LATERAL_UNDULATION_AMPLITUDE = 0.7
        
        # High-fidelity physics
        config.BENDING_STIFFNESS = 45.0
        config.TORSIONAL_STIFFNESS = 22.0
        config.STRETCH_STIFFNESS = 90.0
        config.CENTERLINE_TENSION = 1.2
        
        # Fine-tuned stability
        config.CENTERLINE_DAMPING = 0.7
        config.CENTERLINE_SMOOTHING = 0.05
        
        return config
    
    @staticmethod
    def fast_preview():
        """Fast preview configuration for quick testing"""
        config = CenterlineConfig()
        
        # Large time steps for speed
        config.TIME_STEP = 0.02
        config.MAX_TIME = 3.0
        config.OUTPUT_INTERVAL = 0.2
        
        # Simplified geometry
        config.CENTERLINE_SEGMENTS = 6
        config.LONGITUDINAL_SEGMENTS = 4
        config.RADIAL_SEGMENTS = 3
        
        # Simplified physics
        config.DIRECTIONAL_FORCE_STRENGTH = 10.0
        config.PROPULSIVE_FORCE_AMPLITUDE = 7.0
        config.LATERAL_UNDULATION_AMPLITUDE = 1.0
        
        # Increased damping for stability with large time steps
        config.CENTERLINE_DAMPING = 1.5
        config.DAMPING_FACTOR = 0.8
        
        # Reduced complexity
        config.NOISE_STRENGTH = 0.8
        config.BENDING_STIFFNESS = 35.0
        
        return config
    
    @staticmethod
    def salivary_gland_environment():
        """Configuration tailored for salivary gland environment"""
        config = CenterlineConfig()
        
        # Medium resolution for salivary gland complexity
        config.TIME_STEP = 0.008
        config.MAX_TIME = 12.0
        config.OUTPUT_INTERVAL = 0.08
        
        # Moderate geometry resolution
        config.CENTERLINE_SEGMENTS = 10
        config.LONGITUDINAL_SEGMENTS = 8
        config.RADIAL_SEGMENTS = 6
        
        # Salivary gland specific movement
        config.DIRECTIONAL_FORCE_STRENGTH = 7.0
        config.PROPULSIVE_FORCE_AMPLITUDE = 5.5
        config.WAVE_FREQUENCY = 2.2
        config.LATERAL_UNDULATION_AMPLITUDE = 0.8
        
        # Adjusted for salivary gland viscosity
        config.DAMPING_FACTOR = 0.7
        config.CENTERLINE_DAMPING = 0.8
        
        # Environmental constraints
        config.BOUNDARY_REPULSION_STRENGTH = 8.0
        config.NOISE_STRENGTH = 1.1
        
        return config