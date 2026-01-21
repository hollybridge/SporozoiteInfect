"""
Simulation Configuration Parameters - SIMPLIFIED APPROACH

This file contains all the key parameters that control sporozoite movement,
deformation, and simulation dynamics. Focus on minimal, effective changes.
"""

class SimulationConfig:
    # Time stepping parameters
    TIME_STEP = 0.1
    MAX_TIME = 20.0
    OUTPUT_INTERVAL = 0.5
    
    # Movement parameters - KEEP SIMPLE
    DIRECTIONAL_FORCE_STRENGTH = 8.0  # Moderate forward movement
    MOTILITY_RANGE = (0.6, 1.0)
    MAX_SPEED_RANGE = (3.0, 8.0)
    
    # Undulation parameters - THE KEY TO FIXING DEFORMATION
    UNDULATION_AMPLITUDE = 0.3  # MUCH smaller - barely visible undulation
    UNDULATION_FREQUENCY_RANGE = (0.8, 1.2)
    UNDULATION_FORCE_SCALE = 0.8  # Very gentle undulation forces
    
    # Deformation parameters - SIMPLE AND EFFECTIVE
    SPRING_CONSTANT_BASE = 75.0  # Strong enough to maintain rod shape
    STIFFNESS_RANGE = (0.8, 0.95)  # High stiffness = less deformation
    DAMPING_FACTOR = 0.4
    MASS = 0.1  # Slightly higher mass for stability
    
    # Tissue interaction parameters - KEEP MINIMAL
    TISSUE_RESISTANCE_SCALE = 0.03
    TISSUE_FLOW_SCALE = 0.05
    
    # Immune system parameters
    ENABLE_IMMUNE_RESPONSE = False  # Set to False to disable immune damage
    
    # Direction change parameters
    RANDOM_DIRECTION_PROBABILITY = 0.02  # Less frequent direction changes
    MAX_DIRECTION_CHANGE = 0.2
    
    # Boundary conditions
    BOUNDARY_DAMPING = 0.5
    
    # Physical properties
    SPOROZOITE_LENGTH_RANGE = (10.0, 15.0)
    SPOROZOITE_DIAMETER_RANGE = (0.8, 1.2)
    
    # Mesh resolution
    LONGITUDINAL_SEGMENTS = 12  # Fewer segments for stability
    RADIAL_SEGMENTS = 6

class PresetConfigs:
    """Simplified presets focusing on working solutions"""
    
    @staticmethod
    def minimal_deformation():
        """Minimal deformation while allowing movement - RECOMMENDED"""
        config = SimulationConfig()
        config.TIME_STEP = 0.12
        config.DIRECTIONAL_FORCE_STRENGTH = 10.0
        config.UNDULATION_AMPLITUDE = 0.2  # Barely any undulation
        config.UNDULATION_FORCE_SCALE = 0.5
        config.SPRING_CONSTANT_BASE = 100.0  # Very stiff
        config.STIFFNESS_RANGE = (0.85, 0.95)
        config.DAMPING_FACTOR = 0.3
        return config
    
    @staticmethod
    def realistic_movement():
        """Balanced realistic movement"""
        config = SimulationConfig()
        config.TIME_STEP = 0.1
        config.DIRECTIONAL_FORCE_STRENGTH = 8.0
        config.UNDULATION_AMPLITUDE = 0.3
        config.UNDULATION_FORCE_SCALE = 0.8
        config.SPRING_CONSTANT_BASE = 75.0
        config.STIFFNESS_RANGE = (0.8, 0.9)
        config.DAMPING_FACTOR = 0.4
        return config
    
    @staticmethod
    def fast_movement():
        """Fast movement with shape control"""
        config = SimulationConfig()
        config.TIME_STEP = 0.15
        config.DIRECTIONAL_FORCE_STRENGTH = 15.0
        config.MAX_SPEED_RANGE = (8.0, 15.0)
        config.UNDULATION_AMPLITUDE = 0.25
        config.UNDULATION_FORCE_SCALE = 0.6
        config.SPRING_CONSTANT_BASE = 90.0
        config.STIFFNESS_RANGE = (0.85, 0.95)
        config.DAMPING_FACTOR = 0.25
        return config
    
    @staticmethod
    def gentle_flexible():
        """Gentle movement with slight flexibility"""
        config = SimulationConfig()
        config.TIME_STEP = 0.08
        config.DIRECTIONAL_FORCE_STRENGTH = 6.0
        config.UNDULATION_AMPLITUDE = 0.4
        config.UNDULATION_FORCE_SCALE = 1.0
        config.SPRING_CONSTANT_BASE = 60.0
        config.STIFFNESS_RANGE = (0.75, 0.85)
        config.DAMPING_FACTOR = 0.5
        return config