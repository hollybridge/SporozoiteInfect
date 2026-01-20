"""
Simulation Configuration Parameters

This file contains all the key parameters that control sporozoite movement,
deformation, and simulation dynamics. Adjust these to change behavior.
"""

class SimulationConfig:
    # Time stepping parameters
    TIME_STEP = 0.1  # Increased from 0.01 for faster movement
    MAX_TIME = 20.0
    OUTPUT_INTERVAL = 0.5
    
    # Movement parameters
    DIRECTIONAL_FORCE_STRENGTH = 15.0  # Increased from 2.0 for more movement
    MOTILITY_RANGE = (0.6, 1.0)  # Range for random motility
    MAX_SPEED_RANGE = (5.0, 15.0)  # Increased speed range (micrometers per time step)
    
    # Undulation parameters (sporozoite swimming motion)
    UNDULATION_AMPLITUDE = 2.0  # Reduced from 10.0 to prevent excessive deformation
    UNDULATION_FREQUENCY_RANGE = (0.8, 1.2)  # Hz
    UNDULATION_FORCE_SCALE = 3.0  # Reduced from 10.0
    
    # Deformation parameters
    SPRING_CONSTANT_BASE = 20.0  # Reduced from 50.0 for less stiffness
    STIFFNESS_RANGE = (0.4, 0.8)  # Range for sporozoite stiffness
    DAMPING_FACTOR = 0.3  # Reduced from 0.8 for less velocity loss
    MASS = 0.05  # Reduced mass for more responsiveness
    
    # Tissue interaction parameters
    TISSUE_RESISTANCE_SCALE = 0.05  # How much tissue resists movement
    TISSUE_FLOW_SCALE = 0.1  # How much tissue flow affects sporozoites
    
    # Direction change parameters
    RANDOM_DIRECTION_PROBABILITY = 0.05  # Probability per time step
    MAX_DIRECTION_CHANGE = 0.3  # Maximum radians to change direction
    
    # Boundary conditions
    BOUNDARY_DAMPING = 0.5  # Velocity reduction at boundaries
    
    # Physical properties
    SPOROZOITE_LENGTH_RANGE = (10.0, 15.0)  # micrometers
    SPOROZOITE_DIAMETER_RANGE = (0.8, 1.2)  # micrometers
    
    # Mesh resolution
    LONGITUDINAL_SEGMENTS = 15  # Reduced from 20 for performance
    RADIAL_SEGMENTS = 6  # Reduced from 8 for performance

class PresetConfigs:
    """Predefined configurations for different behaviors"""
    
    @staticmethod
    def fast_movement():
        """Configuration for fast-moving sporozoites"""
        config = SimulationConfig()
        config.TIME_STEP = 0.15
        config.DIRECTIONAL_FORCE_STRENGTH = 25.0
        config.MAX_SPEED_RANGE = (10.0, 20.0)
        config.UNDULATION_AMPLITUDE = 1.5
        config.DAMPING_FACTOR = 0.2
        return config
    
    @staticmethod
    def realistic_movement():
        """Configuration for realistic sporozoite movement"""
        config = SimulationConfig()
        config.TIME_STEP = 0.1
        config.DIRECTIONAL_FORCE_STRENGTH = 12.0
        config.MAX_SPEED_RANGE = (3.0, 8.0)
        config.UNDULATION_AMPLITUDE = 1.8
        config.UNDULATION_FORCE_SCALE = 2.5
        config.DAMPING_FACTOR = 0.4
        return config
    
    @staticmethod
    def slow_deformable():
        """Configuration for slow, highly deformable sporozoites"""
        config = SimulationConfig()
        config.TIME_STEP = 0.05
        config.DIRECTIONAL_FORCE_STRENGTH = 8.0
        config.SPRING_CONSTANT_BASE = 10.0
        config.STIFFNESS_RANGE = (0.2, 0.5)
        config.UNDULATION_AMPLITUDE = 3.0
        config.DAMPING_FACTOR = 0.6
        return config