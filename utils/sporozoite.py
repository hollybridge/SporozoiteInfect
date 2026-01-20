import numpy as np
import random
from typing import Tuple, List
from enum import Enum

class SporozoiteState(Enum):
    SALIVARY_GLAND = "salivary_gland"
    DERMAL_TISSUE = "dermal_tissue"
    MIGRATING = "migrating"
    BLOOD_VESSEL = "blood_vessel"
    INFECTED = "infected"

class Sporozoite:
    """
    Base class representing a Plasmodium sporozoite - a curved, deformable rod
    """
    
    def __init__(self, sporozoite_id: int, initial_position: Tuple[float, float]):
        self.id = sporozoite_id
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        
        # Physical properties of sporozoite
        self.length = random.uniform(10.0, 15.0)  # micrometers
        self.width = random.uniform(0.5, 1.0)     # micrometers
        self.curvature = random.uniform(0.1, 0.3)  # curvature parameter
        
        # Biological properties
        self.motility = random.uniform(0.5, 1.0)  # motility factor
        self.viability = 1.0  # starts fully viable
        self.state = SporozoiteState.SALIVARY_GLAND
        
        # Deformation properties
        self.deformation_factor = 1.0  # 1.0 = no deformation
        self.max_deformation = 2.0
        
        # Movement parameters
        self.max_speed = random.uniform(1.0, 3.0)  # micrometers per time step
        self.direction = random.uniform(0, 2 * np.pi)  # radians
        
    def update_position(self, time_step: float, environment_forces: np.ndarray = None):
        """Update sporozoite position based on motility and environment"""
        if environment_forces is None:
            environment_forces = np.array([0.0, 0.0])
        
        # Calculate movement based on motility and direction
        movement = np.array([
            np.cos(self.direction) * self.max_speed * self.motility,
            np.sin(self.direction) * self.max_speed * self.motility
        ]) * time_step
        
        # Add environmental forces (tissue resistance, flow, etc.)
        total_movement = movement + environment_forces * time_step
        
        self.position += total_movement
        self.velocity = total_movement / time_step if time_step > 0 else np.array([0.0, 0.0])
    
    def deform(self, pressure: float):
        """Simulate deformation due to tissue pressure"""
        deformation_change = pressure * 0.1  # scaling factor
        self.deformation_factor = min(self.max_deformation, 
                                    max(0.5, self.deformation_factor + deformation_change))
    
    def change_direction(self, new_direction: float = None):
        """Change movement direction (random walk or directed)"""
        if new_direction is None:
            # Random direction change
            self.direction += random.uniform(-np.pi/4, np.pi/4)
        else:
            self.direction = new_direction
    
    def reduce_viability(self, factor: float):
        """Reduce viability due to environmental stress"""
        self.viability = max(0.0, self.viability - factor)
    
    def is_viable(self) -> bool:
        """Check if sporozoite is still viable"""
        return self.viability > 0.1
    
    def get_effective_length(self) -> float:
        """Get current effective length considering deformation"""
        return self.length * self.deformation_factor
    
    def get_info(self) -> dict:
        """Get current sporozoite information"""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'state': self.state.value,
            'viability': self.viability,
            'deformation': self.deformation_factor,
            'motility': self.motility,
            'length': self.get_effective_length()
        }