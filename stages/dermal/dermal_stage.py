import numpy as np
import random
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.sporozoite import Sporozoite, SporozoiteState

class SalivaryGland:
    """Represents the mosquito's salivary gland"""
    
    def __init__(self, position: Tuple[float, float], capacity: int = 50):
        self.position = np.array(position)
        self.capacity = capacity
        self.sporozoites: List[Sporozoite] = []
        self.extrusion_rate = 0.3  # probability of extrusion per time step
        
    def add_sporozoite(self, sporozoite: Sporozoite):
        """Add sporozoite to salivary gland"""
        if len(self.sporozoites) < self.capacity:
            sporozoite.state = SporozoiteState.SALIVARY_GLAND
            sporozoite.position = self.position + np.random.normal(0, 1, 2)
            self.sporozoites.append(sporozoite)
    
    def extrude_sporozoites(self) -> List[Sporozoite]:
        """Extrude sporozoites into dermal tissue"""
        extruded = []
        remaining = []
        
        for sporozoite in self.sporozoites:
            if random.random() < self.extrusion_rate:
                # Extrude with some force/pressure
                sporozoite.deform(random.uniform(0.2, 0.5))
                sporozoite.state = SporozoiteState.DERMAL_TISSUE
                
                # Give initial direction towards dermis
                sporozoite.change_direction(random.uniform(0, 2 * np.pi))
                extruded.append(sporozoite)
            else:
                remaining.append(sporozoite)
        
        self.sporozoites = remaining
        return extruded

class DermalTissue:
    """Represents the dermal tissue environment"""
    
    def __init__(self, width: float = 200.0, height: float = 200.0):
        self.width = width
        self.height = height
        self.sporozoites: List[Sporozoite] = []
        
        # Tissue properties
        self.tissue_density = random.uniform(0.7, 1.0)
        self.immune_cell_density = random.uniform(0.1, 0.3)
        self.collagen_fiber_density = random.uniform(0.5, 0.8)
        
    def add_sporozoite(self, sporozoite: Sporozoite):
        """Add sporozoite to dermal tissue"""
        self.sporozoites.append(sporozoite)
    
    def calculate_tissue_resistance(self, position: np.ndarray) -> np.ndarray:
        """Calculate resistance force due to tissue structure"""
        # Simulate collagen fiber network resistance
        resistance_magnitude = self.collagen_fiber_density * 0.5
        
        # Add some randomness to simulate irregular tissue structure
        angle = random.uniform(0, 2 * np.pi)
        resistance = np.array([
            np.cos(angle) * resistance_magnitude,
            np.sin(angle) * resistance_magnitude
        ])
        
        return -resistance  # Resistance opposes movement
    
    def simulate_immune_response(self, sporozoite: Sporozoite):
        """Simulate immune cell interactions"""
        if random.random() < self.immune_cell_density * 0.1:
            # Immune cells can damage sporozoite
            sporozoite.reduce_viability(random.uniform(0.05, 0.15))
    
    def update_sporozoites(self, time_step: float):
        """Update all sporozoites in dermal tissue"""
        active_sporozoites = []
        
        for sporozoite in self.sporozoites:
            if sporozoite.is_viable():
                # Calculate environmental forces
                tissue_resistance = self.calculate_tissue_resistance(sporozoite.position)
                
                # Apply tissue pressure causing deformation
                pressure = self.tissue_density * random.uniform(0.1, 0.3)
                sporozoite.deform(pressure)
                
                # Update position with tissue resistance
                sporozoite.update_position(time_step, tissue_resistance)
                
                # Simulate immune response only if enabled in config
                if hasattr(self, 'config') and self.config.ENABLE_IMMUNE_RESPONSE:
                    self.simulate_immune_response(sporozoite)
                
                # Random direction changes due to tissue obstacles
                if random.random() < 0.1:
                    sporozoite.change_direction()
                
                # Keep sporozoite within tissue bounds
                sporozoite.position[0] = np.clip(sporozoite.position[0], 0, self.width)
                sporozoite.position[1] = np.clip(sporozoite.position[1], 0, self.height)
                
                active_sporozoites.append(sporozoite)
        
        self.sporozoites = active_sporozoites
        return len(active_sporozoites)

class DermalStageSimulation:
    """Main simulation class for the dermal stage"""
    
    def __init__(self, num_sporozoites: int = 30, config=None):
        # Import here to avoid circular imports
        from simulation_config import SimulationConfig
        
        self.config = config or SimulationConfig()
        self.salivary_gland = SalivaryGland((50, 50))
        self.dermal_tissue = DermalTissue(200, 200)
        
        # Pass config to dermal tissue
        self.dermal_tissue.config = self.config
        
        # Create initial sporozoites in salivary gland
        for i in range(num_sporozoites):
            sporozoite = Sporozoite(i, (50, 50))
            self.salivary_gland.add_sporozoite(sporozoite)
        
        self.time = 0.0
        self.time_step = 0.1
        
    def run_simulation_step(self):
        """Run one simulation step"""
        self.time += self.time_step
        
        # Extrude sporozoites from salivary gland
        extruded = self.salivary_gland.extrude_sporozoites()
        for sporozoite in extruded:
            self.dermal_tissue.add_sporozoite(sporozoite)
        
        # Update sporozoites in dermal tissue
        active_count = self.dermal_tissue.update_sporozoites(self.time_step)
        
        return {
            'time': self.time,
            'salivary_gland_count': len(self.salivary_gland.sporozoites),
            'dermal_tissue_count': active_count,
            'extruded_this_step': len(extruded)
        }
    
    def get_sporozoite_positions(self) -> List[dict]:
        """Get positions of all sporozoites"""
        positions = []
        
        # Salivary gland sporozoites
        for sporozoite in self.salivary_gland.sporozoites:
            positions.append(sporozoite.get_info())
        
        # Dermal tissue sporozoites
        for sporozoite in self.dermal_tissue.sporozoites:
            positions.append(sporozoite.get_info())
        
        return positions