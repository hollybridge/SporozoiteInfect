import numpy as np
import random
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.sporozoite import Sporozoite, SporozoiteState

class BloodVessel:
    """Represents a blood vessel in the vascular system"""
    
    def __init__(self, vessel_id: int, start_pos: Tuple[float, float], 
                 end_pos: Tuple[float, float], diameter: float):
        self.id = vessel_id
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.diameter = diameter
        
        # Calculate vessel properties
        self.length = np.linalg.norm(self.end_pos - self.start_pos)
        self.direction = (self.end_pos - self.start_pos) / self.length
        
        # Blood flow properties
        self.flow_velocity = random.uniform(50.0, 200.0)  # micrometers/second
        self.wall_thickness = random.uniform(2.0, 5.0)
        self.permeability = random.uniform(0.1, 0.3)
        
        self.sporozoites: List[Sporozoite] = []
    
    def is_near_vessel(self, position: np.ndarray, detection_range: float = 10.0) -> bool:
        """Check if a position is near this blood vessel"""
        # Calculate distance from point to line segment
        t = np.dot(position - self.start_pos, self.direction)
        t = np.clip(t, 0, self.length)
        closest_point = self.start_pos + t * self.direction
        distance = np.linalg.norm(position - closest_point)
        
        return distance <= detection_range
    
    def can_penetrate(self, sporozoite: Sporozoite) -> bool:
        """Check if sporozoite can penetrate vessel wall"""
        penetration_prob = sporozoite.motility * self.permeability * 0.5
        return random.random() < penetration_prob
    
    def add_sporozoite(self, sporozoite: Sporozoite):
        """Add sporozoite to blood vessel"""
        sporozoite.state = SporozoiteState.BLOOD_VESSEL
        self.sporozoites.append(sporozoite)
        
        # Position sporozoite inside vessel
        t = random.uniform(0, self.length)
        vessel_position = self.start_pos + t * self.direction
        sporozoite.position = vessel_position

class VascularNetwork:
    """Represents the network of blood vessels"""
    
    def __init__(self, width: float = 200.0, height: float = 200.0):
        self.width = width
        self.height = height
        self.vessels: List[BloodVessel] = []
        self.migrating_sporozoites: List[Sporozoite] = []
        
        # Create a simple vascular network
        self._create_vessel_network()
        
        # Vascular environment properties
        self.blood_flow_resistance = random.uniform(0.8, 1.2)
        self.vessel_density = len(self.vessels) / (width * height)
    
    def _create_vessel_network(self):
        """Create a simple network of blood vessels"""
        # Major vessels (arterioles/venules)
        major_vessels = [
            ((20, 100), (180, 100), 8.0),   # Horizontal major vessel
            ((100, 20), (100, 180), 8.0),   # Vertical major vessel
            ((50, 50), (150, 150), 6.0),    # Diagonal vessel
            ((150, 50), (50, 150), 6.0),    # Another diagonal
        ]
        
        # Capillary network
        capillaries = []
        for i in range(8):
            start_x = random.uniform(20, 180)
            start_y = random.uniform(20, 180)
            end_x = start_x + random.uniform(-50, 50)
            end_y = start_y + random.uniform(-50, 50)
            
            # Ensure endpoints are within bounds
            end_x = np.clip(end_x, 20, 180)
            end_y = np.clip(end_y, 20, 180)
            
            capillaries.append(((start_x, start_y), (end_x, end_y), 
                              random.uniform(3.0, 5.0)))
        
        # Create vessel objects
        vessel_id = 0
        for start, end, diameter in major_vessels + capillaries:
            vessel = BloodVessel(vessel_id, start, end, diameter)
            self.vessels.append(vessel)
            vessel_id += 1
    
    def add_migrating_sporozoite(self, sporozoite: Sporozoite):
        """Add sporozoite that's migrating towards blood vessels"""
        sporozoite.state = SporozoiteState.MIGRATING
        self.migrating_sporozoites.append(sporozoite)
    
    def find_nearest_vessel(self, position: np.ndarray) -> BloodVessel:
        """Find the nearest blood vessel to a position"""
        min_distance = float('inf')
        nearest_vessel = None
        
        for vessel in self.vessels:
            # Calculate distance to vessel
            t = np.dot(position - vessel.start_pos, vessel.direction)
            t = np.clip(t, 0, vessel.length)
            closest_point = vessel.start_pos + t * vessel.direction
            distance = np.linalg.norm(position - closest_point)
            
            if distance < min_distance:
                min_distance = distance
                nearest_vessel = vessel
        
        return nearest_vessel
    
    def calculate_chemotactic_force(self, sporozoite: Sporozoite) -> np.ndarray:
        """Calculate chemotactic attraction towards blood vessels"""
        nearest_vessel = self.find_nearest_vessel(sporozoite.position)
        
        if nearest_vessel:
            # Calculate direction towards nearest point on vessel
            t = np.dot(sporozoite.position - nearest_vessel.start_pos, nearest_vessel.direction)
            t = np.clip(t, 0, nearest_vessel.length)
            target_point = nearest_vessel.start_pos + t * nearest_vessel.direction
            
            direction_to_vessel = target_point - sporozoite.position
            distance = np.linalg.norm(direction_to_vessel)
            
            if distance > 0:
                # Chemotactic force decreases with distance
                force_magnitude = 2.0 / (1 + distance * 0.1)
                return (direction_to_vessel / distance) * force_magnitude
        
        return np.array([0.0, 0.0])
    
    def update_migrating_sporozoites(self, time_step: float):
        """Update sporozoites migrating towards blood vessels"""
        remaining_sporozoites = []
        newly_entered = 0
        
        for sporozoite in self.migrating_sporozoites:
            if sporozoite.is_viable():
                # Calculate chemotactic force towards vessels
                chemotactic_force = self.calculate_chemotactic_force(sporozoite)
                
                # Update position with chemotaxis
                sporozoite.update_position(time_step, chemotactic_force)
                
                # Check if sporozoite reached a blood vessel
                vessel_found = False
                for vessel in self.vessels:
                    if vessel.is_near_vessel(sporozoite.position, vessel.diameter/2 + 2.0):
                        if vessel.can_penetrate(sporozoite):
                            vessel.add_sporozoite(sporozoite)
                            vessel_found = True
                            newly_entered += 1
                            break
                
                if not vessel_found:
                    # Continue migrating
                    if random.random() < 0.05:  # Random direction change
                        sporozoite.change_direction()
                    
                    # Keep within bounds
                    sporozoite.position[0] = np.clip(sporozoite.position[0], 0, self.width)
                    sporozoite.position[1] = np.clip(sporozoite.position[1], 0, self.height)
                    
                    remaining_sporozoites.append(sporozoite)
        
        self.migrating_sporozoites = remaining_sporozoites
        return newly_entered
    
    def update_vessel_sporozoites(self, time_step: float):
        """Update sporozoites inside blood vessels"""
        total_in_vessels = 0
        
        for vessel in self.vessels:
            active_sporozoites = []
            
            for sporozoite in vessel.sporozoites:
                if sporozoite.is_viable():
                    # Move with blood flow
                    flow_force = vessel.direction * vessel.flow_velocity * 0.01
                    sporozoite.update_position(time_step, flow_force)
                    
                    # Deformation due to shear stress
                    shear_stress = vessel.flow_velocity * 0.001
                    sporozoite.deform(shear_stress)
                    
                    # Check if sporozoite exits vessel
                    distance_from_start = np.linalg.norm(sporozoite.position - vessel.start_pos)
                    if distance_from_start > vessel.length:
                        # Sporozoite has traveled through vessel
                        sporozoite.state = SporozoiteState.INFECTED
                        # In a full simulation, this could lead to liver infection
                    else:
                        active_sporozoites.append(sporozoite)
            
            vessel.sporozoites = active_sporozoites
            total_in_vessels += len(active_sporozoites)
        
        return total_in_vessels

class VascularStageSimulation:
    """Main simulation class for the vascular stage"""
    
    def __init__(self):
        self.vascular_network = VascularNetwork(200, 200)
        self.time = 0.0
        self.time_step = 0.1
        
        # Statistics
        self.total_entered_vessels = 0
        self.total_successful_infections = 0
    
    def add_sporozoite_from_dermis(self, sporozoite: Sporozoite):
        """Add sporozoite from dermal stage"""
        self.vascular_network.add_migrating_sporozoite(sporozoite)
    
    def run_simulation_step(self):
        """Run one simulation step"""
        self.time += self.time_step
        
        # Update migrating sporozoites
        newly_entered = self.vascular_network.update_migrating_sporozoites(self.time_step)
        self.total_entered_vessels += newly_entered
        
        # Update sporozoites in vessels
        total_in_vessels = self.vascular_network.update_vessel_sporozoites(self.time_step)
        
        return {
            'time': self.time,
            'migrating_count': len(self.vascular_network.migrating_sporozoites),
            'in_vessels_count': total_in_vessels,
            'newly_entered': newly_entered,
            'total_entered': self.total_entered_vessels
        }
    
    def get_sporozoite_positions(self) -> List[dict]:
        """Get positions of all sporozoites"""
        positions = []
        
        # Migrating sporozoites
        for sporozoite in self.vascular_network.migrating_sporozoites:
            positions.append(sporozoite.get_info())
        
        # Sporozoites in vessels
        for vessel in self.vascular_network.vessels:
            for sporozoite in vessel.sporozoites:
                info = sporozoite.get_info()
                info['vessel_id'] = vessel.id
                positions.append(info)
        
        return positions
    
    def get_vessel_info(self) -> List[dict]:
        """Get information about blood vessels"""
        vessel_info = []
        for vessel in self.vascular_network.vessels:
            vessel_info.append({
                'id': vessel.id,
                'start_pos': vessel.start_pos.tolist(),
                'end_pos': vessel.end_pos.tolist(),
                'diameter': vessel.diameter,
                'sporozoite_count': len(vessel.sporozoites)
            })
        return vessel_info