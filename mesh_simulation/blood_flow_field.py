"""
Implicit Blood Flow Environment

This module implements different flow types in blood vessels using an implicit field that provides
resistance, flow, and constraint forces for mesh-based sporozoite simulation.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import random
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator
from enum import Enum

class FlowType(Enum):
    SIMPLE_SHEAR = "simple_shear"
    LAMINAR = "laminar" 
    TURBULENT = "turbulent"

class ImplicitBloodFlowField:
    """
    Represents blood flow in vessels as an implicit field with periodic boundary conditions
    and different flow types (simple shear, laminar, turbulent)
    """
    
    def __init__(self, 
                 domain_size: Tuple[float, float, float] = (150.0, 100.0, 100.0),
                 flow_type: FlowType = FlowType.LAMINAR,
                 inlet_velocity: float = 50.0,  # micrometers/second
                 vessel_diameter: float = 20.0,  # micrometers
                 reynolds_number: float = 100.0,
                 viscosity: float = 0.004,  # Pa⋅s (blood viscosity)
                 hematocrit: float = 0.45):  # volume fraction of red blood cells
        
        self.domain_size = np.array(domain_size)
        self.domain_origin = np.array([0.0, 0.0, 0.0])  # Start from origin for blood vessels
        
        # Flow parameters
        self.flow_type = flow_type
        self.inlet_velocity = inlet_velocity
        self.vessel_diameter = vessel_diameter
        self.reynolds_number = reynolds_number
        self.viscosity = viscosity
        self.hematocrit = hematocrit
        
        # Grid resolution for field sampling
        self.grid_resolution = (120, 50, 50)  # Much higher resolution in flow direction (x) for longer channel
        
        # Flow fields
        self.velocity_field = None
        self.pressure_field = None
        self.shear_rate_field = None
        self.turbulence_intensity_field = None
        
        # Periodic boundary conditions
        self.periodic_x = True  # Flow direction
        self.periodic_y = False  # Cross-flow
        self.periodic_z = False  # Cross-flow
        
        print(f"Initializing blood flow field: {flow_type.value}")
        print(f"  Inlet velocity: {inlet_velocity} μm/s")
        print(f"  Vessel diameter: {vessel_diameter} μm")
        print(f"  Reynolds number: {reynolds_number}")
        
        self._generate_flow_fields()
    
    def _generate_flow_fields(self):
        """Generate blood flow fields based on flow type"""
        # Create coordinate grids
        x = np.linspace(self.domain_origin[0], self.domain_origin[0] + self.domain_size[0], self.grid_resolution[0])
        y = np.linspace(self.domain_origin[1], self.domain_origin[1] + self.domain_size[1], self.grid_resolution[1])
        z = np.linspace(self.domain_origin[2], self.domain_origin[2] + self.domain_size[2], self.grid_resolution[2])
        
        self.grid_coords = (x, y, z)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize fields
        self.velocity_field = np.zeros((*X.shape, 3))  # 3D velocity vector field
        self.pressure_field = np.zeros_like(X)
        self.shear_rate_field = np.zeros_like(X)
        self.turbulence_intensity_field = np.zeros_like(X)
        
        # Generate flow based on type
        if self.flow_type == FlowType.SIMPLE_SHEAR:
            self._generate_simple_shear_flow(X, Y, Z)
        elif self.flow_type == FlowType.LAMINAR:
            self._generate_laminar_flow(X, Y, Z)
        elif self.flow_type == FlowType.TURBULENT:
            self._generate_turbulent_flow(X, Y, Z)
        
        # Apply periodic boundary conditions
        self._apply_periodic_boundaries()
        
        # Create interpolators for efficient field sampling
        self._create_interpolators()
    
    def _generate_simple_shear_flow(self, X, Y, Z):
        """Generate simple shear flow with linear velocity profile"""
        print("Generating simple shear flow...")
        
        # Center coordinates for vessel
        y_center = self.domain_size[1] / 2
        z_center = self.domain_size[2] / 2
        
        for i in range(self.grid_resolution[0]):
            for j in range(self.grid_resolution[1]):
                for k in range(self.grid_resolution[2]):
                    y_pos = Y[i, j, k]
                    z_pos = Z[i, j, k]
                    
                    # Distance from vessel centerline
                    r = np.sqrt((y_pos - y_center)**2 + (z_pos - z_center)**2)
                    
                    # Only inside vessel
                    if r <= self.vessel_diameter / 2:
                        # Simple shear: linear velocity profile
                        # Maximum velocity at center, zero at walls
                        velocity_magnitude = self.inlet_velocity * (1 - r / (self.vessel_diameter / 2))
                        
                        # Flow in x direction
                        self.velocity_field[i, j, k, 0] = velocity_magnitude
                        
                        # Calculate shear rate for simple shear
                        if r > 0:
                            self.shear_rate_field[i, j, k] = self.inlet_velocity / (self.vessel_diameter / 2)
                        
                        # Pressure drops linearly along vessel ((self.domain_size[0] - X[i, j, k] represents the distance from the current point to the inlet of the vessel)
                        pressure_gradient = 8 * self.viscosity * self.inlet_velocity / (self.vessel_diameter / 2)**2
                        self.pressure_field[i, j, k] = pressure_gradient * (self.domain_size[0] - X[i, j, k]) 
    
    def _generate_laminar_flow(self, X, Y, Z):
        """Generate laminar flow with parabolic velocity profile (Poiseuille flow)"""
        print("Generating laminar (Poiseuille) flow...")
        
        # Center coordinates for vessel
        y_center = self.domain_size[1] / 2
        z_center = self.domain_size[2] / 2
        
        for i in range(self.grid_resolution[0]):
            for j in range(self.grid_resolution[1]):
                for k in range(self.grid_resolution[2]):
                    y_pos = Y[i, j, k]
                    z_pos = Z[i, j, k]
                    
                    # Distance from vessel centerline
                    r = np.sqrt((y_pos - y_center)**2 + (z_pos - z_center)**2)
                    
                    # Only inside vessel
                    if r <= self.vessel_diameter / 2:
                        # Parabolic velocity profile for laminar flow
                        r_normalized = r / (self.vessel_diameter / 2)
                        velocity_magnitude = self.inlet_velocity * (1 - r_normalized**2)
                        
                        # Flow in x direction
                        self.velocity_field[i, j, k, 0] = velocity_magnitude
                        
                        # Calculate shear rate (dv/dr)
                        if r > 0:
                            self.shear_rate_field[i, j, k] = 4 * self.inlet_velocity * r / (self.vessel_diameter / 2)**2
                        
                        # Pressure drops according to Hagen-Poiseuille equation
                        pressure_gradient = 32 * self.viscosity * self.inlet_velocity / self.vessel_diameter**2
                        self.pressure_field[i, j, k] = pressure_gradient * (self.domain_size[0] - X[i, j, k])
    
    def _generate_turbulent_flow(self, X, Y, Z):
        """Generate turbulent flow with time-averaged velocity profile and fluctuations"""
        print("Generating turbulent flow...")
        
        # Center coordinates for vessel
        y_center = self.domain_size[1] / 2
        z_center = self.domain_size[2] / 2
        
        for i in range(self.grid_resolution[0]):
            for j in range(self.grid_resolution[1]):
                for k in range(self.grid_resolution[2]):
                    y_pos = Y[i, j, k]
                    z_pos = Z[i, j, k]
                    
                    # Distance from vessel centerline
                    r = np.sqrt((y_pos - y_center)**2 + (z_pos - z_center)**2)
                    
                    # Only inside vessel
                    if r <= self.vessel_diameter / 2:
                        r_normalized = r / (self.vessel_diameter / 2)
                        
                        # Power-law velocity profile for turbulent flow (1/7th power law)
                        if r < self.vessel_diameter / 2:
                            velocity_magnitude = self.inlet_velocity * (1 - r_normalized)**(1/7)
                        else:
                            velocity_magnitude = 0
                        
                        # Add turbulent fluctuations
                        turbulent_intensity = 0.05 * self.inlet_velocity * (1 - r_normalized) * r_normalized
                        self.turbulence_intensity_field[i, j, k] = turbulent_intensity
                        
                        # Random fluctuations in all directions
                        fluctuation_x = np.random.normal(0, turbulent_intensity * 0.8)
                        fluctuation_y = np.random.normal(0, turbulent_intensity * 0.3)
                        fluctuation_z = np.random.normal(0, turbulent_intensity * 0.3)
                        
                        self.velocity_field[i, j, k, 0] = velocity_magnitude + fluctuation_x
                        self.velocity_field[i, j, k, 1] = fluctuation_y
                        self.velocity_field[i, j, k, 2] = fluctuation_z
                        
                        # Higher shear rates in turbulent flow
                        self.shear_rate_field[i, j, k] = 2 * self.inlet_velocity / (self.vessel_diameter / 2) * (1 + turbulent_intensity / self.inlet_velocity)
                        
                        # Turbulent pressure drop (higher than laminar)
                        friction_factor = 0.316 / self.reynolds_number**0.25  # Blasius equation
                        pressure_gradient = friction_factor * 0.5 * 1000 * self.inlet_velocity**2 / self.vessel_diameter  # Approximate
                        self.pressure_field[i, j, k] = pressure_gradient * (self.domain_size[0] - X[i, j, k])
    
    def _apply_periodic_boundaries(self):
        """Apply periodic boundary conditions in the x-direction (flow direction)"""
        if self.periodic_x:
            # Make velocity field periodic in x-direction
            # Inlet (x=0) matches outlet (x=max)
            self.velocity_field[0, :, :, :] = self.velocity_field[-1, :, :, :]
            
            # Pressure field maintains gradient but is continuous
            pressure_drop = self.pressure_field[0, 0, 0] - self.pressure_field[-1, 0, 0]
            print(f"Applied periodic boundary with pressure drop: {pressure_drop:.2f}")
    
    def _create_interpolators(self):
        """Create interpolation functions for efficient field sampling"""
        # Velocity field interpolators (one for each component)
        self.velocity_interpolators = []
        for component in range(3):
            interpolator = RegularGridInterpolator(
                self.grid_coords, self.velocity_field[:,:,:,component],
                bounds_error=False, fill_value=0.0, method='linear'
            )
            self.velocity_interpolators.append(interpolator)
        
        # Pressure interpolator
        self.pressure_interpolator = RegularGridInterpolator(
            self.grid_coords, self.pressure_field,
            bounds_error=False, fill_value=0.0, method='linear'
        )
        
        # Shear rate interpolator
        self.shear_interpolator = RegularGridInterpolator(
            self.grid_coords, self.shear_rate_field,
            bounds_error=False, fill_value=0.0, method='linear'
        )
        
        # Turbulence intensity interpolator
        self.turbulence_interpolator = RegularGridInterpolator(
            self.grid_coords, self.turbulence_intensity_field,
            bounds_error=False, fill_value=0.0, method='linear'
        )
    
    def get_velocity_at_point(self, point: np.ndarray) -> np.ndarray:
        """Get velocity vector at a specific point with vessel boundary checking"""
        # First check if point is inside vessel - if not, return zero velocity
        if not self.is_point_in_vessel(point):
            return np.zeros(3)
        
        # Apply periodic boundary conditions for sampling
        sampling_point = point.copy()
        if self.periodic_x:
            sampling_point[0] = sampling_point[0] % self.domain_size[0]
        
        # Clamp to domain bounds for non-periodic dimensions
        sampling_point[1] = np.clip(sampling_point[1], self.domain_origin[1], 
                                   self.domain_origin[1] + self.domain_size[1])
        sampling_point[2] = np.clip(sampling_point[2], self.domain_origin[2], 
                                   self.domain_origin[2] + self.domain_size[2])
        
        # Sample velocity field components
        velocity_vector = np.zeros(3)
        for i, interpolator in enumerate(self.velocity_interpolators):
            velocity_vector[i] = interpolator(sampling_point)[()]
        
        return velocity_vector
    
    def get_pressure_at_point(self, point: np.ndarray) -> float:
        """Get pressure at a specific point with vessel boundary checking"""
        # Return zero pressure outside vessel
        if not self.is_point_in_vessel(point):
            return 0.0
            
        sampling_point = point.copy()
        if self.periodic_x:
            sampling_point[0] = sampling_point[0] % self.domain_size[0]
        
        sampling_point[1] = np.clip(sampling_point[1], self.domain_origin[1], 
                                   self.domain_origin[1] + self.domain_size[1])
        sampling_point[2] = np.clip(sampling_point[2], self.domain_origin[2], 
                                   self.domain_origin[2] + self.domain_size[2])
        
        return self.pressure_interpolator(sampling_point)[()]
    
    def get_shear_rate_at_point(self, point: np.ndarray) -> float:
        """Get shear rate at a specific point with vessel boundary checking"""
        # Return zero shear rate outside vessel
        if not self.is_point_in_vessel(point):
            return 0.0
            
        sampling_point = point.copy()
        if self.periodic_x:
            sampling_point[0] = sampling_point[0] % self.domain_size[0]
        
        sampling_point[1] = np.clip(sampling_point[1], self.domain_origin[1], 
                                   self.domain_origin[1] + self.domain_size[1])
        sampling_point[2] = np.clip(sampling_point[2], self.domain_origin[2], 
                                   self.domain_origin[2] + self.domain_size[2])
        
        return self.shear_interpolator(sampling_point)[()]
    
    def get_turbulence_intensity_at_point(self, point: np.ndarray) -> float:
        """Get turbulence intensity at a specific point"""
        sampling_point = point.copy()
        if self.periodic_x:
            sampling_point[0] = sampling_point[0] % self.domain_size[0]
        
        sampling_point[1] = np.clip(sampling_point[1], self.domain_origin[1], 
                                   self.domain_origin[1] + self.domain_size[1])
        sampling_point[2] = np.clip(sampling_point[2], self.domain_origin[2], 
                                   self.domain_origin[2] + self.domain_size[2])
        
        return self.turbulence_interpolator(sampling_point)[()]
    
    def get_resistance_at_point(self, point: np.ndarray) -> float:
        """Get flow resistance at a specific point (for compatibility with tissue field interface)"""
        # Resistance in blood vessels is primarily due to viscosity and vessel geometry
        shear_rate = self.get_shear_rate_at_point(point)
        velocity_magnitude = np.linalg.norm(self.get_velocity_at_point(point))
        
        # Higher resistance for higher viscosity and lower velocity
        if velocity_magnitude > 1e-6:
            resistance = self.viscosity * shear_rate / velocity_magnitude
        else:
            resistance = self.viscosity * 10  # High resistance in stagnant regions
        
        # Add hematocrit effect (higher hematocrit = higher resistance)
        resistance *= (1 + self.hematocrit)
        
        return resistance
    
    def get_flow_field_at_point(self, point: np.ndarray) -> np.ndarray:
        """Get flow field at a specific point (alias for velocity for compatibility)"""
        return self.get_velocity_at_point(point)
    
    def is_point_in_vessel(self, point: np.ndarray) -> bool:
        """Check if a point is within the blood vessel"""
        y_center = self.domain_size[1] / 2
        z_center = self.domain_size[2] / 2
        
        r = np.sqrt((point[1] - y_center)**2 + (point[2] - z_center)**2)
        return r <= self.vessel_diameter / 2
    
    def is_point_in_domain(self, point: np.ndarray) -> bool:
        """Check if a point is within the flow domain"""
        return (np.all(point[1:] >= self.domain_origin[1:]) and 
                np.all(point[1:] <= self.domain_origin[1:] + self.domain_size[1:]))
    
    def apply_flow_forces(self, sporozoite_position: np.ndarray, sporozoite_velocity: np.ndarray) -> np.ndarray:
        """Calculate forces on sporozoite due to blood flow - VELOCITY-ADAPTIVE SCALING"""
        if not self.is_point_in_vessel(sporozoite_position):
            return np.zeros(3)
        
        # Get local flow properties
        flow_velocity = self.get_velocity_at_point(sporozoite_position)
        shear_rate = self.get_shear_rate_at_point(sporozoite_position)
        
        # VELOCITY-ADAPTIVE SCALING to prevent force explosion at high velocities
        flow_speed = np.linalg.norm(flow_velocity)
        
        # Calculate adaptive scaling factor based on flow velocity
        if flow_speed <= 1.0:
            # Low velocity: minimal scaling
            velocity_scale_factor = 0.01
        elif flow_speed <= 5.0:
            # Medium velocity: moderate scaling (logarithmic reduction)
            velocity_scale_factor = 0.01 / (1 + np.log10(flow_speed))
        else:
            # High velocity: strong scaling (inverse scaling)
            velocity_scale_factor = 0.01 / (flow_speed * 0.5)
        
        # Drag force (Stokes drag for small particles) - ADAPTIVE SCALING
        relative_velocity = flow_velocity - sporozoite_velocity
        
        # Base drag coefficient (already reduced from original)
        base_drag_coefficient = 6 * np.pi * self.viscosity * 0.5 * 0.01
        
        # Apply velocity-adaptive scaling
        adaptive_drag_coefficient = base_drag_coefficient * velocity_scale_factor
        drag_force = adaptive_drag_coefficient * relative_velocity
        
        # Shear-induced migration (Segre-Silberberg effect) - ADAPTIVE SCALING
        y_center = self.domain_size[1] / 2
        z_center = self.domain_size[2] / 2
        
        radial_direction = np.array([0, 
                                   sporozoite_position[1] - y_center,
                                   sporozoite_position[2] - z_center])
        radial_distance = np.linalg.norm(radial_direction[1:])
        
        if radial_distance > 1e-6:
            radial_direction[1:] = radial_direction[1:] / radial_distance
            
            # Migration force with velocity-adaptive scaling
            base_migration_strength = shear_rate * radial_distance * 0.0001
            adaptive_migration_strength = base_migration_strength * velocity_scale_factor
            migration_force = radial_direction * adaptive_migration_strength
        else:
            migration_force = np.zeros(3)
        
        # Total force with additional velocity-based capping
        total_force = (drag_force + migration_force)
        
        # Cap maximum total force based on velocity to prevent instability
        force_magnitude = np.linalg.norm(total_force)
        if flow_speed <= 1.0:
            max_force_limit = 5.0  # Low velocity limit
        elif flow_speed <= 5.0:
            max_force_limit = 10.0 / flow_speed  # Decreasing limit for medium velocity
        else:
            max_force_limit = 2.0 / flow_speed  # Strong limit for high velocity
        
        if force_magnitude > max_force_limit:
            total_force = total_force * (max_force_limit / force_magnitude)
        
        # Debug output for high velocities
        if flow_speed > 2.0:
            print(f"    HIGH VELOCITY DEBUG - Flow speed: {flow_speed:.2f}, Scale factor: {velocity_scale_factor:.6f}")
            print(f"      Original force mag: {force_magnitude:.2f}, Capped force mag: {np.linalg.norm(total_force):.2f}")
        
        return total_force
    
    def get_field_visualization_data(self) -> Dict:
        """Get field data for visualization"""
        X, Y, Z = np.meshgrid(*self.grid_coords, indexing='ij')
        
        return {
            'coordinates': (X, Y, Z),
            'velocity_field': self.velocity_field,
            'pressure': self.pressure_field,
            'shear_rate': self.shear_rate_field,
            'turbulence_intensity': self.turbulence_intensity_field,
            'grid_coords': self.grid_coords,
            'flow_type': self.flow_type.value,
            'vessel_diameter': self.vessel_diameter,
            'inlet_velocity': self.inlet_velocity,
            'reynolds_number': self.reynolds_number
        }
    
    @classmethod
    def create_simple_shear(cls, domain_size, shear_rate: float = 100.0, **kwargs):
        """Factory method to create simple shear flow"""
        return cls(domain_size=domain_size, 
                  flow_type=FlowType.SIMPLE_SHEAR,
                  inlet_velocity=shear_rate,
                  **kwargs)
    
    @classmethod
    def create_laminar_flow(cls, domain_size, inlet_velocity: float = 50.0, vessel_diameter: float = 20.0, **kwargs):
        """Factory method to create laminar flow"""
        return cls(domain_size=domain_size,
                  flow_type=FlowType.LAMINAR,
                  inlet_velocity=inlet_velocity,
                  vessel_diameter=vessel_diameter,
                  **kwargs)
    
    @classmethod
    def create_turbulent_flow(cls, domain_size, inlet_velocity: float = 100.0, reynolds_number: float = 2500.0, **kwargs):
        """Factory method to create turbulent flow"""
        return cls(domain_size=domain_size,
                  flow_type=FlowType.TURBULENT,
                  inlet_velocity=inlet_velocity,
                  reynolds_number=reynolds_number,
                  **kwargs)