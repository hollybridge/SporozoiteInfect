"""
Implicit Dermal Tissue Environment

This module implements the dermal tissue as an implicit field that provides
resistance, flow, and constraint forces for mesh-based sporozoite simulation.
"""

import numpy as np
from typing import Tuple, Dict
import random
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator

class ImplicitDermalTissue:
    """
    Represents dermal tissue as an implicit field with varying properties
    """
    
    def __init__(self, domain_size: Tuple[float, float, float] = (100.0, 100.0, 50.0)):
        self.domain_size = np.array(domain_size)
        self.domain_origin = -self.domain_size / 2
        
        # Grid resolution for field sampling
        self.grid_resolution = (50, 50, 25)  # cells per dimension
        
        # Tissue properties
        self.collagen_density_field = None
        self.immune_cell_field = None
        self.flow_field = None
        self.pressure_field = None
        
        # Physical parameters
        self.base_viscosity = 0.8
        self.collagen_stiffness = 1.5
        self.immune_activity = 0.3
        
        self._generate_tissue_fields()
    
    def _generate_tissue_fields(self):
        """Generate spatially varying tissue property fields"""
        # Create coordinate grids
        x = np.linspace(self.domain_origin[0], self.domain_origin[0] + self.domain_size[0], self.grid_resolution[0])
        y = np.linspace(self.domain_origin[1], self.domain_origin[1] + self.domain_size[1], self.grid_resolution[1])
        z = np.linspace(self.domain_origin[2], self.domain_origin[2] + self.domain_size[2], self.grid_resolution[2])
        
        self.grid_coords = (x, y, z)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Generate collagen fiber network
        # Create fiber bundles with preferred orientations
        self.collagen_density_field = np.zeros_like(X)
        
        # Add multiple fiber bundle orientations
        for bundle_idx in range(5):
            # Random fiber bundle center and orientation
            center = np.random.uniform(-self.domain_size/4, self.domain_size/4, 3)
            direction = np.random.uniform(-1, 1, 3)
            direction = direction / np.linalg.norm(direction)
            
            # Create fiber bundle using distance from line
            for i in range(self.grid_resolution[0]):
                for j in range(self.grid_resolution[1]):
                    for k in range(self.grid_resolution[2]):
                        point = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                        
                        # Distance from fiber centerline
                        to_point = point - center
                        projected_length = np.dot(to_point, direction)
                        closest_on_line = center + projected_length * direction
                        distance_to_fiber = np.linalg.norm(point - closest_on_line)
                        
                        # Gaussian fiber density
                        fiber_width = random.uniform(5.0, 15.0)
                        fiber_strength = np.exp(-(distance_to_fiber**2) / (2 * fiber_width**2))
                        
                        self.collagen_density_field[i,j,k] += fiber_strength * 0.3
        
        # Add noise for realistic tissue heterogeneity
        noise = np.random.normal(0, 0.1, X.shape)
        self.collagen_density_field += noise
        self.collagen_density_field = np.clip(self.collagen_density_field, 0.1, 1.0)
        
        # Generate immune cell distribution (clustered)
        self.immune_cell_field = np.zeros_like(X)
        
        # Add immune cell clusters
        num_clusters = random.randint(3, 8)
        for cluster_idx in range(num_clusters):
            center = np.random.uniform(self.domain_origin + 10, 
                                     self.domain_origin + self.domain_size - 10)
            cluster_size = random.uniform(8.0, 20.0)
            cluster_strength = random.uniform(0.3, 0.8)
            
            # Create Gaussian cluster
            distances_sq = ((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
            cluster_field = cluster_strength * np.exp(-distances_sq / (2 * cluster_size**2))
            self.immune_cell_field += cluster_field
        
        self.immune_cell_field = np.clip(self.immune_cell_field, 0, 1.0)
        
        # Generate interstitial flow field
        # Simulate lymphatic drainage and capillary flow
        self.flow_field = np.zeros((*X.shape, 3))  # 3D vector field
        
        # Create flow towards lymphatic vessels (simplified as point sinks)
        num_lymphatics = random.randint(2, 4)
        lymphatic_positions = []
        
        for lymph_idx in range(num_lymphatics):
            lymph_pos = np.random.uniform(self.domain_origin + 15,
                                        self.domain_origin + self.domain_size - 15)
            lymphatic_positions.append(lymph_pos)
            
            # Create flow field towards lymphatic vessel
            for i in range(self.grid_resolution[0]):
                for j in range(self.grid_resolution[1]):
                    for k in range(self.grid_resolution[2]):
                        point = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                        
                        # Direction towards lymphatic
                        to_lymphatic = lymph_pos - point
                        distance = np.linalg.norm(to_lymphatic)
                        
                        if distance > 1e-6:
                            # Flow strength decreases with distance
                            flow_strength = 1.0 / (1 + distance * 0.1)
                            flow_direction = to_lymphatic / distance
                            
                            self.flow_field[i,j,k] += flow_direction * flow_strength * 0.5
        
        # Add random turbulence
        turbulence = np.random.normal(0, 0.2, self.flow_field.shape)
        self.flow_field += turbulence
        
        # Generate pressure field (hydrostatic + osmotic)
        # Higher pressure near capillaries, lower near lymphatics
        self.pressure_field = np.ones_like(X) * 0.5  # baseline pressure
        
        # Add pressure variations
        for i in range(3):  # pressure sources (capillaries)
            source_pos = np.random.uniform(self.domain_origin + 10,
                                         self.domain_origin + self.domain_size - 10)
            distances_sq = ((X - source_pos[0])**2 + (Y - source_pos[1])**2 + 
                           (Z - source_pos[2])**2)
            pressure_contribution = 0.3 * np.exp(-distances_sq / 200.0)
            self.pressure_field += pressure_contribution
        
        # Create interpolators for efficient field sampling
        self._create_interpolators()
    
    def _create_interpolators(self):
        """Create interpolation functions for efficient field sampling"""
        self.collagen_interpolator = RegularGridInterpolator(
            self.grid_coords, self.collagen_density_field, 
            bounds_error=False, fill_value=0.5
        )
        
        self.immune_interpolator = RegularGridInterpolator(
            self.grid_coords, self.immune_cell_field,
            bounds_error=False, fill_value=0.1
        )
        
        self.pressure_interpolator = RegularGridInterpolator(
            self.grid_coords, self.pressure_field,
            bounds_error=False, fill_value=0.5
        )
        
        # Flow field interpolators (one for each component)
        self.flow_interpolators = []
        for component in range(3):
            interpolator = RegularGridInterpolator(
                self.grid_coords, self.flow_field[:,:,:,component],
                bounds_error=False, fill_value=0.0
            )
            self.flow_interpolators.append(interpolator)
    
    def get_resistance_at_point(self, point: np.ndarray) -> float:
        """Get tissue resistance at a specific point"""
        # Clamp point to domain
        clamped_point = np.clip(point, 
                               self.domain_origin, 
                               self.domain_origin + self.domain_size)
        
        # Sample collagen density
        collagen_density = self.collagen_interpolator(clamped_point)[()]
        
        # Sample immune cell activity
        immune_activity = self.immune_interpolator(clamped_point)[()]
        
        # Calculate total resistance
        collagen_resistance = collagen_density * self.collagen_stiffness
        immune_resistance = immune_activity * self.immune_activity
        
        total_resistance = self.base_viscosity + collagen_resistance + immune_resistance
        
        return total_resistance
    
    def get_flow_field_at_point(self, point: np.ndarray) -> np.ndarray:
        """Get flow field vector at a specific point"""
        # Clamp point to domain
        clamped_point = np.clip(point,
                               self.domain_origin,
                               self.domain_origin + self.domain_size)
        
        # Sample flow field components
        flow_vector = np.zeros(3)
        for i, interpolator in enumerate(self.flow_interpolators):
            flow_vector[i] = interpolator(clamped_point)[()]
        
        return flow_vector
    
    def get_pressure_at_point(self, point: np.ndarray) -> float:
        """Get pressure at a specific point"""
        clamped_point = np.clip(point,
                               self.domain_origin,
                               self.domain_origin + self.domain_size)
        
        return self.pressure_interpolator(clamped_point)[()]
    
    def get_pressure_gradient_at_point(self, point: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Calculate pressure gradient using finite differences"""
        gradient = np.zeros(3)
        
        for i in range(3):
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += epsilon
            point_minus[i] -= epsilon
            
            pressure_plus = self.get_pressure_at_point(point_plus)
            pressure_minus = self.get_pressure_at_point(point_minus)
            
            gradient[i] = (pressure_plus - pressure_minus) / (2 * epsilon)
        
        return gradient
    
    def is_point_in_domain(self, point: np.ndarray) -> bool:
        """Check if a point is within the tissue domain"""
        return np.all(point >= self.domain_origin) and np.all(point <= self.domain_origin + self.domain_size)
    
    def apply_immune_response(self, sporozoite_position: np.ndarray) -> float:
        """Calculate immune response intensity at sporozoite location"""
        immune_density = self.immune_interpolator(sporozoite_position)[()]
        
        # Immune response reduces sporozoite viability
        damage_rate = immune_density * 0.02  # damage per time step
        
        return damage_rate
    
    def get_field_visualization_data(self) -> Dict:
        """Get field data for visualization"""
        X, Y, Z = np.meshgrid(*self.grid_coords, indexing='ij')
        
        return {
            'coordinates': (X, Y, Z),
            'collagen_density': self.collagen_density_field,
            'immune_cells': self.immune_cell_field,
            'pressure': self.pressure_field,
            'flow_field': self.flow_field,
            'grid_coords': self.grid_coords
        }