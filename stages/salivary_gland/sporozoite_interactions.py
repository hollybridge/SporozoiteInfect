"""
Sporozoite Interaction Manager - 3D Volume Effects with Shifted LJ Potential

This module handles sporozoite-sporozoite interactions in high-density environments.
Implements shifted Lennard-Jones potential, alignment, cohesion, and volume exclusion effects.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import pdist, squareform
import random

class SporozoiteInteractionManager:
    """
    Manages interactions between sporozoites in high-density 3D environments using LJ potential
    """
    
    def __init__(self, 
                 interaction_radius: float = 8.0,  # Radius for neighbor detection
                 lj_epsilon: float = 5.0,          # LJ potential depth (energy scale)
                 lj_sigma: float = 2.0,            # LJ potential length scale
                 lj_cutoff: float = 6.0,           # Cutoff distance for LJ potential
                 alignment_strength: float = 1.0,  # NOT USED - kept for compatibility
                 cohesion_strength: float = 0.5,   # NOT USED - kept for compatibility
                 use_centerline_lj: bool = True):  # Use centerline-based LJ vs vertex-based
        
        self.interaction_radius = interaction_radius
        self.lj_epsilon = lj_epsilon
        self.lj_sigma = lj_sigma
        self.lj_cutoff = lj_cutoff
        self.use_centerline_lj = use_centerline_lj
        
        # Removed: alignment_strength and cohesion_strength are not stored
        # Only LJ forces are used for preventing overlap
        
        # Calculate shifted LJ parameters for continuous potential
        self.lj_shift = self._calculate_lj_shift()
        
        # Spatial acceleration structures
        self.position_tree = None
        self.last_tree_update_time = -1
        self.tree_update_frequency = 0.1  # Update every 0.1 time units
        
        print(f"Initializing sporozoite interaction manager with Shifted LJ Potential:")
        print(f"  Interaction radius: {interaction_radius} μm")
        print(f"  LJ epsilon (energy): {lj_epsilon}")
        print(f"  LJ sigma (length): {lj_sigma} μm")
        print(f"  LJ cutoff: {lj_cutoff} μm")
        print(f"  LJ shift: {self.lj_shift:.3f}")
        print(f"  Centerline-based LJ: {use_centerline_lj}")
    
    def _calculate_lj_shift(self) -> float:
        """Calculate the shift for continuous LJ potential at cutoff"""
        if self.lj_cutoff <= 0:
            return 0.0
        
        r_cut = self.lj_cutoff
        sigma_over_r = self.lj_sigma / r_cut
        lj_at_cutoff = 4 * self.lj_epsilon * (sigma_over_r**12 - sigma_over_r**6)
        return lj_at_cutoff
    
    def calculate_shifted_lj_potential(self, r: float) -> float:
        """Calculate shifted Lennard-Jones potential energy"""
        if r >= self.lj_cutoff or r <= 0:
            return 0.0
        
        sigma_over_r = self.lj_sigma / r
        lj_potential = 4 * self.lj_epsilon * (sigma_over_r**12 - sigma_over_r**6) - self.lj_shift
        
        return lj_potential
    
    def calculate_shifted_lj_force(self, r: float, direction_unit: np.ndarray) -> np.ndarray:
        """Calculate shifted Lennard-Jones force (negative gradient of potential)"""
        if r >= self.lj_cutoff or r <= 1e-8:  # Avoid division by zero
            return np.zeros(3)
        
        sigma_over_r = self.lj_sigma / r
        
        # Force magnitude: F = -dU/dr
        force_magnitude = 4 * self.lj_epsilon / r * (12 * sigma_over_r**12 - 6 * sigma_over_r**6)
        
        # Force vector (repulsive at short distances, attractive at intermediate)
        force_vector = force_magnitude * direction_unit
        
        return force_vector
    
    def calculate_centerline_lj_forces(self, sporozoite: object, neighbors: List[Tuple[int, float]], 
                                      sporozoites: List) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate LJ forces between spine positions of sporozoites (simplified)"""
        total_lj_force = np.zeros(3)
        total_lj_energy = 0.0
        
        # Use spine positions if available, otherwise fall back to center of mass
        if hasattr(sporozoite, 'spine_positions') and sporozoite.spine_positions is not None:
            self_spine_positions = sporozoite.spine_positions
        else:
            # Fallback to center-of-mass LJ
            return self.calculate_com_lj_forces(sporozoite, neighbors, sporozoites)
        
        for neighbor_idx, _ in neighbors:
            neighbor = sporozoites[neighbor_idx]
            
            # Get neighbor spine positions
            if hasattr(neighbor, 'spine_positions') and neighbor.spine_positions is not None:
                neighbor_spine_positions = neighbor.spine_positions
            else:
                continue
            
            # Calculate LJ interactions between spine position pairs
            # Use a subset of spine positions to reduce computation (every 2nd or 3rd point)
            spine_step = max(1, len(self_spine_positions) // 5)  # Use at most 5 spine points
            neighbor_step = max(1, len(neighbor_spine_positions) // 5)
            
            for i in range(0, len(self_spine_positions), spine_step):
                self_pos = self_spine_positions[i]
                
                for j in range(0, len(neighbor_spine_positions), neighbor_step):
                    neighbor_pos = neighbor_spine_positions[j]
                    
                    # Distance between spine positions
                    seg_vector = neighbor_pos - self_pos
                    seg_distance = np.linalg.norm(seg_vector)
                    
                    if seg_distance < 1e-8:  # Avoid division by zero
                        continue
                    
                    # Direction vector
                    direction_unit = seg_vector / seg_distance
                    
                    # Calculate LJ force using standard parameters
                    if seg_distance < self.lj_cutoff:
                        sigma_over_r = self.lj_sigma / seg_distance
                        force_magnitude = 4 * self.lj_epsilon / seg_distance * (12 * sigma_over_r**12 - 6 * sigma_over_r**6)
                        
                        # Weight force by distance along spine (middle sections have more influence)
                        spine_weight = 1.0  # Simplified - equal weight for all spine points
                        
                        lj_force = -force_magnitude * direction_unit * spine_weight  # Negative for repulsion
                        total_lj_force += lj_force
                        
                        # Calculate potential energy
                        lj_energy = 4 * self.lj_epsilon * (sigma_over_r**12 - sigma_over_r**6) - self.lj_shift
                        total_lj_energy += lj_energy * spine_weight
        
        return total_lj_force, total_lj_energy
    
    def calculate_com_lj_forces(self, sporozoite: object, neighbors: List[Tuple[int, float]], 
                               sporozoites: List) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate LJ forces between centers of mass (fallback method)"""
        total_lj_force = np.zeros(3)
        total_lj_energy = 0.0
        
        for neighbor_idx, distance in neighbors:
            neighbor = sporozoites[neighbor_idx]
            
            if distance < 1e-8:  # Avoid division by zero
                continue
            
            # Direction vector
            direction_vector = neighbor.center_position - sporozoite.center_position
            direction_unit = direction_vector / distance
            
            # Calculate LJ force
            lj_force = self.calculate_shifted_lj_force(distance, direction_unit)
            total_lj_force -= lj_force  # Negative for repulsion from neighbor's perspective
            
            # Calculate potential energy
            lj_energy = self.calculate_shifted_lj_potential(distance)
            total_lj_energy += lj_energy
        
        return total_lj_force, total_lj_energy
    
    def calculate_vertex_lj_forces(self, sporozoite: object, neighbors: List[Tuple[int, float]], 
                                  sporozoites: List) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Calculate LJ forces between all vertices (computationally expensive)"""
        center_force = np.zeros(3)
        vertex_forces = [np.zeros(3) for _ in range(len(sporozoite.vertices))]
        total_energy = 0.0
        
        for neighbor_idx, _ in neighbors:
            neighbor = sporozoites[neighbor_idx]
            
            # Calculate forces between all vertex pairs
            for i, self_vertex in enumerate(sporozoite.vertices):
                for neighbor_vertex in neighbor.vertices:
                    # Distance between vertices
                    vertex_vector = neighbor_vertex - self_vertex
                    vertex_distance = np.linalg.norm(vertex_vector)
                    
                    if vertex_distance < 1e-8 or vertex_distance > self.lj_cutoff:
                        continue
                    
                    direction_unit = vertex_vector / vertex_distance
                    
                    # Calculate LJ force on this vertex
                    lj_force = self.calculate_shifted_lj_force(vertex_distance, direction_unit)
                    vertex_forces[i] -= lj_force  # Repulsion
                    
                    # Accumulate center force
                    center_force -= lj_force / len(sporozoite.vertices)
                    
                    # Accumulate energy
                    lj_energy = self.calculate_shifted_lj_potential(vertex_distance)
                    total_energy += lj_energy / (len(sporozoite.vertices) * len(neighbor.vertices))
        
        return center_force, vertex_forces
    
    def apply_interaction_forces(self, sporozoites: List, current_time: float) -> Dict[int, np.ndarray]:
        """Calculate and return ONLY LJ forces for preventing overlap - no other interactions"""
        # Update spatial acceleration structure
        self.update_spatial_tree(sporozoites, current_time)
        
        interaction_forces = {}
        
        for i, sporozoite in enumerate(sporozoites):
            if not sporozoite.is_alive:
                continue
            
            # Find neighbors
            neighbors = self.find_neighbors(i, sporozoites)
            
            # Calculate ONLY LJ forces (for preventing overlap)
            if self.use_centerline_lj:
                lj_force, lj_energy = self.calculate_centerline_lj_forces(sporozoite, neighbors, sporozoites)
            else:
                # Use vertex-based LJ (more expensive)
                center_lj_force, vertex_lj_forces = self.calculate_vertex_lj_forces(sporozoite, neighbors, sporozoites)
                lj_force = center_lj_force
                lj_energy = 0.0
            
            # NO alignment forces - removed
            # NO cohesion forces - removed 
            # NO crowding stress effects - removed
            
            # Total interaction force is ONLY LJ repulsion
            total_force = lj_force
            
            # Store forces for this sporozoite
            interaction_forces[i] = {
                'total': total_force,
                'lj_force': lj_force,
                'lj_energy': lj_energy if self.use_centerline_lj else 0.0,
                'num_neighbors': len(neighbors)
            }
        
        return interaction_forces
    
    def get_lj_statistics(self, sporozoites: List) -> Dict:
        """Get statistics about LJ interactions"""
        if len(sporozoites) < 2:
            return {}
        
        total_energy = 0.0
        min_distance = float('inf')
        max_force = 0.0
        num_interactions = 0
        
        # Calculate pairwise LJ statistics
        for i in range(len(sporozoites)):
            if not sporozoites[i].is_alive:
                continue
            
            neighbors = self.find_neighbors(i, sporozoites)
            
            for neighbor_idx, distance in neighbors:
                if neighbor_idx > i:  # Avoid double counting
                    # Calculate LJ energy for this pair
                    if self.use_centerline_lj:
                        _, lj_energy = self.calculate_centerline_lj_forces(sporozoites[i], [(neighbor_idx, distance)], sporozoites)
                    else:
                        lj_energy = self.calculate_shifted_lj_potential(distance)
                    
                    total_energy += lj_energy
                    min_distance = min(min_distance, distance)
                    
                    # Calculate force magnitude
                    direction = np.array([1, 0, 0])  # Dummy direction
                    force = self.calculate_shifted_lj_force(distance, direction)
                    force_magnitude = np.linalg.norm(force)
                    max_force = max(max_force, force_magnitude)
                    
                    num_interactions += 1
        
        stats = {
            'total_lj_energy': total_energy,
            'min_distance': min_distance if min_distance != float('inf') else 0.0,
            'max_lj_force': max_force,
            'num_lj_interactions': num_interactions,
            'lj_cutoff': self.lj_cutoff,
            'lj_sigma': self.lj_sigma,
            'lj_epsilon': self.lj_epsilon
        }
        
        return stats
    
    def update_spatial_tree(self, sporozoites: List, current_time: float):
        """Update the spatial acceleration structure for neighbor finding"""
        if (self.position_tree is None or 
            current_time - self.last_tree_update_time > self.tree_update_frequency):
            
            positions = np.array([sporo.center_position for sporo in sporozoites])
            if len(positions) > 0:
                self.position_tree = cKDTree(positions)
                self.last_tree_update_time = current_time
    
    def find_neighbors(self, sporozoite_index: int, sporozoites: List) -> List[Tuple[int, float]]:
        """Find neighboring sporozoites within interaction radius"""
        if self.position_tree is None:
            return []
        
        center_pos = sporozoites[sporozoite_index].center_position
        
        # Query neighbors within interaction radius
        neighbor_indices = self.position_tree.query_ball_point(
            center_pos, self.interaction_radius
        )
        
        # Remove self and calculate distances
        neighbors = []
        for idx in neighbor_indices:
            if idx != sporozoite_index:
                distance = np.linalg.norm(
                    sporozoites[idx].center_position - center_pos
                )
                neighbors.append((idx, distance))
        
        return neighbors
    
    def calculate_alignment_force(self, sporozoite: object, neighbors: List[Tuple[int, float]], 
                                  sporozoites: List) -> np.ndarray:
        """Calculate alignment forces to match neighbor orientations"""
        if len(neighbors) == 0:
            return np.zeros(3)
        
        # Average direction of neighbors
        neighbor_directions = []
        weights = []
        
        for neighbor_idx, distance in neighbors:
            neighbor = sporozoites[neighbor_idx]
            neighbor_direction = np.array([
                np.cos(neighbor.direction),
                np.sin(neighbor.direction),
                0
            ])
            
            # Weight by inverse distance (closer neighbors have more influence)
            weight = 1.0 / (distance + 0.1)
            neighbor_directions.append(neighbor_direction * weight)
            weights.append(weight)
        
        if len(neighbor_directions) == 0:
            return np.zeros(3)
        
        # Calculate weighted average direction
        total_weight = sum(weights)
        avg_direction = sum(neighbor_directions) / total_weight
        
        # Normalize
        avg_direction_magnitude = np.linalg.norm(avg_direction)
        if (avg_direction_magnitude > 0.1):
            avg_direction = avg_direction / avg_direction_magnitude
        else:
            return np.zeros(3)
        
        # Current sporozoite direction
        current_direction = np.array([
            np.cos(sporozoite.direction),
            np.sin(sporozoite.direction),
            0
        ])
        
        # Alignment force towards average neighbor direction
        alignment_force = (avg_direction - current_direction) * self.alignment_strength
        
        return alignment_force
    
    def calculate_cohesion_force(self, sporozoite: object, neighbors: List[Tuple[int, float]], 
                                 sporozoites: List) -> np.ndarray:
        """Calculate weak cohesion forces towards group center"""
        if len(neighbors) == 0:
            return np.zeros(3)
        
        # Calculate center of mass of neighbors
        neighbor_positions = []
        weights = []
        
        for neighbor_idx, distance in neighbors:
            neighbor = sporozoites[neighbor_idx]
            # Weight by inverse distance but cap minimum distance
            weight = 1.0 / max(distance, 1.0)
            neighbor_positions.append(neighbor.center_position * weight)
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight > 0:
            group_center = sum(neighbor_positions) / total_weight
        else:
            return np.zeros(3)
        
        # Weak attraction towards group center
        direction_to_center = group_center - sporozoite.center_position
        distance_to_center = np.linalg.norm(direction_to_center)
        
        if distance_to_center > 0.1:
            cohesion_force = direction_to_center * self.cohesion_strength / distance_to_center
        else:
            cohesion_force = np.zeros(3)
        
        return cohesion_force
    
    def calculate_local_density(self, sporozoite: object, neighbors: List[Tuple[int, float]]) -> float:
        """Calculate local sporozoite density around this sporozoite"""
        if len(neighbors) == 0:
            return 0.0
        
        # Count neighbors within different distance shells
        close_neighbors = sum(1 for _, dist in neighbors if dist < self.interaction_radius * 0.3)
        medium_neighbors = sum(1 for _, dist in neighbors if dist < self.interaction_radius * 0.6)
        all_neighbors = len(neighbors)
        
        # Volume of interaction sphere
        interaction_volume = (4/3) * np.pi * (self.interaction_radius**3)
        
        # Density as neighbors per unit volume
        local_density = all_neighbors / interaction_volume
        
        return local_density
    
    def calculate_crowding_stress(self, local_density: float) -> float:
        """Calculate stress due to overcrowding"""
        # Define comfort density (sporozoites per μm³)
        comfort_density = 0.1
        
        if local_density > comfort_density:
            # Stress increases with overcrowding
            stress_factor = (local_density - comfort_density) / comfort_density
            return min(stress_factor, 2.0)  # Cap at 2x stress
        else:
            return 0.0
    
    def get_interaction_statistics(self, sporozoites: List) -> Dict:
        """Get statistics about sporozoite interactions"""
        if len(sporozoites) == 0:
            return {}
        
        # Calculate pairwise distances
        positions = np.array([sporo.center_position for sporo in sporozoites if sporo.is_alive])
        
        if len(positions) < 2:
            return {'num_sporozoites': len(positions)}
        
        distances = pdist(positions)
        
        stats = {
            'num_sporozoites': len(positions),
            'mean_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'std_distance': np.std(distances),
            'close_pairs': np.sum(distances < 2.0),  # Using 2.0 as close distance threshold
            'interaction_pairs': np.sum(distances < self.interaction_radius)
        }
        
        return stats
    
    def detect_clustering(self, sporozoites: List, cluster_threshold: float = 15.0) -> List[List[int]]:
        """Detect clusters of sporozoites"""
        if len(sporozoites) < 2:
            return []
        
        positions = np.array([sporo.center_position for sporo in sporozoites if sporo.is_alive])
        alive_indices = [i for i, sporo in enumerate(sporozoites) if sporo.is_alive]
        
        if len(positions) < 2:
            return []
        
        # Build adjacency matrix based on distance threshold
        distance_matrix = squareform(pdist(positions))
        adjacency = distance_matrix < cluster_threshold
        
        # Find connected components (clusters)
        n = len(positions)
        visited = [False] * n
        clusters = []
        
        def dfs(node, cluster):
            visited[node] = True
            cluster.append(alive_indices[node])
            
            for neighbor in range(n):
                if not visited[neighbor] and adjacency[node, neighbor]:
                    dfs(neighbor, cluster)
        
        for i in range(n):
            if not visited[i]:
                cluster = []
                dfs(i, cluster)
                if len(cluster) > 1:  # Only consider groups of 2 or more
                    clusters.append(cluster)
        
        return clusters