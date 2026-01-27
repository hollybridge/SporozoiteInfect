"""
3D Sporozoite Mesh Visualization

This module provides functions to visualize sporozoite meshes in 3D using matplotlib
and export vertex data for ParaView visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from typing import List, Optional, Tuple
import os
import sys

# Import sporozoite mesh class
sys.path.append(os.path.dirname(__file__))
from sporozoite_mesh import DeformableSporozoiteMesh
from simulation_config import SimulationConfig

class Sporozoite3DPlotter:
    """
    Class for 3D visualization of sporozoite meshes
    """
    
    def __init__(self, figsize=(12, 9)):
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def setup_plot(self, title="Sporozoite 3D Mesh Visualization"):
        """Setup the 3D plot"""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        self.ax.set_xlabel('X Position (μm)', fontsize=12)
        self.ax.set_ylabel('Y Position (μm)', fontsize=12)
        self.ax.set_zlabel('Z Position (μm)', fontsize=12)
        
        return self.fig, self.ax
    
    def plot_sporozoite_mesh(self, sporozoite: DeformableSporozoiteMesh, 
                           show_vertices=True, show_faces=True, 
                           vertex_color='red', face_color='lightblue',
                           face_alpha=0.7, vertex_size=20):
        """
        Plot a single sporozoite mesh in 3D
        
        Args:
            sporozoite: DeformableSporozoiteMesh object
            show_vertices: Whether to show vertex points
            show_faces: Whether to show mesh faces
            vertex_color: Color for vertex points
            face_color: Color for mesh faces
            face_alpha: Transparency of faces (0-1)
            vertex_size: Size of vertex markers
        """
        vertices = sporozoite.vertices
        faces = sporozoite.faces
        
        # Plot mesh faces if requested
        if show_faces and len(faces) > 0:
            # Create face collection
            face_vertices = []
            for face in faces:
                if len(face) >= 3:  # Only valid faces
                    face_coords = vertices[face]
                    face_vertices.append(face_coords)
            
            if face_vertices:
                # Color faces by viability
                viability = sporozoite.viability
                if viability > 0.8:
                    actual_face_color = 'lightgreen'
                elif viability > 0.5:
                    actual_face_color = 'yellow'
                elif viability > 0.2:
                    actual_face_color = 'orange'
                else:
                    actual_face_color = 'red'
                
                poly3d = Poly3DCollection(face_vertices, 
                                        facecolors=actual_face_color,
                                        alpha=face_alpha,
                                        edgecolors='black',
                                        linewidths=0.5)
                self.ax.add_collection3d(poly3d)
        
        # Plot vertices if requested
        if show_vertices:
            # Color vertices by their distance from center (to show deformation)
            center = sporozoite.center_position
            distances = [np.linalg.norm(v - center) for v in vertices]
            
            scatter = self.ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                    c=distances, cmap='viridis', s=vertex_size,
                                    alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Add colorbar for distances
            cbar = plt.colorbar(scatter, ax=self.ax, shrink=0.5, aspect=10)
            cbar.set_label('Distance from Center (μm)', fontsize=10)
        
        # Plot center point
        self.ax.scatter(center[0], center[1], center[2], 
                       c='black', s=100, marker='*', 
                       label=f'Sporozoite {sporozoite.id} Center')
        
        # Plot direction arrow
        direction_vec = np.array([np.cos(sporozoite.direction), 
                                 np.sin(sporozoite.direction), 0]) * 3
        self.ax.quiver(center[0], center[1], center[2],
                      direction_vec[0], direction_vec[1], direction_vec[2],
                      color='red', arrow_length_ratio=0.1, linewidth=2,
                      label=f'Direction (ID: {sporozoite.id})')
    
    def plot_multiple_sporozoites(self, sporozoites: List[DeformableSporozoiteMesh],
                                 show_vertices=True, show_faces=True,
                                 different_colors=True):
        """
        Plot multiple sporozoites in the same 3D plot
        
        Args:
            sporozoites: List of DeformableSporozoiteMesh objects
            show_vertices: Whether to show vertex points
            show_faces: Whether to show mesh faces
            different_colors: Use different colors for each sporozoite
        """
        if not sporozoites:
            print("No sporozoites to plot")
            return
        
        # Define colors for different sporozoites
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                 'lightpink', 'lightgray', 'lightcyan', 'wheat']
        
        for i, sporozoite in enumerate(sporozoites):
            if different_colors:
                face_color = colors[i % len(colors)]
                vertex_color = face_color
            else:
                face_color = 'lightblue'
                vertex_color = 'red'
            
            self.plot_sporozoite_mesh(sporozoite, 
                                    show_vertices=show_vertices,
                                    show_faces=show_faces,
                                    face_color=face_color,
                                    vertex_color=vertex_color,
                                    vertex_size=15)
        
        # Set equal aspect ratio for all sporozoites
        self._set_equal_aspect_ratio(sporozoites)
        
        # Add legend
        self.ax.legend(fontsize=10, loc='upper right')
        
        # Add information text
        info_text = f"Sporozoites: {len(sporozoites)}\n"
        info_text += f"Total vertices: {sum(len(s.vertices) for s in sporozoites)}\n"
        info_text += f"Total faces: {sum(len(s.faces) for s in sporozoites)}"
        
        self.ax.text2D(0.02, 0.98, info_text, transform=self.ax.transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _set_equal_aspect_ratio(self, sporozoites: List[DeformableSporozoiteMesh]):
        """Set equal aspect ratio based on all sporozoites"""
        if not sporozoites:
            return
        
        # Get all vertices
        all_vertices = np.vstack([s.vertices for s in sporozoites])
        
        # Calculate bounds
        min_coords = np.min(all_vertices, axis=0)
        max_coords = np.max(all_vertices, axis=0)
        
        # Add some padding
        padding = 2.0
        self.ax.set_xlim(min_coords[0] - padding, max_coords[0] + padding)
        self.ax.set_ylim(min_coords[1] - padding, max_coords[1] + padding)
        self.ax.set_zlim(min_coords[2] - padding, max_coords[2] + padding)
        
        # Try to set equal aspect ratio
        try:
            self.ax.set_box_aspect([1,1,1])
        except:
            # Fallback for older matplotlib versions
            pass
    
    def save_plot(self, filename: str, dpi=300, bbox_inches='tight'):
        """Save the current plot to file"""
        if self.fig is None:
            print("No plot to save. Call setup_plot() and plot functions first.")
            return
        
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Plot saved to: {filename}")
    
    def show(self):
        """Display the plot"""
        if self.fig is None:
            print("No plot to show. Call setup_plot() and plot functions first.")
            return
        
        plt.tight_layout()
        plt.show()

def create_sample_sporozoite(sporozoite_id=1, position=None):
    """
    Create a sample sporozoite for testing visualization
    
    Args:
        sporozoite_id: ID for the sporozoite
        position: Initial position (default: origin)
    
    Returns:
        DeformableSporozoiteMesh object
    """
    if position is None:
        position = np.array([0.0, 0.0, 0.0])
    
    config = SimulationConfig()
    sporozoite = DeformableSporozoiteMesh(sporozoite_id, position, config)
    
    return sporozoite

def plot_single_sporozoite_example():
    """Example function to plot a single sporozoite"""
    print("Creating sample sporozoite...")
    sporozoite = create_sample_sporozoite(1, np.array([5.0, 2.0, 1.0]))
    
    print("Setting up 3D plot...")
    plotter = Sporozoite3DPlotter()
    plotter.setup_plot("Single Sporozoite Mesh Example")
    
    print("Plotting sporozoite mesh...")
    plotter.plot_sporozoite_mesh(sporozoite, show_vertices=True, show_faces=True)
    
    print("Displaying plot...")
    plotter.show()

def plot_multiple_sporozoites_example():
    """Example function to plot multiple sporozoites"""
    print("Creating multiple sample sporozoites...")
    
    # Create several sporozoites at different positions
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([15.0, 5.0, 2.0]),
        np.array([-10.0, -8.0, 3.0]),
        np.array([8.0, -12.0, -1.0])
    ]
    
    sporozoites = []
    for i, pos in enumerate(positions):
        sporozoite = create_sample_sporozoite(i+1, pos)
        # Vary some properties for visualization
        sporozoite.direction = np.random.uniform(0, 2*np.pi)
        sporozoite.viability = np.random.uniform(0.3, 1.0)
        sporozoites.append(sporozoite)
    
    print("Setting up 3D plot...")
    plotter = Sporozoite3DPlotter(figsize=(14, 10))
    plotter.setup_plot("Multiple Sporozoites Mesh Visualization")
    
    print("Plotting multiple sporozoite meshes...")
    plotter.plot_multiple_sporozoites(sporozoites, 
                                     show_vertices=True, 
                                     show_faces=True,
                                     different_colors=True)
    
    print("Displaying plot...")
    plotter.show()

if __name__ == "__main__":
    print("Sporozoite 3D Mesh Plotter")
    print("1. Single sporozoite example")
    print("2. Multiple sporozoites example")
    
    choice = input("Enter choice (1 or 2, or press Enter for single): ").strip()
    
    if choice == "2":
        plot_multiple_sporozoites_example()
    else:
        plot_single_sporozoite_example()