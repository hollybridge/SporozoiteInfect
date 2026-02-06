"""
3D Sporozoite Mesh Visualization with Step-by-Step Mesh Creation

This module provides functions to visualize sporozoite meshes in 3D using matplotlib,
including step-by-step mesh creation visualization and GIF generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Optional, Tuple, Dict
import os
import sys
from datetime import datetime

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

class MeshCreationVisualizer:
    """
    Class for creating step-by-step visualization of sporozoite mesh creation process
    """
    
    def __init__(self, figsize=(12, 9)):
        self.figsize = figsize
        self.config = SimulationConfig()
        self.steps_data = []
        
    def generate_mesh_steps(self, sporozoite_id=1, position=None):
        """
        Generate all steps of mesh creation process
        
        Returns:
            List of step data dictionaries
        """
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
        
        print(f"Generating mesh creation steps for sporozoite {sporozoite_id}...")
        
        # Initialize sporozoite parameters (copied from DeformableSporozoiteMesh)
        length = np.random.uniform(*self.config.SPOROZOITE_LENGTH_RANGE)
        diameter = np.random.uniform(*self.config.SPOROZOITE_DIAMETER_RANGE)
        n_segments = self.config.LONGITUDINAL_SEGMENTS
        n_radial = self.config.RADIAL_SEGMENTS
        
        self.steps_data = []
        
        # Step 1: Generate curved spine
        print("  Step 1: Generating curved spine...")
        t_values = np.linspace(0, 1, n_segments)
        spine_points = []
        
        for t in t_values:
            x = t * length - length/2
            y = np.sin(t * np.pi) * length * 0.1  # natural curve
            z = 0
            spine_points.append([x, y, z])
        
        spine_points = np.array(spine_points) + position
        
        self.steps_data.append({
            'step': 1,
            'title': 'Step 1: Curved Spine Generation',
            'description': f'Creating {n_segments} spine points with natural curvature',
            'spine_points': spine_points.copy(),
            'vertices': None,
            'cross_sections': None,
            'faces': None,
            'wireframe_edges': None
        })
        
        # Step 2: Generate cross-sections at each spine point
        print("  Step 2: Generating cross-sections...")
        all_vertices = []
        cross_section_data = []
        
        for i, spine_point in enumerate(spine_points):
            # Calculate local coordinate system (same as in DeformableSporozoiteMesh)
            if i == 0:
                tangent = spine_points[i+1] - spine_points[i]
            elif i == len(spine_points) - 1:
                tangent = spine_points[i] - spine_points[i-1]
            else:
                tangent = spine_points[i+1] - spine_points[i-1]
            
            tangent_length = np.linalg.norm(tangent)
            if tangent_length > 1e-8:
                tangent = tangent / tangent_length
            else:
                tangent = np.array([1, 0, 0])
            
            # Create perpendicular vectors
            if abs(tangent[2]) < 0.9:
                normal = np.cross(tangent, [0, 0, 1])
            else:
                normal = np.cross(tangent, [1, 0, 0])
            
            normal_length = np.linalg.norm(normal)
            if normal_length > 1e-8:
                normal = normal / normal_length
            else:
                normal = np.array([0, 1, 0])
            
            binormal = np.cross(tangent, normal)
            binormal_length = np.linalg.norm(binormal)
            if binormal_length > 1e-8:
                binormal = binormal / binormal_length
            else:
                binormal = np.array([0, 0, 1])
            
            # Radius varies along length (tapered ends)
            radius_factor = np.sin(np.pi * i / (n_segments - 1))
            radius_factor = max(0.1, radius_factor)
            radius = diameter/2 * radius_factor
            
            # Create circular cross-section
            cross_section_vertices = []
            for j in range(n_radial):
                angle = 2 * np.pi * j / n_radial
                local_point = (normal * np.cos(angle) + binormal * np.sin(angle)) * radius
                vertex = spine_point + local_point
                all_vertices.append(vertex)
                cross_section_vertices.append(vertex)
            
            cross_section_data.append({
                'center': spine_point,
                'vertices': np.array(cross_section_vertices),
                'radius': radius,
                'normal': normal,
                'binormal': binormal
            })
        
        all_vertices = np.array(all_vertices)
        
        self.steps_data.append({
            'step': 2,
            'title': 'Step 2: Cross-Section Generation',
            'description': f'Creating {n_radial} vertices per spine point ({len(all_vertices)} total vertices)',
            'spine_points': spine_points.copy(),
            'vertices': all_vertices.copy(),
            'cross_sections': cross_section_data.copy(),
            'faces': None,
            'wireframe_edges': None
        })
        
        # Step 3: Create wireframe edges (connecting vertices)
        print("  Step 3: Creating wireframe edges...")
        wireframe_edges = []
        
        # Radial edges (around each cross-section)
        for i in range(n_segments):
            ring_start = i * n_radial
            for j in range(n_radial):
                v1 = ring_start + j
                v2 = ring_start + ((j + 1) % n_radial)
                wireframe_edges.append([v1, v2])
        
        # Longitudinal edges (along spine)
        for i in range(n_segments - 1):
            for j in range(n_radial):
                v1 = i * n_radial + j
                v2 = (i + 1) * n_radial + j
                wireframe_edges.append([v1, v2])
        
        wireframe_edges = np.array(wireframe_edges)
        
        self.steps_data.append({
            'step': 3,
            'title': 'Step 3: Wireframe Edge Creation',
            'description': f'Connecting vertices with {len(wireframe_edges)} edges',
            'spine_points': spine_points.copy(),
            'vertices': all_vertices.copy(),
            'cross_sections': cross_section_data.copy(),
            'faces': None,
            'wireframe_edges': wireframe_edges.copy()
        })
        
        # Step 4: Create triangular faces
        print("  Step 4: Creating triangular faces...")
        faces = []
        
        # Create faces between adjacent cross-sections
        for i in range(n_segments - 1):
            for j in range(n_radial):
                curr_ring = i * n_radial
                next_ring = (i + 1) * n_radial
                curr_j = j
                next_j = (j + 1) % n_radial
                
                v1 = curr_ring + curr_j
                v2 = curr_ring + next_j
                v3 = next_ring + next_j
                v4 = next_ring + curr_j
                
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        faces = np.array(faces)
        
        self.steps_data.append({
            'step': 4,
            'title': 'Step 4: Triangular Face Creation',
            'description': f'Creating {len(faces)} triangular faces',
            'spine_points': spine_points.copy(),
            'vertices': all_vertices.copy(),
            'cross_sections': cross_section_data.copy(),
            'faces': faces.copy(),
            'wireframe_edges': wireframe_edges.copy()
        })
        
        # Step 5: Add end caps
        print("  Step 5: Adding end caps...")
        
        # Add center points for end caps
        vertices_with_caps = np.vstack([all_vertices, [spine_points[0]], [spine_points[-1]]])
        center_start = len(all_vertices)
        center_end = len(all_vertices) + 1
        
        faces_with_caps = faces.copy().tolist()
        
        # Start cap
        for j in range(n_radial):
            next_j = (j + 1) % n_radial
            faces_with_caps.append([center_start, j, next_j])
        
        # End cap
        last_ring_start = (n_segments - 1) * n_radial
        for j in range(n_radial):
            next_j = (j + 1) % n_radial
            faces_with_caps.append([center_end, last_ring_start + next_j, last_ring_start + j])
        
        faces_with_caps = np.array(faces_with_caps)
        
        self.steps_data.append({
            'step': 5,
            'title': 'Step 5: End Cap Addition',
            'description': f'Adding end caps with 2 center points ({len(faces_with_caps)} total faces)',
            'spine_points': spine_points.copy(),
            'vertices': vertices_with_caps.copy(),
            'cross_sections': cross_section_data.copy(),
            'faces': faces_with_caps.copy(),
            'wireframe_edges': wireframe_edges.copy(),
            'end_caps': True
        })
        
        print(f"Generated {len(self.steps_data)} mesh creation steps")
        return self.steps_data
    
    def plot_single_step(self, step_data: dict, ax=None):
        """Plot a single step of mesh creation"""
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        ax.clear()
        ax.set_title(f"{step_data['title']}", fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (μm)', fontsize=12)
        ax.set_ylabel('Y Position (μm)', fontsize=12)
        ax.set_zlabel('Z Position (μm)', fontsize=12)
        
        # Plot based on step
        if step_data['step'] == 1:
            # Step 1: Only spine points
            spine = step_data['spine_points']
            ax.plot(spine[:, 0], spine[:, 1], spine[:, 2], 
                   'ro-', markersize=8, linewidth=3, label='Spine Points')
            ax.scatter(spine[:, 0], spine[:, 1], spine[:, 2], 
                      c='red', s=100, alpha=0.8)
            
        elif step_data['step'] == 2:
            # Step 2: Spine + cross-sections
            spine = step_data['spine_points']
            ax.plot(spine[:, 0], spine[:, 1], spine[:, 2], 
                   'r-', linewidth=2, alpha=0.7, label='Spine')
            
            # Plot cross-sections
            for i, cs_data in enumerate(step_data['cross_sections']):
                vertices = cs_data['vertices']
                # Close the cross-section loop
                vertices_loop = np.vstack([vertices, vertices[0]])
                
                color = plt.cm.viridis(i / len(step_data['cross_sections']))
                ax.plot(vertices_loop[:, 0], vertices_loop[:, 1], vertices_loop[:, 2], 
                       color=color, linewidth=2, alpha=0.8)
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          color=color, s=30, alpha=0.7)
            
            ax.scatter(spine[:, 0], spine[:, 1], spine[:, 2], 
                      c='red', s=80, marker='*', label='Spine Points')
            
        elif step_data['step'] == 3:
            # Step 3: Wireframe
            vertices = step_data['vertices']
            edges = step_data['wireframe_edges']
            
            # Plot wireframe edges
            edge_lines = []
            for edge in edges:
                v1, v2 = edge
                if v1 < len(vertices) and v2 < len(vertices):
                    edge_lines.append([vertices[v1], vertices[v2]])
            
            if edge_lines:
                line_collection = Line3DCollection(edge_lines, colors='blue', 
                                                 linewidths=1, alpha=0.6)
                ax.add_collection3d(line_collection)
            
            # Plot vertices
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      c='blue', s=20, alpha=0.8, label='Vertices')
            
            # Plot spine
            spine = step_data['spine_points']
            ax.plot(spine[:, 0], spine[:, 1], spine[:, 2], 
                   'r-', linewidth=2, alpha=0.7, label='Spine')
            
        elif step_data['step'] in [4, 5]:
            # Step 4/5: Faces
            vertices = step_data['vertices']
            faces = step_data['faces']
            
            # Create face collection
            face_vertices = []
            for face in faces:
                if len(face) >= 3 and all(v < len(vertices) for v in face):
                    face_coords = vertices[face]
                    face_vertices.append(face_coords)
            
            if face_vertices:
                poly3d = Poly3DCollection(face_vertices, 
                                        facecolors='lightblue',
                                        alpha=0.7,
                                        edgecolors='darkblue',
                                        linewidths=0.5)
                ax.add_collection3d(poly3d)
            
            # Highlight end cap centers if present
            if step_data.get('end_caps', False):
                spine = step_data['spine_points']
                ax.scatter([spine[0][0], spine[-1][0]], 
                          [spine[0][1], spine[-1][1]], 
                          [spine[0][2], spine[-1][2]], 
                          c='red', s=100, marker='*', 
                          label='End Cap Centers')
        
        # Add description text
        ax.text2D(0.02, 0.98, step_data['description'], 
                 transform=ax.transAxes, fontsize=10, 
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set axis limits
        if 'vertices' in step_data and step_data['vertices'] is not None:
            vertices = step_data['vertices']
        else:
            vertices = step_data['spine_points']
        
        if len(vertices) > 0:
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            padding = 1.0
            ax.set_xlim(min_coords[0] - padding, max_coords[0] + padding)
            ax.set_ylim(min_coords[1] - padding, max_coords[1] + padding)
            ax.set_zlim(min_coords[2] - padding, max_coords[2] + padding)
        
        # Add legend
        ax.legend(fontsize=10, loc='upper right')
        
        # Set equal aspect ratio
        try:
            ax.set_box_aspect([1,1,1])
        except:
            pass
        
        return ax
    
    def create_step_by_step_plots(self, output_dir="mesh_creation_steps", save_plots=True):
        """Create individual plots for each step"""
        if not self.steps_data:
            print("No steps data available. Run generate_mesh_steps() first.")
            return
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        plots_info = []
        
        for step_data in self.steps_data:
            print(f"Creating plot for {step_data['title']}...")
            
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            self.plot_single_step(step_data, ax)
            
            if save_plots:
                filename = os.path.join(output_dir, f"step_{step_data['step']:02d}.png")
                fig.savefig(filename, dpi=150, bbox_inches='tight')
                plots_info.append({'step': step_data['step'], 'filename': filename})
                print(f"  Saved: {filename}")
            
            plt.close(fig)  # Close to save memory
        
        if save_plots:
            print(f"All step plots saved to: {output_dir}")
        
        return plots_info
    
    def create_animated_gif(self, output_filename=None, duration_per_frame=2.0, 
                          transition_frames=10, final_pause_frames=20):
        """Create an animated GIF showing mesh creation process"""
        if not self.steps_data:
            print("No steps data available. Run generate_mesh_steps() first.")
            return None
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"sporozoite_mesh_creation_{timestamp}.gif"
        
        print(f"Creating animated GIF: {output_filename}")
        print(f"  Duration per frame: {duration_per_frame}s")
        print(f"  Transition frames: {transition_frames}")
        print(f"  Final pause: {final_pause_frames} frames")
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate total frames
        frames_per_step = int(duration_per_frame * 10)  # 10 fps
        total_frames = len(self.steps_data) * frames_per_step + final_pause_frames
        
        def animate(frame):
            # Determine which step we're showing
            step_index = min(frame // frames_per_step, len(self.steps_data) - 1)
            
            # Show the appropriate step
            self.plot_single_step(self.steps_data[step_index], ax)
            
            # Add frame counter
            ax.text2D(0.98, 0.02, f"Frame {frame+1}/{total_frames}", 
                     transform=ax.transAxes, fontsize=8, 
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            return ax.collections + ax.lines + [ax.title]
        
        print("Generating animation frames...")
        anim = FuncAnimation(fig, animate, frames=total_frames, 
                           interval=100, blit=False, repeat=True)
        
        # Save as GIF
        print("Saving GIF (this may take a moment)...")
        writer = PillowWriter(fps=10)
        anim.save(output_filename, writer=writer)
        
        plt.close(fig)
        print(f"Animated GIF saved: {output_filename}")
        return output_filename

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

def create_mesh_creation_visualization():
    """Create step-by-step mesh creation visualization and GIF"""
    print("=== Sporozoite Mesh Creation Visualization ===")
    
    # Create the visualizer
    visualizer = MeshCreationVisualizer(figsize=(14, 10))
    
    # Generate the mesh creation steps
    steps_data = visualizer.generate_mesh_steps(sporozoite_id=1, 
                                              position=np.array([0.0, 0.0, 0.0]))
    
    # Create individual step plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"mesh_creation_steps_{timestamp}"
    
    print("\nCreating individual step plots...")
    plots_info = visualizer.create_step_by_step_plots(output_dir=output_dir, 
                                                     save_plots=True)
    
    # Create animated GIF
    print("\nCreating animated GIF...")
    gif_filename = f"sporozoite_mesh_creation_{timestamp}.gif"
    gif_path = visualizer.create_animated_gif(output_filename=gif_filename,
                                            duration_per_frame=2.0,
                                            transition_frames=10,
                                            final_pause_frames=30)
    
    print(f"\n=== Visualization Complete! ===")
    print(f"Individual plots saved to: {output_dir}/")
    print(f"Animated GIF saved as: {gif_filename}")
    print(f"\nStep-by-step process:")
    for i, step in enumerate(steps_data):
        print(f"  {step['title']}: {step['description']}")
    
    return {
        'output_dir': output_dir,
        'gif_filename': gif_filename,
        'plots_info': plots_info,
        'steps_data': steps_data
    }

def demo_single_step(step_number=1):
    """Demo function to show a specific step of mesh creation"""
    print(f"=== Demonstrating Mesh Creation Step {step_number} ===")
    
    visualizer = MeshCreationVisualizer(figsize=(12, 9))
    steps_data = visualizer.generate_mesh_steps()
    
    if step_number < 1 or step_number > len(steps_data):
        print(f"Invalid step number. Available steps: 1-{len(steps_data)}")
        return
    
    # Show the specific step
    step_data = steps_data[step_number - 1]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    visualizer.plot_single_step(step_data, ax)
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed: {step_data['title']}")
    print(f"Description: {step_data['description']}")

if __name__ == "__main__":
    print("Sporozoite 3D Mesh Plotter - Enhanced with Step-by-Step Visualization")
    print("1. Single sporozoite example")
    print("2. Multiple sporozoites example") 
    print("3. Step-by-step mesh creation visualization + GIF")
    print("4. Demo specific mesh creation step")
    
    choice = input("Enter choice (1-4, or press Enter for option 3): ").strip()
    
    if choice == "1":
        plot_single_sporozoite_example()
    elif choice == "2":
        plot_multiple_sporozoites_example()
    elif choice == "4":
        step_num = input("Enter step number (1-5): ").strip()
        try:
            step_num = int(step_num)
            demo_single_step(step_num)
        except ValueError:
            print("Invalid step number, showing step 1")
            demo_single_step(1)
    else:
        # Default: Create the full step-by-step visualization
        create_mesh_creation_visualization()