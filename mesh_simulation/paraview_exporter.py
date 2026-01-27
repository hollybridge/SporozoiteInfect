"""
ParaView Export for Sporozoite Vertex Data

This module exports sporozoite vertex data in formats compatible with ParaView
for detailed 3D visualization and analysis.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import os
import sys
from typing import List, Optional, Dict, Any
import datetime

# Import sporozoite mesh class
sys.path.append(os.path.dirname(__file__))
from sporozoite_mesh import DeformableSporozoiteMesh
from simulation_config import SimulationConfig

class SporozoiteParaViewExporter:
    """
    Class for exporting sporozoite data to ParaView-compatible formats
    """
    
    def __init__(self, output_dir="paraview_exports"):
        self.output_dir = output_dir
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def export_sporozoite_vertices(self, sporozoite: DeformableSporozoiteMesh, 
                                 filename_prefix: str = None) -> str:
        """
        Export individual sporozoite vertices as VTK point cloud
        
        Args:
            sporozoite: DeformableSporozoiteMesh object
            filename_prefix: Prefix for output filename
        
        Returns:
            Path to exported file
        """
        if filename_prefix is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"sporozoite_{sporozoite.id}_{timestamp}"
        
        filename = os.path.join(self.output_dir, f"{filename_prefix}_vertices.vtp")
        
        # Create VTK points from vertices
        points = vtk.vtkPoints()
        for vertex in sporozoite.vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
        
        # Create polydata for point cloud
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Add vertex data arrays
        self._add_vertex_data_arrays(polydata, sporozoite)
        
        # Write to file
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()
        
        print(f"Exported {len(sporozoite.vertices)} vertices to: {filename}")
        return filename
    
    def export_sporozoite_mesh(self, sporozoite: DeformableSporozoiteMesh,
                              filename_prefix: str = None) -> str:
        """
        Export complete sporozoite mesh (vertices + faces) for ParaView
        
        Args:
            sporozoite: DeformableSporozoiteMesh object
            filename_prefix: Prefix for output filename
        
        Returns:
            Path to exported file
        """
        if filename_prefix is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"sporozoite_{sporozoite.id}_{timestamp}"
        
        filename = os.path.join(self.output_dir, f"{filename_prefix}_mesh.vtp")
        
        # Use the sporozoite's built-in VTK conversion
        polydata = sporozoite.to_vtk_polydata()
        
        # Write to file
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()
        
        print(f"Exported sporozoite mesh to: {filename}")
        return filename
    
    def export_multiple_sporozoites_vertices(self, sporozoites: List[DeformableSporozoiteMesh],
                                           filename_prefix: str = None) -> str:
        """
        Export vertices from multiple sporozoites as a single point cloud
        
        Args:
            sporozoites: List of DeformableSporozoiteMesh objects
            filename_prefix: Prefix for output filename
        
        Returns:
            Path to exported file
        """
        if not sporozoites:
            print("No sporozoites to export")
            return None
        
        if filename_prefix is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"multiple_sporozoites_{len(sporozoites)}_{timestamp}"
        
        filename = os.path.join(self.output_dir, f"{filename_prefix}_all_vertices.vtp")
        
        # Combine all vertices
        all_vertices = []
        all_sporozoite_ids = []
        all_vertex_indices = []
        all_viabilities = []
        all_motilities = []
        all_distances_from_center = []
        
        for sporozoite in sporozoites:
            vertices = sporozoite.vertices
            center = sporozoite.center_position
            
            for i, vertex in enumerate(vertices):
                all_vertices.append(vertex)
                all_sporozoite_ids.append(sporozoite.id)
                all_vertex_indices.append(i)
                all_viabilities.append(sporozoite.viability)
                all_motilities.append(sporozoite.motility)
                
                # Calculate distance from sporozoite center
                distance = np.linalg.norm(vertex - center)
                all_distances_from_center.append(distance)
        
        # Create VTK points
        points = vtk.vtkPoints()
        for vertex in all_vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Add data arrays
        self._add_combined_vertex_arrays(polydata, all_sporozoite_ids, all_vertex_indices,
                                       all_viabilities, all_motilities, all_distances_from_center)
        
        # Write to file
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()
        
        total_vertices = len(all_vertices)
        print(f"Exported {total_vertices} vertices from {len(sporozoites)} sporozoites to: {filename}")
        return filename
    
    def export_multiple_sporozoites_meshes(self, sporozoites: List[DeformableSporozoiteMesh],
                                         filename_prefix: str = None) -> str:
        """
        Export multiple sporozoite meshes as a single multi-block dataset
        
        Args:
            sporozoites: List of DeformableSporozoiteMesh objects
            filename_prefix: Prefix for output filename
        
        Returns:
            Path to exported file
        """
        if not sporozoites:
            print("No sporozoites to export")
            return None
        
        if filename_prefix is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"multiple_sporozoites_{len(sporozoites)}_{timestamp}"
        
        filename = os.path.join(self.output_dir, f"{filename_prefix}_meshes.vtm")
        
        # Create multi-block dataset
        multiblock = vtk.vtkMultiBlockDataSet()
        multiblock.SetNumberOfBlocks(len(sporozoites))
        
        for i, sporozoite in enumerate(sporozoites):
            polydata = sporozoite.to_vtk_polydata()
            multiblock.SetBlock(i, polydata)
            multiblock.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), 
                                        f"Sporozoite_{sporozoite.id}")
        
        # Write to file
        writer = vtk.vtkXMLMultiBlockDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(multiblock)
        writer.Write()
        
        print(f"Exported {len(sporozoites)} sporozoite meshes to: {filename}")
        return filename
    
    def _add_vertex_data_arrays(self, polydata: vtk.vtkPolyData, 
                               sporozoite: DeformableSporozoiteMesh):
        """Add data arrays for single sporozoite vertices"""
        num_vertices = len(sporozoite.vertices)
        
        # Sporozoite ID array
        id_array = vtk.vtkIntArray()
        id_array.SetName("SporozoiteID")
        id_array.SetNumberOfTuples(num_vertices)
        for i in range(num_vertices):
            id_array.SetValue(i, sporozoite.id)
        polydata.GetPointData().AddArray(id_array)
        
        # Vertex index array
        index_array = vtk.vtkIntArray()
        index_array.SetName("VertexIndex")
        index_array.SetNumberOfTuples(num_vertices)
        for i in range(num_vertices):
            index_array.SetValue(i, i)
        polydata.GetPointData().AddArray(index_array)
        
        # Viability array
        viability_array = vtk.vtkFloatArray()
        viability_array.SetName("Viability")
        viability_array.SetNumberOfTuples(num_vertices)
        for i in range(num_vertices):
            viability_array.SetValue(i, sporozoite.viability)
        polydata.GetPointData().SetScalars(viability_array)
        
        # Motility array
        motility_array = vtk.vtkFloatArray()
        motility_array.SetName("Motility")
        motility_array.SetNumberOfTuples(num_vertices)
        for i in range(num_vertices):
            motility_array.SetValue(i, sporozoite.motility)
        polydata.GetPointData().AddArray(motility_array)
        
        # Distance from center array
        distance_array = vtk.vtkFloatArray()
        distance_array.SetName("DistanceFromCenter")
        distance_array.SetNumberOfTuples(num_vertices)
        center = sporozoite.center_position
        for i, vertex in enumerate(sporozoite.vertices):
            distance = np.linalg.norm(vertex - center)
            distance_array.SetValue(i, distance)
        polydata.GetPointData().AddArray(distance_array)
        
        # Velocity magnitude (if available)
        if hasattr(sporozoite, 'velocity') and len(sporozoite.velocity) == num_vertices:
            velocity_array = vtk.vtkFloatArray()
            velocity_array.SetName("VelocityMagnitude")
            velocity_array.SetNumberOfTuples(num_vertices)
            for i, vel in enumerate(sporozoite.velocity):
                vel_mag = np.linalg.norm(vel)
                velocity_array.SetValue(i, vel_mag)
            polydata.GetPointData().AddArray(velocity_array)
    
    def _add_combined_vertex_arrays(self, polydata: vtk.vtkPolyData,
                                   sporozoite_ids: List[int],
                                   vertex_indices: List[int],
                                   viabilities: List[float],
                                   motilities: List[float],
                                   distances: List[float]):
        """Add data arrays for combined sporozoite vertices"""
        num_vertices = len(sporozoite_ids)
        
        # Sporozoite ID array
        id_array = vtk.vtkIntArray()
        id_array.SetName("SporozoiteID")
        id_array.SetNumberOfTuples(num_vertices)
        for i, sid in enumerate(sporozoite_ids):
            id_array.SetValue(i, sid)
        polydata.GetPointData().AddArray(id_array)
        
        # Vertex index array
        index_array = vtk.vtkIntArray()
        index_array.SetName("VertexIndex")
        index_array.SetNumberOfTuples(num_vertices)
        for i, idx in enumerate(vertex_indices):
            index_array.SetValue(i, idx)
        polydata.GetPointData().AddArray(index_array)
        
        # Viability array (use as scalar for coloring)
        viability_array = vtk.vtkFloatArray()
        viability_array.SetName("Viability")
        viability_array.SetNumberOfTuples(num_vertices)
        for i, viab in enumerate(viabilities):
            viability_array.SetValue(i, viab)
        polydata.GetPointData().SetScalars(viability_array)
        
        # Motility array
        motility_array = vtk.vtkFloatArray()
        motility_array.SetName("Motility")
        motility_array.SetNumberOfTuples(num_vertices)
        for i, mot in enumerate(motilities):
            motility_array.SetValue(i, mot)
        polydata.GetPointData().AddArray(motility_array)
        
        # Distance from center array
        distance_array = vtk.vtkFloatArray()
        distance_array.SetName("DistanceFromCenter")
        distance_array.SetNumberOfTuples(num_vertices)
        for i, dist in enumerate(distances):
            distance_array.SetValue(i, dist)
        polydata.GetPointData().AddArray(distance_array)
    
    def create_paraview_instructions(self, exported_files: List[str]) -> str:
        """
        Create a text file with instructions for loading data in ParaView
        
        Args:
            exported_files: List of exported file paths
        
        Returns:
            Path to instructions file
        """
        instructions_file = os.path.join(self.output_dir, "ParaView_Loading_Instructions.txt")
        
        with open(instructions_file, 'w') as f:
            f.write("ParaView Loading Instructions for Sporozoite Data\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Generated on: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            f.write("EXPORTED FILES:\n")
            for i, filepath in enumerate(exported_files, 1):
                f.write(f"{i}. {os.path.basename(filepath)}\n")
            f.write("\n")
            
            f.write("LOADING INSTRUCTIONS:\n")
            f.write("1. Open ParaView\n")
            f.write("2. File -> Open -> Navigate to this directory\n")
            f.write("3. Select the desired .vtp or .vtm file\n")
            f.write("4. Click 'Apply' in the Properties panel\n\n")
            
            f.write("VISUALIZATION TIPS:\n")
            f.write("- For vertex point clouds (.vtp files):\n")
            f.write("  * Change representation to 'Points' or 'Point Gaussian'\n")
            f.write("  * Adjust point size in Display Properties\n")
            f.write("  * Color by 'Viability', 'SporozoiteID', or 'DistanceFromCenter'\n\n")
            
            f.write("- For mesh files:\n")
            f.write("  * Use 'Surface' or 'Surface With Edges' representation\n")
            f.write("  * Color by available data arrays\n")
            f.write("  * Apply transparency for better visualization\n\n")
            
            f.write("- For multi-block datasets (.vtm files):\n")
            f.write("  * Each sporozoite appears as a separate block\n")
            f.write("  * You can show/hide individual sporozoites\n")
            f.write("  * Use the MultiBlock Inspector for block management\n\n")
            
            f.write("DATA ARRAYS AVAILABLE:\n")
            f.write("- SporozoiteID: Unique identifier for each sporozoite\n")
            f.write("- VertexIndex: Index of vertex within sporozoite mesh\n")
            f.write("- Viability: Health/viability of sporozoite (0-1)\n")
            f.write("- Motility: Movement capability (0-1)\n")
            f.write("- DistanceFromCenter: Distance from sporozoite center\n")
            f.write("- VelocityMagnitude: Speed of vertex motion (if available)\n\n")
        
        print(f"Created ParaView instructions: {instructions_file}")
        return instructions_file

def create_sample_sporozoites_for_export(count=3):
    """Create sample sporozoites for testing export functionality"""
    sporozoites = []
    config = SimulationConfig()
    
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([20.0, 5.0, 3.0]),
        np.array([-15.0, -10.0, 2.0])
    ]
    
    for i in range(min(count, len(positions))):
        sporozoite = DeformableSporozoiteMesh(i+1, positions[i], config)
        # Vary properties for testing
        sporozoite.viability = np.random.uniform(0.4, 1.0)
        sporozoite.direction = np.random.uniform(0, 2*np.pi)
        sporozoites.append(sporozoite)
    
    return sporozoites

def export_examples():
    """Example function demonstrating various export options"""
    print("Creating sample sporozoites...")
    sporozoites = create_sample_sporozoites_for_export(3)
    
    print("Setting up exporter...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exporter = SporozoiteParaViewExporter(f"paraview_exports_{timestamp}")
    
    exported_files = []
    
    print("\nExporting individual sporozoite vertices...")
    for sporozoite in sporozoites:
        filename = exporter.export_sporozoite_vertices(sporozoite, f"example_{timestamp}")
        exported_files.append(filename)
    
    print("\nExporting individual sporozoite meshes...")
    for sporozoite in sporozoites:
        filename = exporter.export_sporozoite_mesh(sporozoite, f"example_{timestamp}")
        exported_files.append(filename)
    
    print("\nExporting combined vertices...")
    filename = exporter.export_multiple_sporozoites_vertices(sporozoites, f"example_{timestamp}")
    exported_files.append(filename)
    
    print("\nExporting combined meshes...")
    filename = exporter.export_multiple_sporozoites_meshes(sporozoites, f"example_{timestamp}")
    exported_files.append(filename)
    
    print("\nCreating ParaView instructions...")
    exporter.create_paraview_instructions(exported_files)
    
    print(f"\nAll exports completed! Check directory: {exporter.output_dir}")

if __name__ == "__main__":
    print("Sporozoite ParaView Exporter")
    export_examples()