#!/usr/bin/env python3

import sys
import os
import yaml
import json
import subprocess
from datetime import datetime
import tempfile

import rclpy
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker

import numpy as np
import trimesh
import open3d as o3d

class MeshToStlToGcodeConverter(Node):
    # Scale factor for mesh (10000 = 10000%)
    SCALE_FACTOR = 10000.0
    
    def __init__(self):
        super().__init__('mesh_to_stl_to_gcode_converter_node')
        
        # Create subscribers for mesh
        self.mesh_subscription = self.create_subscription(
            Mesh,
            '/mesh',
            self.mesh_callback,
            10
        )
        
        # Publisher for the modified mesh
        self.modified_mesh_publisher = self.create_publisher(
            Mesh,
            '/modified_mesh',
            10
        )
        
        # Publisher for visualization marker
        self.marker_publisher = self.create_publisher(
            Marker,
            '/mesh_marker',
            10
        )
        
        # Timer for continuous mesh publishing (10Hz)
        self.publish_timer = self.create_timer(0.1, self.publish_meshes)
        
        # Store latest meshes for continuous publishing
        self.latest_mesh = None
        self.latest_modified_mesh = None
        
        # Status publisher
        self.status_publisher = self.create_publisher(
            String,
            '/mesh_processing_status',
            10
        )
        
        # Set up directories for STL and G-code files
        self.stl_directory = '/home/izzypopiolek/ldlidar_ros_ws/src/data_processing/resource/outputs/stl'
        self.gcode_directory = '/home/izzypopiolek/ldlidar_ros_ws/src/data_processing/resource/outputs/gcode'
        
        # Create directories if they don't exist
        os.makedirs(self.stl_directory, exist_ok=True)
        os.makedirs(self.gcode_directory, exist_ok=True)
        
        # Path to CuraEngine executable
        self.cura_executable_path = 'cura-slicer'
        
        self.get_logger().info(f'Using STL directory: {self.stl_directory}')
        self.get_logger().info(f'Using G-code directory: {self.gcode_directory}')
        self.get_logger().info('Mesh to STL to G-code converter node initialized')

    def publish_meshes(self):
        """Continuously publish meshes at 10Hz for RViz visualization"""
        if self.latest_mesh is not None and self.latest_modified_mesh is not None:
            self.modified_mesh_publisher.publish(self.latest_modified_mesh)
            
            # Create and publish marker for visualization
            marker = self.create_mesh_marker(self.latest_modified_mesh)
            self.marker_publisher.publish(marker)

    def create_mesh_marker(self, mesh_msg):
        """Create a visualization marker from a mesh message"""
        marker = Marker()
        marker.header.frame_id = "map"  # Adjust frame as needed
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Add vertices for each triangle
        for triangle in mesh_msg.triangles:
            v1 = mesh_msg.vertices[triangle.vertex_indices[0]]
            v2 = mesh_msg.vertices[triangle.vertex_indices[1]]
            v3 = mesh_msg.vertices[triangle.vertex_indices[2]]
            
            marker.points.extend([v1, v2, v3])
        
        return marker

    def ros_mesh_to_trimesh(self, ros_mesh):
        """Convert ROS mesh message to Trimesh format"""
        self.get_logger().info('Converting mesh to Trimesh format')
        
        try:
            # Extract vertices and faces from ROS mesh
            vertices = [[v.x, v.y, v.z] for v in ros_mesh.vertices]
            faces = [[idx for idx in triangle.vertex_indices] for triangle in ros_mesh.triangles]
            
            # Create Trimesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh
            
        except Exception as e:
            self.get_logger().error(f'Error converting to Trimesh: {str(e)}')
            return None
            
    def trimesh_to_ros(self, mesh):
        """Convert Trimesh to ROS mesh message"""
        try:
            ros_mesh = Mesh()
            
            # Convert vertices
            for vertex in mesh.vertices:
                point = Point()
                point.x = float(vertex[0])
                point.y = float(vertex[1])
                point.z = float(vertex[2])
                ros_mesh.vertices.append(point)
            
            # Convert faces
            for face in mesh.faces:
                triangle = MeshTriangle()
                triangle.vertex_indices = [int(idx) for idx in face]
                ros_mesh.triangles.append(triangle)
                
            return ros_mesh
            
        except Exception as e:
            self.get_logger().error(f'Error converting to ROS mesh: {str(e)}')
            return None

    def modify_mesh(self, mesh_msg):
        """Modify mesh using Trimesh operations"""
        self.get_logger().info('Modifying mesh...')
        
        try:
            # Convert ROS mesh to Trimesh
            mesh = self.ros_mesh_to_trimesh(mesh_msg)
            if mesh is None:
                return None
                
            # Process mesh to merge vertices and clean up
            mesh.process()
            
            # Fill holes
            mesh.fill_holes()
            
            # Fix normals for consistent winding
            mesh.fix_normals()
            
            # Remove duplicate faces and unreferenced vertices
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Ensure mesh is watertight
            if not mesh.is_watertight:
                self.get_logger().warning('Mesh is not watertight after modifications')
            
            self.get_logger().info('Mesh modification completed')
            
            # Convert back to ROS mesh
            return self.trimesh_to_ros(mesh)
            
        except Exception as e:
            self.get_logger().error(f'Error modifying mesh: {str(e)}')
            return None

    def mesh_callback(self, mesh_msg):
        """Handle incoming mesh messages"""
        if self.latest_mesh is not None:
            # We've already processed a mesh, ignore new ones
            return
            
        self.get_logger().info('Received mesh message')
        
        # Store original mesh
        self.latest_mesh = mesh_msg
        
        # First, modify the mesh
        modified_mesh = self.modify_mesh(mesh_msg)
        if modified_mesh is not None:
            # Store modified mesh
            self.latest_modified_mesh = modified_mesh
            
            # Convert original mesh to STL
            original_stl = self.convert_mesh_to_stl(mesh_msg, "original")
            if original_stl:
                # Convert original STL to G-code
                self.convert_stl_to_gcode(original_stl)
            
            # Convert modified mesh to STL
            modified_stl = self.convert_mesh_to_stl(modified_mesh, "modified")
            if modified_stl:
                # Convert modified STL to G-code
                self.convert_stl_to_gcode(modified_stl)
            
            # Destroy the subscription since we don't need more meshes
            self.mesh_subscription.destroy()
            self.get_logger().info('Successfully processed mesh, unsubscribed from topic')
        else:
            self.get_logger().error('Failed to modify mesh')
            status_msg = String()
            status_msg.data = "Failed to modify mesh"
            self.status_publisher.publish(status_msg)

    def convert_mesh_to_stl(self, mesh_msg, prefix=""):
        """Convert ROS mesh message to STL file"""
        if mesh_msg is None or len(mesh_msg.vertices) == 0:
            self.get_logger().error(f'Cannot convert empty {prefix} mesh')
            status_msg = String()
            status_msg.data = f"Cannot convert empty {prefix} mesh"
            self.status_publisher.publish(status_msg)
            return None
        
        # Convert ROS Mesh message to Trimesh
        self.get_logger().info(f'Converting {prefix} mesh to Trimesh format')
        mesh = self.ros_mesh_to_trimesh(mesh_msg)
        
        if mesh is None or len(mesh.vertices) == 0:
            self.get_logger().error(f'Failed to convert {prefix} mesh to Trimesh format')
            status_msg = String()
            status_msg.data = f"Failed to convert {prefix} mesh"
            self.status_publisher.publish(status_msg)
            return None
            
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stl_filename = f'{prefix}_mesh_{timestamp}.stl'
        stl_filepath = os.path.join(self.stl_directory, stl_filename)
        
        try:
            # Scale the mesh using Trimesh's efficient scaling
            scale_factor = self.SCALE_FACTOR / 100.0  # Convert percentage to scale factor
            self.get_logger().info(f'Scaling {prefix} mesh by {self.SCALE_FACTOR}% (factor: {scale_factor:.2f})')
            mesh.vertices *= scale_factor
            
            # Save the scaled mesh to STL
            mesh.export(stl_filepath)
            self.get_logger().info(f'Successfully saved scaled {prefix} mesh to {stl_filepath}')
            
            # Clean up old files (keep last 10)
            self.cleanup_old_files(self.stl_directory, '.stl', keep_last=10)
            
            return stl_filepath
            
        except Exception as e:
            self.get_logger().error(f'Failed to save {prefix} mesh to STL: {str(e)}')
            status_msg = String()
            status_msg.data = f"Failed to save {prefix} mesh"
            self.status_publisher.publish(status_msg)
            return None

    def convert_stl_to_gcode(self, stl_filepath):
        """Convert an STL file to G-code using Cura"""
        if not os.path.exists(stl_filepath):
            self.get_logger().error(f'STL file does not exist: {stl_filepath}')
            status_msg = String()
            status_msg.data = "G-code generation failed: STL file not found"
            self.status_publisher.publish(status_msg)
            return None
            
        # Generate output G-code filename based on the STL filename
        gcode_filename = os.path.splitext(os.path.basename(stl_filepath))[0] + '.gcode'
        gcode_filepath = os.path.join(self.gcode_directory, gcode_filename)
        
        try:
            # Basic Cura command without scaling (since we scaled in Trimesh)
            cmd = [
                self.cura_executable_path,
                stl_filepath,
                '-o', gcode_filepath
            ]
            
            process = subprocess.Popen(cmd, 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True)
                                      
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                self.get_logger().error(f'Cura failed: {stderr}')
                status_msg = String()
                status_msg.data = "G-code generation failed: Cura error"
                self.status_publisher.publish(status_msg)
                return None
            
            self.get_logger().info(f'Successfully created G-code at {gcode_filepath}')
            
            # Clean up old files (keep last 10)
            self.cleanup_old_files(self.gcode_directory, '.gcode', keep_last=10)
            
            return gcode_filepath
            
        except Exception as e:
            self.get_logger().error(f'Error running Cura: {str(e)}')
            status_msg = String()
            status_msg.data = "G-code generation failed: System error"
            self.status_publisher.publish(status_msg)
            return None

    def cleanup_old_files(self, directory, extension, keep_last=10):
        """Clean up old files in the given directory, keeping only the most recent ones"""
        try:
            # Get all files with the given extension
            files = [f for f in os.listdir(directory) if f.endswith(extension)]
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
            
            # Remove old files, keeping only the specified number
            for old_file in files[keep_last:]:
                try:
                    os.remove(os.path.join(directory, old_file))
                    self.get_logger().debug(f'Removed old file: {old_file}')
                except Exception as e:
                    self.get_logger().warning(f'Failed to remove old file {old_file}: {str(e)}')
        except Exception as e:
            self.get_logger().warning(f'Error during file cleanup in {directory}: {str(e)}')
    
def main(args=None):
    rclpy.init(args=args)
    converter = MeshToStlToGcodeConverter()
    rclpy.spin(converter)

if __name__ == '__main__':
    main()