import os
import sys
import yaml
import json
import subprocess
from datetime import datetime
import tempfile

import rclpy
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle
from std_srvs.srv import Trigger
from std_msgs.msg import String, Bool

import numpy as np
import open3d as o3d

class MeshToStlToGcodeConverter(Node):
    def __init__(self):
        super().__init__('mesh_to_stl_to_gcode_converter_node')
        
        # Declare parameters
        self.declare_parameter('output_directory', '/tmp/ros_print_files')
        self.declare_parameter('stl_subdirectory', 'stl')
        self.declare_parameter('gcode_subdirectory', 'gcode')
        self.declare_parameter('cura_config_path', '')
        self.declare_parameter('cura_executable_path', 'cura-slicer')
        self.declare_parameter('filename_prefix', 'ros_mesh')
        self.declare_parameter('subscribe_topic', 'modified_mesh')  # Can be 'mesh' or 'modified_mesh'
        
        # Get parameters
        self.output_directory = self.get_parameter('output_directory').value
        self.stl_subdirectory = self.get_parameter('stl_subdirectory').value
        self.gcode_subdirectory = self.get_parameter('gcode_subdirectory').value
        self.cura_config_path = self.get_parameter('cura_config_path').value
        self.cura_executable_path = self.get_parameter('cura_executable_path').value
        self.filename_prefix = self.get_parameter('filename_prefix').value
        self.subscribe_topic = self.get_parameter('subscribe_topic').value
        
        # Create output directories if they don't exist
        self.stl_directory = os.path.join(self.output_directory, self.stl_subdirectory)
        self.gcode_directory = os.path.join(self.output_directory, self.gcode_subdirectory)
        
        for directory in [self.output_directory, self.stl_directory, self.gcode_directory]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.get_logger().info(f'Created directory: {directory}')
        
        # Load Cura config if exists
        self.cura_config = {}
        if self.cura_config_path and os.path.exists(self.cura_config_path):
            try:
                with open(self.cura_config_path, 'r') as f:
                    self.cura_config = yaml.safe_load(f)
                self.get_logger().info(f'Loaded Cura config from {self.cura_config_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load Cura config: {str(e)}')
        
        # Flag to track if we have already processed a mesh
        self.processed_first_mesh = False
        
        # Create a subscription to the mesh topic
        self.mesh_subscription = self.create_subscription(
            Mesh,
            self.subscribe_topic,
            self.mesh_callback,
            10)
        
        # Service to trigger the full conversion pipeline for the last received mesh
        self.convert_service = self.create_service(
            Trigger, 
            'convert_mesh_to_gcode', 
            self.convert_service_callback)
        
        # Service to only convert mesh to STL
        self.mesh_to_stl_service = self.create_service(
            Trigger, 
            'convert_mesh_to_stl_only', 
            self.mesh_to_stl_service_callback)
        
        # Service to slice an existing STL file
        self.stl_to_gcode_service = self.create_service(
            Trigger, 
            'convert_stl_to_gcode', 
            self.stl_to_gcode_service_callback)
        
        # Publishers for conversion status
        self.stl_status_publisher = self.create_publisher(
            String, 
            'stl_conversion_status', 
            10)
        
        self.gcode_status_publisher = self.create_publisher(
            String, 
            'gcode_conversion_status', 
            10)
        
        # Store the last received mesh and most recent STL path
        self.last_mesh = None
        self.last_stl_path = None
        
        self.get_logger().info('Mesh to STL to G-code Converter Node has been initialized')
        self.get_logger().info(f'Using Cura at: {self.cura_executable_path}')
        
    # def mesh_callback(self, msg):
    #     """Handle received mesh messages"""
    #     self.get_logger().info('Received mesh, ready for processing')
    #     self.last_mesh = msg
        
    #     # Auto-process the complete pipeline (mesh -> STL -> G-code)
    #     stl_path = self.convert_mesh_to_stl(msg)
    #     if stl_path:
    #         self.convert_stl_to_gcode(stl_path)
    
    def mesh_callback(self, msg):
        """Handle received mesh messages - process only the first one automatically"""
        self.get_logger().info('Received mesh message')
        self.last_mesh = msg
        
        # Auto-process only if this is the first mesh we've received
        if not self.processed_first_mesh:
            self.get_logger().info('Processing first mesh automatically')
            stl_path = self.convert_mesh_to_stl(msg)
            if stl_path:
                self.convert_stl_to_gcode(stl_path)
                self.processed_first_mesh = True
                self.get_logger().info('First mesh has been processed. Further meshes will be stored but not automatically processed.')
        else:
            self.get_logger().info('Mesh stored. First mesh already processed. Use service calls for manual processing.')
    
    def convert_service_callback(self, request, response):
        """Service callback to run the complete pipeline"""
        if self.last_mesh is None:
            response.success = False
            response.message = "No mesh has been received yet"
            return response
        
        stl_path = self.convert_mesh_to_stl(self.last_mesh)
        if not stl_path:
            response.success = False
            response.message = "Failed to convert mesh to STL"
            return response
            
        gcode_path = self.convert_stl_to_gcode(stl_path)
        if gcode_path:
            response.success = True
            response.message = f"Complete pipeline succeeded. STL: {stl_path}, G-code: {gcode_path}"
        else:
            response.success = False
            response.message = f"STL created at {stl_path}, but G-code conversion failed"
        
        return response
    
    def mesh_to_stl_service_callback(self, request, response):
        """Service callback to convert mesh to STL only"""
        if self.last_mesh is None:
            response.success = False
            response.message = "No mesh has been received yet"
            return response
        
        stl_path = self.convert_mesh_to_stl(self.last_mesh)
        if stl_path:
            response.success = True
            response.message = f"Mesh converted to STL: {stl_path}"
        else:
            response.success = False
            response.message = "Failed to convert mesh to STL"
        
        return response
    
    def stl_to_gcode_service_callback(self, request, response):
        """Service callback to convert the last created STL to G-code"""
        if self.last_stl_path is None or not os.path.exists(self.last_stl_path):
            response.success = False
            response.message = "No valid STL file available for conversion"
            return response
        
        gcode_path = self.convert_stl_to_gcode(self.last_stl_path)
        if gcode_path:
            response.success = True
            response.message = f"STL converted to G-code: {gcode_path}"
        else:
            response.success = False
            response.message = "Failed to convert STL to G-code"
        
        return response
    
    def convert_mesh_to_stl(self, mesh_msg):
        """Convert a ROS mesh to STL file and return the filepath"""
        if mesh_msg is None or len(mesh_msg.vertices) == 0:
            self.get_logger().error('Cannot convert empty mesh')
            status_msg = String()
            status_msg.data = "ERROR: Empty mesh, conversion failed"
            self.stl_status_publisher.publish(status_msg)
            return None
        
        # Convert ROS Mesh message to Open3D TriangleMesh
        o3d_mesh = self.ros_mesh_to_o3d(mesh_msg)
        
        # Generate a filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stl_filename = f"{self.filename_prefix}_{timestamp}.stl"
        stl_filepath = os.path.join(self.stl_directory, stl_filename)
        
        try:
            # Ensure the mesh has normals
            if len(o3d_mesh.triangles) > 0:
                o3d_mesh.compute_vertex_normals()
                o3d_mesh.compute_triangle_normals()
            
            # Save the mesh as STL
            o3d.io.write_triangle_mesh(stl_filepath, o3d_mesh, write_ascii=False)
            
            self.get_logger().info(f'Mesh saved as STL: {stl_filepath}')
            
            # Publish status
            status_msg = String()
            status_msg.data = f"SUCCESS: Mesh converted to STL: {stl_filepath}"
            self.stl_status_publisher.publish(status_msg)
            
            # Store this as the last STL path
            self.last_stl_path = stl_filepath
            
            return stl_filepath
            
        except Exception as e:
            self.get_logger().error(f'Failed to save STL: {str(e)}')
            status_msg = String()
            status_msg.data = f"ERROR: Failed to save STL: {str(e)}"
            self.stl_status_publisher.publish(status_msg)
            return None
    
    def convert_stl_to_gcode(self, stl_filepath):
        """Convert an STL file to G-code using Cura"""
        if not stl_filepath or not os.path.exists(stl_filepath):
            self.get_logger().error(f'STL file does not exist: {stl_filepath}')
            status_msg = String()
            status_msg.data = f"ERROR: STL file does not exist: {stl_filepath}"
            self.gcode_status_publisher.publish(status_msg)
            return None
            
        # Generate output G-code filename based on the STL filename
        stl_basename = os.path.basename(stl_filepath)
        gcode_filename = os.path.splitext(stl_basename)[0] + '.gcode'
        gcode_filepath = os.path.join(self.gcode_directory, gcode_filename)
        
        # Build the new Cura command
        cmd = [self.cura_executable_path, stl_filepath, f"--output={gcode_filepath}"]
        
        # Add config file if available - Note: You might need to adjust this part
        # depending on how your cura-slicer supports configuration
        if self.cura_config and os.path.exists(self.cura_config_path):
            cmd.extend(["--config", self.cura_config_path])
        
        # Run Cura
        try:
            self.get_logger().info(f'Running Cura command: {" ".join(cmd)}')
            
            status_msg = String()
            status_msg.data = f"STARTED: Converting {stl_filepath} to G-code"
            self.gcode_status_publisher.publish(status_msg)
            
            process = subprocess.Popen(cmd, 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True)
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                self.get_logger().error(f'Cura failed with code {process.returncode}: {stderr}')
                self.get_logger().info(f'Output: {stdout}')
                status_msg = String()
                status_msg.data = f"ERROR: Slicing failed: {stderr}"
                self.gcode_status_publisher.publish(status_msg)
                return None
            
            self.get_logger().info(f'Successfully created G-code at {gcode_filepath}')
            self.get_logger().info(f'Cura output: {stdout}')
            
            # Publish success status
            status_msg = String()
            status_msg.data = f"SUCCESS: STL converted to G-code: {gcode_filepath}"
            self.gcode_status_publisher.publish(status_msg)
            
            return gcode_filepath
            
        except Exception as e:
            self.get_logger().error(f'Error running Cura: {str(e)}')
            status_msg = String()
            status_msg.data = f"ERROR: Error running Cura: {str(e)}"
            self.gcode_status_publisher.publish(status_msg)
            return None
    

    
    def ros_mesh_to_o3d(self, ros_mesh):
        """Convert from ROS shape_msgs/Mesh to Open3D TriangleMesh"""
        vertices = []
        triangles = []
        
        # Extract vertices
        for vertex in ros_mesh.vertices:
            vertices.append([vertex.x, vertex.y, vertex.z])
        
        # Extract triangles
        for triangle in ros_mesh.triangles:
            triangles.append([
                triangle.vertex_indices[0],
                triangle.vertex_indices[1],
                triangle.vertex_indices[2]
            ])
        
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        
        # Compute normals for better visualization
        if len(triangles) > 0:
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
        return mesh
    
    def create_cura_config_file(self):
        """Create a temporary configuration file for Cura"""
        fd, path = tempfile.mkstemp(suffix='.json')
        
        try:
            # Cura uses JSON format for config
            with os.fdopen(fd, 'w') as tmp:
                # Extract just the Cura section from our config
                cura_config = self.cura_config.get('cura', {})
                json.dump(cura_config, tmp)
                
            return path
        except Exception as e:
            os.close(fd)
            os.remove(path)
            raise e

def main(args=None):
    rclpy.init(args=args)
    converter = MeshToStlToGcodeConverter()
    rclpy.spin(converter)
    
    # Destroy the node explicitly
    converter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
# def convert_stl_to_gcode(self, stl_filepath):
    #     """Convert an STL file to G-code using Cura"""
    #     if not stl_filepath or not os.path.exists(stl_filepath):
    #         self.get_logger().error(f'STL file does not exist: {stl_filepath}')
    #         status_msg = String()
    #         status_msg.data = f"ERROR: STL file does not exist: {stl_filepath}"
    #         self.gcode_status_publisher.publish(status_msg)
    #         return None
            
    #     # Generate output G-code filename based on the STL filename
    #     stl_basename = os.path.basename(stl_filepath)
    #     gcode_filename = os.path.splitext(stl_basename)[0] + '.gcode'
    #     gcode_filepath = os.path.join(self.gcode_directory, gcode_filename)
        
    #     # Create a config file for Cura if we have configurations
    #     config_file = None
    #     if self.cura_config:
    #         try:
    #             config_file = self.create_cura_config_file()
    #         except Exception as e:
    #             self.get_logger().error(f'Failed to create Cura config file: {str(e)}')
        
    #     # Build the Cura command
    #     cmd = [self.cura_executable_path]
        
    #     # Check if the executable is a flatpak
    #     if 'flatpak' in self.cura_executable_path:
    #         cmd = [self.cura_executable_path, 'run', 'com.ultimaker.cura']
            
    #     # Add slicing parameters
    #     cmd.extend(['-l', stl_filepath, '-o', gcode_filepath, '--slice', '--force'])
        
    #     # Add config file if available
    #     if config_file:
    #         cmd.extend(['-j', config_file])
        
    #     # Run Cura
    #     try:
    #         self.get_logger().info(f'Running Cura command: {" ".join(cmd)}')
            
    #         status_msg = String()
    #         status_msg.data = f"STARTED: Converting {stl_filepath} to G-code"
    #         self.gcode_status_publisher.publish(status_msg)
            
    #         process = subprocess.Popen(cmd, 
    #                                   stdout=subprocess.PIPE, 
    #                                   stderr=subprocess.PIPE,
    #                                   universal_newlines=True)
            
    #         stdout, stderr = process.communicate()
            
    #         if process.returncode != 0:
    #             self.get_logger().error(f'Cura failed with code {process.returncode}: {stderr}')
    #             self.get_logger().info(f'Output: {stdout}')
    #             status_msg = String()
    #             status_msg.data = f"ERROR: Slicing failed: {stderr}"
    #             self.gcode_status_publisher.publish(status_msg)
                
    #             # Cleanup temp files
    #             if config_file and os.path.exists(config_file):
    #                 os.remove(config_file)
                    
    #             return None
            
    #         self.get_logger().info(f'Successfully created G-code at {gcode_filepath}')
            
    #         # Cleanup temp files
    #         if config_file and os.path.exists(config_file):
    #             os.remove(config_file)
            
    #         # Publish success status
    #         status_msg = String()
    #         status_msg.data = f"SUCCESS: STL converted to G-code: {gcode_filepath}"
    #         self.gcode_status_publisher.publish(status_msg)
            
    #         return gcode_filepath
            
    #     except Exception as e:
    #         self.get_logger().error(f'Error running Cura: {str(e)}')
    #         status_msg = String()
    #         status_msg.data = f"ERROR: Error running Cura: {str(e)}"
    #         self.gcode_status_publisher.publish(status_msg)
            
    #         # Cleanup temp files
    #         if config_file and os.path.exists(config_file):
    #             os.remove(config_file)
                
    #         return None