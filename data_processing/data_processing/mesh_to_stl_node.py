#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from shape_msgs.msg import Mesh
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np

class MeshToStlToGcodeConverter(Node):
    def __init__(self):
        super().__init__('mesh_to_stl_to_gcode_converter')
        
        # Create output directories
        self.OUTPUT_STL_DIR = os.path.join(os.path.dirname(__file__), '..', 'resource', 'outputs', 'stl')
        self.OUTPUT_GCODE_DIR = os.path.join(os.path.dirname(__file__), '..', 'resource', 'outputs', 'gcode')
        os.makedirs(self.OUTPUT_STL_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_GCODE_DIR, exist_ok=True)
        
        # Initialize subscriptions
        self.mesh_subscription = self.create_subscription(
            Mesh,
            '/mesh',  # Changed to match memory
            self.mesh_callback,
            10)
        self.modified_mesh_subscription = self.create_subscription(
            Mesh,
            '/modified_mesh',  # Changed to match memory
            self.modified_mesh_callback,
            10)
            
        # Initialize publishers
        self.status_publisher = self.create_publisher(
            String,
            'data_process_status',
            10)
        self.mesh_marker_publisher = self.create_publisher(
            Marker,
            '/mesh_marker',  # Changed to match memory
            10)
        self.modified_mesh_marker_publisher = self.create_publisher(
            Marker,
            '/modified_mesh_marker',  # Changed to match memory
            10)
            
        # Initialize state
        self.SCALE_FACTOR = 1000.0  # Convert meters to millimeters
        self.processed_original = False
        self.processed_modified = False
        self.original_mesh = None
        self.modified_mesh = None
        
        # Create timer for continuous visualization (10Hz)
        self.create_timer(0.1, self.publish_visualization)
        
        self.get_logger().info('Mesh to STL to G-code converter node initialized')
        
    def publish_visualization(self):
        """Publish visualization markers at 10Hz"""
        now = self.get_clock().now()
        
        if self.original_mesh is not None:
            marker = self.create_mesh_marker(self.original_mesh, "map", [0.0, 1.0, 0.0])  # Green for original
            marker.header.stamp = now.to_msg()
            self.mesh_marker_publisher.publish(marker)
            
        if self.modified_mesh is not None:
            marker = self.create_mesh_marker(self.modified_mesh, "map", [0.0, 0.0, 1.0])  # Blue for modified
            marker.header.stamp = now.to_msg()
            self.modified_mesh_marker_publisher.publish(marker)
            
    def create_mesh_marker(self, mesh_msg, frame_id, color):
        """Create visualization marker from mesh"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mesh"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # Set marker color
        marker.color = ColorRGBA()
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.5
        
        # Add vertices for each triangle
        for triangle in mesh_msg.triangles:
            for idx in triangle.vertex_indices:
                vertex = mesh_msg.vertices[idx]
                point = Point()
                point.x = vertex.x
                point.y = vertex.y
                point.z = vertex.z
                marker.points.append(point)
                
                # Add color for each vertex
                vertex_color = ColorRGBA()
                vertex_color.r = color[0]
                vertex_color.g = color[1]
                vertex_color.b = color[2]
                vertex_color.a = 0.5
                marker.colors.append(vertex_color)
                
        return marker
        
    def validate_mesh(self, ros_mesh, prefix):
        """
        Validate mesh before processing.
        Returns (is_valid, message)
        """
        try:
            # Check if mesh has vertices and triangles
            if not ros_mesh.vertices or not ros_mesh.triangles:
                return False, f"{prefix} mesh is empty"
                
            # Check if vertices are all zeros
            all_zeros = True
            for vertex in ros_mesh.vertices:
                if abs(vertex.x) > 1e-6 or abs(vertex.y) > 1e-6 or abs(vertex.z) > 1e-6:
                    all_zeros = False
                    break
            if all_zeros:
                return False, f"{prefix} mesh contains all zero vertices"
                
            # Convert to numpy arrays for validation
            vertices = np.array([[v.x, v.y, v.z] for v in ros_mesh.vertices])
            triangles = np.array([[t.vertex_indices[0], t.vertex_indices[1], t.vertex_indices[2]] 
                                for t in ros_mesh.triangles])
                                
            # Check for invalid vertex indices
            if np.any(triangles >= len(vertices)):
                return False, f"{prefix} mesh contains invalid vertex indices"
                
            # Check for degenerate triangles
            for triangle in triangles:
                v1 = vertices[triangle[0]]
                v2 = vertices[triangle[1]]
                v3 = vertices[triangle[2]]
                
                # Calculate edge vectors
                e1 = v2 - v1
                e2 = v3 - v1
                
                # Calculate triangle area using cross product
                area = np.linalg.norm(np.cross(e1, e2)) / 2
                if area < 1e-10:  # Area threshold for degenerate triangles
                    return False, f"{prefix} mesh contains degenerate triangles"
                    
            # Check if mesh is manifold (each edge should be shared by exactly two triangles)
            edges = {}  # Dictionary to store edge counts
            for triangle in triangles:
                # Add all edges of the triangle
                for i in range(3):
                    edge = tuple(sorted([triangle[i], triangle[(i+1)%3]]))
                    edges[edge] = edges.get(edge, 0) + 1
                    
            # Check edge counts
            non_manifold_edges = [edge for edge, count in edges.items() if count != 2]
            if non_manifold_edges:
                return False, f"{prefix} mesh is not watertight (has {len(non_manifold_edges)} non-manifold edges)"
                
            return True, f"{prefix} mesh is valid"
            
        except Exception as e:
            return False, f"Error validating {prefix} mesh: {str(e)}"
            
    def mesh_callback(self, msg):
        """Handle incoming original mesh messages"""
        if not self.processed_original:
            # Validate mesh first
            is_valid, message = self.validate_mesh(msg, "Original")
            self.get_logger().info(message)
            
            if is_valid:
                self.get_logger().info('Converting original mesh to STL...')
                self.convert_mesh_to_stl(msg, 'original')
                self.processed_original = True
                self.original_mesh = msg  # Store for visualization
                self.get_logger().info('Original mesh processing complete, node running and doing nothing')
            else:
                status_msg = String()
                status_msg.data = f"Failed to process original mesh: {message}"
                self.status_publisher.publish(status_msg)
            
    def modified_mesh_callback(self, msg):
        """Handle incoming modified mesh messages"""
        if not self.processed_modified:
            # Validate mesh first
            is_valid, message = self.validate_mesh(msg, "Modified")
            self.get_logger().info(message)
            
            if is_valid:
                self.get_logger().info('Converting modified mesh to STL...')
                self.convert_mesh_to_stl(msg, 'modified')
                self.processed_modified = True
                self.modified_mesh = msg  # Store for visualization
                self.get_logger().info('Modified mesh processing complete, node running and doing nothing')
            else:
                status_msg = String()
                status_msg.data = f"Failed to process modified mesh: {message}"
                self.status_publisher.publish(status_msg)
            
    def convert_mesh_to_stl(self, ros_mesh, prefix):
        """Convert ROS mesh to STL and generate G-code"""
        try:
            # Convert ROS mesh vertices to numpy array and scale
            vertices = []
            self.get_logger().info(f'Scaling {prefix} mesh vertices by {self.SCALE_FACTOR}x')
            for vertex in ros_mesh.vertices:
                vertices.append([
                    vertex.x * self.SCALE_FACTOR,
                    vertex.y * self.SCALE_FACTOR,
                    vertex.z * self.SCALE_FACTOR
                ])
            vertices = np.array(vertices)
            
            # Convert triangle indices to numpy array
            triangles = []
            for triangle in ros_mesh.triangles:
                triangles.append([
                    triangle.vertex_indices[0],
                    triangle.vertex_indices[1],
                    triangle.vertex_indices[2]
                ])
            triangles = np.array(triangles)
            
            # Save to STL file
            stl_filename = os.path.join(self.OUTPUT_STL_DIR, f'{prefix}_mesh.stl')
            self.save_to_stl(vertices, triangles, stl_filename)
            self.get_logger().info(f'Saved {prefix} mesh to {stl_filename}')
            
            # Generate G-code
            gcode_filename = os.path.join(self.OUTPUT_GCODE_DIR, f'{prefix}_mesh.gcode')
            self.generate_gcode(stl_filename, gcode_filename)
            self.get_logger().info(f'Generated G-code: {gcode_filename}')
            
            # Publish status
            status_msg = String()
            status_msg.data = f"Completed processing {prefix} mesh"
            self.status_publisher.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing {prefix} mesh: {str(e)}')
            
    def save_to_stl(self, vertices, triangles, filename):
        """Save mesh data to binary STL file"""
        # Binary STL format
        header = np.zeros(80, dtype='uint8')
        num_triangles = len(triangles)
        
        with open(filename, 'wb') as f:
            f.write(header)
            f.write(np.array(num_triangles, dtype=np.uint32))
            
            # For each triangle
            for i in range(num_triangles):
                triangle = triangles[i]
                v1 = vertices[triangle[0]]
                v2 = vertices[triangle[1]]
                v3 = vertices[triangle[2]]
                
                # Calculate normal
                normal = np.cross(v2 - v1, v3 - v1)
                if np.any(normal):  # Check if normal is non-zero
                    normal = normal / np.linalg.norm(normal)
                
                # Write normal and vertices (12 x 32-bit floats)
                data = np.zeros(12, dtype=np.float32)
                data[0:3] = normal
                data[3:6] = v1
                data[6:9] = v2
                data[9:12] = v3
                data.tofile(f)
                
                # Write attribute byte count (2 bytes)
                f.write(np.zeros(2, dtype=np.uint16))
                
    def generate_gcode(self, stl_file, output_file):
        """Generate G-code from STL file using slic3r"""
        try:
            # Basic slic3r command with common 3D printing parameters
            cmd = [
                'slic3r',
                '--nozzle-diameter', '0.4',
                '--filament-diameter', '1.75',
                '--layer-height', '0.2',
                '--fill-density', '20%',
                '--temperature', '200',
                '--bed-temperature', '60',
                '--output', output_file,
                stl_file
            ]
            
            # Execute slic3r
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Slic3r error: {result.stderr}")
                
        except Exception as e:
            self.get_logger().error(f'Error generating G-code: {str(e)}')
            raise

def main(args=None):
    rclpy.init(args=args)
    node = MeshToStlToGcodeConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()