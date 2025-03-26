import sys
import os

import rclpy
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, String

import numpy as np
import trimesh

class MeshModifier(Node):
    def __init__(self):
        super().__init__('mesh_modifier_node')
        
        # Create publishers
        self.modified_mesh_publisher = self.create_publisher(
            Mesh, 
            '/modified_mesh', 
            10)
        self.modified_marker_publisher = self.create_publisher(
            Marker, 
            '/modified_mesh_marker', 
            10)
        self.status_publisher = self.create_publisher(
            String,
            'data_process_status', 
            10)
        self.mesh_ready_publisher = self.create_publisher(
            String,
            'mesh_ready_status', 
            10)
        
        # Create subscribers
        self.mesh_subscription = self.create_subscription(
            Mesh,
            'mesh',
            self.mesh_callback,
            10)
        self.status_subscription = self.create_subscription(
            String,
            'data_process_status',
            self.status_callback,
            10)
        
        # Initialize state
        self.received_mesh = None
        self.received_completion_signal = False
        self.processed_mesh = False
        self.modified_mesh = None
        self.modified_marker = None
        
        self.get_logger().info('Mesh Modifier Node has been initialized')
        
    def status_callback(self, msg):
        if msg.data == "Mesh generation completed, publishing visualization" and not self.processed_mesh:
            self.received_completion_signal = True
            self.process_mesh_if_ready()
            
    def mesh_callback(self, msg):
        if not self.processed_mesh:
            self.received_mesh = msg
            self.process_mesh_if_ready()
            
    def process_mesh_if_ready(self):
        if self.received_completion_signal and self.received_mesh is not None and not self.processed_mesh:
            # Send status update
            status_msg = String()
            status_msg.data = "Starting mesh modification..."
            self.status_publisher.publish(status_msg)
            self.get_logger().info('Starting mesh modification')
            
            # Start timer
            start_time = self.get_clock().now()
            
            # Convert ROS Mesh message to Trimesh
            self.get_logger().info('Converting ROS mesh to Trimesh format...')
            mesh = self.ros_mesh_to_trimesh(self.received_mesh)
            if mesh is None:
                return
                
            self.get_logger().info(f'Conversion complete. Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} triangles')
            
            # Process mesh to merge vertices and clean up
            mesh.process()
            
            # Fill holes
            mesh.fill_holes()
            
            # Fix normals for consistent winding
            mesh.fix_normals()
            
            # Remove duplicate faces and unreferenced vertices
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Create cylindrical hole
            self.get_logger().info('Creating cylindrical hole...')
            
            # Calculate hole parameters based on mesh dimensions
            bounds = mesh.bounds
            mesh_height = bounds[1][2] - bounds[0][2]
            mesh_min_dim = min(bounds[1] - bounds[0])
            
            # Create cylinder for boolean operation
            radius = mesh_min_dim * 0.05  # 5% of smallest dimension
            height = mesh_height * 1.5  # Ensure it goes through the mesh
            
            # Create cylinder at mesh center
            center = mesh.centroid
            cylinder = trimesh.creation.cylinder(radius=radius, height=height)
            
            # Move cylinder to mesh center
            cylinder.apply_translation(center)
            
            # Perform boolean difference
            try:
                self.get_logger().info(f'Performing boolean subtraction with cylinder (r={radius:.3f}, h={height:.3f})')
                modified_mesh = mesh.difference(cylinder)
                
                if modified_mesh is None or len(modified_mesh.vertices) == 0:
                    self.get_logger().error('Boolean operation failed')
                    return
                    
                # Convert back to ROS messages
                self.get_logger().info('Converting modified mesh back to ROS format...')
                self.modified_mesh = self.trimesh_to_ros(modified_mesh)
                self.modified_marker = self.create_mesh_marker(self.modified_mesh, "map")
                self.get_logger().info('ROS message conversion complete')
                
                # Calculate processing time
                end_time = self.get_clock().now()
                duration = (end_time - start_time).nanoseconds / 1e9  # Convert to seconds
                
                # Log timing info
                self.get_logger().info(f'Mesh modification completed in {duration:.2f} seconds')
                
                # Send completion status
                status_msg.data = "Mesh modification completed"
                self.status_publisher.publish(status_msg)
                
                # Set up continuous publishing at 10Hz
                self.processed_mesh = True
                
                # Publish initial messages
                self.get_logger().info('Publishing modified mesh...')
                self.modified_mesh_publisher.publish(self.modified_mesh)
                self.modified_marker_publisher.publish(self.modified_marker)
                
                # Create timer after all processing is done
                timer_period = 1/10.0  # 10Hz
                self.timer = self.create_timer(timer_period, self.timer_callback)
                self.get_logger().info('Continuous publishing timer created')
                
            except Exception as e:
                self.get_logger().error(f'Error during mesh modification: {str(e)}')
                return
            
    def timer_callback(self):
        """Publish mesh and marker at 10Hz for continuous visualization"""
        if self.processed_mesh and self.modified_mesh is not None and self.modified_marker is not None:
            # Update marker timestamp
            self.modified_marker.header.stamp = self.get_clock().now().to_msg()
            
            # Publish both mesh and marker
            self.modified_mesh_publisher.publish(self.modified_mesh)
            self.modified_marker_publisher.publish(self.modified_marker)
            
            # Continuously publish ready status at 10Hz
            ready_msg = String()
            ready_msg.data = "ready"
            self.mesh_ready_publisher.publish(ready_msg)

    def ros_mesh_to_trimesh(self, ros_mesh):
        """Convert ROS mesh message to Trimesh format"""
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
            
    def create_mesh_marker(self, mesh_msg, frame_id):
        """Create a visualization marker from a mesh message"""
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
        
        # Set marker color (blue)
        marker.color = ColorRGBA()
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
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
                color = ColorRGBA()
                color.r = 0.0
                color.g = 0.0
                color.b = 1.0
                color.a = 0.5
                marker.colors.append(color)
                
        return marker

def main(args=None):
    rclpy.init(args=args)
    node = MeshModifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()