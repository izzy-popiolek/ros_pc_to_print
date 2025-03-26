#!/usr/bin/env python3
import rclpy
import open3d as o3d
import numpy as np
import os
import traceback
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from rclpy.node import Node

class PointCloudFileToMeshNode(Node):
    def __init__(self):
        super().__init__('pointcloud_file_to_mesh_converter')
        
        # Declare parameters
        self.declare_parameter('pointcloud_file', '')  # Path to point cloud file
        self.declare_parameter('publish_rate', 1.0)    # How often to publish the mesh (Hz)
        self.declare_parameter('use_simple_mesh', True)  # Use simple mesh generation instead of Poisson
        self.declare_parameter('scale_factor', 1.0)    # Scale factor for visualization
        
        # Publisher for visualization
        self.mesh_pub = self.create_publisher(
            Marker, 
            '/mesh_visualization', 
            10
        )
        
        # Publisher for a simple reference cube
        self.ref_pub = self.create_publisher(
            Marker,
            '/reference_marker',
            10
        )
        
        # Get the file path parameter
        self.pointcloud_file = self.get_parameter('pointcloud_file').get_parameter_value().string_value
        self.use_simple_mesh = self.get_parameter('use_simple_mesh').get_parameter_value().bool_value
        self.scale_factor = self.get_parameter('scale_factor').get_parameter_value().double_value
        
        self.get_logger().info(f'Node initialized with parameters:')
        self.get_logger().info(f'  Point cloud file: {self.pointcloud_file}')
        self.get_logger().info(f'  Use simple mesh: {self.use_simple_mesh}')
        self.get_logger().info(f'  Scale factor: {self.scale_factor}')
        
        # Check if file exists
        if not self.pointcloud_file:
            self.get_logger().error('No point cloud file specified. Use --ros-args -p pointcloud_file:=/path/to/file.pcd')
            return
            
        file_exists = os.path.exists(self.pointcloud_file)
        self.get_logger().info(f'Point cloud file exists: {file_exists}')
        
        if not file_exists:
            self.get_logger().error(f'Point cloud file not found: {self.pointcloud_file}')
            return
            
        # Process the file once at startup
        self.process_pointcloud_file()
        
        # Create a timer to periodically publish the reference cube
        self.ref_timer = self.create_timer(1.0, self.publish_reference_cube)
        self.get_logger().info('Reference cube timer created')
        
        # Create a timer to periodically publish the mesh
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_mesh)
        self.get_logger().info('Mesh publishing timer created')
        
        self.mesh = None  # Will store the generated mesh
        self.counter = 0  # Counter for logging

    def process_pointcloud_file(self):
        try:
            self.get_logger().info(f'Loading point cloud file: {self.pointcloud_file}')
            
            # Load the point cloud file based on extension
            file_extension = os.path.splitext(self.pointcloud_file)[1].lower()
            
            pcd = None
            if file_extension == '.pcd':
                pcd = o3d.io.read_point_cloud(self.pointcloud_file)
            elif file_extension == '.ply':
                pcd = o3d.io.read_point_cloud(self.pointcloud_file)
            elif file_extension == '.xyz' or file_extension == '.txt':
                pcd = o3d.io.read_point_cloud(self.pointcloud_file, format='xyz')
            else:
                self.get_logger().error(f'Unsupported file format: {file_extension}')
                return
            
            # Check if point cloud loaded successfully    
            if pcd is None:
                self.get_logger().error('Failed to load point cloud file')
                return
                
            num_points = len(pcd.points)
            self.get_logger().info(f'Loaded {num_points} points')
            
            if num_points == 0:
                self.get_logger().error('Loaded point cloud is empty')
                return
                
            # Print some basic info about the point cloud
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()
            self.get_logger().info(f'Point cloud bounds:')
            self.get_logger().info(f'  Min: ({min_bound[0]:.2f}, {min_bound[1]:.2f}, {min_bound[2]:.2f})')
            self.get_logger().info(f'  Max: ({max_bound[0]:.2f}, {max_bound[1]:.2f}, {max_bound[2]:.2f})')
            
            # Estimate normals if they don't exist
            if not pcd.has_normals():
                self.get_logger().info('Estimating point cloud normals')
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, 
                        max_nn=30
                    )
                )
                self.get_logger().info('Normals estimated successfully')
            else:
                self.get_logger().info('Point cloud already has normals')
            
            # Choose mesh generation method
            if self.use_simple_mesh:
                # Use Alpha Shapes (simpler than Poisson)
                self.get_logger().info('Using Alpha Shapes for mesh generation')
                alpha = 0.3  # This is a parameter you might need to tune
                self.get_logger().info(f'Alpha parameter: {alpha}')
                
                try:
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                    self.get_logger().info(f'Alpha Shapes generated {len(mesh.triangles)} triangles')
                    
                    # If Alpha Shape failed, try convex hull as a fallback
                    if len(mesh.triangles) == 0:
                        self.get_logger().info('Alpha Shapes produced no triangles, trying convex hull')
                        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(pcd)
                        self.get_logger().info(f'Convex hull generated {len(mesh.triangles)} triangles')
                except Exception as e:
                    self.get_logger().error(f'Error generating Alpha Shapes: {str(e)}')
                    # Fallback to convex hull
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(pcd)
                    self.get_logger().info(f'Fallback convex hull generated {len(mesh.triangles)} triangles')
            else:
                # Use Poisson reconstruction (more complex but better quality)
                self.get_logger().info('Generating mesh using Poisson reconstruction')
                try:
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd,
                        depth=8  # Reduced from 9 for faster processing
                    )
                    self.get_logger().info(f'Poisson reconstruction generated {len(mesh.triangles)} triangles')
                    
                    # Optional: Remove low density vertices
                    vertices_to_remove = densities < np.quantile(densities, 0.1)
                    mesh.remove_vertices_by_mask(vertices_to_remove)
                    self.get_logger().info(f'After density filtering: {len(mesh.triangles)} triangles')
                except Exception as e:
                    self.get_logger().error(f'Error in Poisson reconstruction: {str(e)}')
                    # Fallback to convex hull
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(pcd)
                    self.get_logger().info(f'Fallback convex hull generated {len(mesh.triangles)} triangles')
            
            # Check if mesh generation was successful
            if len(mesh.triangles) == 0:
                self.get_logger().error('Mesh generation produced 0 triangles. Creating a simple cube instead.')
                mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
                mesh.compute_vertex_normals()
                self.get_logger().info(f'Created simple cube with {len(mesh.triangles)} triangles')
            
            # Save the mesh for publishing
            self.mesh = mesh
            
            # Optional: Save the mesh to a file
            output_file = os.path.splitext(self.pointcloud_file)[0] + '_mesh.ply'
            o3d.io.write_triangle_mesh(output_file, mesh)
            self.get_logger().info(f'Mesh saved to: {output_file}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud file: {str(e)}')
            self.get_logger().error(traceback.format_exc())

    def publish_reference_cube(self):
        # Create a reference cube marker at the origin
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.ref_pub.publish(marker)

    def publish_mesh(self):
        self.counter += 1
        
        if self.counter % 10 == 0:  # Log every 10 publishes to avoid spam
            self.get_logger().info(f'Publishing mesh attempt #{self.counter}')
        
        if self.mesh is None:
            self.get_logger().warn('No mesh available to publish')
            return
            
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 1
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.7
        marker.color.b = 0.7
        
        try:
            # Convert mesh triangles to marker points
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            
            if len(triangles) == 0:
                self.get_logger().warn('Mesh has 0 triangles')
                return
                
            if self.counter % 10 == 0:
                self.get_logger().info(f'Processing {len(triangles)} triangles')
            
            for triangle in triangles:
                for vertex_idx in triangle:
                    try:
                        point = Point()
                        point.x = float(vertices[vertex_idx][0]) * self.scale_factor
                        point.y = float(vertices[vertex_idx][1]) * self.scale_factor
                        point.z = float(vertices[vertex_idx][2]) * self.scale_factor
                        marker.points.append(point)
                    except IndexError as e:
                        self.get_logger().error(f'Index error accessing vertex: {str(e)}')
                    except Exception as e:
                        self.get_logger().error(f'Error processing vertex: {str(e)}')
            
            if len(marker.points) == 0:
                self.get_logger().warn('No points added to marker')
                return
                
            if self.counter % 10 == 0:
                self.get_logger().info(f'Publishing marker with {len(marker.points)} points')
            
            self.mesh_pub.publish(marker)
            
            if self.counter == 1:
                self.get_logger().info('First mesh publish complete')
                
        except Exception as e:
            self.get_logger().error(f'Error publishing mesh: {str(e)}')
            self.get_logger().error(traceback.format_exc())

def main():
    print("Starting node...")
    rclpy.init()
    print("ROS initialized")
    
    node = PointCloudFileToMeshNode()
    print("Node created")
    
    try:
        print("Starting to spin")
        rclpy.spin(node)
        print("Spin ended")
    except Exception as e:
        print(f"Exception in spin: {str(e)}")
        traceback.print_exc()
    finally:
        print("Shutting down")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()