#!/usr/bin/env python3
import rclpy
import open3d as o3d
import numpy as np
import os
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point
from rclpy.node import Node
import std_msgs.msg

class PointCloudFileToMeshNode(Node):
    def __init__(self):
        super().__init__('pointcloud_file_to_mesh_converter')
        
        # Declare parameters
        self.declare_parameter('pointcloud_file', '')  # Path to point cloud file
        self.declare_parameter('publish_rate', 1.0)    # How often to publish the mesh (Hz)
        
        # Publisher for mesh visualization
        self.mesh_pub = self.create_publisher(
            Marker, 
            '/mesh_visualization', 
            10
        )
        
        # Publisher for original point cloud visualization
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/original_pointcloud',
            10
        )
        
        # Get the file path parameter
        self.pointcloud_file = self.get_parameter('pointcloud_file').get_parameter_value().string_value
        
        if not self.pointcloud_file:
            self.get_logger().error('No point cloud file specified. Use --ros-args -p pointcloud_file:=/path/to/file.pcd')
            return
            
        if not os.path.exists(self.pointcloud_file):
            self.get_logger().error(f'Point cloud file not found: {self.pointcloud_file}')
            return
            
        # Process the file once at startup
        self.process_pointcloud_file()
        
        # Create a timer to periodically publish the mesh and point cloud
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_visualizations)
        
        self.mesh = None  # Will store the generated mesh
        self.pcd = None   # Will store the original point cloud

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
                
            if len(pcd.points) == 0:
                self.get_logger().error('Loaded point cloud is empty')
                return
                
            self.get_logger().info(f'Loaded {len(pcd.points)} points')
            
            # Save the original point cloud
            self.pcd = pcd
            
            # Estimate normals if they don't exist
            if not pcd.has_normals():
                self.get_logger().info('Estimating point cloud normals')
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, 
                        max_nn=30
                    )
                )
            
            # Create mesh using Poisson reconstruction
            self.get_logger().info('Generating mesh using Poisson reconstruction')
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=9
            )
            
            # Optional: Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Save the mesh for publishing
            self.mesh = mesh
            
            # Optional: Save the mesh to a file
            output_file = os.path.splitext(self.pointcloud_file)[0] + '_mesh.ply'
            o3d.io.write_triangle_mesh(output_file, mesh)
            self.get_logger().info(f'Mesh saved to: {output_file}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud file: {str(e)}')

    def publish_visualizations(self):
        # Publish both the mesh and the original point cloud
        self.publish_mesh()
        self.publish_pointcloud()

    def publish_mesh(self):
        if self.mesh is None:
            return
            
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 1.0
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        
        # Convert mesh triangles to marker points
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        
        for triangle in triangles:
            for vertex_idx in triangle:
                point = Point()
                point.x = float(vertices[vertex_idx][0])
                point.y = float(vertices[vertex_idx][1])
                point.z = float(vertices[vertex_idx][2])
                marker.points.append(point)
        
        self.mesh_pub.publish(marker)
        self.get_logger().debug('Published mesh visualization')

    def publish_pointcloud(self):
        if self.pcd is None:
            return
            
        # Convert Open3D point cloud to ROS PointCloud2
        points = np.asarray(self.pcd.points)
        num_points = len(points)
        
        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header.frame_id = "base_link"
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.height = 1
        cloud_msg.width = num_points
        
        # Define point fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Add color fields if available
        if self.pcd.has_colors():
            colors = np.asarray(self.pcd.colors)
            points_with_colors = np.hstack([points, colors * 255]).astype(np.float32)
            fields.extend([
                PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
            ])
            point_data = points_with_colors.tobytes()
            cloud_msg.point_step = 24  # 6 * 4 bytes
        else:
            # If no colors, create a single color for better visualization
            points_with_single_color = np.hstack([
                points, 
                np.ones((num_points, 3), dtype=np.float32) * np.array([0.0, 1.0, 0.0], dtype=np.float32)
            ]).astype(np.float32)
            fields.extend([
                PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
            ])
            point_data = points_with_single_color.tobytes()
            cloud_msg.point_step = 24  # 6 * 4 bytes
        
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.row_step = cloud_msg.point_step * num_points
        cloud_msg.is_dense = True
        cloud_msg.data = point_data
        
        # Publish the point cloud
        self.pointcloud_pub.publish(cloud_msg)
        self.get_logger().debug('Published original point cloud')

def main():
    rclpy.init()
    node = PointCloudFileToMeshNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()