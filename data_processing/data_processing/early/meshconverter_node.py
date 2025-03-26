#! /usr/bin/env python3

import rclpy  # Use rclpy instead of rospy
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class PointCloudToMeshNode:
    def __init__(self):
        rclpy.init()  # Initialize ROS 2
        
        # Create a node
        self.node = rclpy.create_node('pointcloud_to_mesh_converter')
        
        # Subscribe to point cloud topic
        self.point_cloud_sub = self.node.create_subscription(
            PointCloud2,
            '/input_pointcloud', 
            self.pointcloud_callback,
            10  # QoS (queue size)
        )
        
        # Publisher for visualization (optional)
        self.mesh_pub = self.node.create_publisher(
            Marker, 
            '/mesh_visualization', 
            10  # QoS (queue size)
        )

    def pointcloud_callback(self, cloud_msg):
        # Convert ROS PointCloud2 to numpy array
        points_list = []
        for point in pc2.read_points(cloud_msg, skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        
        if not points_list:
            self.node.get_logger().warn("Received empty point cloud")
            return

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points_list))
        
        try:
            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, 
                    max_nn=30
                )
            )
            
            # Create mesh using Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=9
            )
            
            # Optional: Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Publish mesh visualization
            self.publish_mesh_marker(mesh)
            
        except Exception as e:
            self.node.get_logger().error(f"Error in mesh generation: {str(e)}")

    def publish_mesh_marker(self, mesh):
        marker = Marker()
        marker.header.frame_id = "map"  # Adjust frame_id as needed
        marker.header.stamp = self.node.get_clock().now().to_msg()
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
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        for triangle in triangles:
            for vertex_idx in triangle:
                point = Point()
                point.x = vertices[vertex_idx][0]
                point.y = vertices[vertex_idx][1]
                point.z = vertices[vertex_idx][2]
                marker.points.append(point)
        
        self.mesh_pub.publish(marker)

def main():
    node = PointCloudToMeshNode()
    rclpy.spin(node.node)  # ROS 2 spin loop

if __name__ == '__main__':
    main()

