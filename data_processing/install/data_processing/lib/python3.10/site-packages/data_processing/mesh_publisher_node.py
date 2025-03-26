import sys
import os

import rclpy 
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from std_msgs.msg import ColorRGBA, String
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from scipy.spatial import Delaunay
from sensor_msgs_py import point_cloud2

import numpy as np
import open3d as o3d

class MESHPublisher(Node):

    def __init__(self):
        super().__init__('mesh_publisher_node')

        # Create publishers for both raw mesh data and visualization
        self.mesh_publisher = self.create_publisher(Mesh, 'mesh', 10)
        self.marker_publisher = self.create_publisher(Marker, 'mesh_marker', 10)
        self.status_publisher = self.create_publisher(String, 'data_process_status', 10)

        # Create subscriber to continuous point cloud
        self.point_cloud_sub = self.create_subscription(
            sensor_msgs.PointCloud2,
            'continuous_pcd',
            self.point_cloud_callback,
            10
        )
        
        # Create timer for continuous publishing at 10Hz
        self.create_timer(0.1, self.timer_callback)
        
        # Initialize state
        self.points = None
        self.mesh = None
        self.marker = None
        self.processed_mesh = False
        self.received_first_cloud = False
        
        self.get_logger().info('Mesh Publisher Node has been initialized')
            
    def point_cloud_callback(self, cloud_msg):
        if not self.processed_mesh:
            # Process point cloud only once
            if not self.received_first_cloud:
                status_msg = String()
                status_msg.data = "Received point cloud data"
                self.status_publisher.publish(status_msg)
                self.get_logger().info('Received point cloud data')
                self.received_first_cloud = True

                # Convert PointCloud2 to numpy array
                points_list = []
                for point in point_cloud2.read_points(cloud_msg, skip_nans=True):
                    points_list.append([point[0], point[1], point[2]])
                self.points = np.array(points_list)
                self.get_logger().info(f'Converted point cloud, shape: {self.points.shape}')
                
                # Process mesh once
                self.process_mesh()
                
            else:
                # Continue publishing both mesh data and visualization marker at 10Hz
                self.mesh_publisher.publish(self.mesh)
                self.marker_publisher.publish(self.marker)
        else:
            # Continue publishing both mesh data and visualization marker at 10Hz
            self.mesh_publisher.publish(self.mesh)
            self.marker_publisher.publish(self.marker)
            
    def timer_callback(self):
        """Publish mesh and marker at regular intervals (10Hz)"""
        if self.processed_mesh and self.mesh is not None and self.marker is not None:
            # Publish mesh for processing
            self.mesh_publisher.publish(self.mesh)
            # Publish marker for visualization
            self.marker_publisher.publish(self.marker)
            
    def process_mesh(self):
        if not self.processed_mesh and self.points is not None:
            # Send status update
            status_msg = String()
            status_msg.data = "Starting mesh generation..."
            self.status_publisher.publish(status_msg)
            self.get_logger().info('Starting mesh generation')
            
            # Start timer
            start_time = self.get_clock().now()
            
            # Process the mesh
            self.mesh = point_cloud_to_mesh(self.points, method='poisson')
            self.marker = create_mesh_marker(self, self.mesh, "map")
            
            # Calculate processing time
            end_time = self.get_clock().now()
            duration = (end_time - start_time).nanoseconds / 1e9  # Convert to seconds
            
            # Log timing info
            self.get_logger().info(f'Mesh generation completed in {duration:.2f} seconds')
            
            # Send completion status - keep exact phrase for downstream nodes
            status_msg.data = "Mesh generation completed, publishing visualization"
            self.status_publisher.publish(status_msg)
            
            # Send timing info as separate message
            timing_msg = String()
            timing_msg.data = f"Mesh generation took {duration:.2f} seconds"
            self.status_publisher.publish(timing_msg)
            
            self.processed_mesh = True
            
def point_cloud_to_mesh(points, method = 'poisson', clean_mesh=True):
    """ Creates a high-quality mesh message from a point cloud using Open3D
    
    Args:
        points: Nx3 array of xyz positions
        method: Reconstruction method ('ball_pivoting', 'poisson', 'alpha_shape', or 'delaunay')
        clean_mesh: Whether to clean the mesh after reconstruction (remove duplicates, etc.)
    
    Returns:
        shape_msgs/Mesh message
    """
    import numpy as np
    import open3d as o3d
    from shape_msgs.msg import Mesh, MeshTriangle
    from geometry_msgs.msg import Point
    from scipy.spatial import Delaunay
    
    # Input validation
    if points.shape[0] < 4:
        print("Warning: Too few points for proper meshing")
        return create_empty_mesh_with_vertices(points)
    
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Compute normals (required for some reconstruction methods)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # Create mesh using the requested method
    o3d_mesh = None
    if method == 'ball_pivoting':
        # Ball pivoting is good for dense, uniform point clouds
        # Try different radii to capture different levels of detail
        radii = [0.02, 0.04, 0.08, 0.16]  # Adjust based on point cloud scale
        o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
    
    elif method == 'poisson':
        # Poisson is good for closed surfaces with good normal estimates
        o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=12, width=0, scale=1.1, linear_fit=False)
        
        # Crop the mesh to focus on high-confidence regions
        # This removes parts of the mesh that are likely artifacts
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox = bbox.scale(1.2, bbox.get_center())  # Expand bbox slightly
        o3d_mesh = o3d_mesh.crop(bbox)
    
    elif method == 'alpha_shape':
        # Alpha shape is good for non-uniform point clouds
        # We'll compute an appropriate alpha value based on point density
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        alpha = avg_dist * 2.0  # A good heuristic starting point
        o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha)
    
    elif method == 'delaunay':
        # Fallback to simple Delaunay triangulation
        # Find best projection plane using PCA
        # This works better than just using XY coordinates
        _, _, vh = np.linalg.svd(points - np.mean(points, axis=0))
        # Use the first two principal components as the projection plane
        proj_matrix = vh[0:2].T
        points_2d = np.dot(points - np.mean(points, axis=0), proj_matrix)
        
        # Create Delaunay triangulation
        tri = Delaunay(points_2d)
        
        # Create Open3D mesh from Delaunay triangles
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(points)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
    
    else:
        print(f"Unknown method '{method}', falling back to Delaunay triangulation")
        # Recursive call with known method
        return point_cloud_to_mesh(points, method='delaunay')
    
    # If mesh creation failed or produced an empty mesh, try another method
    if o3d_mesh is None or len(o3d_mesh.triangles) == 0:
        print(f"Mesh creation with {method} failed, trying Delaunay triangulation")
        return point_cloud_to_mesh(points, method='delaunay')
    
    # Clean up the mesh if requested
    if clean_mesh:
        # Remove duplicated vertices
        o3d_mesh.remove_duplicated_vertices()
        # Remove duplicated triangles
        o3d_mesh.remove_duplicated_triangles()
        # Remove degenerate triangles
        o3d_mesh.remove_degenerate_triangles()
        # Remove unreferenced vertices
        o3d_mesh.remove_unreferenced_vertices()
        
        # Optional: Filter out triangles with long edges
        if len(o3d_mesh.triangles) > 0:
            # Convert to numpy for faster processing
            vertices = np.asarray(o3d_mesh.vertices)
            triangles = np.asarray(o3d_mesh.triangles)
            
            # Calculate edge lengths for each triangle
            valid_triangles = []
            
            # Calculate average distance between points to set threshold
            avg_dist = np.mean(pcd.compute_nearest_neighbor_distance()) * 10.0
            
            for tri in triangles:
                v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                edge1 = np.linalg.norm(v0 - v1)
                edge2 = np.linalg.norm(v1 - v2)
                edge3 = np.linalg.norm(v2 - v0)
                
                # Keep triangle if all edges are below threshold
                if edge1 < avg_dist and edge2 < avg_dist and edge3 < avg_dist:
                    valid_triangles.append(tri)
            
            # Update mesh with filtered triangles
            if len(valid_triangles) > 0:
                o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(valid_triangles))
            else:
                print("Warning: All triangles filtered out, keeping original mesh")
    
    # Convert to ROS message
    mesh = Mesh()
    
    # Add vertices
    mesh.vertices = []
    for vertex in o3d_mesh.vertices:
        mesh_point = Point()
        mesh_point.x = float(vertex[0])
        mesh_point.y = float(vertex[1])
        mesh_point.z = float(vertex[2])
        mesh.vertices.append(mesh_point)
    
    # Add triangles
    mesh.triangles = []
    for triangle in o3d_mesh.triangles:
        mesh_triangle = MeshTriangle()
        mesh_triangle.vertex_indices = [int(triangle[0]), int(triangle[1]), int(triangle[2])]
        mesh.triangles.append(mesh_triangle)
    
    print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    return mesh

def create_empty_mesh_with_vertices(points):
    """Creates a mesh with just vertices (no triangles) when meshing fails"""
    from shape_msgs.msg import Mesh
    from geometry_msgs.msg import Point
    
    mesh = Mesh()
    mesh.vertices = []
    for point in points:
        mesh_point = Point()
        mesh_point.x = float(point[0])
        mesh_point.y = float(point[1])
        mesh_point.z = float(point[2])
        mesh.vertices.append(mesh_point)
    mesh.triangles = []
    
    return mesh

def create_mesh_marker(self, mesh, frame_id):
    """
    Converts a shape_msgs/Mesh to a visualization_msgs/Marker for RViz display
    
    Args:
        mesh: shape_msgs/Mesh message
        frame_id: The coordinate frame for the marker
        
    Returns:
        visualization_msgs/Marker message ready to be published
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    
    # Set color (adjust as needed)
    marker.color.r = 0.5
    marker.color.g = 0.5
    marker.color.b = 1.0
    marker.color.a = 1.0
    
    # Convert mesh triangles to marker points
    for triangle in mesh.triangles:
        idx1, idx2, idx3 = triangle.vertex_indices
        marker.points.append(mesh.vertices[idx1])
        marker.points.append(mesh.vertices[idx2])
        marker.points.append(mesh.vertices[idx3])
        
        # Add colors for each vertex if desired
        marker.colors.append(ColorRGBA(r=0.5, g=0.5, b=1.0, a=1.0))
        marker.colors.append(ColorRGBA(r=0.5, g=0.5, b=1.0, a=1.0))
        marker.colors.append(ColorRGBA(r=0.5, g=0.5, b=1.0, a=1.0))
    
    return marker



def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    mesh_publisher = MESHPublisher()
    rclpy.spin(mesh_publisher)
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mesh_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
