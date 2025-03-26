# TODO - ideally this is just subscribed to the same topic that is publishing the point cloud message

import sys
import os

import rclpy 
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from std_msgs.msg import ColorRGBA
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from scipy.spatial import Delaunay

import numpy as np
import open3d as o3d

class MESHPublisher(Node):

    def __init__(self):
        super().__init__('mesh_publisher_node')

        # This executable expectes the first argument to be the path to a 
        # point cloud file. I.e. when you run it with ros:
        # ros2 run pcd_publisher pcd_publisher_node /path/to/ply
        assert len(sys.argv) > 1, "No ply file given."
        assert os.path.exists(sys.argv[1]), "File doesn't exist."
        pcd_path = sys.argv[1]

        # I use Open3D to read point clouds and meshes. It's a great library!
        pcd = o3d.io.read_point_cloud(pcd_path)
        # I then convert it into a numpy array.
        self.points = np.asarray(pcd.points)
        print(self.points.shape)
        
        # I create a publisher that publishes sensor_msgs.PointCloud2 to the 
        # topic 'pcd'. The value '10' refers to the history_depth, which I 
        # believe is related to the ROS1 concept of queue size. 
        # Read more here: 
        # http://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
        self.mesh_publisher = self.create_publisher(Mesh, 'mesh', 10)
        self.marker_publisher = self.create_publisher(Marker, 'mesh_marker', 10)
        timer_period = 1/30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # This rotation matrix is used for visualization purposes. It rotates
        # the point cloud on each timer callback. 
        # self.R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, np.pi/48])
        self.R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0])

              
                
    def timer_callback(self):
        # For visualization purposes, I rotate the point cloud with self.R 
        # to make it spin. 
        self.points = self.points @ self.R
        # Here I use the point_cloud() function to convert the numpy array 
        # into a sensor_msgs.PointCloud2 object. The second argument is the 
        # name of the frame the point cloud will be represented in. The default
        # (fixed) frame in RViz is called 'map'
        # self.mesh = point_cloud_to_mesh(self.points)
        self.mesh = point_cloud_to_mesh(self.points, method = 'poisson')
        # Then I publish the PointCloud2 object 
        self.mesh_publisher.publish(self.mesh)

        # Create marker from mesh
        self.marker = create_mesh_marker(self, self.mesh, "map")  # Replace "map" with your frame

        # Publish the marker
        self.marker_publisher.publish(self.marker)

# def point_cloud(points, parent_frame):
#     """ Creates a point cloud message.
#     Args:
#         points: Nx3 array of xyz positions.
#         parent_frame: frame in which the point cloud is defined
#     Returns:
#         sensor_msgs/PointCloud2 message

#     Code source:
#         https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0

#     References:
#         http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
#         http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
#         http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html

#     """
#     # In a PointCloud2 message, the point cloud is stored as an byte 
#     # array. In order to unpack it, we also include some parameters 
#     # which desribes the size of each individual point.
#     ros_dtype = sensor_msgs.PointField.FLOAT32
#     dtype = np.float32
#     itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

#     data = points.astype(dtype).tobytes() 

#     # The fields specify what the bytes represents. The first 4 bytes 
#     # represents the x-coordinate, the next 4 the y-coordinate, etc.
#     fields = [sensor_msgs.PointField(
#         name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
#         for i, n in enumerate('xyz')]

#     # The PointCloud2 message also has a header which specifies which 
#     # coordinate frame it is represented in. 
#     header = std_msgs.Header(frame_id=parent_frame)

#     return sensor_msgs.PointCloud2(
#         header=header,
#         height=1, 
#         width=points.shape[0],
#         is_dense=False,
#         is_bigendian=False,
#         fields=fields,
#         point_step=(itemsize * 3), # Every point consists of three float32s.
#         row_step=(itemsize * 3 * points.shape[0]),
#         data=data
#     )

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

# def point_cloud_to_mesh(points):
#     """ Creates a mesh message from a point cloud
#     Args:
#         points: Nx3 array of xyz positions.
#         parent_frame: frame in which the point cloud is defined
#     Returns:
#         shape_msgs/Mesh message
    
#     References:
#         https://docs.ros.org/en/noetic/api/shape_msgs/html/msg/Mesh.html
#     """
    
#     # Create the mesh message
#     mesh = Mesh()
    
#     # Convert the points to geometry_msgs/Point for the vertices
#     mesh.vertices = []
#     for point in points:
#         mesh_point = Point()
#         mesh_point.x = float(point[0])
#         mesh_point.y = float(point[1])
#         mesh_point.z = float(point[2])
#         mesh.vertices.append(mesh_point)
    
#     # Since we only have points without connectivity information,
#     # we could either:
#     # 1. Return just the vertices without triangles
#     # 2. Attempt to create a mesh using a triangulation algorithm
    
#     # For option 1 (just return vertices without triangles):
#     mesh.triangles = []
    
#     # For option 2, you'd need a triangulation algorithm
#     # A basic approach might be to use Delaunay triangulation if points form a 2.5D surface
#     # This would require additional libraries like scipy:
    
#     # from scipy.spatial import Delaunay
#     try:
#         # Only works well for 2.5D surfaces (not full 3D)
#         # Project points to 2D for triangulation
#         points_2d = points[:, 0:2]  # Use XY coordinates
#         tri = Delaunay(points_2d)
        
#         # Create triangles
#         for simplex in tri.simplices:
#             triangle = MeshTriangle()
#             triangle.vertex_indices = [int(simplex[0]), int(simplex[1]), int(simplex[2])]
#             mesh.triangles.append(triangle)
#     except Exception as e:
#         # Fallback if triangulation fails
#         print(f"Triangulation failed: {e}")
#         mesh.triangles = []
    
#     # Note: The parent_frame should be set in the message header
#     # However, shape_msgs/Mesh doesn't have a header field
#     # You'll need to set this when publishing using a stamped message type or TF
    
#     # return shape_msgs/Mesh(
#     #     triangles = shape_msgs/MeshTriangle(),
#     #     vertices = geometry_msgs/Point()
#     # )
#     return mesh

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
