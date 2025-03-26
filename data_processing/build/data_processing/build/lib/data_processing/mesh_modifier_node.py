import sys
import os

import rclpy
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, String

import numpy as np
import open3d as o3d
import trimesh

class MeshModifier(Node):
    def __init__(self):
        super().__init__('mesh_modifier_node')
        
        # Create publishers
        self.modified_mesh_publisher = self.create_publisher(
            Mesh, 
            'modified_mesh', 
            10)
        self.modified_marker_publisher = self.create_publisher(
            Marker, 
            'modified_mesh_marker', 
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
            
            # Convert ROS Mesh message to Open3D TriangleMesh
            self.get_logger().info('Converting ROS mesh to Open3D format...')
            o3d_mesh = self.ros_mesh_to_o3d(self.received_mesh)
            self.get_logger().info(f'Conversion complete. Mesh has {len(o3d_mesh.vertices)} vertices and {len(o3d_mesh.triangles)} triangles')
            
            # Get mesh dimensions to calculate hole depth
            mesh_bbox = o3d_mesh.get_axis_aligned_bounding_box()
            mesh_extent = mesh_bbox.get_extent()
            hole_depth = min(mesh_extent) / 2.0
            self.get_logger().info(f'Calculated hole depth: {hole_depth:.3f} (half of shortest dimension)')
            
            # Add a simple cylindrical hole with smaller diameter
            self.get_logger().info('Creating cylindrical hole...')
            modified_o3d_mesh = self.add_cylindrical_hole(o3d_mesh, 
                                                        diameter=0.1,  # 10cm diameter
                                                        depth=hole_depth)  # Half of mesh depth
            
            # Convert back to ROS messages
            self.get_logger().info('Converting modified mesh back to ROS format...')
            self.modified_mesh = self.o3d_to_ros_mesh(modified_o3d_mesh)
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
            
            # Send timing info as separate message
            timing_msg = String()
            timing_msg.data = f"Mesh modification took {duration:.2f} seconds"
            self.status_publisher.publish(timing_msg)
            
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
    
    def smooth_mesh(self, mesh, iterations=5, lambda_value=0.5):
        """
        Smooth a mesh using Laplacian smoothing
        
        Args:
            mesh: Open3D TriangleMesh object
            iterations: Number of smoothing iterations to perform
            lambda_value: Strength of the smoothing (0-1)
            
        Returns:
            Smoothed Open3D TriangleMesh
        """
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            self.get_logger().warning('Empty mesh, cannot smooth')
            return mesh
            
        # Make a copy of the mesh to avoid modifying the original
        smoothed_mesh = o3d.geometry.TriangleMesh()
        smoothed_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        smoothed_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
        
        # Copy vertex colors if they exist
        if hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0:
            smoothed_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
        
        # Apply Laplacian smoothing with correct parameter names
        smoothed_mesh = smoothed_mesh.filter_smooth_laplacian(iterations, lambda_value)
        
        # Ensure normals are computed for the smoothed mesh
        smoothed_mesh.compute_vertex_normals()
        smoothed_mesh.compute_triangle_normals()
        
        return smoothed_mesh
    
    def add_female_bolt_hole(self, mesh, diameter=0.4, depth=0.8, thread_pitch=0.1, num_threads=10, center=None):
        """
        Add a female bolt hole (threaded hole) to the mesh centered at a specific point.
        
        Parameters:
        -----------
        mesh : o3d.geometry.TriangleMesh
            The input mesh to modify
        diameter : float
            Diameter of the bolt hole
        depth : float
            Depth of the bolt hole
        thread_pitch : float
            Distance between thread loops
        num_threads : int
            Number of thread loops
        center : np.array(3) or None
            Center point of the bolt hole (if None, uses mesh center)
            
        Returns:
        --------
        o3d.geometry.TriangleMesh
            The modified mesh with a bolt hole
        """
        import copy
        import numpy as np
        
        # Make a copy of the input mesh to avoid modifying the original
        result_mesh = copy.deepcopy(mesh)
        
        # Calculate the center of the mesh if not provided
        if center is None:
            center = result_mesh.get_center()
        
        # Create a bolt hole cylinder
        radius = diameter / 2.0
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=depth)
        
        # Add thread details to the cylinder
        cylinder_vertices = np.asarray(cylinder.vertices)
        cylinder_normals = np.asarray(cylinder.vertex_normals)
        
        # Calculate parameters for threading
        thread_depth = radius * 0.15  # Thread depth relative to radius
        
        # Create threading pattern
        for i, vertex in enumerate(cylinder_vertices):
            # Only modify vertices on the side surface (not end caps)
            dist_from_center = np.sqrt(vertex[0]**2 + vertex[1]**2)
            
            # Check if this vertex is on the side of the cylinder (not end caps)
            if abs(dist_from_center - radius) < 0.001:
                # Calculate height along cylinder
                height_percent = (vertex[2] + depth/2) / depth
                
                # Create thread pattern using sine function
                angle = height_percent * depth * (2 * np.pi / thread_pitch) * num_threads
                thread_offset = thread_depth * np.sin(angle)
                
                # Calculate direction from center axis to vertex (normalized)
                direction = np.array([vertex[0], vertex[1], 0])
                direction = direction / np.linalg.norm(direction)
                
                # Apply thread pattern by moving vertex inward
                cylinder_vertices[i][0] += direction[0] * thread_offset
                cylinder_vertices[i][1] += direction[1] * thread_offset
        
        # Update cylinder mesh with the threaded vertices
        cylinder.vertices = o3d.utility.Vector3dVector(cylinder_vertices)
        cylinder.compute_vertex_normals()
        
        # Transform the cylinder to be oriented along the mesh's principal axes
        # Assuming the threaded hole should go into the mesh from its "top"
        
        # Compute mesh principal axes
        mesh_bbox = mesh.get_axis_aligned_bounding_box()
        mesh_extent = mesh_bbox.get_extent()
        
        # Determine the principal axis (longest dimension is z)
        principal_axes = np.eye(3)
        if mesh_extent[0] > mesh_extent[1] and mesh_extent[0] > mesh_extent[2]:
            # x is longest
            rotation = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]
            ])
            principal_axes = np.matmul(principal_axes, rotation)
        elif mesh_extent[1] > mesh_extent[0] and mesh_extent[1] > mesh_extent[2]:
            # y is longest
            rotation = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])
            principal_axes = np.matmul(principal_axes, rotation)
        
        # Compute transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = principal_axes
        
        # Position cylinder at center of mesh
        transformation[:3, 3] = center
        
        # Move cylinder upward by half its depth to properly position it
        surface_offset = depth / 2
        principal_axis = principal_axes[:, 2]  # The z axis in the rotated frame
        transformation[:3, 3] += principal_axis * surface_offset
        
        # Apply the transformation
        cylinder.transform(transformation)
        
        # Boolean difference operation (mesh - cylinder = mesh with hole)
        # Open3D doesn't have direct boolean operations, so we use mesh.sample_points_uniformly
        # to create a point cloud and then reconstruct the mesh
        
        # Sample the meshes uniformly with high density
        self.get_logger().info("Sampling points from meshes for boolean operation...")
        mesh_points = mesh.sample_points_uniformly(number_of_points=100000)
        cylinder_points = cylinder.sample_points_uniformly(number_of_points=20000)
        
        # Convert to numpy arrays for processing
        mesh_points_arr = np.asarray(mesh_points.points)
        cylinder_points_arr = np.asarray(cylinder_points.points)
        
        # Keep only points that are outside the cylinder
        # For each point in the mesh, check if it's inside the cylinder
        self.get_logger().info("Computing boolean difference operation...")
        mesh_filtered_points = []
        
        # Create a translation to standardize the cylinder position
        cylinder_center = cylinder.get_center()
        cylinder_translation = -cylinder_center + np.array([0, 0, 0])
        cylinder_axis = principal_axis
        
        for point in mesh_points_arr:
            # Translate the point to cylinder's local coordinate system
            local_point = point + cylinder_translation
            
            # Project the point onto the cylinder axis
            projected_length = np.dot(local_point, cylinder_axis)
            
            # Check if point is within the cylinder height
            if abs(projected_length) > depth/2:
                # Point is outside the cylinder height
                mesh_filtered_points.append(point)
                continue
            
            # Calculate distance from point to cylinder axis
            axis_point = projected_length * cylinder_axis
            distance_to_axis = np.linalg.norm(local_point - axis_point)
            
            # Check if point is outside the cylinder radius
            if distance_to_axis > radius:
                mesh_filtered_points.append(point)
        
        # Create a new point cloud from the filtered points
        filtered_points = o3d.geometry.PointCloud()
        filtered_points.points = o3d.utility.Vector3dVector(np.array(mesh_filtered_points))
        
        # Add the internal thread points from the cylinder (creating the female thread)
        thread_points = []
        for point in cylinder_points_arr:
            # Translate the point to cylinder's local coordinate system
            local_point = point + cylinder_translation
            
            # Project the point onto the cylinder axis
            projected_length = np.dot(local_point, cylinder_axis)
            
            # Calculate distance from point to cylinder axis
            axis_point = projected_length * cylinder_axis
            distance_to_axis = np.linalg.norm(local_point - axis_point)
            
            # Only keep points that are within the cylinder but not at the exact radius
            # This preserves the thread pattern
            if abs(projected_length) <= depth/2 and distance_to_axis < radius and abs(distance_to_axis - radius) > 0.01:
                thread_points.append(point)
        
        # Add the thread points to the filtered points
        all_points = np.vstack([mesh_filtered_points, thread_points])
        filtered_points.points = o3d.utility.Vector3dVector(all_points)
        
        # Estimate normals for the point cloud
        filtered_points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Reconstruct the mesh using Poisson reconstruction
        self.get_logger().info("Reconstructing mesh with Poisson reconstruction...")
        reconstructed_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            filtered_points, depth=9, width=0, scale=1.1, linear_fit=False)
        
        # Filter out low-density vertices
        density_threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_threshold
        reconstructed_mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Clean up the mesh
        self.get_logger().info("Cleaning up the final mesh...")
        reconstructed_mesh.remove_degenerate_triangles()
        reconstructed_mesh.remove_duplicated_triangles()
        reconstructed_mesh.remove_duplicated_vertices()
        reconstructed_mesh.remove_non_manifold_edges()
        
        return reconstructed_mesh
    
    def add_simple_bolt_hole(self, mesh, diameter=0.4, depth=0.8, thread_pitch=0.1, num_threads=10, center=None):
        """
        Add a female bolt hole (threaded hole) to the mesh with minimal mesh reconstruction.
        
        Parameters:
        -----------
        mesh : o3d.geometry.TriangleMesh
            The input mesh to modify
        diameter : float
            Diameter of the bolt hole
        depth : float
            Depth of the bolt hole
        thread_pitch : float
            Distance between thread loops
        num_threads : int
            Number of thread loops
        center : np.array(3) or None
            Center point of the bolt hole (if None, uses mesh center)
            
        Returns:
        --------
        o3d.geometry.TriangleMesh
            The modified mesh with a bolt hole
        """
        import copy
        import numpy as np
        
        # Make a copy of the input mesh to avoid modifying the original
        result_mesh = copy.deepcopy(mesh)
        
        # Calculate the center of the mesh if not provided
        if center is None:
            center = result_mesh.get_center()
        
        # Create a bolt hole cylinder with threads
        radius = diameter / 2.0
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=depth)
        
        # Add thread details to the cylinder
        cylinder_vertices = np.asarray(cylinder.vertices)
        
        # Calculate parameters for threading
        thread_depth = radius * 0.15  # Thread depth relative to radius
        
        # Create threading pattern
        for i, vertex in enumerate(cylinder_vertices):
            # Only modify vertices on the side surface (not end caps)
            dist_from_center = np.sqrt(vertex[0]**2 + vertex[1]**2)
            
            # Check if this vertex is on the side of the cylinder (not end caps)
            if abs(dist_from_center - radius) < 0.001:
                # Calculate height along cylinder
                height_percent = (vertex[2] + depth/2) / depth
                
                # Create thread pattern using sine function
                angle = height_percent * depth * (2 * np.pi / thread_pitch) * num_threads
                thread_offset = thread_depth * np.sin(angle)
                
                # Calculate direction from center axis to vertex (normalized)
                direction = np.array([vertex[0], vertex[1], 0])
                direction = direction / np.linalg.norm(direction)
                
                # Apply thread pattern by moving vertex inward
                cylinder_vertices[i][0] += direction[0] * thread_offset
                cylinder_vertices[i][1] += direction[1] * thread_offset
        
        # Update cylinder mesh with the threaded vertices
        cylinder.vertices = o3d.utility.Vector3dVector(cylinder_vertices)
        cylinder.compute_vertex_normals()
        
        # Align cylinder with mesh principal axes
        mesh_bbox = mesh.get_axis_aligned_bounding_box()
        mesh_extent = mesh_bbox.get_extent()
        
        # Find principal axis (use Z by default, or choose longest dimension)
        principal_axis = np.array([0, 0, 1])  # Default Z-axis
        
        # Position cylinder at center of mesh
        transformation = np.eye(4)
        transformation[:3, 3] = center
        
        # Apply the transformation
        cylinder.transform(transformation)
        
        # Boolean operation: we'll use mesh cutting approach
        self.get_logger().info("Creating hole in mesh...")
        
        # The cylinder defines the region to remove
        # We'll perform simplified hole cutting by:
        # 1. Taking a point within each triangle and checking if it's inside the cylinder
        # 2. If yes, mark the triangle for removal
        
        # Extract mesh data
        mesh_vertices = np.asarray(result_mesh.vertices)
        mesh_triangles = np.asarray(result_mesh.triangles)
        
        # Get cylinder bounding parameters
        cylinder_center = cylinder.get_center()
        cylinder_radius = radius
        
        # Keep track of triangles to remove
        triangles_to_keep = []
        
        # For each triangle, check if it intersects with the cylinder
        for i, triangle in enumerate(mesh_triangles):
            # Get triangle vertices
            v1 = mesh_vertices[triangle[0]]
            v2 = mesh_vertices[triangle[1]]
            v3 = mesh_vertices[triangle[2]]
            
            # Calculate triangle center (approximate)
            triangle_center = (v1 + v2 + v3) / 3.0
            
            # Check if triangle center is inside cylinder
            # 1. Vector from cylinder center to triangle center
            vector_to_triangle = triangle_center - cylinder_center
            
            # 2. Project onto principal axis
            projection_length = np.dot(vector_to_triangle, principal_axis)
            
            # 3. Check if within cylinder height
            if abs(projection_length) <= depth / 2:
                # 4. Check if within cylinder radius
                # Calculate perpendicular distance to cylinder axis
                projected_point = cylinder_center + projection_length * principal_axis
                perpendicular_distance = np.linalg.norm(triangle_center - projected_point)
                
                # If outside cylinder radius, keep the triangle
                if perpendicular_distance > cylinder_radius:
                    triangles_to_keep.append(triangle)
            else:
                # Outside cylinder height, keep the triangle
                triangles_to_keep.append(triangle)
        
        # Create a new mesh with the remaining triangles
        result_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles_to_keep))
        
        # Create a mesh for the thread internal surface
        thread_mesh = cylinder
        
        # Combine the meshes
        final_mesh = result_mesh + thread_mesh
        
        # Clean up the mesh
        self.get_logger().info("Cleaning up the final mesh...")
        final_mesh.remove_degenerate_triangles()
        final_mesh.remove_duplicated_triangles()
        final_mesh.remove_duplicated_vertices()
        final_mesh.compute_vertex_normals()
        
        return final_mesh
    
    def ensure_watertight(self, mesh):
        """Helper function to ensure a mesh is watertight before boolean operations."""
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Fill holes if any exist
        hole_count = len(mesh.get_non_manifold_edges())
        if hole_count > 0:
            self.get_logger().info(f'Filling {hole_count} holes in mesh')
            mesh.fill_holes()
        
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh

    def add_cylindrical_hole(self, mesh, diameter=0.4, depth=0.8, center=None):
        """
        Add a cylindrical hole through the mesh using trimesh boolean operations.
        """
        import numpy as np
        import trimesh
        
        self.get_logger().info('Starting cylindrical hole creation...')
        
        # Convert Open3D mesh to trimesh
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        mesh_trim = trimesh.Trimesh(vertices=vertices, faces=triangles)
        
        # Calculate the center if not provided
        if center is None:
            center = mesh.get_center()
        self.get_logger().info(f'Hole center calculated: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]')
        
        # Get mesh dimensions
        mesh_bbox = mesh_trim.bounding_box
        mesh_extent = mesh_bbox.extents
        max_extent = np.max(mesh_extent) * 1.5
        radius = diameter / 2.0
        
        # Create cylinder
        self.get_logger().info(f'Creating cylinder with radius {radius:.3f} and height {max_extent:.3f}')
        cylinder = trimesh.creation.cylinder(
            radius=radius,
            height=max_extent,
            sections=32  # Number of segments for smoother cylinder
        )
        
        # Move cylinder to mesh center
        cylinder.apply_translation(center)
        
        # Perform boolean difference
        self.get_logger().info('Performing boolean subtraction...')
        try:
            mesh_with_hole = mesh_trim.difference(cylinder, engine='scad')
            self.get_logger().info('Boolean operation completed successfully')
        except Exception as e:
            self.get_logger().error(f'Boolean operation failed with scad: {str(e)}')
            try:
                self.get_logger().info('Retrying with blender engine...')
                mesh_with_hole = mesh_trim.difference(cylinder, engine='blender')
                self.get_logger().info('Boolean operation completed with blender engine')
            except Exception as e:
                self.get_logger().error(f'Boolean operation failed with blender: {str(e)}')
                try:
                    self.get_logger().info('Retrying with path engine...')
                    mesh_with_hole = mesh_trim.difference(cylinder, engine='path')
                    self.get_logger().info('Boolean operation completed with path engine')
                except Exception as e:
                    self.get_logger().error(f'All boolean operations failed: {str(e)}')
                    return mesh  # Return original mesh if all attempts fail
        
        # Convert back to Open3D mesh
        result_mesh = o3d.geometry.TriangleMesh()
        result_mesh.vertices = o3d.utility.Vector3dVector(mesh_with_hole.vertices)
        result_mesh.triangles = o3d.utility.Vector3iVector(mesh_with_hole.faces)
        
        # Ensure normals are computed
        result_mesh.compute_vertex_normals()
        result_mesh.compute_triangle_normals()
        
        self.get_logger().info(f'Final mesh created with {len(result_mesh.vertices)} vertices and {len(result_mesh.triangles)} triangles')
        return result_mesh
    
    def o3d_to_ros_mesh(self, o3d_mesh):
        """Convert from Open3D TriangleMesh to ROS shape_msgs/Mesh"""
        mesh_msg = Mesh()
        
        # Convert vertices
        for vertex in o3d_mesh.vertices:
            point = Point()
            point.x = float(vertex[0])
            point.y = float(vertex[1])
            point.z = float(vertex[2])
            mesh_msg.vertices.append(point)
        
        # Convert triangles
        for triangle in o3d_mesh.triangles:
            mesh_triangle = MeshTriangle()
            mesh_triangle.vertex_indices = [int(idx) for idx in triangle]
            mesh_msg.triangles.append(mesh_triangle)
            
        return mesh_msg
    
    def create_mesh_marker(self, mesh, frame_id):
        """
        Converts a shape_msgs/Mesh to a visualization_msgs/Marker for RViz display
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "modified_mesh"
        marker.id = 1
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        # Set transform
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set scale (1:1 mapping)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # Set color (orange for better visibility)
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque
        
        # Convert mesh triangles to marker points
        for triangle in mesh.triangles:
            for idx in triangle.vertex_indices:
                if idx < len(mesh.vertices):
                    vertex = mesh.vertices[idx]
                    point = Point()
                    point.x = vertex.x
                    point.y = vertex.y
                    point.z = vertex.z
                    marker.points.append(point)
        
        return marker

def main(args=None):
    rclpy.init(args=args)
    mesh_modifier = MeshModifier()
    rclpy.spin(mesh_modifier)
    
    # Destroy the node explicitly
    mesh_modifier.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()