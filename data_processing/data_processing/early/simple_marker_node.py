#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
import time

class SimpleMarkerNode(Node):
    def __init__(self):
        super().__init__('simple_marker_node')
        
        # Create a publisher for the marker
        self.marker_pub = self.create_publisher(
            Marker, 
            '/test_marker', 
            10
        )
        
        # Print startup message
        self.get_logger().info('Simple marker node initialized')
        
        # Create a timer to publish the marker periodically
        self.timer = self.create_timer(1.0, self.publish_marker)
        self.get_logger().info('Timer created')
        
        # Counter for debugging
        self.count = 0

    def publish_marker(self):
        self.count += 1
        self.get_logger().info(f'Publishing marker #{self.count}')
        
        # Create a simple cube marker
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Set the position (origin)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set the scale (1 meter cube)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # Set the color (red)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Alpha (transparency)
        
        # Publish the marker
        self.marker_pub.publish(marker)
        self.get_logger().info('Marker published')

def main():
    print("Starting node...")
    rclpy.init()
    print("ROS initialized")
    
    node = SimpleMarkerNode()
    print("Node created")
    
    try:
        print("Starting to spin")
        rclpy.spin(node)
        print("Spin ended")
    except Exception as e:
        print(f"Exception in spin: {str(e)}")
    finally:
        print("Shutting down")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()