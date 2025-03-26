import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    rviz_config_dir = os.path.join(get_package_share_directory(
        'data_processing'), 'config', 'config.rviz')
    assert os.path.exists(rviz_config_dir)

    slicer_config = os.path.join(get_package_share_directory(
        'data_processing'), 'config', 'cura.yaml')
    assert os.path.exists(slicer_config)

    ply_path = os.path.join(get_package_share_directory(
        'data_processing'), 'resource', 'crater_pc.ply') #rock_surface_structured is some good shit
    assert os.path.exists(ply_path)

    output_dir = "src/data_processing/resource/outputs"

    return LaunchDescription([
        Node(package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            output='screen'
        ),
        Node(package='data_processing',
            executable='pcd_publisher_node',
            name='pcd_publisher_node',
            output='screen',
            arguments=[ply_path],
        ),
        Node(package='data_processing',
            executable='mesh_publisher_node',
            name='mesh_publisher_node',
            output='screen',
            arguments=[ply_path],
        ),
        Node(package='data_processing',
            executable='mesh_modifier_node',
            name='mesh_modifier_node',
            output='screen',
            arguments=[ply_path],
        ),
        Node(
            package='data_processing',
            executable='mesh_to_stl_node',
            name='mesh_to_stl_node',
            parameters=[{
                'subscribe_topic': 'modified_mesh',
                'output_directory': output_dir,
                'stl_subdirectory': 'stl',
                'gcode_subdirectory': 'gcode',
                'cura_executable_path': "cura-slicer",
                'filename_prefix': 'ros_part'
            }],
            output='screen'
        )
    ])

