from setuptools import setup
from glob import glob
import os

package_name = 'data_processing'  # use your actual package name

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launcher.launch.py']),
        # During installation, we need to copy the launch files
        (os.path.join('share', package_name, "launch"), glob('launch/*.launch.py')),
        # Same with the RViz configuration file.
        (os.path.join('share', package_name, "config"), glob('config/*')),
        # And the ply files.
        (os.path.join('share', package_name, "resource"), glob('resource/*.ply')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'open3d',
        'trimesh',
    ],
    python_requires='>=3.6',
    zip_safe=True,
    maintainer='IZZY POPIOLEK',
    maintainer_email='izzypopiolek@email.com',
    description='Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pcd_publisher_node = data_processing.pcd_publisher_node:main',
            'mesh_publisher_node = data_processing.mesh_publisher_node:main',
            'mesh_modifier_node = data_processing.mesh_modifier_node:main',
            'mesh_to_stl_node = data_processing.mesh_to_stl_node:main',
        ],
    },
)
