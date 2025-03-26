import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/izzypopiolek/ldlidar_ros_ws/src/data_processing/install/data_processing'
