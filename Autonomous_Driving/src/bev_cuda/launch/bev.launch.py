from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    param_file = os.path.join(
        get_package_share_directory('bev_cuda'),
        'param',
        'bev.param.yaml'
    )

    return LaunchDescription([
        Node(
            package='bev_cuda',
            executable='bev_node',
            name='bev_node',
            parameters=[param_file],
            output='screen'
        )
    ])
