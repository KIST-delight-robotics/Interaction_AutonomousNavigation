from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_name = 'dwa_nav'
    pkg_share = get_package_share_directory(pkg_name)

    param_file = os.path.join(
        pkg_share,
        'param',
        'dwa_command_node.param.yaml'
    )

    return LaunchDescription([
        Node(
            package=pkg_name,
            executable='dwa_node',   
            name='dwa_command_node',
            parameters=[param_file],
            output='screen'
        )
    ])
