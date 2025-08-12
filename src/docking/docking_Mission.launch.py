from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='docking',
            executable='docking_Cont',
            name='docking_Cont',
            output='screen'
        ),
        Node(
            package='docking',
            executable='docking_Guid',
            name='docking_Guid',
            output='screen'
        ),
        Node(
            package='docking',
            executable='docking_Navi',
            name='docking_Navi',
            output='screen'
        )
    ])
