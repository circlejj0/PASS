from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pass_hopping',
            executable='hopping_Cont',
            name='hopping_Cont',
            output='screen'
        ),
        Node(
            package='pass_hopping',
            executable='hopping_Navi',
            name='hopping_Navi',
            output='screen'
        )
    ])
