#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mavros 사용해서 hopping

import rclpy
import math
# import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pyproj import Transformer
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import NavSatFix, Imu
from visualization_msgs.msg import Marker, MarkerArray
# from std_msgs.msg import ColorRGBA
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import transforms3d.euler as euler

class mavroshopping(Node):
    def __init__(self):
        super().__init__('mavros_waypoint_follower')

        self.current_state = None
        self.current_pose_gps = None
        self.current_yaw_rad = None
        
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True)
        self.origin_utm = None
        self.waypoints_local = []
        
        self.next_obj = 0
        self.prev_heading_error = 0.0

        self.declare_parameter('arrival_radius', 10.0)
        self.declare_parameter('target_speed', 0.1)
        self.declare_parameter('kp', 30.0)
        self.declare_parameter('kd', 10.0)
        
        self.arrival_radius = self.get_parameter('arrival_radius').value
        self.target_speed = self.get_parameter('target_speed').value
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value

        waypoints_lonlat = [
            (129.107100, 35.133504)
        ]
        self.origin_utm = self.transformer.transform(waypoints_lonlat[0][0], waypoints_lonlat[0][1])
        self.waypoints_local = [(x - self.origin_utm[0], y - self.origin_utm[1]) for x, y in [self.transformer.transform(lon, lat) for lon, lat in waypoints_lonlat]]

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Sub
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, qos_profile)
        self.gps_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, qos_profile)
        self.imu_sub = self.create_subscription(Imu, '/mavros/imu/data', self.imu_callback, qos_profile)

        # Pub
        self.velocity_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.path_pub = self.create_publisher(Path, '/boat_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/boat_waypoints', 10)
        
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        self.status = "INITIALIZING"
        self.get_logger().info('Node initialized. Waiting for MAVROS connection and GPS...')
        self.pause_timer = None
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def state_callback(self, msg):
        self.current_state = msg

    def gps_callback(self, msg):
        self.current_pose_gps = msg

    def imu_callback(self, msg):
        q = msg.orientation
        _, _, self.current_yaw_rad = euler.quat2euler([q.w, q.x, q.y, q.z])

    def arm_and_set_mode(self):
        arm_req = CommandBool.Request()
        arm_req.value = True
        self.arming_client.call_async(arm_req)
        self.get_logger().info("Arming command sent...")

        mode_req = SetMode.Request()
        mode_req.custom_mode = 'GUIDED'
        self.set_mode_client.call_async(mode_req)
        self.get_logger().info("Set mode to GUIDED command sent...")

    def control_loop(self):
        if self.status == "PAUSED":
            return

        if self.status == "INITIALIZING":
            if self.current_state is None or self.current_pose_gps is None or self.current_yaw_rad is None:
                return
            
            if not self.current_state.connected:
                self.get_logger().warn("MAVROS not connected to FCU. Retrying...")
                return

            self.get_logger().info("Connection established. Attempting to arm and set mode.")
            self.arm_and_set_mode()
            self.status = "ARMING"

        elif self.status == "ARMING":
            if self.current_state.armed and self.current_state.mode == 'GUIDED':
                self.get_logger().info("Vehicle armed and in GUIDED mode. Starting mission.")
                self.status = "MISSION"
            else:
                self.arm_and_set_mode()
                self.get_logger().info(f"Waiting for arm/guided mode. Current mode: {self.current_state.mode}, Armed: {self.current_state.armed}")
            return

        elif self.status == "MISSION":
            if self.current_pose_gps is None or self.current_yaw_rad is None:
                return

            self.publish_waypoints(self.waypoints_local, self.next_obj)

            current_utm = self.transformer.transform(self.current_pose_gps.longitude, self.current_pose_gps.latitude)
            current_local = (current_utm[0] - self.origin_utm[0], current_utm[1] - self.origin_utm[1])
            
            goal_local = self.waypoints_local[self.next_obj]
            distance_to_goal = math.sqrt((goal_local[0] - current_local[0])**2 + (goal_local[1] - current_local[1])**2)
            
            self.get_logger().info(f"Waypoint {self.next_obj}: Distance = {distance_to_goal:.1f} m")

            if distance_to_goal < self.arrival_radius:
                self.get_logger().info(f"--- Waypoint {self.next_obj} Reached! ---")
                self.next_obj += 1
                if self.next_obj >= len(self.waypoints_local):
                    self.status = "DONE"
                    return
                self.publish_velocity(0.0, 0.0)
                self.status = "PAUSED"
                self.pause_timer = self.create_timer(7.0, self.resume_mission) # 7초 후 미션 재개
                return
            
            desired_heading_rad = math.atan2(goal_local[0] - current_local[0], goal_local[1] - current_local[1])
            heading_error = desired_heading_rad - self.current_yaw_rad
            while heading_error > math.pi: heading_error -= 2 * math.pi
            while heading_error < -math.pi: heading_error += 2 * math.pi
            
            turn_rate_rad_s = self.kp * heading_error + self.kd * (heading_error - self.prev_heading_error)
            self.prev_heading_error = heading_error
            
            self.publish_velocity(self.target_speed, turn_rate_rad_s)
            
            self.publish_path(current_local[0], current_local[1], self.current_yaw_rad)
            
        elif self.status == "DONE":
            self.get_logger().info("Mission complete. Stopping vehicle.")
            self.publish_velocity(0.0, 0.0)

    def resume_mission(self):
        """일시 정지 타이머 콜백"""
        self.get_logger().info("Resuming mission...")
        self.pause_timer.cancel()
        self.pause_timer = None
        self.status = "MISSION"

    def publish_velocity(self, velocity_x, yaw_rate_rad_s):
        vel_msg = Twist()
        vel_msg.linear.x = velocity_x
        vel_msg.angular.z = yaw_rate_rad_s
        self.velocity_pub.publish(vel_msg)

    def publish_path(self, utm_x, utm_y, yaw_rad):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = utm_x
        pose.pose.position.y = utm_y
        
        q = euler.euler2quat(0, 0, yaw_rad)
        pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z = q[0], q[1], q[2], q[3]

        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    def publish_waypoints(self, waypoints_local, next_obj_index):
        marker_array = MarkerArray()
        for i, (wx, wy) in enumerate(waypoints_local):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns, marker.id, marker.type, marker.action = "waypoints", i, Marker.SPHERE, Marker.ADD
            marker.pose.position.x, marker.pose.position.y = wx, wy
            marker.scale.x, marker.scale.y, marker.scale.z = 2.0, 2.0, 2.0
            marker.color.a = 1.0

            if i == next_obj_index:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = mavroshopping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nNode interrupted by user.")
    finally:
        node.get_logger().info("Shutting down. Sending stop command.")
        node.publish_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
