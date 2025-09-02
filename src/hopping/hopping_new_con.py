#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mavros 사용해서 hopping

import rclpy
import math
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pyproj import Transformer

# ROS 2 메시지 타입
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, Twist
from sensor_msgs.msg import NavSatFix, Imu
from visualization_msgs.msg import Marker, MarkerArray
# from std_msgs.msg import ColorRGBA

# MAVROS 메시지 및 서비스 타입
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

import transforms3d.euler as euler

class DronekitToMavrosNode(Node):
    def __init__(self):
        super().__init__('mavros_waypoint_follower')

        # --- 상태 변수 ---
        self.current_state = None
        self.current_pose_gps = None
        self.current_yaw_rad = None
        
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True)
        self.origin_utm = None
        self.waypoints_local = []
        
        self.next_obj = 0
        self.prev_heading_error = 0.0

        # --- 파라미터 ---
        self.declare_parameter('arrival_radius', 10.0)
        self.declare_parameter('target_speed', 0.1)
        self.declare_parameter('kp', 30.0)
        self.declare_parameter('kd', 10.0)
        
        self.arrival_radius = self.get_parameter('arrival_radius').value
        self.target_speed = self.get_parameter('target_speed').value
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value

        waypoints_lonlat = [
            (129.104375, 35.133902), (129.104595, 35.133731),
            (129.106874, 35.133931), (129.106810, 35.134896),
            (129.105754, 35.134853)
        ]
        self.origin_utm = self.transformer.transform(waypoints_lonlat[0][0], waypoints_lonlat[0][1])
        self.waypoints_local = [(x - self.origin_utm[0], y - self.origin_utm[1]) for x, y in [self.transformer.transform(lon, lat) for lon, lat in waypoints_lonlat]]

        # --- ROS 2 인터페이스 ---
        # QoS 설정 수정: MAVROS와의 통신을 위해 RELIABILITY를 BEST_EFFORT로 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 구독자 (Subscribers)
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, qos_profile)
        self.gps_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, qos_profile)
        self.imu_sub = self.create_subscription(Imu, '/mavros/imu/data', self.imu_callback, qos_profile)

        # 발행자 (Publishers)
        self.velocity_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.path_pub = self.create_publisher(Path, '/boat_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/boat_waypoints', 10)
        
        # 서비스 클라이언트 (Service Clients)
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # 시각화를 위한 Path 메시지 초기화
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        # --- 상태 머신 및 메인 루프 ---
        self.status = "INITIALIZING"
        self.get_logger().info('Node initialized. Waiting for MAVROS connection and GPS...')
        # `time.sleep` 대신 사용될 타이머.
        self.pause_timer = None
        self.control_timer = self.create_timer(0.1, self.control_loop)

    # --- 콜백 함수들 ---
    def state_callback(self, msg):
        self.current_state = msg

    def gps_callback(self, msg):
        self.current_pose_gps = msg

    def imu_callback(self, msg):
        q = msg.orientation
        _, _, self.current_yaw_rad = euler.quat2euler([q.w, q.x, q.y, q.z])

    # --- 제어 및 상태 관리 함수 ---
    def arm_and_set_mode(self):
        # Arming 요청
        arm_req = CommandBool.Request()
        arm_req.value = True
        self.arming_client.call_async(arm_req)
        self.get_logger().info("Arming command sent...")

        # GUIDED 모드 변경 요청
        mode_req = SetMode.Request()
        mode_req.custom_mode = 'GUIDED'
        self.set_mode_client.call_async(mode_req)
        self.get_logger().info("Set mode to GUIDED command sent...")

    def control_loop(self):
        # 웨이포인트 도착 후 일시 정지 상태인 경우, 제어 로직을 실행하지 않음
        if self.status == "PAUSED":
            return

        # 상태 머신
        if self.status == "INITIALIZING":
            if self.current_state is None or self.current_pose_gps is None or self.current_yaw_rad is None:
                return # 아직 모든 정보가 들어오지 않음
            
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
                # 계속 Arming 및 모드 변경 시도
                self.arm_and_set_mode()
                self.get_logger().info(f"Waiting for arm/guided mode. Current mode: {self.current_state.mode}, Armed: {self.current_state.armed}")
            return # 미션 시작 전까지는 아래 로직 실행 안 함

        elif self.status == "MISSION":
            # 모든 정보가 있는지 최종 확인
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
                # 다음 웨이포인트 가기 전 잠시 정지
                self.publish_velocity(0.0, 0.0)
                self.status = "PAUSED"
                self.pause_timer = self.create_timer(7.0, self.resume_mission) # 7초 후 미션 재개
                return
            
            desired_heading_rad = math.atan2(goal_local[0] - current_local[0], goal_local[1] - current_local[1])
            heading_error = desired_heading_rad - self.current_yaw_rad
            # 각도 오차를 -pi ~ pi 범위로 정규화
            while heading_error > math.pi: heading_error -= 2 * math.pi
            while heading_error < -math.pi: heading_error += 2 * math.pi
            
            turn_rate_rad_s = self.kp * heading_error + self.kd * (heading_error - self.prev_heading_error)
            self.prev_heading_error = heading_error
            
            self.publish_velocity(self.target_speed, turn_rate_rad_s)
            
            self.publish_path(current_local[0], current_local[1], self.current_yaw_rad)
            
        elif self.status == "DONE":
            self.get_logger().info("Mission complete. Stopping vehicle.")
            self.publish_velocity(0.0, 0.0)
            # 노드를 종료하거나 대기 상태로 전환할 수 있음

    def resume_mission(self):
        """일시 정지 타이머 콜백"""
        self.get_logger().info("Resuming mission...")
        self.pause_timer.cancel()
        self.pause_timer = None
        self.status = "MISSION"

    # --- 발행 함수들 ---
    def publish_velocity(self, velocity_x, yaw_rate_rad_s):
        vel_msg = Twist()
        vel_msg.linear.x = velocity_x
        vel_msg.angular.z = yaw_rate_rad_s # MAVROS는 rad/s 단위를 사용
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
    node = DronekitToMavrosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nNode interrupted by user.")
    finally:
        node.get_logger().info("Shutting down. Sending stop command.")
        # 노드가 종료되기 전 마지막으로 정지 명령을 보냄
        node.publish_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
