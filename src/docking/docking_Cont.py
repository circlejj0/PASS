#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pyproj import Transformer
from std_msgs.msg import Float64, Float64MultiArray, String
from sensor_msgs.msg import Imu, LaserScan

class DockingCont(Node):
    def __init__(self):
        super().__init__('docking_cont')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Sub
        self.create_subscription(Float64, '/error_psi', self.e_psi_callback, qos_profile)
        self.create_subscription(String, '/nav/status', self.status_callback, 10)
        self.create_subscription(Float64, 'visual_error', self.visual_error_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/UTM_Latlot', self.utm_callback, qos_profile)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, qos_profile)
        self.create_subscription(LaserScan, '/wamv/sensors/lidar/front_lidar/scan', self.lidar_callback, 10)
        
        # Pub
        self.left_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.status_pub = self.create_publisher(String, '/nav/status', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.process)
        
        # 변수 설정
        self.mode = "WAYPOINT_FOLLOWING"
        self.visual_error, self.prev_visual_error = 0.0, 0.0
        self.current_yaw_rad, self.wamv_x, self.wamv_y = 0.0, 0.0, 0.0
        self.min_forward_distance, self.target_yaw_rad = 999.0, None
        self.docking_start_time, self.prev_error_nav = None, 0.0
        self.error_psi = 0.0
        
        # 제어 파라미터
        self.kp_nav, self.kd_nav, self.base_thrust_nav = 10.0, 5.0, 550.0
        self.kp_rot = 200.0
        self.LIDAR_CAUTION_DIST, self.LIDAR_STOP_DIST = 7.0, 3.5
        self.DOCKING_MAX_SPEED = 60.0
        self.Kp_visual, self.Kd_visual = 0.25, 0.2
        
        # 도크 방향 (회전하기 위함)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")
        dock_lonlat = (150.67441330657957, -33.72255187124071)
        self.dock_utm_x, self.dock_utm_y = transformer.transform(dock_lonlat[1], dock_lonlat[0])
    
    # Callback Functions
    def e_psi_callback(self, msg): self.error_psi = msg.data
    def utm_callback(self, msg): self.wamv_x, self.wamv_y = msg.data[0], msg.data[1]
    def imu_callback(self, msg: Imu):
        q = msg.orientation
        self.current_yaw_rad = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
    def status_callback(self, msg):
        if msg.data == "ARRIVED_P1" and self.mode == "WAYPOINT_FOLLOWING":
            self.mode = "ROTATING"
            self.get_logger().info("MODE CHANGE: Moving to docking spot -> Rotating")
            self.stop_wamv()
    def lidar_callback(self, msg: LaserScan):
        center_index = len(msg.ranges) // 2
        view_range_idx = int(math.radians(10) / msg.angle_increment)
        start_idx, end_idx = max(0, center_index - view_range_idx), min(len(msg.ranges), center_index + view_range_idx)
        forward_ranges = msg.ranges[start_idx : end_idx]
        valid_ranges = [r for r in forward_ranges if r > msg.range_min and r < msg.range_max]
        self.min_forward_distance = min(valid_ranges) if valid_ranges else 999.0
    def visual_error_callback(self, msg: Float64):
        self.visual_error = msg.data

    # 주기 시작
    def process(self):
        if self.mode == "WAYPOINT_FOLLOWING": self.navigate_to_waypoint()
        elif self.mode == "ROTATING": self.rotate_in_place()
        elif self.mode == "HOLDING_FOR_SCAN": self.hold_for_scan()
        elif self.mode == "DOCKING_MANEUVER": self.dock_with_lidar_and_vision()
        elif self.mode == "MISSION_COMPLETE": self.stop_wamv()

    # 도킹 시작 지점 이동
    def navigate_to_waypoint(self):
        turn_val = self.kp_nav * self.error_psi + self.kd_nav * (self.error_psi - self.prev_error_nav)
        self.prev_error_nav = self.error_psi
        self.publish_thrust(self.base_thrust_nav - turn_val, self.base_thrust_nav + turn_val)

    # 도킹 시작 지점에서 제자리 회전
    def rotate_in_place(self):
        if self.target_yaw_rad is None:
            dx, dy = self.dock_utm_x - self.wamv_x, self.dock_utm_y - self.wamv_y
            self.target_yaw_rad = math.atan2(dy, dx)
        error_yaw = self.target_yaw_rad - self.current_yaw_rad
        error_yaw = (error_yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(math.degrees(error_yaw)) < 3.0:
            self.get_logger().info("Rotation Complete -> Holding for scan")
            self.mode, self.docking_start_time = "HOLDING_FOR_SCAN", self.get_clock().now()
            self.status_pub.publish(String(data="ARRIVED_SCAN_P1"))
            self.stop_wamv()
            return
        turn = self.kp_rot * error_yaw
        self.publish_thrust(-turn, turn)

    # 이미지 센싱을 위한 정지
    def hold_for_scan(self):
        self.stop_wamv()
        elapsed = (self.get_clock().now() - self.docking_start_time).nanoseconds * 1e-9
        if elapsed >= 3.0:
            self.get_logger().info("Scan Complete -> DOCKING_MANEUVER")
            self.mode = "DOCKING_MANEUVER"

    # 도킹
    """
    def dock_with_lidar_and_vision(self):
        if self.min_forward_distance <= self.LIDAR_STOP_DIST:
            self.get_logger().info(f"LIDAR STOP! Distance: {self.min_forward_distance:.2f}m. MISSION COMPLETE.")
            self.mode = "MISSION_COMPLETE"
            self.stop_wamv()
            return

        base_speed = 0.0
        if self.min_forward_distance < self.LIDAR_CAUTION_DIST:
            speed_ratio = (self.min_forward_distance - self.LIDAR_STOP_DIST) / (self.LIDAR_CAUTION_DIST - self.LIDAR_STOP_DIST)
            base_speed = max(0.0, self.DOCKING_MAX_SPEED * speed_ratio)
        else: 
            base_speed = self.DOCKING_MAX_SPEED

        error_derivative = self.visual_error - self.prev_visual_error
        turn_adjustment = self.Kp_visual * self.visual_error + self.Kd_visual * error_derivative

        self.prev_visual_error = self.visual_error

        left_thrust = base_speed + turn_adjustment
        right_thrust = base_speed - turn_adjustment
        self.publish_thrust(left_thrust, right_thrust)
        """
        
    def stop_wamv(self): self.publish_thrust(0.0, 0.0)

    def publish_thrust(self, left, right):
        l_msg, r_msg = Float64(), Float64()
        l_msg.data, r_msg.data = float(left), float(right)
        self.left_thruster_pub.publish(l_msg)
        self.right_thruster_pub.publish(r_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DockingCont()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
