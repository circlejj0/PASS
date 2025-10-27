#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pyproj import Transformer
from std_msgs.msg import Float64, Float64MultiArray, String
from sensor_msgs.msg import Imu, LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math

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
        self.create_subscription(Float64MultiArray, 'visual_errors', self.visual_errors_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/UTM_Latlot', self.utm_callback, qos_profile)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, qos_profile)
        self.create_subscription(LaserScan, '/wamv/sensors/lidar/front_lidar/scan', self.lidar_callback, 10)
        
        # Pub
        self.left_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.status_pub = self.create_publisher(String, '/nav/status', 10)
        self.visual_path_pub = self.create_publisher(Marker, 'wamv_visual_path', 10)
        # --- [추가된 부분 1] ---
        self.dock_gate_pub = self.create_publisher(Marker, 'dock_gate_marker', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.process)
        
        # 변수 설정
        self.mode = "WAYPOINT_FOLLOWING"
        self.circle_error, self.triangle_error = 0.0, 0.0
        self.prev_circle_error, self.prev_triangle_error = 0.0, 0.0
        self.current_yaw_rad, self.wamv_x, self.wamv_y = 0.0, 0.0, 0.0
        self.min_forward_distance, self.target_yaw_rad = 999.0, None
        self.docking_start_time, self.prev_error_nav = None, 0.0
        self.error_psi = 0.0
        self.image_width = 640.0
        self.camera_fov_rad = math.radians(60.0)
        
        # 제어 파라미터
        self.kp_nav, self.kd_nav, self.base_thrust_nav = 10.0, 5.0, 550.0
        self.kp_rot = 200.0
        self.LIDAR_CAUTION_DIST, self.LIDAR_STOP_DIST = 7.0, 3.5
        self.DOCKING_MAX_SPEED = 60.0
        self.Kp_visual = 1.5
        self.Kd_visual = 2.0
        self.ALIGNMENT_THRESHOLD = 2.0
        
        # --- [추가된 부분 2] ---
        # 좌표 변환기
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")
        
        # 도크 방향 (회전하기 위함)
        dock_lonlat = (150.67447312373196, -33.72246602165311)
        self.dock_utm_x, self.dock_utm_y = transformer.transform(dock_lonlat[1], dock_lonlat[0])

        # 시각화할 도크 게이트의 두 지점 GPS 좌표
        dock_gate_A_lonlat = (150.67452, -33.72248) # 예시 좌표 1
        dock_gate_B_lonlat = (150.67454, -33.72242) # 예시 좌표 2
        
        # GPS를 UTM으로 미리 변환
        self.dock_gate_A_utm = transformer.transform(dock_gate_A_lonlat[1], dock_gate_A_lonlat[0])
        self.dock_gate_B_utm = transformer.transform(dock_gate_B_lonlat[1], dock_gate_B_lonlat[0])

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
    def visual_errors_callback(self, msg: Float64MultiArray):
        if len(msg.data) >= 2:
            self.circle_error = msg.data[0]
            self.triangle_error = msg.data[1]

    # 주기 시작
    def process(self):
        if self.mode == "WAYPOINT_FOLLOWING": self.navigate_to_waypoint()
        elif self.mode == "ROTATING": self.rotate_in_place()
        elif self.mode == "HOLDING_FOR_SCAN": self.hold_for_scan()
        elif self.mode == "DOCKING_MANEUVER": self.dock_with_triangle_and_move()
        elif self.mode == "MISSION_COMPLETE": self.stop_wamv()
        
        self.publish_visual_path()
        # --- [추가된 부분 3] ---
        self.publish_dock_gate_marker()

    def navigate_to_waypoint(self):
        turn_val = self.kp_nav * self.error_psi + self.kd_nav * (self.error_psi - self.prev_error_nav)
        self.prev_error_nav = self.error_psi
        self.publish_thrust(self.base_thrust_nav - turn_val, self.base_thrust_nav + turn_val)

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

    def hold_for_scan(self):
        self.stop_wamv()
        elapsed = (self.get_clock().now() - self.docking_start_time).nanoseconds * 1e-9
        if elapsed >= 3.0:
            self.get_logger().info("Scan Complete -> DOCKING_MANEUVER")
            self.mode = "DOCKING_MANEUVER"

    def dock_with_triangle_and_move(self):
        base_speed = 0.0
        if self.min_forward_distance < self.LIDAR_CAUTION_DIST:
            speed_ratio = (self.min_forward_distance - self.LIDAR_STOP_DIST) / (self.LIDAR_CAUTION_DIST - self.LIDAR_STOP_DIST)
            base_speed = max(0.0, self.DOCKING_MAX_SPEED * speed_ratio)
        else:
            base_speed = self.DOCKING_MAX_SPEED

        if abs(self.triangle_error) < self.ALIGNMENT_THRESHOLD:
            self.stop_wamv()
            self.get_logger().info("Triangle Aligned. Stopping WAMV.")
            self.mode = "MISSION_COMPLETE"
            return
        
        turn_adjustment = self.Kp_visual * self.triangle_error + self.Kd_visual * (self.triangle_error - self.prev_triangle_error)
        self.prev_triangle_error = self.triangle_error
        
        left_thrust = base_speed + turn_adjustment
        right_thrust = base_speed - turn_adjustment
        self.publish_thrust(left_thrust, right_thrust)
        
    def stop_wamv(self): self.publish_thrust(0.0, 0.0)

    def publish_thrust(self, left, right):
        l_msg, r_msg = Float64(), Float64()
        l_msg.data, r_msg.data = float(left), float(right)
        self.left_thruster_pub.publish(l_msg)
        self.right_thruster_pub.publish(r_msg)

    def publish_visual_path(self):
        if self.min_forward_distance > 100 or self.circle_error == 0.0:
            return

        angle_offset = (self.circle_error / (self.image_width / 2)) * (self.camera_fov_rad / 2)
        
        marker = Marker()
        marker.header.frame_id = "wamv/wamv/base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "visual_path"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        start_point = Point(x=0.0, y=0.0, z=0.0)
        end_point = Point()
        end_point.x = self.min_forward_distance * math.cos(angle_offset)
        end_point.y = self.min_forward_distance * math.sin(angle_offset)
        end_point.z = 0.0
        
        marker.points.append(start_point)
        marker.points.append(end_point)
        
        marker.scale.x = 0.1
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        self.visual_path_pub.publish(marker)

    # --- [추가된 부분 4] ---
    def publish_dock_gate_marker(self):
        marker = Marker()
        marker.header.frame_id = "map" # 고정된 좌표계 사용
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "docking_gate"
        marker.id = 1 # 다른 마커와 ID가 겹치지 않게 설정
        marker.type = Marker.LINE_STRIP # 선 타입
        marker.action = Marker.ADD

        # 선의 두께 설정
        marker.scale.x = 0.3 # 라인 두께 (미터)

        # 선의 색상 설정 (파란색)
        marker.color.a = 1.0 # Alpha (투명도)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        # 선을 구성할 두 지점 생성
        p1 = Point()
        p1.x = self.dock_gate_A_utm[0]
        p1.y = self.dock_gate_A_utm[1]
        p1.z = 0.0

        p2 = Point()
        p2.x = self.dock_gate_B_utm[0]
        p2.y = self.dock_gate_B_utm[1]
        p2.z = 0.0

        marker.points.append(p1)
        marker.points.append(p2)

        self.dock_gate_pub.publish(marker)


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
