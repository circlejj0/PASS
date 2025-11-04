#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [!!] 수정된 DockingCont (제어 노드)
# [!!] 3단계 도크 접근 속도 파라미터 분리

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64, Float64MultiArray, String
from sensor_msgs.msg import Imu, LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math
from pyproj import Transformer

class DockingCont(Node):
    def __init__(self):
        super().__init__('docking_cont')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Sub (기존과 동일)
        self.create_subscription(Float64, '/error_psi', self.e_psi_callback, qos_profile)
        self.create_subscription(String, '/nav/status', self.status_callback, 10)
        self.create_subscription(Float64MultiArray, 'visual_errors', self.visual_errors_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/UTM_Latlot', self.utm_callback, qos_profile)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, qos_profile)
        self.create_subscription(LaserScan, '/wamv/sensors/lidar/front_lidar/scan', self.lidar_callback, 10)
        self.create_subscription(Float64, '/target_yaw', self.target_yaw_callback, 10)
        self.create_subscription(Float64, '/docking/target_index', self.index_callback, qos_profile)

        # Pub (기존과 동일)
        self.left_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.status_pub = self.create_publisher(String, '/nav/status', 10) 
        self.visual_path_pub = self.create_publisher(Marker, 'wamv_visual_path', 10)
        self.dock_gate_pub = self.create_publisher(Marker, 'dock_gate_marker', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.process)
        
        # 변수 설정
        self.mode = "WAYPOINT_FOLLOWING" 
        self.circle_error, self.triangle_error, self.rectangle_error = 0.0, 0.0, 0.0 
        self.prev_circle_error, self.prev_triangle_error, self.prev_rectangle_error = 0.0, 0.0, 0.0
        self.current_yaw_rad, self.wamv_x, self.wamv_y = 0.0, 0.0, 0.0
        self.min_forward_distance = 999.0
        self.target_yaw_from_navi = None 
        self.target_yaw_rad_internal = None 
        self.prev_error_nav = 0.0
        self.error_psi = 0.0
        self.image_width = 640.0
        self.camera_fov_rad = math.radians(60.0)
        self.target_dock_index = -1.0 
        
        # [!!] 제어 파라미터 (속도 분리)
        self.declare_parameter('kp_nav', 10.0)
        self.declare_parameter('kd_nav', 5.0)
        self.declare_parameter('base_thrust_nav', 550.0) # [!!] 1단계(P1) 이동 속도
        self.declare_parameter('base_thrust_dock_nav', 300.0) # [!!] 3단계(도크) 이동 속도 (느리게)
        
        self.kp_nav = self.get_parameter('kp_nav').value
        self.kd_nav = self.get_parameter('kd_nav').value
        self.base_thrust_nav = self.get_parameter('base_thrust_nav').value
        self.base_thrust_dock_nav = self.get_parameter('base_thrust_dock_nav').value # [!!] 새 변수
        
        self.declare_parameter('kp_rot', 200.0)
        self.kp_rot = self.get_parameter('kp_rot').value
        
        # (기타 파라미터 기존과 동일)
        self.LIDAR_CAUTION_DIST, self.LIDAR_STOP_DIST = 7.0, 3.5
        self.DOCKING_MAX_SPEED = 60.0
        self.Kp_visual = 1.5
        self.Kd_visual = 2.0
        self.ALIGNMENT_THRESHOLD = 2.0
        
        # Dock Gate 마커용 좌표 (기존과 동일)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")
        dock_gate_A_lonlat = (150.67452, -33.72248) 
        dock_gate_B_lonlat = (150.67454, -33.72242)
        self.dock_gate_A_utm = transformer.transform(dock_gate_A_lonlat[1], dock_gate_A_lonlat[0])
        self.dock_gate_B_utm = transformer.transform(dock_gate_B_lonlat[1], dock_gate_B_lonlat[0])


    # Callback Functions (e_psi_callback, utm_callback, imu_callback, target_yaw_callback, index_callback - 기존과 동일)
    def e_psi_callback(self, msg): self.error_psi = msg.data
    def utm_callback(self, msg): self.wamv_x, self.wamv_y = msg.data[0], msg.data[1]
    def imu_callback(self, msg: Imu):
        q = msg.orientation
        self.current_yaw_rad = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
    def target_yaw_callback(self, msg: Float64):
        self.get_logger().info(f"Received new target yaw: {math.degrees(msg.data):.1f} deg")
        self.target_yaw_from_navi = msg.data
    def index_callback(self, msg: Float64):
        self.target_dock_index = msg.data
        self.get_logger().info(f"Received target dock index: {self.target_dock_index}")


    # [!!] status_callback (상태 변경 로직 수정)
    def status_callback(self, msg):
        self.get_logger().info(f"Received status from Navi: {msg.data}")
        
        # 1단계 항해 완료
        if msg.data == "ARRIVED_P1" and self.mode == "WAYPOINT_FOLLOWING":
            self.mode = "ROTATING_P1"
            self.get_logger().info("MODE CHANGE: -> ROTATING_P1")
            self.target_yaw_rad_internal = None
            self.stop_wamv()
        
        # 2단계 스캔 시작 (Cont는 무시)
        elif msg.data == "ARRIVED_SCAN_P1": 
            pass 
        
        # 2단계 인덱스 확정 (3단계 항해 시작)
        elif msg.data == "TARGET_ORDER_CONFIRMED" and self.mode == "HOLDING_FOR_SCAN":
            self.get_logger().info(f"MODE CHANGE: Target confirmed (Index: {self.target_dock_index}). Starting Dock Navigation.")
            # [!!] 3단계를 위한 'NAV_TO_DOCK' 새 상태로 변경
            self.mode = "NAV_TO_DOCK" 
            self.get_logger().info("MODE CHANGE: -> NAV_TO_DOCK")
        
        # [!!] 3단계 항해 완료
        elif msg.data == "ARRIVED_DOCK_NAV" and self.mode == "NAV_TO_DOCK":
            self.get_logger().info(f"MODE CHANGE: Arrived at Dock Nav Point {self.target_dock_index}.")
            self.mode = "ROTATING_DOCK_FACE" # [!!] 3단계 회전 모드
            self.get_logger().info("MODE CHANGE: -> ROTATING_DOCK_FACE")
            self.target_yaw_rad_internal = None
            self.stop_wamv()

    # (lidar_callback, visual_errors_callback - 기존과 동일)
    def lidar_callback(self, msg: LaserScan):
        center_index = len(msg.ranges) // 2
        view_range_idx = int(math.radians(10) / msg.angle_increment)
        start_idx, end_idx = max(0, center_index - view_range_idx), min(len(msg.ranges), center_index + view_range_idx)
        forward_ranges = msg.ranges[start_idx : end_idx]
        valid_ranges = [r for r in forward_ranges if r > msg.range_min and r < msg.range_max]
        self.min_forward_distance = min(valid_ranges) if valid_ranges else 999.0
    def visual_errors_callback(self, msg: Float64MultiArray):
        if len(msg.data) >= 3:
            self.circle_error = msg.data[0]
            self.triangle_error = msg.data[1]
            self.rectangle_error = msg.data[2]
        else:
            self.circle_error = 0.0
            self.triangle_error = 0.0
            self.rectangle_error = 0.0


    # [!!] process (메인 상태 머신 수정)
    def process(self):
        
        # 1단계 항해 (P1)
        if self.mode == "WAYPOINT_FOLLOWING": 
            self.navigate_to_waypoint()
            
        # 1단계 회전 (P2)
        elif self.mode == "ROTATING_P1": 
            self.rotate_in_place(next_mode="HOLDING_FOR_SCAN", completion_status="ROTATION_P1_COMPLETE")
        
        # 2단계 스캔 대기
        elif self.mode == "HOLDING_FOR_SCAN":
            self.stop_wamv()
            self.get_logger().info("Holding for scan... (waiting for Navi's confirmation)", throttle_duration_sec=5)
            pass

        # [!!] 3단계 항해 (Dock Nav) - 새 상태
        elif self.mode == "NAV_TO_DOCK":
            self.navigate_to_dock() # [!!] 새 함수 호출

        # 3단계 회전 (Dock Face)
        elif self.mode == "ROTATING_DOCK_FACE":
            self.rotate_in_place(next_mode="READY_FOR_LIDAR_APPROACH", completion_status="ROTATION_DOCK_COMPLETE")

        # 4단계 LiDAR 접근 대기
        elif self.mode == "READY_FOR_LIDAR_APPROACH":
            self.stop_wamv()
            self.get_logger().info(f"Ready for Lidar Approach (Step 4). Index {self.target_dock_index}. Holding...", throttle_duration_sec=5)
        
        # (아직 사용 안함)
        elif self.mode == "DOCKING_MANEUVER":
            self.dock_with_target_shape() 
            
        elif self.mode == "MISSION_COMPLETE": 
            self.stop_wamv()
            self.get_logger().info("Mission Complete. Holding position.", throttle_duration_sec=5)
        
        self.publish_visual_path()
        self.publish_dock_gate_marker()

    # (navigate_to_waypoint - 기존과 동일. 1단계에서만 사용됨)
    def navigate_to_waypoint(self):
        turn_val = self.kp_nav * self.error_psi + self.kd_nav * (self.error_psi - self.prev_error_nav)
        self.prev_error_nav = self.error_psi
        self.publish_thrust(self.base_thrust_nav - turn_val, self.base_thrust_nav + turn_val)

    # [!!] 3단계 도크 접근을 위한 새 함수
    def navigate_to_dock(self):
        turn_val = self.kp_nav * self.error_psi + self.kd_nav * (self.error_psi - self.prev_error_nav)
        self.prev_error_nav = self.error_psi
        # [!!] 1단계 속도(base_thrust_nav) 대신 3단계 속도(base_thrust_dock_nav) 사용
        self.publish_thrust(self.base_thrust_dock_nav - turn_val, self.base_thrust_dock_nav + turn_val)

    # (rotate_in_place - 기존과 동일)
    def rotate_in_place(self, next_mode, completion_status):
        if self.target_yaw_rad_internal is None:
            if self.target_yaw_from_navi is None:
                self.get_logger().warn("Waiting for target yaw from Navi...")
                self.stop_wamv()
                return
            self.target_yaw_rad_internal = self.target_yaw_from_navi 
            
        error_yaw = self.target_yaw_rad_internal - self.current_yaw_rad
        error_yaw = (error_yaw + math.pi) % (2 * math.pi) - math.pi
        
        if abs(math.degrees(error_yaw)) < 3.0:
            self.get_logger().info(f"Rotation Complete. Publishing: {completion_status}")
            self.mode = next_mode
            self.status_pub.publish(String(data=completion_status))
            self.stop_wamv()
            self.target_yaw_from_navi = None 
            self.target_yaw_rad_internal = None 
            return
            
        turn = self.kp_rot * error_yaw
        self.publish_thrust(-turn, turn)

    # (dock_with_target_shape - 기존과 동일. 아직 사용 안함)
    def dock_with_target_shape(self):
        base_speed = 0.0
        if self.min_forward_distance < self.LIDAR_CAUTION_DIST:
            speed_ratio = (self.min_forward_distance - self.LIDAR_STOP_DIST) / (self.LIDAR_CAUTION_DIST - self.LIDAR_STOP_DIST)
            base_speed = max(0.0, self.DOCKING_MAX_SPEED * speed_ratio)
        else:
            base_speed = self.DOCKING_MAX_SPEED
            
        target_error = 0.0
        prev_target_error = 0.0
        
        # [!!] (참고: 이 로직은 나중에 4단계에서 완성해야 함)
        if self.target_dock_index == 0:
            target_error = self.circle_error
            prev_target_error = self.prev_circle_error
        elif self.target_dock_index == 1:
            target_error = self.triangle_error
            prev_target_error = self.prev_triangle_error
        elif self.target_dock_index == 2:
            target_error = self.rectangle_error
            prev_target_error = self.prev_rectangle_error

        if abs(target_error) < self.ALIGNMENT_THRESHOLD:
            self.stop_wamv()
            self.get_logger().info("Target Aligned. Stopping WAMV.")
            self.mode = "MISSION_COMPLETE"
            return
        
        turn_adjustment = self.Kp_visual * target_error + self.Kd_visual * (target_error - prev_target_error)
        
        if self.target_dock_index == 0: self.prev_circle_error = target_error
        elif self.target_dock_index == 1: self.prev_triangle_error = target_error
        elif self.target_dock_index == 2: self.prev_rectangle_error = target_error

        left_thrust = base_speed + turn_adjustment
        right_thrust = base_speed - turn_adjustment
        self.publish_thrust(left_thrust, right_thrust)
            
    # (stop_wamv, publish_thrust, publish_visual_path, publish_dock_gate_marker - 기존과 동일)
    def stop_wamv(self): self.publish_thrust(0.0, 0.0)
    def publish_thrust(self, left, right):
        l_msg, r_msg = Float64(), Float64()
        l_msg.data, r_msg.data = float(left), float(right)
        self.left_thruster_pub.publish(l_msg)
        self.right_thruster_pub.publish(r_msg)
    def publish_visual_path(self):
        if self.min_forward_distance > 100 or self.circle_error == 0.0: return
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
        marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
        self.visual_path_pub.publish(marker)
    def publish_dock_gate_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "docking_gate"
        marker.id = 1 
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.3
        marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0
        p1 = Point(); p1.x = self.dock_gate_A_utm[0]; p1.y = self.dock_gate_A_utm[1]; p1.z = 0.0
        p2 = Point(); p2.x = self.dock_gate_B_utm[0]; p2.y = self.dock_gate_B_utm[1]; p2.z = 0.0
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
