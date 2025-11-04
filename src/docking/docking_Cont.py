#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [!!] 수정된 DockingCont (제어 노드)
# [!!] 5초 정지 및 10m 후진 로직 추가

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

        # (Sub/Pub 선언부는 기존과 동일)
        self.create_subscription(Float64, '/error_psi', self.e_psi_callback, qos_profile)
        self.create_subscription(String, '/nav/status', self.status_callback, 10)
        self.create_subscription(Float64MultiArray, 'visual_errors', self.visual_errors_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/UTM_Latlot', self.utm_callback, qos_profile)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, qos_profile)
        self.create_subscription(LaserScan, '/wamv/sensors/lidars/lidar_wamv_sensor/scan', self.lidar_callback, 10)
        self.create_subscription(Float64, '/target_yaw', self.target_yaw_callback, 10)
        self.create_subscription(Float64, '/docking/target_index', self.index_callback, qos_profile)
        self.left_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.status_pub = self.create_publisher(String, '/nav/status', 10) 
        self.visual_path_pub = self.create_publisher(Marker, 'wamv_visual_path', 10)
        self.dock_gate_pub = self.create_publisher(Marker, 'dock_gate_marker', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.process)
        
        # (변수 설정)
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
        
        self.align_success_timer = None 
        self.prev_visual_error = 0.0
        
        # (파라미터 선언부)
        self.declare_parameter('kp_nav', 10.0)
        self.declare_parameter('kd_nav', 5.0)
        self.declare_parameter('base_thrust_nav', 550.0)
        self.declare_parameter('base_thrust_dock_nav', 200.0) 
        self.kp_nav = self.get_parameter('kp_nav').value
        self.kd_nav = self.get_parameter('kd_nav').value
        self.base_thrust_nav = self.get_parameter('base_thrust_nav').value
        self.base_thrust_dock_nav = self.get_parameter('base_thrust_dock_nav').value
        
        self.declare_parameter('kp_rot', 200.0) 
        self.kp_rot = self.get_parameter('kp_rot').value
        
        # 3-2단계 (시각 회전) 및 4단계 (최종 접근)용 PD 게인
        self.declare_parameter('kp_visual_rot', 0.1)
        self.declare_parameter('kd_visual_rot', 0.8) 
        self.declare_parameter('visual_align_threshold_px', 10.0)
        self.declare_parameter('max_visual_rot_thrust', 200.0) 
        
        # 4단계 (최종 접근) 파라미터
        self.declare_parameter('base_thrust_final_approach', 80.0) 
        self.declare_parameter('final_stop_distance_lidar', 4.0) # [!!] 4미터 앞에서 정지
        
        # N초 유지 파라미터
        self.declare_parameter('visual_align_hold_sec', 2.0)
        
        # [!!] --- 5, 6단계 (정지 및 후진) 파라미터 추가 ---
        self.declare_parameter('hold_duration_after_dock', 5.0) # 5초 정지
        self.declare_parameter('reverse_thrust', 150.0) # 후진 추력
        self.declare_parameter('target_reverse_distance', 10.0) # 10m 후진
        
        self.kp_visual_rot = self.get_parameter('kp_visual_rot').value
        self.kd_visual_rot = self.get_parameter('kd_visual_rot').value
        self.visual_align_threshold_px = self.get_parameter('visual_align_threshold_px').value
        self.max_visual_rot_thrust = self.get_parameter('max_visual_rot_thrust').value
        self.visual_align_hold_sec = self.get_parameter('visual_align_hold_sec').value
        self.base_thrust_final_approach = self.get_parameter('base_thrust_final_approach').value
        self.final_stop_distance_lidar = self.get_parameter('final_stop_distance_lidar').value
        
        # [!!] --- 새 파라미터 로드 ---
        self.hold_duration_after_dock = self.get_parameter('hold_duration_after_dock').value
        self.reverse_thrust = self.get_parameter('reverse_thrust').value
        self.target_reverse_distance = self.get_parameter('target_reverse_distance').value
        
        # (Dock Gate 마커 좌표는 기존과 동일)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")
        dock_gate_A_lonlat = (150.67452, -33.72248) 
        dock_gate_B_lonlat = (150.67454, -33.72242)
        self.dock_gate_A_utm = transformer.transform(dock_gate_A_lonlat[1], dock_gate_A_lonlat[0])
        self.dock_gate_B_utm = transformer.transform(dock_gate_B_lonlat[1], dock_gate_B_lonlat[0])


    # (Callback 함수들은 기존과 동일)
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
    def lidar_callback(self, msg: LaserScan):
        # (LiDAR 콜백은 기존과 동일)
        center_index = len(msg.ranges) // 2
        view_range_idx = int(math.radians(10) / msg.angle_increment)
        start_idx, end_idx = max(0, center_index - view_range_idx), min(len(msg.ranges), center_index + view_range_idx)
        forward_ranges = msg.ranges[start_idx : end_idx]
        valid_ranges = [r for r in forward_ranges if r > msg.range_min and r < msg.range_max]
        self.min_forward_distance = min(valid_ranges) if valid_ranges else 999.0
    def visual_errors_callback(self, msg: Float64MultiArray):
        # (visual_errors 콜백은 기존과 동일)
        if len(msg.data) >= 3:
            self.circle_error = msg.data[0]
            self.triangle_error = msg.data[1]
            self.rectangle_error = msg.data[2]
        else:
            self.circle_error = 0.0
            self.triangle_error = 0.0
            self.rectangle_error = 0.0

    # [!!] --- status_callback 수정 (새 상태 수신) ---
    def status_callback(self, msg):
        self.get_logger().info(f"Received status from Navi: {msg.data}")
        
        if msg.data == "ARRIVED_P1" and self.mode == "WAYPOINT_FOLLOWING":
            self.mode = "ROTATING_P1"
            self.get_logger().info("MODE CHANGE: -> ROTATING_P1")
            self.target_yaw_rad_internal = None
            self.stop_wamv()
        
        elif msg.data == "ARRIVED_SCAN_P1": 
            pass 
        
        elif msg.data == "TARGET_ORDER_CONFIRMED" and self.mode == "HOLDING_FOR_SCAN":
            self.get_logger().info(f"MODE CHANGE: Target confirmed (Index: {self.target_dock_index}). Starting Dock Navigation.")
            self.mode = "NAV_TO_DOCK" 
            self.get_logger().info("MODE CHANGE: -> NAV_TO_DOCK")
        
        elif msg.data == "ARRIVED_DOCK_NAV" and self.mode == "NAV_TO_DOCK":
            self.get_logger().info(f"MODE CHANGE: Arrived at Dock Nav Point {self.target_dock_index}.")
            self.mode = "ROTATING_DOCK_FACE" # 1차 (GPS) 회전 모드
            self.get_logger().info("MODE CHANGE: -> ROTATING_DOCK_FACE")
            self.target_yaw_rad_internal = None
            self.stop_wamv()
        
        # [!!] Navi가 5초 정지 또는 후진 상태로 변경되어도 Cont는 본인의 모드를 유지 (Cont가 Navi에게 신호를 보내기 때문)
        elif msg.data == "DOCKING_COMPLETE":
            pass
        elif msg.data == "START_REVERSING":
            pass
        elif msg.data == "REVERSE_COMPLETE":
            pass


    # [!!] --- process (메인 상태 머신 수정) ---
    def process(self):
        
        if self.mode == "WAYPOINT_FOLLOWING": 
            self.navigate_to_waypoint()
            
        elif self.mode == "ROTATING_P1": 
            self.rotate_in_place(next_mode="HOLDING_FOR_SCAN", completion_status="ROTATION_P1_COMPLETE")
        
        elif self.mode == "HOLDING_FOR_SCAN":
            self.stop_wamv()
            self.get_logger().info("Holding for scan... (waiting for Navi's confirmation)", throttle_duration_sec=5)
            pass

        elif self.mode == "NAV_TO_DOCK":
            self.navigate_to_dock()

        # 3-1단계: 1차 (GPS) 회전
        elif self.mode == "ROTATING_DOCK_FACE":
            self.rotate_in_place(
                next_mode="ROTATING_VISUAL_ALIGN", 
                completion_status="ROTATION_DOCK_COMPLETE"
            )

        # 3-2단계: 2차 (시각적) 회전
        elif self.mode == "ROTATING_VISUAL_ALIGN":
            self.rotate_for_visual_alignment(
                next_mode="READY_FOR_LIDAR_APPROACH",
                completion_status="ROTATION_VISUAL_COMPLETE"
            )

        # 4단계: 최종 접근 (YOLO+LIDAR 기반)
        elif self.mode == "READY_FOR_LIDAR_APPROACH":
            self.final_approach_with_visual_steering()
        
        # [!!] 5단계: 5초 정지
        elif self.mode == "HOLDING_5_SEC":
            self.hold_after_dock()

        # [!!] 6단계: 10m 후진
        elif self.mode == "REVERSING":
            self.reverse_from_dock()
            
        elif self.mode == "MISSION_COMPLETE": 
            self.stop_wamv()
            self.get_logger().info("Mission Complete. Holding position.", throttle_duration_sec=5)
        
        self.publish_visual_path()
        self.publish_dock_gate_marker()

    # (navigate_to_waypoint, navigate_to_dock 기존과 동일)
    def navigate_to_waypoint(self):
        turn_val = self.kp_nav * self.error_psi + self.kd_nav * (self.error_psi - self.prev_error_nav)
        self.prev_error_nav = self.error_psi
        self.publish_thrust(self.base_thrust_nav - turn_val, self.base_thrust_nav + turn_val)
    def navigate_to_dock(self):
        turn_val = self.kp_nav * self.error_psi + self.kd_nav * (self.error_psi - self.prev_error_nav)
        self.prev_error_nav = self.error_psi
        self.publish_thrust(self.base_thrust_dock_nav - turn_val, self.base_thrust_dock_nav + turn_val)

    # (rotate_in_place 기존과 동일 - D제어 변수 리셋 포함)
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
            self.get_logger().info(f"Rotation Complete (GPS). Publishing: {completion_status}")
            self.mode = next_mode
            self.status_pub.publish(String(data=completion_status))
            self.stop_wamv()
            self.target_yaw_from_navi = None 
            self.target_yaw_rad_internal = None 
            
            if next_mode == "ROTATING_VISUAL_ALIGN":
                self.align_success_timer = None 
                self.prev_visual_error = 0.0 
            return
            
        turn = self.kp_rot * error_yaw
        self.publish_thrust(-turn, turn)

    # (rotate_for_visual_alignment 기존과 동일 - PD제어 및 N초 유지)
    def rotate_for_visual_alignment(self, next_mode, completion_status):
        target_pixel_error = 0.0
        target_name = "None"
        
        if self.target_dock_index == 0.0:
            target_pixel_error = self.circle_error
            target_name = "circle"
        elif self.target_dock_index == 1.0:
            target_pixel_error = self.triangle_error
            target_name = "triangle"
        elif self.target_dock_index == 2.0:
            target_pixel_error = self.rectangle_error
            target_name = "rectangle"
        else:
            self.get_logger().warn("Visual align mode, but target index is unknown! Stopping.", throttle_duration_sec=5)
            self.stop_wamv()
            return

        if target_pixel_error == 0.0:
            self.get_logger().warn(f"Visual align mode, but target '{target_name}' (Index {self.target_dock_index}) is not detected! Stopping.", throttle_duration_sec=5)
            self.stop_wamv()
            self.align_success_timer = None 
            self.prev_visual_error = 0.0 
            return

        if abs(target_pixel_error) < self.visual_align_threshold_px:
            if self.align_success_timer is None:
                self.get_logger().info(f"Target aligned. Holding for {self.visual_align_hold_sec} sec...")
                self.align_success_timer = self.get_clock().now()
            
            duration = (self.get_clock().now() - self.align_success_timer).nanoseconds / 1e9
            
            if duration >= self.visual_align_hold_sec:
                self.get_logger().info(f"Visual alignment complete (Held for {duration:.1f} sec).")
                self.stop_wamv()
                self.mode = next_mode 
                self.status_pub.publish(String(data=completion_status))
                self.align_success_timer = None 
                self.prev_visual_error = 0.0 # D제어 리셋
                return
            else:
                self.stop_wamv()
                self.get_logger().info(f"Holding alignment... ({duration:.1f} / {self.visual_align_hold_sec} sec)", throttle_duration_sec=1)
                self.prev_visual_error = target_pixel_error 
                return 
            
        if self.align_success_timer is not None:
            self.get_logger().info("Alignment lost! Resuming PD-control.")
            self.align_success_timer = None

        error_derivative = target_pixel_error - self.prev_visual_error
        p_turn = self.kp_visual_rot * target_pixel_error
        d_turn = self.kd_visual_rot * error_derivative
        turn = -(p_turn + d_turn)
        self.prev_visual_error = target_pixel_error
        
        turn = max(min(turn, self.max_visual_rot_thrust), -self.max_visual_rot_thrust)
        
        self.publish_thrust(-turn, turn)
        self.get_logger().info(f"Visual Aligning... PxError: {target_pixel_error:.1f}, D_Error: {error_derivative:.1f}, Turn: {turn:.1f}", throttle_duration_sec=1)
    
    
    # [!!] --- 4단계 최종 접근 함수 (로직 수정) ---
    def final_approach_with_visual_steering(self):
        
        # [!!] --- 1. LIDAR 정지 조건 (가장 중요) ---
        if self.min_forward_distance < self.final_stop_distance_lidar:
            self.get_logger().info(f"LIDAR Stop! Distance: {self.min_forward_distance:.2f}m. Docking complete.")
            self.stop_wamv()
            # [!!] 5초 정지 모드로 변경
            self.mode = "HOLDING_5_SEC"
            # [!!] Navi에게 도킹 완료 신호 전송
            self.status_pub.publish(String(data="DOCKING_COMPLETE")) 
            self.prev_visual_error = 0.0 # 리셋
            self.align_success_timer = None # [!!] 5초 타이머를 위해 리셋
            return
        # [!!] -------------------------------------
            
        target_pixel_error = 0.0
        is_target_detected = False
        target_name = "None"
        
        # 2. 인덱스에 맞는 에러 값 선택 및 감지 확인
        if self.target_dock_index == 0.0 and self.circle_error != 0.0:
            target_pixel_error = self.circle_error
            is_target_detected = True
            target_name = "circle"
        elif self.target_dock_index == 1.0 and self.triangle_error != 0.0:
            target_pixel_error = self.triangle_error
            is_target_detected = True
            target_name = "triangle"
        elif self.target_dock_index == 2.0 and self.rectangle_error != 0.0:
            target_pixel_error = self.rectangle_error
            is_target_detected = True
            target_name = "rectangle"

        # 3. 타겟 감지 여부에 따라 행동 결정
        if is_target_detected:
            # [!!] 타겟 감지됨 -> PD 제어로 조향 + 느린 전진
            error_derivative = target_pixel_error - self.prev_visual_error
            p_turn = self.kp_visual_rot * target_pixel_error
            d_turn = self.kd_visual_rot * error_derivative
            turn = -(p_turn + d_turn)
            self.prev_visual_error = target_pixel_error
            
            turn = max(min(turn, self.max_visual_rot_thrust), -self.max_visual_rot_thrust)
            
            base_speed = self.base_thrust_final_approach
            left_thrust = base_speed - turn  
            right_thrust = base_speed + turn 
            
            self.publish_thrust(left_thrust, right_thrust)
            self.get_logger().info(f"Final Approach... Steering (Target: {target_name}, PxError: {target_pixel_error:.1f}, Lidar: {self.min_forward_distance:.2f}m)", throttle_duration_sec=1)
            
        else:
            # [!!] 타겟 손실 -> LIDAR가 멈출 때까지 조향 없이 직진
            self.get_logger().warn(f"Target lost during final approach! Continuing blind straight (Lidar: {self.min_forward_distance:.2f}m)...", throttle_duration_sec=1)
            base_speed = self.base_thrust_final_approach
            self.publish_thrust(base_speed, base_speed)
            self.prev_visual_error = 0.0 # D제어 리셋
            
    # [!!] ----------------------------------------------------

    # [!!] --- 이 함수를 새로 추가합니다 (5단계: 5초 정지) ---
    def hold_after_dock(self):
        # 5초 타이머
        if self.align_success_timer is None:
            self.get_logger().info("Docked. Starting 5 second hold...")
            self.align_success_timer = self.get_clock().now()
        
        duration = (self.get_clock().now() - self.align_success_timer).nanoseconds / 1e9
        
        if duration >= self.hold_duration_after_dock:
            # [!!] 5초 경과 -> 후진 모드로 변경
            self.get_logger().info(f"Hold complete ({duration:.1f} sec). Starting reverse.")
            self.mode = "REVERSING"
            self.status_pub.publish(String(data="START_REVERSING"))
            self.align_success_timer = None # 타이머 리셋
            return
        else:
            # 5초가 아직 안 지났으므로, 제자리 정지
            self.stop_wamv()
            self.get_logger().info(f"Holding docked position... ({duration:.1f} / {self.hold_duration_after_dock} sec)", throttle_duration_sec=1)

    # [!!] --- 이 함수를 새로 추가합니다 (6단계: 10m 후진) ---
    def reverse_from_dock(self):
        # [!!] LiDAR 거리가 10m에 도달했는지 확인 (전방 LiDAR 기준)
        if self.min_forward_distance >= self.target_reverse_distance:
            self.get_logger().info(f"Reverse complete. Distance: {self.min_forward_distance:.2f}m.")
            self.stop_wamv()
            self.mode = "MISSION_COMPLETE"
            self.status_pub.publish(String(data="REVERSE_COMPLETE"))
        else:
            # [!!] 후진 (음수 추력)
            thrust = -self.reverse_thrust 
            self.publish_thrust(thrust, thrust)
            self.get_logger().info(f"Reversing... (Dist: {self.min_forward_distance:.2f}m / {self.target_reverse_distance}m)", throttle_duration_sec=1)

            
    # (stop_wamv, publish_thrust, publish_visual_path, publish_dock_gate_marker는 기존과 동일)
    def stop_wamv(self): self.publish_thrust(0.0, 0.0)
    def publish_thrust(self, left, right):
        l_msg, r_msg = Float64(), Float64()
        l_msg.data, r_msg.data = float(left), float(right)
        self.left_thruster_pub.publish(l_msg)
        self.right_thruster_pub.publish(r_msg)
    def publish_visual_path(self):
        if self.min_forward_distance > 100 or self.circle_error == 0.0: return
        angle_offset = (self.circle_error / (self.image_width / 2)) * (self.camera_fov_rad / 2)
        marker = Marker(); marker.header.frame_id = "wamv/wamv/base_link"; marker.header.stamp = self.get_clock().now().to_msg(); marker.ns = "visual_path"; marker.id = 0; marker.type = Marker.ARROW; marker.action = Marker.ADD
        start_point = Point(x=0.0, y=0.0, z=0.0); end_point = Point(); end_point.x = self.min_forward_distance * math.cos(angle_offset); end_point.y = self.min_forward_distance * math.sin(angle_offset); end_point.z = 0.0
        marker.points.append(start_point); marker.points.append(end_point); marker.scale.x = 0.1; marker.scale.y = 0.5; marker.scale.z = 0.5; marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
        self.visual_path_pub.publish(marker)
    def publish_dock_gate_marker(self):
        marker = Marker(); marker.header.frame_id = "map"; marker.header.stamp = self.get_clock().now().to_msg(); marker.ns = "docking_gate"; marker.id = 1; marker.type = Marker.LINE_STRIP; marker.action = Marker.ADD; marker.scale.x = 0.3
        marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0
        p1 = Point(); p1.x = self.dock_gate_A_utm[0]; p1.y = self.dock_gate_A_utm[1]; p1.z = 0.0; p2 = Point(); p2.x = self.dock_gate_B_utm[0]; p2.y = self.dock_gate_B_utm[1]; p2.z = 0.0
        marker.points.append(p1); marker.points.append(p2)
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
