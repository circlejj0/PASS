#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [!!] 수정된 DockingCont (제어 노드)
# [!!] P1 회전 후 HOLDING_FOR_SCAN 상태 진입
# [!!] Navi로부터 확정된 도크 인덱스(0,1,2)를 수신

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

        # Sub
        self.create_subscription(Float64, '/error_psi', self.e_psi_callback, qos_profile)
        self.create_subscription(String, '/nav/status', self.status_callback, 10)
        self.create_subscription(Float64MultiArray, 'visual_errors', self.visual_errors_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/UTM_Latlot', self.utm_callback, qos_profile)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, qos_profile)
        self.create_subscription(LaserScan, '/wamv/sensors/lidar/front_lidar/scan', self.lidar_callback, 10)
        self.create_subscription(Float64, '/target_yaw', self.target_yaw_callback, 10)
        # --- [!!] 여기: Navi가 보낸 최종 인덱스를 구독 ---
        self.create_subscription(Float64, '/docking/target_index', self.index_callback, qos_profile)


        # Pub
        self.left_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.status_pub = self.create_publisher(String, '/nav/status', 10) # [!!] 회전 완료 신호 보낼 때 사용
        self.visual_path_pub = self.create_publisher(Marker, 'wamv_visual_path', 10)
        self.dock_gate_pub = self.create_publisher(Marker, 'dock_gate_marker', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.process)
        
        # 변수 설정
        # [!!] 상태 머신 수정
        self.mode = "WAYPOINT_FOLLOWING" 
        
        # [!!] 3개 도형 에러 변수
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
        
        # --- [!!] 여기: Navi로부터 받을 최종 도크 인덱스 저장 변수 ---
        self.target_dock_index = -1.0 
        
        # 제어 파라미터
        self.kp_nav, self.kd_nav, self.base_thrust_nav = 10.0, 5.0, 550.0
        self.kp_rot = 200.0
        self.LIDAR_CAUTION_DIST, self.LIDAR_STOP_DIST = 7.0, 3.5
        self.DOCKING_MAX_SPEED = 60.0
        self.Kp_visual = 1.5
        self.Kd_visual = 2.0
        self.ALIGNMENT_THRESHOLD = 2.0
        
        # [!!] Dock Gate 마커용 좌표 (임시)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")
        dock_gate_A_lonlat = (150.67452, -33.72248) 
        dock_gate_B_lonlat = (150.67454, -33.72242)
        self.dock_gate_A_utm = transformer.transform(dock_gate_A_lonlat[1], dock_gate_A_lonlat[0])
        self.dock_gate_B_utm = transformer.transform(dock_gate_B_lonlat[1], dock_gate_B_lonlat[0])


    # Callback Functions
    def e_psi_callback(self, msg): self.error_psi = msg.data
    def utm_callback(self, msg): self.wamv_x, self.wamv_y = msg.data[0], msg.data[1]
    def imu_callback(self, msg: Imu):
        q = msg.orientation
        self.current_yaw_rad = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))

    def target_yaw_callback(self, msg: Float64):
        self.get_logger().info(f"Received new target yaw: {math.degrees(msg.data):.1f} deg")
        self.target_yaw_from_navi = msg.data

    # --- [!!] 여기: 인덱스 수신 콜백 ---
    def index_callback(self, msg: Float64):
        self.target_dock_index = msg.data
        self.get_logger().info(f"Received target dock index: {self.target_dock_index}")

    # [!!] Navi로부터 오는 상태 변경 신호를 처리
    def status_callback(self, msg):
        self.get_logger().info(f"Received status from Navi: {msg.data}")
        if msg.data == "ARRIVED_P1" and self.mode == "WAYPOINT_FOLLOWING":
            self.mode = "ROTATING_P1"
            self.get_logger().info("MODE CHANGE: -> ROTATING_P1")
            self.target_yaw_rad_internal = None
            self.stop_wamv()
        
        # [!!] Guid(YOLO)가 활성화되는 신호 (Cont는 무시)
        elif msg.data == "ARRIVED_SCAN_P1": 
            pass # Cont는 이 신호에 반응할 필요 없음
        
        # [!!] Navi가 인덱스 판단을 마쳤다는 신호
        elif msg.data == "TARGET_ORDER_CONFIRMED" and self.mode == "HOLDING_FOR_SCAN":
            # [!!] 인덱스 변수가 잘 수신되었는지 한 번 더 확인
            self.get_logger().info(f"MODE CHANGE: Target order confirmed. Stored Index is {self.target_dock_index}.")
            
            # [!!] (다음 단계: GPS 이동)을 준비하는 새 상태로 변경
            self.mode = "READY_FOR_DOCK_NAV" 
            self.get_logger().info("MODE CHANGE: -> READY_FOR_DOCK_NAV")


    def lidar_callback(self, msg: LaserScan):
        center_index = len(msg.ranges) // 2
        view_range_idx = int(math.radians(10) / msg.angle_increment)
        start_idx, end_idx = max(0, center_index - view_range_idx), min(len(msg.ranges), center_index + view_range_idx)
        forward_ranges = msg.ranges[start_idx : end_idx]
        valid_ranges = [r for r in forward_ranges if r > msg.range_min and r < msg.range_max]
        self.min_forward_distance = min(valid_ranges) if valid_ranges else 999.0
        
    def visual_errors_callback(self, msg: Float64MultiArray):
        # [!!] 3개의 에러 데이터를 수신
        # [!!] Guid의 self.targets 순서와 일치
        # [!!] (['circle', 'triangle', 'rectangle'])
        if len(msg.data) >= 3:
            self.circle_error = msg.data[0]
            self.triangle_error = msg.data[1]
            self.rectangle_error = msg.data[2]
        else:
            self.circle_error = 0.0
            self.triangle_error = 0.0
            self.rectangle_error = 0.0

    # 주기 시작
    def process(self):
        # [!!] 상태 머신 수정
        if self.mode == "WAYPOINT_FOLLOWING": 
            self.navigate_to_waypoint()
            
        elif self.mode == "ROTATING_P1": 
            # [!!] 회전 완료 시, 다음 상태를 HOLDING_FOR_SCAN로 지정
            self.rotate_in_place(next_mode="HOLDING_FOR_SCAN", completion_status="ROTATION_P1_COMPLETE")
        
        elif self.mode == "HOLDING_FOR_SCAN":
            # [!!] Navi가 'TARGET_ORDER_CONFIRMED'를 보낼 때까지 정지 상태 유지
            self.stop_wamv()
            self.get_logger().info("Holding for scan... (waiting for Navi's confirmation)", throttle_duration_sec=5)
            # [!!] (status_callback에서 다음 상태로 변경됨)
            pass

        # --- [!!] 여기: 다음 단계(GPS 이동)를 대기하는 상태 ---
        elif self.mode == "READY_FOR_DOCK_NAV":
            # [!!] 사용자님의 3번 계획(GPS 이동) 코드가 여기에 들어갈 것임
            # [!!] 지금은 그 전까지 정지 상태 유지
            self.stop_wamv()
            self.get_logger().info(f"Ready for Dock Navigation (Index: {self.target_dock_index}). Holding...", throttle_duration_sec=5)
        
        elif self.mode == "DOCKING_MANEUVER":
            # [!!] (아직 사용 안함. 추후 '어떤 에러'를 쓸지 선택하는 로직 필요)
            self.dock_with_target_shape() 
            
        elif self.mode == "MISSION_COMPLETE": 
            self.stop_wamv()
            self.get_logger().info("Mission Complete. Holding position.", throttle_duration_sec=5)
        

        self.publish_visual_path()
        self.publish_dock_gate_marker()

    def navigate_to_waypoint(self):
        # error_psi는 Navi 노드에서 계산되어 /error_psi 토픽으로 수신됨
        turn_val = self.kp_nav * self.error_psi + self.kd_nav * (self.error_psi - self.prev_error_nav)
        self.prev_error_nav = self.error_psi
        self.publish_thrust(self.base_thrust_nav - turn_val, self.base_thrust_nav + turn_val)

    def rotate_in_place(self, next_mode, completion_status):
        if self.target_yaw_rad_internal is None:
            if self.target_yaw_from_navi is None:
                self.get_logger().warn("Waiting for target yaw from Navi...")
                self.stop_wamv()
                return
            # Navi로부터 받은 각도를 내부 목표 각도로 설정
            self.target_yaw_rad_internal = self.target_yaw_from_navi 
            
        error_yaw = self.target_yaw_rad_internal - self.current_yaw_rad
        error_yaw = (error_yaw + math.pi) % (2 * math.pi) - math.pi
        
        if abs(math.degrees(error_yaw)) < 3.0:
            self.get_logger().info(f"Rotation Complete. Publishing: {completion_status}")
            self.mode = next_mode
            self.status_pub.publish(String(data=completion_status)) # [!!] Navi에게 완료 신호 전송
            self.stop_wamv()
            
            # [!!] 다음 임무를 위해 리셋
            self.target_yaw_from_navi = None 
            self.target_yaw_rad_internal = None 
            return
            
        turn = self.kp_rot * error_yaw
        self.publish_thrust(-turn, turn)

    # [!!] (이 함수는 아직 사용되지 않음 - 기존 dock_with_triangle...에서 이름만 변경)
    def dock_with_target_shape(self):
        # (기존 코드와 동일 - self.triangle_error를 하드코딩하여 사용 중)
        # (추후 self.target_dock_index에 따라 사용할 에러 변수를 선택해야 함)
        
        base_speed = 0.0
        if self.min_forward_distance < self.LIDAR_CAUTION_DIST:
            speed_ratio = (self.min_forward_distance - self.LIDAR_STOP_DIST) / (self.LIDAR_CAUTION_DIST - self.LIDAR_STOP_DIST)
            base_speed = max(0.0, self.DOCKING_MAX_SPEED * speed_ratio)
        else:
            base_speed = self.DOCKING_MAX_SPEED

        # [!!] (주의: 지금은 하드코딩된 self.triangle_error를 따름)
        target_error = self.triangle_error 
        prev_target_error = self.prev_triangle_error
        
        if abs(target_error) < self.ALIGNMENT_THRESHOLD:
            self.stop_wamv()
            self.get_logger().info("Target Aligned. Stopping WAMV.")
            self.mode = "MISSION_COMPLETE"
            return
        
        turn_adjustment = self.Kp_visual * target_error + self.Kd_visual * (target_error - prev_target_error)
        self.prev_triangle_error = target_error # [!!] (이 부분도 수정 필요)
        
        left_thrust = base_speed + turn_adjustment
        right_thrust = base_speed - turn_adjustment
        self.publish_thrust(left_thrust, right_thrust)
            
    def stop_wamv(self): self.publish_thrust(0.0, 0.0)

    def publish_thrust(self, left, right):
        l_msg, r_msg = Float64(), Float64()
        l_msg.data, r_msg.data = float(left), float(right)
        self.left_thruster_pub.publish(l_msg)
        self.right_thruster_pub.publish(r_msg)

    # (시각화 함수 - 기존 코드와 동일)
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

    def publish_dock_gate_marker(self):
        marker = Marker()
        marker.header.frame_id = "map" # 고정된 좌표계 사용
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "docking_gate"
        marker.id = 1 # 다른 마커와 ID가 겹치지 않게 설정
        marker.type = Marker.LINE_STRIP # 선 타입
        marker.action = Marker.ADD

        marker.scale.x = 0.3 # 라인 두께 (미터)
        marker.color.a = 1.0 # Alpha (투명도)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0

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
