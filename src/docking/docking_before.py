#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 3-Node(Guid, Navi, Cont) 로직을 MAVROS 기반 단일 노드로 통합
# MAVROS의 /mavros/setpoint_velocity/cmd_vel_unstamped 토픽을 사용
# [!!] Ouster LIDAR (PointCloud2)를 사용하도록 수정

import rclpy
import math
import cv2
import os
import numpy as np
import transforms3d.euler as euler

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pyproj import Transformer
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
from cv_bridge import CvBridge

# MAVROS 메시지
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

# 기존 센서 메시지
from sensor_msgs.msg import NavSatFix, Imu, Image
from sensor_msgs.msg import PointCloud2 # [!!] PointCloud2 임포트
import sensor_msgs_py.point_cloud2 as pc2 # [!!] PointCloud2 파서 임포트
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker


class MavrosDockingController(Node):
    def __init__(self):
        super().__init__('mavros_docking_controller')

        # --- FSM 상태 정의 ---
        self.mission_status = "INITIALIZING" 
        # INITIALIZING -> ARMING -> NAV_TO_P1 -> ROTATING_P1 -> SCANNING_FOR_TARGETS ->
        # NAV_TO_DOCK -> ROTATING_DOCK_FACE -> ROTATING_VISUAL_ALIGN ->
        # FINAL_APPROACH -> HOLDING_AT_DOCK -> REVERSING -> MISSION_COMPLETE

        # --- MAVROS 및 센서 변수 ---
        self.current_state = None
        self.current_pose_gps = None
        self.current_yaw_rad = 0.0
        self.current_local = (0.0, 0.0) # 현재 로컬 UTM 좌표
        self.min_forward_distance = 999.0
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True)
        self.origin_utm = None # GPS 원점 (첫 번째 P1 지점)

        # --- 비전(Guid) 변수 ---
        self.bridge = CvBridge()
        self.visual_errors = {} # 예: {'circle': -120.5, 'triangle': 10.2, 'rectangle': 150.0}
        self.target_order_str = None # 예: "circle,rectangle,triangle"
        self.order_determined = False
        self.image_center_x = 0
        self.image_width = 0

        # --- 임무(Navi) 변수 ---
        self.target_index = -1 # 0, 1, 2 중 목표 도크 인덱스
        self.selected_dock_nav_local = None
        self.selected_dock_face_local = None
        self.p1_local = None
        self.p2_local = None
        self.dock_nav_points_local = []
        self.dock_face_points_local = []

        # --- 제어(Cont) 변수 ---
        self.prev_heading_error = 0.0
        self.prev_visual_error = 0.0
        self.target_yaw_rad_internal = None # 회전 상태용 임시 목표 Yaw
        self.align_success_timer = None
        self.pause_timer = None # 도킹 후 정지 및 후진용 타이머

        # ==================================================================
        # 1. 파라미터 선언 (Navi, Guid, Cont 파라미터 통합)
        # ==================================================================

        # --- Guid(비전) 파라미터 ---
        self.declare_parameter('camera_topic', '/flir_camera/image_raw')
        self.declare_parameter('show_vision', True)
        self.declare_parameter('targets', ['circle', 'square', 'triangle'])
        self.declare_parameter('yolo_model_pkg', 'docking_final') # [!] 패키지 이름
        self.declare_parameter('yolo_model_path', 'weights/docking.pt')

        # --- Navi(항법) 파라미터 ---
        self.declare_parameter('target_shape', 'circle') # [!] 최종 목표 도형
        self.declare_parameter('point_1_inspection_lonlat', [129.107267, 35.132566])
        self.declare_parameter('point_2_inspection_facing_lonlat', [129.107388, 35.132576])
        self.declare_parameter('dock_0_nav_lonlat', [129.107582, 35.132693])
        self.declare_parameter('dock_0_face_lonlat', [129.107667, 35.132726])
        self.declare_parameter('dock_1_nav_lonlat', [129.107502, 35.132588])
        self.declare_parameter('dock_1_face_lonlat', [129.107604, 35.132604])
        self.declare_parameter('dock_2_nav_lonlat', [129.107685, 35.132477])
        self.declare_parameter('dock_2_face_lonlat', [129.107789, 35.132511])
        
        # --- Cont(제어) 파라미터 ---
        self.declare_parameter('arrival_radius', 0.5) # m, 경유점 도착 반경
        self.declare_parameter('rotation_threshold_rad', math.radians(3.0)) # rad, 회전 완료 오차

        # 속도 (m/s)
        self.declare_parameter('nav_speed_ms', 1.0) # m/s, P1으로 갈 때 속도
        self.declare_parameter('nav_dock_speed_ms', 0.5) # m/s, 도크로 갈 때 속도
        self.declare_parameter('final_approach_speed_ms', 0.2) # m/s, 최종 접근 속도
        self.declare_parameter('reverse_speed_ms', 0.3) # m/s, 후진 속도

        # PD 게인 (항법) - 목표 각속도(rad/s) 생성용
        self.declare_parameter('kp_nav_yaw_rate', 1.5)
        self.declare_parameter('kd_nav_yaw_rate', 0.5)
        
        # P 게인 (GPS 회전) - 목표 각속도(rad/s) 생성용
        self.declare_parameter('kp_rot_yaw_rate', 1.0) 

        # PD 게인 (시각적 회전/접근) - 목표 각속도(rad/s) 생성용
        self.declare_parameter('kp_visual_yaw_rate', 0.005) # 픽셀 오차 -> rad/s
        self.declare_parameter('kd_visual_yaw_rate', 0.001)
        self.declare_parameter('visual_align_threshold_px', 15.0) # px
        self.declare_parameter('visual_align_hold_sec', 2.0) # sec

        # [!!] LIDAR 파라미터 변경
        self.declare_parameter('pointcloud_topic', '/ouster/points') # [!!] Ouster 토픽 예시
        self.declare_parameter('lidar_fov_deg', 10.0) # [!!] 정면 수평 탐지 각도 (좌우 5도)
        self.declare_parameter('lidar_z_min', -0.5)   # [!!] LIDAR 기준, 물 표면 반사 무시
        self.declare_parameter('lidar_z_max', 1.0)    # [!!] LIDAR 기준, 너무 높은 장애물 무시
        
        # 도킹 완료 파라미터
        self.declare_parameter('final_stop_distance_lidar', 4.0) # m
        self.declare_parameter('hold_duration_after_dock', 5.0) # sec
        self.declare_parameter('target_reverse_distance', 10.0) # m

        # ==================================================================
        # 2. 파라미터 로드 및 좌표 변환
        # ==================================================================
        
        gp = self.get_parameter
        self.target_shape = gp('target_shape').value.lower()
        self.targets_to_find = [t.lower() for t in gp('targets').value]
        if len(self.targets_to_find) != 3:
            self.get_logger().error("targets 파라미터는 3개여야 합니다. (예: ['circle', 'triangle', 'rectangle'])")
            return

        # --- 좌표 변환 (Lon/Lat -> Local UTM) ---
        p1_lonlat = gp('point_1_inspection_lonlat').value
        self.origin_utm = self.transformer.transform(p1_lonlat[0], p1_lonlat[1])
        
        self.p1_local = self.convert_gps_to_local(p1_lonlat[0], p1_lonlat[1])
        p2_lonlat = gp('point_2_inspection_facing_lonlat').value
        self.p2_local = self.convert_gps_to_local(p2_lonlat[0], p2_lonlat[1])

        for i in range(3):
            nav_lonlat = gp(f'dock_{i}_nav_lonlat').value
            face_lonlat = gp(f'dock_{i}_face_lonlat').value
            nav_local = self.convert_gps_to_local(nav_lonlat[0], nav_lonlat[1])
            face_local = self.convert_gps_to_local(face_lonlat[0], face_lonlat[1])
            self.dock_nav_points_local.append(nav_local)
            self.dock_face_points_local.append(face_local)

        self.get_logger().info(f"Origin (P1) set to UTM: {self.origin_utm}")
        self.get_logger().info(f"Target shape: {self.target_shape}")

        # --- 제어 게인 로드 ---
        self.arrival_radius = gp('arrival_radius').value
        self.rotation_threshold_rad = gp('rotation_threshold_rad').value
        self.nav_speed_ms = gp('nav_speed_ms').value
        self.nav_dock_speed_ms = gp('nav_dock_speed_ms').value
        self.kp_nav_yaw_rate = gp('kp_nav_yaw_rate').value
        self.kd_nav_yaw_rate = gp('kd_nav_yaw_rate').value
        self.kp_rot_yaw_rate = gp('kp_rot_yaw_rate').value
        self.kp_visual_yaw_rate = gp('kp_visual_yaw_rate').value
        self.kd_visual_yaw_rate = gp('kd_visual_yaw_rate').value
        self.visual_align_threshold_px = gp('visual_align_threshold_px').value
        self.visual_align_hold_sec = gp('visual_align_hold_sec').value
        self.final_approach_speed_ms = gp('final_approach_speed_ms').value
        self.reverse_speed_ms = gp('reverse_speed_ms').value
        self.final_stop_distance_lidar = gp('final_stop_distance_lidar').value
        self.hold_duration_after_dock = gp('hold_duration_after_dock').value
        self.target_reverse_distance = gp('target_reverse_distance').value
        self.show_vision = gp('show_vision').value
        
        # --- PointCloud 파라미터 로드 ---
        self.pc_topic = gp('pointcloud_topic').value
        self.pc_fov_rad = math.radians(gp('lidar_fov_deg').value / 2.0) # (절반 각도, rad)
        self.pc_z_min = gp('lidar_z_min').value
        self.pc_z_max = gp('lidar_z_max').value

        # --- YOLO 모델 로드 ---
        pkg_share = get_package_share_directory(gp('yolo_model_pkg').value)
        model_path = os.path.join(pkg_share, gp('yolo_model_path').value)
        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f"YOLO model loaded from: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            return

        # ==================================================================
        # 3. MAVROS 및 센서 Sub/Pub/Client 설정
        # ==================================================================

        # --- QoS ---
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- MAVROS Subscribers ---
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, qos_best_effort)
        self.gps_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, qos_best_effort)
        self.imu_sub = self.create_subscription(Imu, '/mavros/imu/data', self.imu_callback, qos_best_effort)
        
        # --- 센서 Subscribers ---
        self.image_sub = self.create_subscription(Image, gp('camera_topic').value, self.image_cb, qos_best_effort)
        self.pc_sub = self.create_subscription(
            PointCloud2, 
            self.pc_topic, 
            self.pointcloud_callback, 
            qos_best_effort # Ouster는 데이터 양이 많으므로 BEST_EFFORT
        )

        # --- MAVROS Publisher ---
        self.velocity_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.status_pub_for_debug = self.create_publisher(String, '~/mission_status', qos_reliable) # 디버깅용
        self.marker_pub = self.create_publisher(Marker, '~/waypoints_viz', qos_reliable) # 디버깅용

        # --- MAVROS Clients ---
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # --- 메인 제어 루프 ---
        self.control_timer = self.create_timer(0.1, self.control_loop) # 10Hz

        self.get_logger().info('MavrosDockingController node initialized. Waiting for MAVROS connection...')


    # ==================================================================
    # 4. 콜백 함수 (센서 데이터 수신)
    # ==================================================================

    def state_callback(self, msg):
        self.current_state = msg

    def gps_callback(self, msg):
        self.current_pose_gps = msg
        if self.origin_utm is None:
            return
        # GPS 수신 시 즉시 로컬 좌표로 변환
        self.current_local = self.convert_gps_to_local(msg.longitude, msg.latitude)

    def imu_callback(self, msg):
        q = msg.orientation
        _, _, self.current_yaw_rad = euler.quat2euler([q.w, q.x, q.y, q.z])

    def pointcloud_callback(self, msg: PointCloud2):
        """
        Ouster LIDAR (PointCloud2) 데이터를 처리하여 정면의 최소 거리를 계산합니다.
        LIDAR가 +X 전방, +Y 좌측, +Z 상단을 향하도록 장착되었다고 가정합니다.
        """
        min_x_distance = 999.0
        
        try:
            # 'x', 'y', 'z' 필드만 읽어옵니다.
            # [!] Ouster 포인트 클라우드가 'x', 'y', 'z' 필드를 가지고 있는지 Rviz로 확인해야 합니다.
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = p[0], p[1], p[2]

                # 1. Z (높이) 필터링: 너무 높거나 낮은(물 반사) 지점 무시
                if z < self.pc_z_min or z > self.pc_z_max:
                    continue
                
                # 2. X (전방 거리) 필터링: 후방 지점 무시
                if x <= 0:
                    continue

                # 3. Y (수평) 필터링: 정면 FOV(예: 10도) 내의 지점만 선택
                #    math.atan2(y, x)로 수평 각도 계산
                angle = math.atan2(y, x)
                if abs(angle) < self.pc_fov_rad:
                    # 모든 필터를 통과한 지점 중 가장 가까운 '전방(x)' 거리 탐색
                    if x < min_x_distance:
                        min_x_distance = x
                        
        except Exception as e:
            # PointCloud2 파싱 중 오류 발생 (예: 데이터 형식 불일치)
            self.get_logger().warn(f"Failed to read PointCloud2 data: {e}", throttle_duration_sec=5)
            return

        # 최종 계산된 최소 전방 거리를 업데이트
        self.min_forward_distance = min_x_distance

    def image_cb(self, msg: Image):
        # 비전 처리가 필요한 상태가 아니면 CPU 절약을 위해 반환
        if self.mission_status not in ["SCANNING_FOR_TARGETS", "ROTATING_VISUAL_ALIGN", "FINAL_APPROACH"]:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w, _ = frame.shape
            self.image_center_x = w // 2
            self.image_width = w
        except Exception as e:
            self.get_logger().warn(f"CV Bridge error: {e}")
            return

        results = self.model(frame, verbose=False)

        # 에러 및 중심점 딕셔너리 초기화
        current_errors = {target: 0.0 for target in self.targets_to_find}
        target_centers = {} # (cx, cy) 저장용

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls_name = self.model.names[cls_id].lower()
                conf = float(box.conf.item())

                if cls_name in self.targets_to_find and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # 픽셀 에러 계산 (중심점으로부터의 거리)
                    current_errors[cls_name] = float(cx - self.image_center_x)
                    target_centers[cls_name] = (cx, cy) # 중심 좌표 저장
                    
                    if self.show_vision:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"{cls_name} ({current_errors[cls_name]:.1f}px)", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # self.visual_errors를 원자적(atomic)으로 업데이트 (제어 루프와의 충돌 방지)
        self.visual_errors = current_errors

        # --- [중요] 타겟 순서 판단 로직 (SCANNING 상태에서만 1회 실행) ---
        if self.mission_status == "SCANNING_FOR_TARGETS" and not self.order_determined:
            # 1. 3개의 타겟이 모두 감지되었는지 확인
            all_found = all(t in target_centers for t in self.targets_to_find)

            if all_found:
                self.order_determined = True # 래치(latch) 잠금!
                
                # 2. target_centers 딕셔너리를 x좌표(cx) 기준으로 정렬
                sorted_targets = sorted(target_centers.items(), key=lambda item: item[1][0])
                
                # 3. 정렬된 순서대로 이름만 추출
                ordered_names = [name for name, center in sorted_targets]
                self.target_order_str = ",".join(ordered_names)
                
                self.get_logger().info(f"--- [!!] All targets detected! Order: {self.target_order_str} [!!] ---")

                # 4. 내 목표 타겟이 몇 번째 인덱스인지 찾기
                if self.target_shape in ordered_names:
                    self.target_index = ordered_names.index(self.target_shape)
                    self.selected_dock_nav_local = self.dock_nav_points_local[self.target_index]
                    self.selected_dock_face_local = self.dock_face_points_local[self.target_index]

                    self.get_logger().info(f"--- [!!] Target '{self.target_shape}' found at index {self.target_index} [!!] ---")
                    
                    # 5. FSM 상태 변경!
                    self.mission_status = "NAV_TO_DOCK"
                    self.prev_heading_error = 0.0 # PD 제어기 리셋
                else:
                    self.get_logger().warn(f"Target shape '{self.target_shape}' not found in detected order: {self.target_order_str}. Rescanning.")
                    self.order_determined = False # 래치 해제, 다시 스캔
        
        if self.show_vision:
            cv2.line(frame, (self.image_center_x, 0), (self.image_center_x, h), (0, 255, 255), 1)
            cv2.imshow("YOLO Detection", frame)
            cv2.waitKey(1)

    # ==================================================================
    # 5. MAVROS 헬퍼 함수
    # ==================================================================

    def arm_and_set_mode(self):
        # [!] 서비스가 준비되었는지 확인 (중요)
        if not self.arming_client.wait_for_service(timeout_sec=1.0) or \
           not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("MAVROS services not available. Retrying...")
            return False

        # Arming 요청
        arm_req = CommandBool.Request()
        arm_req.value = True
        self.arming_client.call_async(arm_req)
        self.get_logger().info("Arming command sent...")

        # GUIDED 모드 설정 요청
        mode_req = SetMode.Request()
        mode_req.custom_mode = 'GUIDED' # ArduRover/ArduSub의 경우 'GUIDED'
        self.set_mode_client.call_async(mode_req)
        self.get_logger().info("Set mode to GUIDED command sent...")
        return True

    def publish_velocity(self, velocity_x, yaw_rate_rad_s):
        """Twist 메시지를 /mavros/setpoint_velocity/cmd_vel_unstamped에 발행"""
        vel_msg = Twist()
        vel_msg.linear.x = velocity_x       # 전진 속도 (m/s)
        vel_msg.angular.z = yaw_rate_rad_s # 각속도 (rad/s)
        self.velocity_pub.publish(vel_msg)

    def convert_gps_to_local(self, lon, lat):
        """GPS 좌표를 (0,0) 원점 기준 로컬 좌표로 변환"""
        if self.origin_utm is None:
            self.get_logger().warn("Origin UTM is not set. Cannot convert GPS.")
            return (0.0, 0.0)
        
        utm_x, utm_y = self.transformer.transform(lon, lat)
        return (utm_x - self.origin_utm[0], utm_y - self.origin_utm[1])


    # ==================================================================
    # 6. 제어 로직 헬퍼 함수
    # ==================================================================

    def execute_pd_heading_control(self, goal_local, speed_ms):
        """목표 지점을 향한 PD '방향' 제어 및 '고정' 속도 전진"""
        if goal_local is None:
            self.publish_velocity(0.0, 0.0)
            return

        distance_to_goal = math.sqrt((goal_local[0] - self.current_local[0])**2 + (goal_local[1] - self.current_local[1])**2)
        
        # 목표 방위각 (Desired Heading)
        desired_heading_rad = math.atan2(goal_local[1] - self.current_local[1], goal_local[0] - self.current_local[0])
        
        # 방위각 오차 (-pi ~ +pi)
        heading_error = desired_heading_rad - self.current_yaw_rad
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi

        # PD 제어 (목표 각속도 계산)
        heading_error_derivative = heading_error - self.prev_heading_error
        turn_rate_rad_s = self.kp_nav_yaw_rate * heading_error + self.kd_nav_yaw_rate * heading_error_derivative
        self.prev_heading_error = heading_error
        
        self.publish_velocity(speed_ms, turn_rate_rad_s)
        
        # 디버깅
        self.get_logger().info(f"Navigating... Dist: {distance_to_goal:.1f}m, Err: {math.degrees(heading_error):.1f}deg, YawRate: {turn_rate_rad_s:.2f}rad/s", throttle_duration_sec=1)
        self.publish_waypoint_marker(goal_local, (0.0, 1.0, 0.0)) # 녹색


    def execute_rotation(self, target_yaw_rad, next_status):
        """지정된 Yaw로 제자리 회전 (P 제어)"""
        if target_yaw_rad is None:
            self.get_logger().warn("Rotation target yaw is not set!")
            self.publish_velocity(0.0, 0.0)
            return

        # Yaw 오차 계산 (-pi ~ +pi)
        error_yaw = target_yaw_rad - self.current_yaw_rad
        error_yaw = (error_yaw + math.pi) % (2 * math.pi) - math.pi

        # 목표 도달 확인
        if abs(error_yaw) < self.rotation_threshold_rad:
            self.get_logger().info(f"Rotation Complete. Error: {math.degrees(error_yaw):.1f} deg. -> Changing status to {next_status}")
            self.mission_status = next_status
            self.publish_velocity(0.0, 0.0)
            self.target_yaw_rad_internal = None # 임시 목표 Yaw 리셋
            self.prev_heading_error = 0.0       # PD 리셋
            self.prev_visual_error = 0.0        # PD 리셋
            self.align_success_timer = None     # 타이머 리셋
            return
        
        # P 제어 (목표 각속도 계산)
        turn_rate = self.kp_rot_yaw_rate * error_yaw
        
        self.publish_velocity(0.0, turn_rate)
        self.get_logger().info(f"Rotating... Target: {math.degrees(target_yaw_rad):.1f}, Current: {math.degrees(self.current_yaw_rad):.1f}, Error: {math.degrees(error_yaw):.1f} deg", throttle_duration_sec=1)


    # ==================================================================
    # 7. 메인 제어 루프 (FSM)
    # ==================================================================

    def control_loop(self):
        # 0. 디버깅용 상태 발행
        self.status_pub_for_debug.publish(String(data=self.mission_status))

        # 1. INITIALIZING: MAVROS 연결 및 센서 대기
        if self.mission_status == "INITIALIZING":
            if self.current_state is None or self.current_pose_gps is None:
                self.get_logger().info("Waiting for MAVROS connection and GPS/IMU...", throttle_duration_sec=5)
                return
            
            if not self.current_state.connected:
                self.get_logger().warn("MAVROS not connected to FCU. Retrying...", throttle_duration_sec=5)
                return
            
            self.get_logger().info("Connection established. Attempting to arm and set mode.")
            if self.arm_and_set_mode():
                self.mission_status = "ARMING"
            return

        # 2. ARMING: Arming 및 GUIDED 모드 대기
        elif self.mission_status == "ARMING":
            if self.current_state.armed and self.current_state.mode == 'GUIDED':
                self.get_logger().info("Vehicle armed and in GUIDED mode. Starting mission.")
                self.mission_status = "NAV_TO_P1"
                self.prev_heading_error = 0.0 # PD 제어기 리셋
            else:
                self.get_logger().info(f"Waiting for arm/guided mode. Current mode: {self.current_state.mode}, Armed: {self.current_state.armed}", throttle_duration_sec=2)
                self.arm_and_set_mode() # 계속 요청
            return

        # --- 모든 센서 데이터가 준비되었는지 확인 ---
        if self.current_local is None or self.current_yaw_rad is None:
            self.get_logger().warn("Waiting for sensor data...", throttle_duration_sec=5)
            self.publish_velocity(0.0, 0.0) # 안전 정지
            return

        # 3. NAV_TO_P1: P1(스캔 지점)으로 항해
        elif self.mission_status == "NAV_TO_P1":
            goal_local = self.p1_local
            distance_to_goal = math.sqrt((goal_local[0] - self.current_local[0])**2 + (goal_local[1] - self.current_local[1])**2)
            
            if distance_to_goal < self.arrival_radius:
                self.get_logger().info("--- Arrived at P1 (Scan Point) ---")
                # P2(바라볼 지점)을 향한 목표 Yaw 계산
                self.target_yaw_rad_internal = math.atan2(self.p2_local[1] - self.current_local[1], self.p2_local[0] - self.current_local[0])
                self.mission_status = "ROTATING_P1"
                self.publish_velocity(0.0, 0.0)
                return
            
            self.execute_pd_heading_control(goal_local, self.nav_speed_ms)

        # 4. ROTATING_P1: P2를 바라보도록 1차 회전 (GPS 기반)
        elif self.mission_status == "ROTATING_P1":
            self.execute_rotation(self.target_yaw_rad_internal, next_status="SCANNING_FOR_TARGETS")
            self.publish_waypoint_marker(self.p2_local, (1.0, 0.0, 0.0)) # 빨간색

        # 5. SCANNING_FOR_TARGETS: 정지 상태로 YOLO 순서 판단 대기
        elif self.mission_status == "SCANNING_FOR_TARGETS":
            self.publish_velocity(0.0, 0.0) # 제자리 정지
            self.get_logger().info("Scanning for targets... (Waiting for image_cb to find order)", throttle_duration_sec=5)
            # [!] 실제 상태 변경은 image_cb 콜백에서 비동기적으로 발생함
            
        # 6. NAV_TO_DOCK: 선택된 도크 (NAV 지점)로 항해
        elif self.mission_status == "NAV_TO_DOCK":
            if self.selected_dock_nav_local is None:
                self.get_logger().error("NAV_TO_DOCK state but no target selected! Reverting to SCAN.")
                self.mission_status = "SCANNING_FOR_TARGETS"
                self.order_determined = False # 스캔 재시도
                return

            goal_local = self.selected_dock_nav_local
            distance_to_goal = math.sqrt((goal_local[0] - self.current_local[0])**2 + (goal_local[1] - self.current_local[1])**2)
            
            if distance_to_goal < self.arrival_radius:
                self.get_logger().info(f"--- Arrived at Dock {self.target_index} (Nav Point) ---")
                # 도크 정면(FACE 지점)을 향한 목표 Yaw 계산
                self.target_yaw_rad_internal = math.atan2(self.selected_dock_face_local[1] - self.current_local[1], self.selected_dock_face_local[0] - self.current_local[0])
                self.mission_status = "ROTATING_DOCK_FACE"
                self.publish_velocity(0.0, 0.0)
                return
            
            self.execute_pd_heading_control(goal_local, self.nav_dock_speed_ms) # 더 느린 속도

        # 7. ROTATING_DOCK_FACE: 도크 정면을 바라보도록 2차 회전 (GPS 기반)
        elif self.mission_status == "ROTATING_DOCK_FACE":
            self.execute_rotation(self.target_yaw_rad_internal, next_status="ROTATING_VISUAL_ALIGN")
            self.publish_waypoint_marker(self.selected_dock_face_local, (1.0, 0.0, 0.0)) # 빨간색

        # 8. ROTATING_VISUAL_ALIGN: 3차 회전 (시각적 정렬)
        elif self.mission_status == "ROTATING_VISUAL_ALIGN":
            target_pixel_error = self.visual_errors.get(self.target_shape, 0.0)
            
            if target_pixel_error == 0.0:
                self.get_logger().warn(f"Visual align mode, but target '{self.target_shape}' not detected! Stopping.", throttle_duration_sec=2)
                self.publish_velocity(0.0, 0.0)
                self.align_success_timer = None # 타이머 리셋
                self.prev_visual_error = 0.0    # D제어 리셋
                return

            # N초간 정렬 유지 조건
            if abs(target_pixel_error) < self.visual_align_threshold_px:
                if self.align_success_timer is None:
                    self.get_logger().info(f"Target aligned. Holding for {self.visual_align_hold_sec} sec...")
                    self.align_success_timer = self.get_clock().now()
                
                duration = (self.get_clock().now() - self.align_success_timer).nanoseconds / 1e9
                
                if duration >= self.visual_align_hold_sec:
                    self.get_logger().info(f"--- Visual alignment complete (Held for {duration:.1f} sec) ---")
                    self.publish_velocity(0.0, 0.0)
                    self.mission_status = "FINAL_APPROACH"
                    self.align_success_timer = None
                    self.prev_visual_error = 0.0 # D제어 리셋
                    return
                else:
                    self.publish_velocity(0.0, 0.0) # 정렬 유지 (정지)
                    self.get_logger().info(f"Holding alignment... ({duration:.1f} / {self.visual_align_hold_sec} sec)", throttle_duration_sec=1)
                    self.prev_visual_error = target_pixel_error # D항 초기화
                    return
            
            # 정렬이 틀어졌으면 타이머 리셋
            if self.align_success_timer is not None:
                self.get_logger().info("Alignment lost! Resuming PD-control.")
                self.align_success_timer = None

            # 시각적 PD 제어 (픽셀 에러 -> 각속도)
            error_derivative = target_pixel_error - self.prev_visual_error
            p_turn = self.kp_visual_yaw_rate * target_pixel_error
            d_turn = self.kd_visual_yaw_rate * error_derivative
            yaw_rate = -(p_turn + d_turn) # [!] 픽셀이 + (오른쪽)이면, - (시계방향) 회전 필요
            self.prev_visual_error = target_pixel_error
            
            self.publish_velocity(0.0, yaw_rate)
            self.get_logger().info(f"Visual Aligning... PxError: {target_pixel_error:.1f}, YawRate: {yaw_rate:.2f}rad/s", throttle_duration_sec=1)

        # 9. FINAL_APPROACH: 최종 접근 (LIDAR + 시각)
        elif self.mission_status == "FINAL_APPROACH":
            # 1. LIDAR 정지 조건 (최우선)
            # [!] self.min_forward_distance는 pointcloud_callback이 채워줌
            if self.min_forward_distance < self.final_stop_distance_lidar:
                self.get_logger().info(f"--- LIDAR Stop! Distance: {self.min_forward_distance:.2f}m. Docking complete. ---")
                self.publish_velocity(0.0, 0.0)
                self.mission_status = "HOLDING_AT_DOCK"
                self.prev_visual_error = 0.0
                # N초 정지 타이머 시작
                self.pause_timer = self.create_timer(self.hold_duration_after_dock, self.resume_after_hold)
                return

            # 2. 시각적 조향
            target_pixel_error = self.visual_errors.get(self.target_shape, 0.0)
            yaw_rate = 0.0
            
            if target_pixel_error != 0.0:
                # 타겟 감지됨: PD 조향
                error_derivative = target_pixel_error - self.prev_visual_error
                p_turn = self.kp_visual_yaw_rate * target_pixel_error
                d_turn = self.kd_visual_yaw_rate * error_derivative
                yaw_rate = -(p_turn + d_turn)
                self.prev_visual_error = target_pixel_error
                self.get_logger().info(f"Final Approach... Steering. PxError: {target_pixel_error:.1f}, Lidar: {self.min_forward_distance:.2f}m", throttle_duration_sec=1)
            else:
                # 타겟 손실: 조향 없이 직진
                self.get_logger().warn(f"Target lost during final approach! Continuing blind straight (Lidar: {self.min_forward_distance:.2f}m)...", throttle_duration_sec=1)
                self.prev_visual_error = 0.0 # D제어 리셋

            # 3. 전진 속도 및 조향 각속도 발행
            self.publish_velocity(self.final_approach_speed_ms, yaw_rate)

        # 10. HOLDING_AT_DOCK: 도킹 후 N초 정지
        elif self.mission_status == "HOLDING_AT_DOCK":
            self.publish_velocity(0.0, 0.0)
            self.get_logger().info(f"Docked. Holding position for {self.hold_duration_after_dock} sec...", throttle_duration_sec=2)
            # [!] 상태 변경은 self.pause_timer의 콜백(resume_after_hold)에서 발생
        
        # 11. REVERSING: N미터 후진
        elif self.mission_status == "REVERSING":
            # [!] self.min_forward_distance는 pointcloud_callback이 채워줌
            if self.min_forward_distance >= self.target_reverse_distance:
                self.get_logger().info(f"--- Reverse complete. Distance: {self.min_forward_distance:.2f}m. ---")
                self.publish_velocity(0.0, 0.0)
                self.mission_status = "MISSION_COMPLETE"
                return

            self.publish_velocity(-self.reverse_speed_ms, 0.0) # [!] 음수 속도
            self.get_logger().info(f"Reversing... (Dist: {self.min_forward_distance:.2f}m / {self.target_reverse_distance}m)", throttle_duration_sec=1)
        
        # 12. MISSION_COMPLETE: 임무 완료
        elif self.mission_status == "MISSION_COMPLETE":
            self.publish_velocity(0.0, 0.0)
            self.get_logger().info("Mission Complete. Holding position.", throttle_duration_sec=5)

    def resume_after_hold(self):
        """5초 정지 타이머 콜백"""
        self.get_logger().info("Hold complete. Starting reverse maneuver.")
        self.pause_timer.cancel()
        self.pause_timer = None
        self.mission_status = "REVERSING"
        
    def publish_waypoint_marker(self, goal_local, color_rgb):
        marker = Marker()
        marker.header.frame_id = "map" # [!] 'map' 프레임 사용 (origin_utm 기준)
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type, marker.action = "waypoints", 0, Marker.SPHERE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y = goal_local[0], goal_local[1]
        marker.scale.x, marker.scale.y, marker.scale.z = 2.0, 2.0, 2.0
        marker.color.a = 0.8
        marker.color.r, marker.color.g, marker.color.b = color_rgb
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = MavrosDockingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nNode interrupted by user.")
    finally:
        node.get_logger().info("Shutting down. Sending stop command.")
        node.publish_velocity(0.0, 0.0) # 비상 정지
        if node.show_vision:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
