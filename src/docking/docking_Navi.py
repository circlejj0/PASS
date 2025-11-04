#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [!!] 수정된 DockingNavi (항법/지휘 노드)
# [!!] 'target_shape' 파라미터를 기준으로 인덱스(0,1,2)를 찾아 발행

import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pyproj import Transformer
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray, String

class DockingNavi(Node):
    def __init__(self):
        super().__init__('docking_navi')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.declare_parameter('point_1_inspection_lonlat', [150.67433227316755, -33.72264401144996])
        self.declare_parameter('point_2_inspection_facing_lonlat', [150.67439938281387, -33.722560796687716])
        self.declare_parameter('lookahead_distance', 5.0)
        self.declare_parameter('arrive_thr', 2.0)
        self.declare_parameter('los_kp', 1.0)
        
        # --- [!!] 여기: 원하는 타겟 도형을 파라미터로 선언 ---
        self.declare_parameter('target_shape', 'circle') # 예: 'circle', 'triangle', 'rectangle'
        
        # Sub
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, qos)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, qos)
        self.create_subscription(String, '/nav/status', self.status_callback, 10) # Cont 신호 수신
        # [!!] Guid로부터 순서 정보를 받기 위한 구독자
        self.create_subscription(String, '/docking/target_order', self.order_callback, 10)

        # Pub
        self.epsi_pub = self.create_publisher(Float64, '/error_psi', 10) 
        self.dist_pub = self.create_publisher(Float64, '/distance', 10)
        self.utm_pub = self.create_publisher(Float64MultiArray, '/UTM_Latlot', 10)
        self.yaw_pub = self.create_publisher(Float64, '/yaw', 10)
        self.status_pub = self.create_publisher(String, '/nav/status', 10) # Cont와 Guid에게 신호 전송
        self.target_yaw_pub = self.create_publisher(Float64, '/target_yaw', 10)
        # --- [!!] 여기: 찾은 인덱스를 Cont에게 보낼 발행자 ---
        self.index_pub = self.create_publisher(Float64, '/docking/target_index', qos)

        # Timer
        self.timer = self.create_timer(0.1, self.process)

        # 변수 설정
        self.gps_data = None
        self.latitude, self.longitude = 0.0, 0.0
        self.rad, self.degree = 0.0, 0.0
        self.base_x, self.base_y = 0.0, 0.0
        self.error_psi, self.distance = 0.0, 0.0
        
        # [!!] 상태 머신 확장: P1회전 -> 스캔 -> 다음준비
        self.mission_status = "NAV_TO_P1"
        self.target_order = None # [!!] Guid로부터 받을 순서 저장 변수
        # --- [!!] 여기: 파라미터에서 원하는 도형 이름 가져오기 ---
        self.target_shape = self.get_parameter('target_shape').value.lower()
        self.target_index = -1 # 찾은 인덱스 저장용
        self.get_logger().info(f"Main target shape to find: '{self.target_shape}'")

        # 좌표 변환
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")
        p1_lonlat = self.get_parameter('point_1_inspection_lonlat').value
        p2_lonlat = self.get_parameter('point_2_inspection_facing_lonlat').value
        self.point_1_utm = self.transformer.transform(p1_lonlat[1], p1_lonlat[0])
        self.point_2_utm = self.transformer.transform(p2_lonlat[1], p2_lonlat[0])
        
        self.get_logger().info(f"P1 (Nav): {self.point_1_utm}")
        self.get_logger().info(f"P2 (Rot): {self.point_2_utm}")

        # 제어 파라미터
        self.kp_los = self.get_parameter('los_kp').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.arrive_thr = self.get_parameter('arrive_thr').value

    # GPS Data Sub
    def gps_callback(self, msg):
        self.gps_data = msg
        self.latitude, self.longitude = msg.latitude, msg.longitude
        if self.gps_data:
            self.base_x, self.base_y = self.transformer.transform(self.latitude, self.longitude)
            self.utm_pub.publish(Float64MultiArray(data=[self.base_x, self.base_y]))

    # IMU Data Sub
    def imu_callback(self, msg):
        q = msg.orientation
        self.rad = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        self.degree = math.degrees(self.rad)
        self.yaw_pub.publish(Float64(data=self.rad))
        
    # [!!] Cont로부터 오는 상태 변경 신호를 처리
    def status_callback(self, msg: String):
        if msg.data == "ROTATION_P1_COMPLETE" and self.mission_status == "WAIT_FOR_ROTATION_1":
            self.get_logger().info("Navi: Cont R-P1 complete. Starting SCANNING.")
            # [!!] 상태를 스캔 중(SCANNING)으로 변경
            self.mission_status = "SCANNING_FOR_TARGETS"
            # [!!] Guid 노드 활성화 신호 전송!
            self.status_pub.publish(String(data="ARRIVED_SCAN_P1"))
    
    # [!!] Guid로부터 오는 순서 문자열(예: "rect,circle,tri") 처리
    def order_callback(self, msg: String):
        # [!!] 스캔 상태일 때, 그리고 아직 인덱스를 못 찾았을 때만 실행
        if self.mission_status == "SCANNING_FOR_TARGETS" and self.target_index == -1:
            
            self.target_order = msg.data # 예: "rectangle,circle,triangle"
            self.get_logger().info(f"Navi: Received target order string: {self.target_order}")
            
            # 1. 쉼표(,)를 기준으로 잘라서 리스트로 만듦
            ordered_list = self.target_order.split(',')
            
            # 2. 파라미터로 받은 self.target_shape이 리스트에 있는지 확인
            if self.target_shape in ordered_list:
                # 3. 있다면, 그것의 인덱스(위치)를 찾음 (0, 1, 또는 2)
                self.target_index = ordered_list.index(self.target_shape)
                
                # 4. 로그 출력 및 인덱스 발행!
                self.get_logger().info(f"--- [!!] Target '{self.target_shape}' found at position {self.target_index + 1} (index {self.target_index}) [!!] ---")
                self.index_pub.publish(Float64(data=float(self.target_index)))
                
                # 5. 다음 상태로 전환 및 Cont에게 신호 전송
                self.mission_status = "READY_FOR_DOCKING"
                self.status_pub.publish(String(data="TARGET_ORDER_CONFIRMED"))
            
            else:
                # 3. 없다면, 경고 로그 출력
                self.get_logger().warn(f"Target shape '{self.target_shape}' not found in detected order: {self.target_order}.")
                self.get_logger().warn("Check 'target_shape' parameter in Navi and 'targets' parameter in Guid.")


    # [!!] 주기 실행 (메인 상태 머신)
    def process(self):
        if self.gps_data is None:
            return

        # [!!] 상태 머신
        if self.mission_status == "NAV_TO_P1":
            goal_x, goal_y = self.point_1_utm
            self.calculate_guidance(goal_x, goal_y) # epsi, distance 계산
            
            self.epsi_pub.publish(Float64(data=self.error_psi))
            self.dist_pub.publish(Float64(data=self.distance))

            if self.distance < self.arrive_thr:
                self.get_logger().info("Navi: Arrived at P1. Publishing target yaw for P2.")
                self.status_pub.publish(String(data="ARRIVED_P1"))
                
                target_yaw = math.atan2(self.point_2_utm[1] - self.base_y, self.point_2_utm[0] - self.base_x)
                self.target_yaw_pub.publish(Float64(data=target_yaw))
                
                self.mission_status = "WAIT_FOR_ROTATION_1"
        
        elif self.mission_status == "WAIT_FOR_ROTATION_1":
            # Cont가 회전하고 완료 신호를 보낼 때까지 대기
            pass

        elif self.mission_status == "SCANNING_FOR_TARGETS":
            # [!!] Cont는 HOLDING_FOR_SCAN 상태일 것임
            # [!!] epsi를 0으로 발행하여 Cont가 멈추도록 함
            self.epsi_pub.publish(Float64(data=0.0))
            self.get_logger().info("Scanning for targets... (waiting for Guid)", throttle_duration_sec=5)
            # [!!] (order_callback에서 다음 상태로 변경됨)
            pass

        elif self.mission_status == "READY_FOR_DOCKING":
            # [!!] 인덱스를 찾았고, Cont가 다음 행동(GPS 이동)을 준비할 상태
            self.epsi_pub.publish(Float64(data=0.0))
            self.get_logger().info(f"Target order is confirmed (Index: {self.target_index}). Ready for next move (Dock Nav).", throttle_duration_sec=5)
            # [!!] (이 상태가 유지됨. 다음 단계(3)에서 이 상태를 변경할 것임)
            pass
        
        elif self.mission_status == "MISSION_COMPLETE":
             self.epsi_pub.publish(Float64(data=0.0))
             self.get_logger().info("Mission Complete. Holding position.", throttle_duration_sec=5)
             pass


    # 항법 에러 계산 (LOS)
    def calculate_guidance(self, goal_x, goal_y):
        self.distance = math.sqrt((goal_x - self.base_x)**2 + (goal_y - self.base_y)**2)
        
        way_x, way_y = goal_x - self.base_x, goal_y - self.base_y
        way_len = math.sqrt(way_x**2 + way_y**2)
        if way_len == 0: 
            self.error_psi = 0.0
            return

        lookahead_x = self.base_x + (way_x / way_len) * self.lookahead_distance
        lookahead_y = self.base_y + (way_y / way_len) * self.lookahead_distance
        los_angle_deg = math.degrees(math.atan2(lookahead_y - self.base_y, lookahead_x - self.base_x))
        
        psi_error = (los_angle_deg - self.degree + 180) % 360 - 180
        self.error_psi = self.kp_los * psi_error

def main(args=None):
    rclpy.init(args=args)
    node = DockingNavi()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
