#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [!!] 수정된 DockingNavi (항법/지휘 노드)
# [!!] 5초 정지 및 후진 상태 추가

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

        # (파라미터 선언부는 기존과 동일)
        # --- [!!] 1단계 좌표 (스캔 지점) ---
        self.declare_parameter('point_1_inspection_lonlat', [150.67433227316755, -33.72264401144996])
        self.declare_parameter('point_2_inspection_facing_lonlat', [150.67439938281387, -33.722560796687716])
        # --- [!!] 3단계 좌표 (도크 0, 1, 2) ---
        self.declare_parameter('dock_0_nav_lonlat', [150.67435382189106, -33.72252171445921])
        self.declare_parameter('dock_0_face_lonlat', [150.6744257637786, -33.722430096440945])
        self.declare_parameter('dock_1_nav_lonlat', [150.6744280305074, -33.722527561336065])
        self.declare_parameter('dock_1_face_lonlat', [150.67448022601252, -33.72245881810486])
        self.declare_parameter('dock_2_nav_lonlat', [150.6744766434662, -33.72256357543034])
        self.declare_parameter('dock_2_face_lonlat', [150.67453523579022, -33.72248888351512])
        # --- [!!] 제어 및 타겟 파라미터 ---
        self.declare_parameter('lookahead_distance', 5.0)
        self.declare_parameter('arrive_thr', 2.0)
        self.declare_parameter('los_kp', 1.0)
        self.declare_parameter('target_shape', 'rectangle') 

        # (Sub/Pub 선언부는 기존과 동일)
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, qos)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, qos)
        self.create_subscription(String, '/nav/status', self.status_callback, 10) 
        self.create_subscription(String, '/docking/target_order', self.order_callback, 10)
        self.epsi_pub = self.create_publisher(Float64, '/error_psi', 10) 
        self.dist_pub = self.create_publisher(Float64, '/distance', 10)
        self.utm_pub = self.create_publisher(Float64MultiArray, '/UTM_Latlot', 10)
        self.yaw_pub = self.create_publisher(Float64, '/yaw', 10)
        self.status_pub = self.create_publisher(String, '/nav/status', 10) 
        self.target_yaw_pub = self.create_publisher(Float64, '/target_yaw', 10)
        self.index_pub = self.create_publisher(Float64, '/docking/target_index', qos)

        # Timer
        self.timer = self.create_timer(0.1, self.process)

        # (변수 설정 및 좌표 변환은 기존과 동일)
        self.gps_data = None
        self.latitude, self.longitude = 0.0, 0.0
        self.rad, self.degree = 0.0, 0.0
        self.base_x, self.base_y = 0.0, 0.0
        self.error_psi, self.distance = 0.0, 0.0
        self.mission_status = "NAV_TO_P1"
        self.target_shape = self.get_parameter('target_shape').value.lower()
        self.target_index = -1 
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")
        p1_lonlat = self.get_parameter('point_1_inspection_lonlat').value
        p2_lonlat = self.get_parameter('point_2_inspection_facing_lonlat').value
        self.point_1_utm = self.transformer.transform(p1_lonlat[1], p1_lonlat[0])
        self.point_2_utm = self.transformer.transform(p2_lonlat[1], p2_lonlat[0])
        self.dock_nav_points_utm = []
        self.dock_face_points_utm = []
        for i in range(3):
            nav_lonlat = self.get_parameter(f'dock_{i}_nav_lonlat').value
            face_lonlat = self.get_parameter(f'dock_{i}_face_lonlat').value
            nav_utm = self.transformer.transform(nav_lonlat[1], nav_lonlat[0])
            face_utm = self.transformer.transform(face_lonlat[1], face_lonlat[0])
            self.dock_nav_points_utm.append(nav_utm)
            self.dock_face_points_utm.append(face_utm)
        self.selected_dock_nav_utm = None
        self.selected_dock_face_utm = None
        self.kp_los = self.get_parameter('los_kp').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.arrive_thr = self.get_parameter('arrive_thr').value

    # (gps_callback, imu_callback은 기존과 동일)
    def gps_callback(self, msg):
        self.gps_data = msg
        self.latitude, self.longitude = msg.latitude, msg.longitude
        if self.gps_data:
            self.base_x, self.base_y = self.transformer.transform(self.latitude, self.longitude)
            self.utm_pub.publish(Float64MultiArray(data=[self.base_x, self.base_y]))
    def imu_callback(self, msg):
        q = msg.orientation
        self.rad = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        self.degree = math.degrees(self.rad)
        self.yaw_pub.publish(Float64(data=self.rad))
        
    # [!!] --- status_callback 수정 (3개 상태 추가) ---
    def status_callback(self, msg: String):
        # 1단계 회전 완료
        if msg.data == "ROTATION_P1_COMPLETE" and self.mission_status == "WAIT_FOR_ROTATION_1":
            self.get_logger().info("Navi: Cont R-P1 complete. Starting SCANNING.")
            self.mission_status = "SCANNING_FOR_TARGETS"
            self.status_pub.publish(String(data="ARRIVED_SCAN_P1"))
            
        # 3-1단계: GPS 회전 완료
        elif msg.data == "ROTATION_DOCK_COMPLETE" and self.mission_status == "WAIT_FOR_DOCK_ROTATION":
            self.get_logger().info("Navi: Cont GPS-Align complete. Starting Visual Align.")
            self.mission_status = "WAIT_FOR_VISUAL_ALIGN" 
    
        # 3-2단계: 시각적 회전 완료
        elif msg.data == "ROTATION_VISUAL_COMPLETE" and self.mission_status == "WAIT_FOR_VISUAL_ALIGN":
            self.get_logger().info("Navi: Cont Visual-Align complete. Ready for Lidar Approach (Step 4).")
            self.mission_status = "READY_FOR_LIDAR_APPROACH"

        # [!!] 4단계: 도킹 완료 (Cont가 LiDAR 거리 보고 판단)
        elif msg.data == "DOCKING_COMPLETE" and self.mission_status == "READY_FOR_LIDAR_APPROACH":
            self.get_logger().info("Navi: Cont reported docking complete. Holding for 5 sec.")
            self.mission_status = "HOLDING_AFTER_DOCK" # [!!] 5초 정지 상태

        # [!!] 5단계: 5초 정지 완료 (Cont가 타이머 보고 판단)
        elif msg.data == "START_REVERSING" and self.mission_status == "HOLDING_AFTER_DOCK":
            self.get_logger().info("Navi: Hold complete. Starting reverse maneuver.")
            self.mission_status = "REVERSING_FROM_DOCK" # [!!] 후진 상태

        # [!!] 6단계: 후진 완료 (Cont가 LiDAR 거리 보고 판단)
        elif msg.data == "REVERSE_COMPLETE" and self.mission_status == "REVERSING_FROM_DOCK":
            self.get_logger().info("Navi: Cont reported reverse complete. Mission Finished.")
            self.mission_status = "MISSION_COMPLETE" # [!!] 임무 완료

    # (order_callback은 기존과 동일)
    def order_callback(self, msg: String):
        if self.mission_status == "SCANNING_FOR_TARGETS" and self.target_index == -1:
            self.target_order = msg.data 
            self.get_logger().info(f"Navi: Received target order string: {self.target_order}")
            ordered_list = self.target_order.split(',')
            if self.target_shape in ordered_list:
                self.target_index = ordered_list.index(self.target_shape)
                self.get_logger().info(f"--- [!!] Target '{self.target_shape}' found at index {self.target_index} [!!] ---")
                self.index_pub.publish(Float64(data=float(self.target_index)))
                self.selected_dock_nav_utm = self.dock_nav_points_utm[self.target_index]
                self.selected_dock_face_utm = self.dock_face_points_utm[self.target_index]
                self.mission_status = "NAV_TO_DOCK"
                self.status_pub.publish(String(data="TARGET_ORDER_CONFIRMED"))
            else:
                self.get_logger().warn(f"Target shape '{self.target_shape}' not found in detected order: {self.target_order}.")

    # [!!] --- process 함수 수정 (3개 상태 추가) ---
    def process(self):
        if self.gps_data is None:
            return

        # 1단계: P1으로 항해
        if self.mission_status == "NAV_TO_P1":
            goal_x, goal_y = self.point_1_utm
            self.calculate_guidance(goal_x, goal_y)
            self.epsi_pub.publish(Float64(data=self.error_psi))
            self.dist_pub.publish(Float64(data=self.distance))
            if self.distance < self.arrive_thr:
                self.get_logger().info("Navi: Arrived at P1. Publishing target yaw for P2.")
                self.status_pub.publish(String(data="ARRIVED_P1"))
                target_yaw = math.atan2(self.point_2_utm[1] - self.base_y, self.point_2_utm[0] - self.base_x)
                self.target_yaw_pub.publish(Float64(data=target_yaw))
                self.mission_status = "WAIT_FOR_ROTATION_1"
        
        # 1단계: 회전 대기
        elif self.mission_status == "WAIT_FOR_ROTATION_1":
            pass 

        # 2단계: 스캔 대기
        elif self.mission_status == "SCANNING_FOR_TARGETS":
            self.epsi_pub.publish(Float64(data=0.0))
            self.get_logger().info("Scanning for targets... (waiting for Guid)", throttle_duration_sec=5)
            pass 

        # 3단계: 도크 항해
        elif self.mission_status == "NAV_TO_DOCK":
            if self.selected_dock_nav_utm is None:
                self.get_logger().error("NAV_TO_DOCK state but no target selected!")
                return
            goal_x, goal_y = self.selected_dock_nav_utm
            self.calculate_guidance(goal_x, goal_y)
            self.epsi_pub.publish(Float64(data=self.error_psi))
            self.dist_pub.publish(Float64(data=self.distance))

            if self.distance < self.arrive_thr:
                self.get_logger().info(f"Navi: Arrived at Dock {self.target_index} Nav Point.")
                self.status_pub.publish(String(data="ARRIVED_DOCK_NAV"))
                
                face_goal = self.selected_dock_face_utm
                target_yaw = math.atan2(face_goal[1] - self.base_y, face_goal[0] - self.base_x)
                self.target_yaw_pub.publish(Float64(data=target_yaw))
                
                self.mission_status = "WAIT_FOR_DOCK_ROTATION" 

        # 3-1단계: GPS 회전 대기
        elif self.mission_status == "WAIT_FOR_DOCK_ROTATION":
            self.epsi_pub.publish(Float64(data=0.0)) 
            self.get_logger().info("Waiting for Cont GPS alignment...", throttle_duration_sec=5)
            pass

        # 3-2단계: 시각적 회전 대기
        elif self.mission_status == "WAIT_FOR_VISUAL_ALIGN":
            self.epsi_pub.publish(Float64(data=0.0)) 
            self.get_logger().info("Waiting for Cont VISUAL alignment...", throttle_duration_sec=5)
            pass

        # 4단계: 최종 접근 대기 (Cont가 수행)
        elif self.mission_status == "READY_FOR_LIDAR_APPROACH":
            self.epsi_pub.publish(Float64(data=0.0)) # 조향은 Cont가 하므로 epsi 0
            self.get_logger().info(f"Ready for Lidar Approach (Step 4). Index {self.target_index}.", throttle_duration_sec=5)
            pass # Cont가 DOCKING_COMPLETE 신호 보낼 때까지 대기
        
        # [!!] 5단계: 5초 정지
        elif self.mission_status == "HOLDING_AFTER_DOCK":
            self.epsi_pub.publish(Float64(data=0.0)) # 정지
            self.get_logger().info("Docked. Holding 5 sec...", throttle_duration_sec=5)
            pass # Cont가 START_REVERSING 신호 보낼 때까지 대기

        # [!!] 6단계: 후진
        elif self.mission_status == "REVERSING_FROM_DOCK":
            self.epsi_pub.publish(Float64(data=0.0)) # 후진은 직진(epsi 0)
            self.get_logger().info("Reversing from dock...", throttle_duration_sec=5)
            pass # Cont가 REVERSE_COMPLETE 신호 보낼 때까지 대기
            
        # [!!] 7단계: 임무 완료
        elif self.mission_status == "MISSION_COMPLETE":
             self.epsi_pub.publish(Float64(data=0.0))
             self.get_logger().info("Mission Complete. Holding position.", throttle_duration_sec=5)
             pass

    # (calculate_guidance는 기존과 동일)
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
