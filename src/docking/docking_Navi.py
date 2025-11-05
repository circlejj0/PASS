#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pyproj import Transformer
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray, String
import transforms3d.euler as euler # [!! MAVROS !!] MAVROS IMU는 쿼터니언이므로 변환 라이브러리 사용

class DockingNavi(Node):
    def __init__(self):
        super().__init__('docking_navi')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # (파라미터 선언부는 기존과 동일)
        self.declare_parameter('point_1_inspection_lonlat', [150.67433227316755, -33.72264401144996])
        self.declare_parameter('point_2_inspection_facing_lonlat', [150.67439938281387, -33.722560796687716])
        self.declare_parameter('dock_0_nav_lonlat', [150.67435382189106, -33.72252171445921])
        self.declare_parameter('dock_0_face_lonlat', [150.6744257637786, -33.722430096440945])
        self.declare_parameter('dock_1_nav_lonlat', [150.6744280305074, -33.722527561336065])
        self.declare_parameter('dock_1_face_lonlat', [150.67448022601252, -33.72245881810486])
        self.declare_parameter('dock_2_nav_lonlat', [150.6744766434662, -33.72256357543034])
        self.declare_parameter('dock_2_face_lonlat', [150.67453523579022, -33.72248888351512])
        self.declare_parameter('lookahead_distance', 5.0)
        self.declare_parameter('arrive_thr', 2.0)
        self.declare_parameter('los_kp', 1.0)
        self.declare_parameter('target_shape', 'rectangle') 

        # [!! MAVROS !!] MAVROS의 GPS/IMU 토픽으로 변경
        self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, qos)
        self.create_subscription(Imu, '/mavros/imu/data', self.imu_callback, qos)
        
        self.create_subscription(String, '/nav/status', self.status_callback, 10) 
        self.create_subscription(String, '/docking/target_order', self.order_callback, 10)
        self.epsi_pub = self.create_publisher(Float64, '/error_psi', 10) 
        self.dist_pub = self.create_publisher(Float64, '/distance', 10)
        self.utm_pub = self.create_publisher(Float64MultiArray, '/UTM_Latlot', 10)
        self.yaw_pub = self.create_publisher(Float64, '/yaw', 10)
        self.status_pub = self.create_publisher(String, '/nav/status', 10) 
        self.target_yaw_pub = self.create_publisher(Float64, '/target_yaw', 10)
        self.index_pub = self.create_publisher(Float64, '/docking/target_index', qos)

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

    def gps_callback(self, msg):
        self.gps_data = msg
        self.latitude, self.longitude = msg.latitude, msg.longitude
        if self.gps_data:
            self.base_x, self.base_y = self.transformer.transform(self.latitude, self.longitude)
            self.utm_pub.publish(Float64MultiArray(data=[self.base_x, self.base_y]))

    def imu_callback(self, msg):
        q = msg.orientation
        # [!! MAVROS !!] 쿼터니언에서 Yaw(rad) 추출 (NED 기준이므로 VRX와 동일하게 작동)
        _, _, self.rad = euler.quat2euler([q.w, q.x, q.y, q.z])
        
        # [!! MAVROS !!] Yaw가 -pi ~ +pi 범위이므로 0~360도 범위가 아님.
        # VRX IMU가 0~360을 줬다면 변환이 필요하지만,
        # 기존 코드의 atan2 yaw 계산과 LOS guidance의 각도 차이 계산(%)을 보면
        # -pi ~ +pi 범위(rad)와 0~360 범위(deg)가 혼용되어 있음.
        # 여기서는 MAVROS의 NED 기준 Yaw(rad)를 그대로 사용하고, 
        # degree가 필요한 calculate_guidance에서만 변환하도록 함.
        
        # 기존 VRX IMU 콜백:
        # q = msg.orientation
        # self.rad = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        # self.degree = math.degrees(self.rad)
        # self.yaw_pub.publish(Float64(data=self.rad))
        
        # [!! MAVROS !!] euler.quat2euler로 추출한 self.rad를 사용
        self.degree = math.degrees(self.rad)
        self.yaw_pub.publish(Float64(data=self.rad))

        
    # (status_callback은 기존과 동일)
    def status_callback(self, msg: String):
        if msg.data == "ROTATION_P1_COMPLETE" and self.mission_status == "WAIT_FOR_ROTATION_1":
            self.get_logger().info("Navi: Cont R-P1 complete. Starting SCANNING.")
            self.mission_status = "SCANNING_FOR_TARGETS"
            self.status_pub.publish(String(data="ARRIVED_SCAN_P1"))
            
        elif msg.data == "ROTATION_DOCK_COMPLETE" and self.mission_status == "WAIT_FOR_DOCK_ROTATION":
            self.get_logger().info("Navi: Cont GPS-Align complete. Starting Visual Align.")
            self.mission_status = "WAIT_FOR_VISUAL_ALIGN" 
    
        elif msg.data == "ROTATION_VISUAL_COMPLETE" and self.mission_status == "WAIT_FOR_VISUAL_ALIGN":
            self.get_logger().info("Navi: Cont Visual-Align complete. Ready for Lidar Approach (Step 4).")
            self.mission_status = "READY_FOR_LIDAR_APPROACH"

        elif msg.data == "DOCKING_COMPLETE" and self.mission_status == "READY_FOR_LIDAR_APPROACH":
            self.get_logger().info("Navi: Cont reported docking complete. Holding for 5 sec.")
            self.mission_status = "HOLDING_AFTER_DOCK" 

        elif msg.data == "START_REVERSING" and self.mission_status == "HOLDING_AFTER_DOCK":
            self.get_logger().info("Navi: Hold complete. Starting reverse maneuver.")
            self.mission_status = "REVERSING_FROM_DOCK" 

        elif msg.data == "REVERSE_COMPLETE" and self.mission_status == "REVERSING_FROM_DOCK":
            self.get_logger().info("Navi: Cont reported reverse complete. Mission Finished.")
            self.mission_status = "MISSION_COMPLETE" 

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

    # (process 함수는 기존과 동일)
    def process(self):
        if self.gps_data is None:
            return

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
        
        elif self.mission_status == "WAIT_FOR_ROTATION_1":
            pass 

        elif self.mission_status == "SCANNING_FOR_TARGETS":
            self.epsi_pub.publish(Float64(data=0.0))
            self.get_logger().info("Scanning for targets... (waiting for Guid)", throttle_duration_sec=5)
            pass 

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

        elif self.mission_status == "WAIT_FOR_DOCK_ROTATION":
            self.epsi_pub.publish(Float64(data=0.0)) 
            self.get_logger().info("Waiting for Cont GPS alignment...", throttle_duration_sec=5)
            pass

        elif self.mission_status == "WAIT_FOR_VISUAL_ALIGN":
            self.epsi_pub.publish(Float64(data=0.0)) 
            self.get_logger().info("Waiting for Cont VISUAL alignment...", throttle_duration_sec=5)
            pass

        elif self.mission_status == "READY_FOR_LIDAR_APPROACH":
            self.epsi_pub.publish(Float64(data=0.0))
            self.get_logger().info(f"Ready for Lidar Approach (Step 4). Index {self.target_index}.", throttle_duration_sec=5)
            pass 
        
        elif self.mission_status == "HOLDING_AFTER_DOCK":
            self.epsi_pub.publish(Float64(data=0.0))
            self.get_logger().info("Docked. Holding 5 sec...", throttle_duration_sec=5)
            pass 

        elif self.mission_status == "REVERSING_FROM_DOCK":
            self.epsi_pub.publish(Float64(data=0.0)) 
            self.get_logger().info("Reversing from dock...", throttle_duration_sec=5)
            pass
            
        elif self.mission_status == "MISSION_COMPLETE":
             self.epsi_pub.publish(Float64(data=0.0))
             self.get_logger().info("Mission Complete. Holding position.", throttle_duration_sec=5)
             pass

    # (calculate_guidance는 기존과 동일)
    # [!! MAVROS !!] 참고: self.degree는 -180 ~ +180 범위일 수 있음. 
    # los_angle_deg 역시 atan2로 계산되므로 -180 ~ +180 범위임.
    # (A - B + 180) % 360 - 180 로직은 두 각도의 범위를 정규화하므로
    # 기존 코드 그대로 사용해도 문제 없음.
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
