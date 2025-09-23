#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile
from pyproj import Transformer
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray, String

class DockingNavi(Node):
    def __init__(self):
        super().__init__('docking_navi')
        qos = QoSProfile(depth=10)

        # Sub
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.latlot_listener_callback, qos)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.psi_listener_callback, qos)

        # Pub
        self.epsi_pub = self.create_publisher(Float64, '/error_psi', 10)
        self.dist_pub = self.create_publisher(Float64, '/distance', 10)
        self.utm_pub = self.create_publisher(Float64MultiArray, '/UTM_Latlot', 10)
        self.yaw_pub = self.create_publisher(Float64, '/yaw', 10)
        self.status_pub = self.create_publisher(String, '/nav/status', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.process)

        # 변수 설정
        self.gps_data, self.arrived_sent = None, False
        self.latitude, self.longitude = 0.0, 0.0
        self.rad, self.degree = 0.0, 0.0
        self.base_x, self.base_y = 0.0, 0.0
        self.error_psi, self.distance = 0.0, 0.0

        # 좌표 변환
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")

        # 목표 좌표 (도착 시작 지점 좌표임. 여기서 도크쪽으로 회전)
        waypoint_lonlat = [(150.674357383792, -33.7226209717351)]
        self.waypoints = [self.transformer.transform(lat, lon) for lon, lat in waypoint_lonlat]

        # 제어 파라미터
        self.kp, self.lookahead_distance, self.arrive_thr = 1.0, 5.0, 2.0

    # GPS Data Sub
    def latlot_listener_callback(self, msg):
        self.gps_data = msg
        self.latitude, self.longitude = msg.latitude, msg.longitude

    # IMU Data Sub
    def psi_listener_callback(self, msg):
        q = msg.orientation
        self.rad = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        self.degree = self.rad * 180 / math.pi

    # 주기 실행
    def process(self):
        if self.gps_data is None or self.arrived_sent:
            return

        self.base_x, self.base_y = self.transformer.transform(self.latitude, self.longitude)
        self.calculate_guidance()

        if self.distance < self.arrive_thr:
            self.status_pub.publish(String(data="ARRIVED_P1"))
            self.arrived_sent = True
            return

        self.epsi_pub.publish(Float64(data=self.error_psi))
        self.dist_pub.publish(Float64(data=self.distance))
        self.utm_pub.publish(Float64MultiArray(data=[self.base_x, self.base_y]))
        self.yaw_pub.publish(Float64(data=self.rad))

    # 도형 방향 계산
    def calculate_guidance(self):
        goal_x, goal_y = self.waypoints[0]
        self.distance = math.sqrt((goal_x - self.base_x)**2 + (goal_y - self.base_y)**2)
        
        way_x, way_y = goal_x - self.base_x, goal_y - self.base_y
        way_len = math.sqrt(way_x**2 + way_y**2)
        if way_len == 0: return

        lookahead_x = self.base_x + (way_x / way_len) * self.lookahead_distance
        lookahead_y = self.base_y + (way_y / way_len) * self.lookahead_distance
        los_angle = math.atan2(lookahead_y - self.base_y, lookahead_x - self.base_x) * 180 / math.pi
        
        psi_error = (los_angle - self.degree + 180) % 360 - 180
        self.error_psi = self.kp * psi_error

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
