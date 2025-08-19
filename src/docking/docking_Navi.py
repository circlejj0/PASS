#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile
from pyproj import Transformer
from sensor_msgs.msg import NavSatFix, Imu, LaserScan
from std_msgs.msg import Float64, Float64MultiArray, String

class DockingNavi(Node):
    def __init__(self):
        super().__init__('vrx_navigation')
        qos = QoSProfile(depth=10)

        # Sub
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.latlot_listener_callback, qos)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.psi_listener_callback, qos)

        # Pub
        self.epsi_pub = self.create_publisher(Float64, '/error_psi', 10)
        self.dist_pub = self.create_publisher(Float64, '/distance', 10)
        self.utm_pub = self.create_publisher(Float64MultiArray, '/UTM_Latlot', 10)
        self.yaw_pub = self.create_publisher(Float64, '/yaw', 10)
        self.next_obj_pub = self.create_publisher(Float64, '/next_obj', 10)
        self.status_pub = self.create_publisher(String, '/nav/status', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.process)

        # 변수 설정
        self.gps_data = None
        self.latitude = None
        self.longitude = None
        self.rad = 0.0
        self.error_psi = 0.0
        self.distance = 0.0
        self.arrived_sent = False

        # 좌표 변환
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")

        # 목표 좌표 (도킹 시작 위치)
        waypoint_lonlat = [(150.67427803618017,  -33.72272135580442),
                            (150.67441330657957,  -33.72255187124071)]
        self.waypoints = [self.transformer.transform(lat, lon) for lon, lat in waypoint_lonlat]

        # 제어 파라미터
        self.next_obj = 0
        self.close_distance = 7
        self.kp = 1.0
        self.lookahead_distance = 5.0
        self.arrive_thr = 3.0

    # Callback 함수들
    def latlot_listener_callback(self, msg):
        self.gps_data = msg
        self.latitude = msg.latitude
        self.longitude = msg.longitude

    def psi_listener_callback(self, msg):
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w
        self.rad = self.cal_yaw(x, y, z, w)
    
    # 주기 실행
    def process(self):
        if self.gps_data is None:
            return
        
        self.change_lonlat_UTM()
        self.cal_psi()
        self.moving_obs_point()

        # Data pub
        self.epsi_pub.publish(Float64(data=self.error_psi))
        self.dist_pub.publish(Float64(data=self.distance))
        self.utm_pub.publish(Float64MultiArray(data=[self.base_x, self.base_y]))
        self.yaw_pub.publish(Float64(data=self.rad))

        # 도킹시작지점에 도착하면 Guidance 코드로 넘길 신호 1회 전송
        is_last_wp = (self.next_obj == len(self.waypoints) - 1)
        if is_last_wp and (self.distance < self.arrive_thr) and (not self.arrived_sent):
            self.status_pub.publish(String(data="ARRIVED_SCAN_P1"))
            self.arrived_sent = True

    # 보조 함수들
    def cal_yaw(self, x, y, z, w):
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    
    def cal_distance(self, x1, x2, y1, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def cal_psi(self):
        self.degree = (self.rad * 180) / math.pi

    def change_lonlat_UTM(self):
        self.base_x, self.base_y = self.transformer.transform(self.latitude, self.longitude)

    def moving_obs_point(self):
        goal_x, goal_y = self.waypoints[self.next_obj]
        self.distance = self.cal_distance(self.base_x, goal_x, self.base_y, goal_y)
        
        # LOS 기반 목표점 계산
        way_x = goal_x - self.base_x
        way_y = goal_y - self.base_y
        way_length = math.sqrt(way_x**2 + way_y**2)
        lookahead_x = self.base_x + (way_x / way_length) * self.lookahead_distance
        lookahead_y = self.base_y + (way_y / way_length) * self.lookahead_distance
        los_angle = math.atan2(lookahead_y - self.base_y, lookahead_x - self.base_x) * 180 / math.pi
        psi_error = (los_angle - self.degree + 180) % 360 - 180 
        self.error_psi = self.kp * psi_error 

        # waypoint reset
        if self.distance < self.close_distance:
            if self.next_obj < len(self.waypoints) - 1:
                self.next_obj += 1


        # next_obj pub
        self.next_obj_pub.publish(Float64(data=float(self.next_obj)))



def main(args=None):
    rclpy.init(args=args)
    node = DockingNavi()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__': 
    main()
