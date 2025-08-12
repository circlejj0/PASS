#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile
from pyproj import Transformer
from sensor_msgs.msg import NavSatFix, Imu, LaserScan
import numpy as np
from std_msgs.msg import Float64, Float64MultiArray

class WamvNavigation(Node):
    def __init__(self):
        super().__init__('vrx_navigation')
        qos = QoSProfile(depth=10)

        self.latlot_subscription = self.create_subscription(
            NavSatFix,
            '/wamv/sensors/gps/gps/fix',
            self.latlot_listener_callback,
            qos_profile=qos)
        
        self.psi_subscription = self.create_subscription(
            Imu,
            '/wamv/sensors/imu/imu/data',
            self.psi_listener_callback,
            qos_profile=qos)
        
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/wamv/sensors/lidars/lidar_wamv_sensor/scan',
            self.scan_listener_callback,
            qos_profile=qos)

        self.timer = self.create_timer(0.1, self.process)
        self.gps_data = None
        self.latitude = None
        self.longitude = None
        self.obs_x = None
        self.obs_y = None

        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")
        waypoint_lonlat = [(150.6736090321409,  -33.72228691262119),
                            (150.67340718360654,  -33.72228570361426),
                            (150.6736090265203,  -33.72228691262119)]
        self.waypoints = [self.transformer.transform(lat, lon) for lon, lat in waypoint_lonlat]

        self.next_obj_pub = self.create_publisher(Float64, '/next_obj', 10)
        self.next_obj = 0
        self.close_distance = 7
        self.error_psi = 0.0
        self.distance = 0.0
        self.kp = 1.0
        self.lookahead_distance = 5.0

    def e_psi_pub(self):
        publisher = self.create_publisher(Float64, '/error_psi', 10)
        msg = Float64()
        msg.data = self.error_psi
        publisher.publish(msg)
    
    def distance_pub(self):
        publisher = self.create_publisher(Float64, '/distance', 10)
        msg = Float64()
        msg.data = self.distance
        publisher.publish(msg)
    
    def UTM_pub(self):
        publisher = self.create_publisher(Float64MultiArray, '/UTM_Latlot', 10)
        msg = Float64MultiArray()
        msg.data = [self.base_x, self.base_y]
        publisher.publish(msg)

    def yaw_pub(self):
        publisher = self.create_publisher(Float64, '/yaw', 10)
        msg = Float64()
        msg.data = self.rad
        publisher.publish(msg)


    def latlot_listener_callback(self, msg):
        self.gps_data = msg
        self.latitude = self.gps_data.latitude
        self.longitude = self.gps_data.longitude

    def psi_listener_callback(self, msg):
        self.psi_data = msg
        self.x = self.psi_data.orientation.x
        self.y = self.psi_data.orientation.y
        self.z = self.psi_data.orientation.z
        self.w = self.psi_data.orientation.w
        self.rad = self.cal_yaw(self.x, self.y, self.z, self.w)
        
    def scan_listener_callback(self, msg):
        self.scan_data = msg
        self.obs_data = self.scan_data.ranges
        self.angle_min = self.scan_data.angle_min
        self.angle_max = self.scan_data.angle_max
        self.angle_increment = self.scan_data.angle_increment

    # def transform_rotation(self, x, y):
    #     rotation_matrix = np.array([
    #         [np.cos(self.rad), -np.sin(self.rad)],
    #         [np.sin(self.rad), np.cos(self.rad)]
    #          ])

    #     point = np.array([x, y])

    #     self.transform = np.dot(rotation_matrix, point)
    #     return self.transform
    
    def process(self):
        if self.gps_data is None:
            return
        
        self.change_lonlat_UTM()
        self.cal_psi()
        self.moving_obs_point()
        self.e_psi_pub()
        self.distance_pub()
        self.UTM_pub()
        self.yaw_pub()
        # self.transform_rotation(self.obs_x, self.obs_y)

    def cal_yaw(self, x, y, z, w):
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    
    def cal_distance(self, x1, x2, y1, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def cal_psi(self):
        self.degree = (self.rad * 180) / math.pi


    def change_lonlat_UTM(self):
        self.start_x, self.start_y = 284479.95, 6266152.67
        # self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")
        self.base_x, self.base_y = self.transformer.transform(self.latitude, self.longitude)
        # self.base_x, self.base_y = self.transformer.transform(self.latittude, self.longitude)

    def moving_obs_point(self):
        goal_x, goal_y = self.waypoints[self.next_obj]
        self.distance = self.cal_distance(self.base_x, goal_x, self.base_y, goal_y)
        
        way_x = goal_x - self.base_x
        way_y = goal_y - self.base_y
        way_length = math.sqrt(way_x**2 + way_y**2)
        lookahead_x = self.base_x + (way_x / way_length) * self.lookahead_distance
        lookahead_y = self.base_y + (way_y / way_length) * self.lookahead_distance

        los_angle = math.atan2(lookahead_y - self.base_y, lookahead_x - self.base_x) * 180 / math.pi
        psi_error = (los_angle - self.degree + 180) % 360 - 180 
        self.error_psi = self.kp * psi_error 

        print(f"next_obj: {self.next_obj}, distance: {self.distance:.2f}, error_psi: {self.error_psi:.2f}")

        # === ✅ waypoint index update ===
        # === ✅ waypoint index update ===
        if self.distance < self.close_distance:
            # 마지막 목표점이면 증가하지 말고 그대로 멈춤 유지
            if self.next_obj < len(self.waypoints) - 1:
                self.next_obj += 1
            # 마지막 waypoint에 도달한 상태를 유지하기 위해 self.next_obj 그대로 1로 유지


        # === ✅ next_obj 퍼블리시 ===
        msg = Float64()
        msg.data = float(self.next_obj)
        self.next_obj_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = WamvNavigation()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__': 
    main()
