#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from pyproj import Transformer
from sensor_msgs.msg import NavSatFix, Imu
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point  
from std_msgs.msg import Float64, Float64MultiArray, String

class DockingGuid(Node):
    def __init__(self):
        super().__init__('wamv_perception')
        qos = QoSProfile(depth=10)

        self.gps_sub = self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, qos)
        self.imu_sub = self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, qos)
        self.obs_marker_sub = self.create_subscription(Marker, '/obs_center_point', self.obs_callback, qos)
        self.circle_marker_sub = self.create_subscription(Marker, '/circle_load', self.circle_callback, qos)

        self.error_psi_pub = self.create_publisher(Float64, '/error_psi', 10)
        self.distance_pub = self.create_publisher(Float64, '/distance', 10)
        self.yaw_pub = self.create_publisher(Float64, '/yaw', 10)
        self.utm_pub = self.create_publisher(Float64MultiArray, 'UTM_Latlot', 10)

        self.obs_pub = self.create_publisher(Marker, '/filtered_obs', 10)
        self.circle_pub = self.create_publisher(Marker, '/transformed_circle', 10)

        self.goal_x = 325000.0
        self.goal_y = 4098000.0

        self.create_subscription(String, '/nav/status', self.status_cb, 10)
        self.arrived_scan_p1 = False

        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True)

    def status_cb(self, msg):
        if msg.data == "ARRIVED_SCAN_P1" and not self.arrived_scan_p1:
            self.arrived_scan_p1 = True
            self.get_logger().info("[Guidance] 스캔 포인트 도착 신호 수신 ✅  -> Guidance 모드 로직 시작")


    def gps_callback(self, msg):
        lon, lat = msg.longitude, msg.latitude
        x, y = self.transformer.transform(lon, lat)
        utm_msg = Float64MultiArray()
        utm_msg.data = [x, y]
        self.utm_pub.publish(utm_msg)

        dx = self.goal_x - x
        dy = self.goal_y - y
        distance = math.hypot(dx, dy)
        self.distance_pub.publish(Float64(data=distance))

    def imu_callback(self, msg):
        q = msg.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.yaw_pub.publish(Float64(data=yaw))

    def obs_callback(self, msg):
        self.obs_pub.publish(msg)

    def circle_callback(self, msg):
        new_marker = Marker()
        new_marker.header = msg.header
        new_marker.ns = msg.ns
        new_marker.id = msg.id
        new_marker.type = msg.type
        new_marker.action = msg.action
        new_marker.pose = msg.pose
        new_marker.scale = msg.scale
        new_marker.color = msg.color
        new_marker.frame_locked = msg.frame_locked
        new_marker.lifetime = msg.lifetime

        new_marker.points = []

        for p in msg.points:
            x, y, z = self.transformer.transform(p.y, p.x, p.z)
            pt = Marker().points[0]
            pt.x, pt.y, pt.z = x, y, z
            new_marker.points.append(pt)

        self.circle_pub.publish(new_marker)

def main(args=None):
    rclpy.init(args=args)
    node = DockingGuid()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
