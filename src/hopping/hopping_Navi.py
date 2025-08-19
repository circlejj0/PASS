#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import math
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile
from pyproj import Transformer

from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker


class HoppingNavi(Node):
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

        # RViz 시각화용
        self.path_pub = self.create_publisher(Path, '/wamv/path', 10)      
        self.marker_pub = self.create_publisher(Marker, '/wamv/waypoints', 10) 

        # 변수 설정
        self.gps_data = None
        self.latitude = None
        self.longitude = None
        self.rad = 0.0
        self.error_psi = 0.0
        self.distance = 0.0
        self.arrived_sent = False
        self.waiting = False
        self.wait_start_time = None
        self.wait_duration = 5.0

        # 좌표 변환 (WGS84 -> UTM Zone 56S)
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756")

        # Hopping 좌표
        waypoint_lonlat = [
            (150.67422274290925, -33.722449333791765),
            (150.6739268456858, -33.72229115102343),
            (150.6741442298346, -33.72172120901166)
        ]
        utm_waypoints = [self.transformer.transform(lat, lon) for lon, lat in waypoint_lonlat]

        # 원점 기준 상대좌표 변환
        self.origin_x, self.origin_y = utm_waypoints[0]
        self.waypoints = [(wx - self.origin_x, wy - self.origin_y) for (wx, wy) in utm_waypoints]

        # 제어 파라미터
        self.next_obj = 0
        self.close_distance = 4.0
        self.kp = 1.0
        self.lookahead_distance = 2.0
        self.arrive_thr = 1.0

        # Path 메시지 초기화
        self.path_msg = Path()
        self.path_msg.header.frame_id = "wamv/wamv/base_link"

        # Timer
        self.timer = self.create_timer(0.1, self.process)

        # Waypoint marker 최초 publish
        self.publish_waypoints_marker()

    # Waypoint Marker
    def publish_waypoints_marker(self):
        marker = Marker()
        marker.header.frame_id = "wamv/wamv/base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        for (x, y) in self.waypoints:
            p = Point()
            p.x, p.y, p.z = x, y, 0.0
            marker.points.append(p)

        self.marker_pub.publish(marker)

    # Callback 함수들
    def latlot_listener_callback(self, msg):
        self.gps_data = msg
        self.latitude = msg.latitude
        self.longitude = msg.longitude

    def psi_listener_callback(self, msg):
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.rad = self.cal_yaw(x, y, z, w)

    # 주기 실행
    def process(self):
        if self.gps_data is None:
            return

        # 대기 상태 → 정지
        if self.waiting:
            self.epsi_pub.publish(Float64(data=0.0))
            self.dist_pub.publish(Float64(data=-1.0))
            self.yaw_pub.publish(Float64(data=self.rad))
            self.utm_pub.publish(Float64MultiArray(data=[self.base_x, self.base_y]))
            self.next_obj_pub.publish(Float64(data=float(self.next_obj)))

            if time.time() - self.wait_start_time >= self.wait_duration:
                self.waiting = False
                self.next_obj += 1
                self.get_logger().info(f"🚤 Resume after waiting, next waypoint {self.next_obj}")
            return

        # 좌표 변환 및 제어 계산
        self.change_lonlat_UTM()
        self.cal_psi()
        self.moving_obs_point()

        # Pub
        self.epsi_pub.publish(Float64(data=self.error_psi))
        self.dist_pub.publish(Float64(data=self.distance))
        self.utm_pub.publish(Float64MultiArray(data=[self.base_x, self.base_y]))
        self.yaw_pub.publish(Float64(data=self.rad))

        # 이동 경로 Path 업데이트
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "wamv/wamv/base_link"
        pose.pose.position.x = self.base_x
        pose.pose.position.y = self.base_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(self.rad / 2.0)
        pose.pose.orientation.w = math.cos(self.rad / 2.0)

        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

        # Waypoint marker 갱신
        self.publish_waypoints_marker()

        # 마지막 waypoint 도착 처리
        is_last_wp = (self.next_obj == len(self.waypoints) - 1)
        if is_last_wp and (self.distance < self.arrive_thr) and (not self.arrived_sent):
            self.status_pub.publish(String(data="ARRIVED_SCAN_P1"))
            self.dist_pub.publish(Float64(data=-1.0))
            self.arrived_sent = True

    # 보조 함수들
    def cal_yaw(self, x, y, z, w):
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

    def cal_distance(self, x1, x2, y1, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def cal_psi(self):
        self.degree = (self.rad * 180) / math.pi

    def change_lonlat_UTM(self):
        x, y = self.transformer.transform(self.latitude, self.longitude)
        self.base_x = x - self.origin_x
        self.base_y = y - self.origin_y

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

        # 호핑 지점 도착 → 잠시 대기
        if self.distance < self.close_distance:
            if self.next_obj < len(self.waypoints) - 1:
                self.waiting = True
                self.wait_start_time = time.time()
                self.get_logger().info(f"🛑 Arrived waypoint {self.next_obj}, waiting {self.wait_duration} sec...")

        # next_obj pub
        self.next_obj_pub.publish(Float64(data=float(self.next_obj)))


def main(args=None):
    rclpy.init(args=args)
    node = HoppingNavi()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
