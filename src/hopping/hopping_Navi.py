#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import math
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy
)
from pyproj import Transformer
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker


class HoppingNavi(Node):
    def __init__(self):
        super().__init__('hopping_Navi')

        # QoS
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Sub
        self.create_subscription(NavSatFix, '/mavros/global_position/raw/fix', self.latlot_listener_callback, sensor_qos)
        self.create_subscription(Imu, '/mavros/imu/data', self.psi_listener_callback, sensor_qos)

        # Pub
        self.epsi_pub = self.create_publisher(Float64, '/error_psi', 10)
        self.dist_pub = self.create_publisher(Float64, '/distance', 10)
        self.utm_pub = self.create_publisher(Float64MultiArray, '/UTM_Latlot', 10)
        self.yaw_pub = self.create_publisher(Float64, '/yaw', 10)
        self.next_obj_pub = self.create_publisher(Float64, '/next_obj', 10)

        # RViz 시각화용 Pub
        self.path_pub = self.create_publisher(Path, '/mk/path', 10)      
        self.marker_pub = self.create_publisher(Marker, '/mk/waypoints', 10) 

        # 변수 설정
        self.gps_data = None
        self.latitude = None
        self.longitude = None
        self.rad = 0.0
        self.next_obj = 0

        # 좌표 변환
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True)

        # Hopping 좌표
        waypoint_lonlat = [
            (129.106815, 35.134922),
            (129.104100, 35.134699),
            (129.104206, 35.133883),
            (129.104570, 35.133616),
            (129.106949, 35.133797),
            (129.107077, 35.132835)
        ]
        utm_waypoints = [self.transformer.transform(lon, lat) for lon, lat in waypoint_lonlat]

        self.origin_x, self.origin_y = utm_waypoints[0]
        self.waypoints = [(wx - self.origin_x, wy - self.origin_y) for (wx, wy) in utm_waypoints]

        # Path 메시지 초기화
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        # Timer
        self.timer = self.create_timer(0.5, self.process)

        # Waypoint marker 최초 publish
        self.publish_waypoints_marker()

        self.get_logger().info("HoppingNavi node started")

    # Waypoint Marker
    def publish_waypoints_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
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
        self.get_logger().info(f"Published {len(self.waypoints)} waypoints")

    # Callback 함수들
    def latlot_listener_callback(self, msg):
        self.gps_data = msg
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        self.get_logger().info(f"GPS update: lat={self.latitude}, lon={self.longitude}")

    def psi_listener_callback(self, msg):
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.rad = self.cal_yaw(x, y, z, w)

    # 주기 실행
    def process(self):
        if self.gps_data is None:
            self.get_logger().warn("No GPS data yet")
            return

        # 좌표 변환 및 제어 계산
        try:
            self.change_lonlat_UTM()
        except Exception as e:
            self.get_logger().error(f"UTM transform failed: {e}")
            return

        self.cal_psi()
        self.moving_obs_point()

        # Path 업데이트
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = self.base_x
        pose.pose.position.y = self.base_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(self.rad / 2.0)
        pose.pose.orientation.w = math.cos(self.rad / 2.0)

        self.path_msg.header.stamp = pose.header.stamp
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

        self.get_logger().info(f"Published path point: ({self.base_x:.2f}, {self.base_y:.2f})")

    # 보조 함수들
    def cal_yaw(self, x, y, z, w):
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

    def cal_psi(self):
        self.degree = (self.rad * 180) / math.pi

    def change_lonlat_UTM(self):
        x, y = self.transformer.transform(self.longitude, self.latitude)
        self.base_x = x - self.origin_x
        self.base_y = y - self.origin_y
        self.get_logger().info(f"UTM pos: x={self.base_x:.2f}, y={self.base_y:.2f}")

    def moving_obs_point(self):
        goal_x, goal_y = self.waypoints[self.next_obj]
        self.distance = math.sqrt((goal_x - self.base_x)**2 + (goal_y - self.base_y)**2)


def main(args=None):
    rclpy.init(args=args)
    node = HoppingNavi()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
