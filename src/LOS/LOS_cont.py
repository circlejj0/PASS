#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from pyproj import Transformer
from std_msgs.msg import Float64, Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def cal_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class WamvControl(Node):
    def __init__(self):
        super().__init__('wamv_control')
        qos = QoSProfile(depth=10)

        # 구독
        self.create_subscription(Float64, '/error_psi', self.e_psi_callback, qos_profile=qos)
        self.create_subscription(Float64, '/distance', self.distance_callback, qos_profile=qos)
        self.create_subscription(Float64, '/yaw', self.yaw_callback, qos_profile=qos)
        self.create_subscription(Float64MultiArray, 'UTM_Latlot', self.utm_callback, qos_profile=qos)
        self.create_subscription(Marker, '/obs_center_point', self.obs_callback, qos_profile=qos)
        self.create_subscription(Marker, '/circle_load', self.circle_callback, qos_profile=qos)

        # 퍼블리시
        self.left_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)



        self.timer = self.create_timer(0.1, self.process)

        # 내부 상태
        self.error_psi = 0.0
        self.distance = 0.0
        self.yaw = 0.0
        self.wamv_x = 0.0
        self.wamv_y = 0.0
        self.center_of_obs = []

        self.kp = 10.0
        self.kd = 5.0
        self.prev_error = 0.0
        self.close_distance = 3
        self.right_thrust = 750
        self.left_thrust = 750

        self.avoid_mode = False
        self.avoid_target = None
        self.current_idx = 0
        self.path_points = []

        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True)
        self.docking_hold = False
        self.docking_start_time = None
        self.next_obj = 0  # 초기 목표 index
        self.create_subscription(Float64, '/next_obj', self.next_obj_callback, qos_profile=qos)



    def e_psi_callback(self, msg):
        self.error_psi = msg.data

    def distance_callback(self, msg):
        self.distance = msg.data

    def yaw_callback(self, msg):
        self.yaw = msg.data

    def utm_callback(self, msg):
        self.wamv_x = msg.data[0]
        self.wamv_y = msg.data[1]                           ###################### 콜백

    def obs_callback(self, msg):
        self.center_of_obs = msg.points

    def circle_callback(self, msg):
        self.path_points = []
        for p in msg.points:
            x, y, z = self.transformer.transform(p.y, p.x, p.z)  # (lon, lat)
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = z
            self.path_points.append(pt)

    def process(self):
        if self.next_obj == 1 and self.distance < self.close_distance and not self.docking_hold:
            self.docking_hold = True
            self.docking_start_time = self.get_clock().now()
            self.stop_wamv()
            return

        if self.docking_hold:
            elapsed = (self.get_clock().now() - self.docking_start_time).nanoseconds * 1e-9
            if elapsed >= 3.0:
                self.get_logger().info("도킹 완료")
                return
            else:
                self.stop_wamv()
                return

        if self.avoid_mode:
            self.follow_circle_path()
            return

        if self.avoid_check():
            return

        self.go_straight()




    def avoid_check(self):
        if self.avoid_mode or not self.center_of_obs:
            return False

        front = [p for p in self.center_of_obs if p.x > 0.0 and abs(p.y) < 2.0 and math.hypot(p.x, p.y) < 13.0]
        if not front:
            return False                                                                                             ############ 회피 판단

        close_obs = min(front, key=lambda p: math.hypot(p.x, p.y))
        if math.hypot(close_obs.x, close_obs.y) < 10.0 and self.path_points:
            self.avoid_mode = True
            self.current_idx = 0
            self.get_logger().info("회피 경로 진입")
            return True

        return False

    def follow_circle_path(self):
        if self.current_idx >= len(self.path_points):
            self.get_logger().info("회피 종료")
            self.avoid_mode = False
            return

        target = self.path_points[self.current_idx]
        dx = target.x - self.wamv_x
        dy = target.y - self.wamv_y
        psi_d = math.atan2(dy, dx)
        yaw_err = cal_angle(psi_d - self.yaw)                                              ############ 반경 추종, (사용 X)

        turn_val = self.kp * yaw_err + self.kd * (yaw_err - self.prev_error)
        self.prev_error = yaw_err
        self.publish_thrust(750 - turn_val, 750 + turn_val)

        if math.hypot(dx, dy) < 1.0:
            self.current_idx += 1

    def go_straight(self):
        turn_val = self.kp * self.error_psi + self.kd * (self.error_psi - self.prev_error)
        self.prev_error = self.error_psi                                                        ####### 직진, (PID 제어)
        self.publish_thrust(self.left_thrust - turn_val, self.right_thrust + turn_val)

    def stop_wamv(self):
        self.publish_thrust(0.0, 0.0)

    def publish_thrust(self, left, right):
        l_msg, r_msg = Float64(), Float64()
        l_msg.data, r_msg.data = float(left), float(right)
        self.left_thruster_pub.publish(l_msg)
        self.right_thruster_pub.publish(r_msg)
        self.get_logger().info(f"[추력] 좌: {left:.1f}, 우: {right:.1f}")

    def next_obj_is_last(self):
    # WamvNavigation 노드에서 사용하는 waypoint 인덱스와 공유하거나 아래와 같이 직접 정의
    # 예: 총 2개의 경유점 중 마지막이면 True 반환
        return self.current_idx == len(self.path_points) - 1
    
    def next_obj_callback(self, msg):
        self.next_obj = int(round(msg.data))





def main(args=None):
    rclpy.init(args=args)
    node = WamvControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
