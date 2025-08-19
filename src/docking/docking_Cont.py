#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float64, Float64MultiArray
from visualization_msgs.msg import Marker

def cal_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

class DockingCont(Node):
    def __init__(self):
        super().__init__('wamv_control')
        qos = QoSProfile(depth=10)

        # Sub
        self.create_subscription(Float64, '/error_psi', self.e_psi_callback, qos)
        self.create_subscription(Float64, '/distance', self.distance_callback, qos)
        self.create_subscription(Float64, '/yaw', self.yaw_callback, qos)
        self.create_subscription(Float64MultiArray, 'UTM_Latlot', self.utm_callback, qos)
        self.create_subscription(Float64, '/next_obj', self.next_obj_callback, qos)

        # Pub
        self.left_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.process)

        # 변수 설정
        self.error_psi = 0.0
        self.distance = 0.0
        self.yaw = 0.0
        self.wamv_x = 0.0
        self.wamv_y = 0.0
        self.center_of_obs = []
        self.path_points = []
        self.next_obj = 0

        # 제어 파라미터
        self.kp = 10.0
        self.kd = 5.0
        self.prev_error = 0.0
        self.close_distance = 3
        self.right_thrust = 550
        self.left_thrust = 550

        # 상태 플래그
        self.avoid_mode = False
        self.current_idx = 0
        self.docking_hold = False
        self.docking_start_time = None

    def e_psi_callback(self, msg): self.error_psi = msg.data
    def distance_callback(self, msg): self.distance = msg.data
    def yaw_callback(self, msg): self.yaw = msg.data
    def utm_callback(self, msg): self.wamv_x, self.wamv_y = msg.data[0], msg.data[1]
    def next_obj_callback(self, msg): self.next_obj = int(round(msg.data))

    # 주기 실행
    def process(self):
        # 도킹 시작 지점에서 멈춤
        if self.next_obj == 1 and self.distance < self.close_distance and not self.docking_hold:
            self.docking_hold = True
            self.docking_start_time = self.get_clock().now()
            self.stop_wamv()
            return

        # 도킹 시작 지점 위치 유지
        if self.docking_hold:
            elapsed = (self.get_clock().now() - self.docking_start_time).nanoseconds * 1e-9
            if elapsed >= 3.0:
                self.get_logger().info("이미지 센싱 위치 도달")
                return
            else:
                self.stop_wamv()
                return


        self.go_straight()

    # 직진 제어
    def go_straight(self):
        turn_val = self.kp * self.error_psi + self.kd * (self.error_psi - self.prev_error)
        self.prev_error = self.error_psi
        self.publish_thrust(self.left_thrust - turn_val, self.right_thrust + turn_val)

    # 정지
    def stop_wamv(self):
        self.publish_thrust(0.0, 0.0)

    # 추력 pub
    def publish_thrust(self, left, right):
        l_msg, r_msg = Float64(), Float64()
        l_msg.data, r_msg.data = float(left), float(right)
        self.left_thruster_pub.publish(l_msg)
        self.right_thruster_pub.publish(r_msg)
        self.get_logger().info(f"[추력] 좌: {left:.1f}, 우: {right:.1f}")

def main(args=None):
    rclpy.init(args=args)
    node = DockingCont()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
