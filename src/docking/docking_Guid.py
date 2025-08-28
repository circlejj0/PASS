#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DockingGuid(Node):
    def __init__(self):
        super().__init__('wamv_perception')

        self.declare_parameter('camera_topic', '/wamv/sensors/cameras/front_left_camera_sensor/optical/image_raw')
        self.declare_parameter('show', True)
        # 탐지할 도형 지정
        self.declare_parameter('target_shape', 'rectangle')
        # 탐지할 색깔 지정
        self.declare_parameter('target_shape_color', 'red')

        gp = self.get_parameter
        self.camera_topic = gp('camera_topic').value
        self.show = bool(gp('show').value)
        self.target_shape = gp('target_shape').value.lower()
        self.target_shape_color = gp('target_shape_color').value.lower()

        self.arrived = False
        self.create_subscription(String, '/nav/status', self.status_cb, 10)
        self.dir_pub = self.create_publisher(String, 'object_direction', 10)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.camera_topic, self.image_cb, 10)

        self.get_logger().info(f"DockingGuid Node started. Looking for a '{self.target_shape_color} {self.target_shape}'.")

    def status_cb(self, msg: String):
        if msg.data == "ARRIVED_SCAN_P1" and not self.arrived:
            self.arrived = True
            self.get_logger().info("[Guidance] ARRIVED_SCAN_P1 -> Start publishing directions")

    def detect_shape(self, contour):
        shape = "unidentified"
        peri = cv2.arcLength(contour, True)
        # 컨투어
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # 꼭짓점 개수를 이용하여 도형 탐지
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        elif len(approx) > 4:
            shape = "circle"
            
        return shape

    def image_cb(self, msg: Image):
        if not self.arrived:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        
        # 이진화 -> 그레이스케일로 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 목표 색상 튜닝
        if self.target_shape_color == 'yellow':
            lower_color = np.array([20, 50, 50])
            upper_color = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower_color, upper_color)

        elif self.target_shape_color == 'red':
            lower1 = np.array([0, 50, 50])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 50, 50])
            upper2 = np.array([179, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.add(mask1, mask2)

        elif self.target_shape_color == 'blue':
            lower_color = np.array([110, 50, 50])
            upper_color = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_color, upper_color)
        else:
            mask = np.zeros(frame.shape[:2], dtype='uint8')

        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

        # 컨투어 검출
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        direction_msg = String(data="none")

        if contours:
            # 면적이 가장 큰 컨투어 선택
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 최소 면적 이상으로
            if cv2.contourArea(largest_contour) > 150:
                shape = self.detect_shape(largest_contour)
                
                # 탐지된 도형이 원하는 도형인지 확인함
                if shape == self.target_shape:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # 방향 결정
                        center_tolerance = w * 0.15 # 화면 중앙 영역의 폭 (15%)
                        if cx < (w / 2) - center_tolerance:
                            direction_msg.data = "left"
                        elif cx > (w / 2) + center_tolerance:
                            direction_msg.data = "right"
                        else:
                            direction_msg.data = "center"

                        # 시각화과정
                        if self.show:
                            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
                            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                            cv2.putText(frame, f"{self.target_shape_color} {shape}: {direction_msg.data}",
                                        (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.dir_pub.publish(direction_msg)

        if self.show:
            cv2.imshow("Result Frame", frame)
            cv2.imshow("Cleaned Mask", mask_cleaned)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DockingGuid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
