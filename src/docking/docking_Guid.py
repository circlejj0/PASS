#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class DockingGuid(Node):
    def __init__(self):
        super().__init__('docking_guid')

        self.declare_parameter('camera_topic', '/wamv/sensors/cameras/front_left_camera_sensor/optical/image_raw')
        self.declare_parameter('show', True)
        self.declare_parameter('target_shape', 'rectangle')
        self.declare_parameter('target_shape_color', 'red')

        gp = self.get_parameter
        self.camera_topic = gp('camera_topic').value
        self.show = bool(gp('show').value)
        self.target_shape = gp('target_shape').value.lower()
        self.target_shape_color = gp('target_shape_color').value.lower()

        self.arrived = False

        # Sub
        self.create_subscription(String, '/nav/status', self.status_cb, 10)
        self.create_subscription(Image, self.camera_topic, self.image_cb, 10)
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Pub
        self.error_pub = self.create_publisher(Float64, 'visual_error', qos_profile)
        
        self.bridge = CvBridge()

        self.get_logger().info(f"DockingGuid Node started. Looking for a '{self.target_shape_color} {self.target_shape}'.")

    def status_cb(self, msg: String):
        if msg.data == "ARRIVED_SCAN_P1" and not self.arrived:
            self.arrived = True
            self.get_logger().info("[Guidance] ARRIVED_SCAN_P1 -> Start publishing error")

    def detect_shape(self, contour):
        shape = "unidentified"
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) == 3: shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        elif len(approx) > 4: shape = "circle"
        return shape

    def image_cb(self, msg: Image):
        if not self.arrived: return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 색깔 필터링
        if self.target_shape_color == 'red':
            lower1, upper1 = np.array([0, 100, 100]), np.array([10, 255, 255])
            lower2, upper2 = np.array([170, 100, 100]), np.array([179, 255, 255])
            mask1, mask2 = cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2)
            mask = cv2.add(mask1, mask2)
        else: mask = np.zeros(frame.shape[:2], dtype='uint8')

        # 모폴로지 연산
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        visual_error_msg = Float64(data=0.0)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 150:
                shape = self.detect_shape(largest_contour)
                if shape == self.target_shape:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        # 화면 중앙과 목표 도형의 픽셀 오차 계산
                        error = cx - (w / 2)
                        visual_error_msg.data = float(error)

                        if self.show:
                            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
                            cv2.circle(frame, (cx, int(M["m01"] / M["m00"])), 7, (0, 0, 255), -1)
                            text = f"Error: {error:.1f} pixels"
                            cv2.putText(frame, text, (cx - 70, int(M["m01"] / M["m00"]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.error_pub.publish(visual_error_msg)

        if self.show:
            cv2.imshow("Result Frame", frame)
            cv2.imshow("Cleaned Mask", mask_cleaned)
            cv2.waitKey(1)
            
def main(args=None):
    rclpy.init(args=args)
    node = DockingGuid()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
