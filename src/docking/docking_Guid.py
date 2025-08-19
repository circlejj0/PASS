#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DockingGuid(Node):
    def __init__(self):
        super().__init__('wamv_perception')

        # Parameter
        self.declare_parameter('camera_topic', '/wamv/sensors/cameras/front_left_camera_sensor/optical/image_raw')
        self.declare_parameter('show', False)  

        gp = self.get_parameter
        self.camera_topic = gp('camera_topic').value
        self.show = bool(gp('show').value)

        # 도착 신호 받음
        self.arrived = False
        self.create_subscription(String, '/nav/status', self.status_cb, 10)

        # Pub
        self.dir_pub = self.create_publisher(String, 'object_direction', 10)

        # Sub
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.camera_topic, self.image_cb, 10)

    # Navigation 코드에서 도착 신호 받았을 때 실행됨
    def status_cb(self, msg: String):
        if msg.data == "ARRIVED_SCAN_P1" and not self.arrived:
            self.arrived = True
            self.get_logger().info("[Guidance] ARRIVED_SCAN_P1 -> start publishing directions")

    # 카메라 Callback
    def image_cb(self, msg: Image):
        if not self.arrived:
            return

        # ROS Image를 OpenCV BGR 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # ========== 임시 ==========
        h, w = frame.shape[:2]
        cx = w // 2

        if cx < w/3: direction = "left"
        elif cx > 2*w/3: direction = "right"
        else: direction = "center"
        # ========== 임시 ===========

def main(args=None):
    rclpy.init(args=args)
    node = DockingGuid()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
