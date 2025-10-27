#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory
import os

class DockingGuid(Node):
    def __init__(self):
        super().__init__('docking_Guid')

        self.declare_parameter('camera_topic', '/wamv/sensors/cameras/front_left_camera_sensor/optical/image_raw')
        self.declare_parameter('show', True)
        self.declare_parameter('targets', ['circle', 'triangle'])
        
        gp = self.get_parameter
        self.camera_topic = gp('camera_topic').value
        self.show = bool(gp('show').value)
        self.targets = [t.lower() for t in gp('targets').value]
        self.arrived = False

        self.create_subscription(String, '/nav/status', self.status_cb, 10)
        self.create_subscription(Image, self.camera_topic, self.image_cb, 10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.errors_pub = self.create_publisher(Float64MultiArray, 'visual_errors', qos_profile)
        self.bridge = CvBridge()

        pkg_share = get_package_share_directory('pass_docking')
        model_path = os.path.join(pkg_share, 'weights', 'vrx_dock.pt')
        self.model = YOLO(model_path)

        self.get_logger().info(f"YOLO model loaded from: {model_path}")
        self.get_logger().info(f"Looking for targets: {self.targets}")

    def status_cb(self, msg: String):
        if msg.data == "ARRIVED_SCAN_P1" and not self.arrived:
            self.arrived = True
            self.get_logger().info("[Guidance] ARRIVED_SCAN_P1 -> Start publishing errors")

    def image_cb(self, msg: Image):
        if not self.arrived:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        center_x = w // 2

        results = self.model(frame, verbose=False)
        
        errors = {target: 0.0 for target in self.targets}
        target_centers = {}

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls_name = self.model.names[cls_id].lower()
                conf = float(box.conf.item())

                if cls_name in self.targets and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    errors[cls_name] = float(cx - center_x)
                    target_centers[cls_name] = (cx, cy)
                    
                    if self.show:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        text = f"{cls_name} ({conf:.2f})"
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        errors_msg = Float64MultiArray()
        errors_msg.data = [errors.get('circle', 0.0), errors.get('triangle', 0.0)]
        self.errors_pub.publish(errors_msg)

        if self.show:
            cv2.line(frame, (center_x, 0), (center_x, h), (0, 255, 255), 1)
            if 'circle' in target_centers:
                cv2.line(frame, (center_x, h), target_centers['circle'], (255, 0, 0), 2)
            if 'triangle' in target_centers:
                cv2.line(frame, (center_x, h), target_centers['triangle'], (0, 0, 255), 2)
            
            cv2.imshow("YOLO Detection", frame)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DockingGuid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
