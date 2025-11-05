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

        # [!! MAVROS !!] 실제 보트의 카메라 토픽으로 변경해야 합니다.
        self.declare_parameter('camera_topic', '/camera/image_raw') 
        self.declare_parameter('show', True)
        self.declare_parameter('targets', ['circle', 'triangle', 'rectangle']) 
        
        gp = self.get_parameter
        # [!! MAVROS !!] 파라미터에서 카메라 토픽을 읽어옵니다.
        self.camera_topic = gp('camera_topic').value
        self.show = bool(gp('show').value)
        self.targets = [t.lower() for t in gp('targets').value]
        
        self.arrived = False 
        self.order_determined = False

        self.create_subscription(String, '/nav/status', self.status_cb, 10)
        
        # [!! MAVROS !!] 파라미터로 받은 실제 카메라 토픽을 구독합니다.
        self.create_subscription(Image, self.camera_topic, self.image_cb, 10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.errors_pub = self.create_publisher(Float64MultiArray, 'visual_errors', qos_profile)
        self.order_pub = self.create_publisher(String, '/docking/target_order', qos_profile)
        
        self.bridge = CvBridge()

        pkg_share = get_package_share_directory('pass_docking') 
        model_path = os.path.join(pkg_share, 'weights', 'vrx_dock.pt')
        self.model = YOLO(model_path)

        self.get_logger().info(f"YOLO model loaded from: {model_path}")
        self.get_logger().info(f"Looking for targets: {self.targets}")
        if len(self.targets) != 3:
            self.get_logger().warn(f"Warning: Expected 3 targets, but got {len(self.targets)}. Order logic may fail.")

    def status_cb(self, msg: String):
        if msg.data == "ARRIVED_SCAN_P1" and not self.arrived:
            self.arrived = True
            self.get_logger().info("[Guidance] ARRIVED_SCAN_P1 -> Start YOLO detection and error publishing")

    def image_cb(self, msg: Image):
        # (이하 로직은 VRX와 동일)
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
        
        all_found = all(t in target_centers for t in self.targets)
        
        if all_found and not self.order_determined:
            self.order_determined = True 
            
            sorted_targets = sorted(target_centers.items(), key=lambda item: item[1][0])
            ordered_names = [name for name, center in sorted_targets]
            order_str = ",".join(ordered_names)
            
            self.get_logger().info(f"--- [!!] All targets detected! Order: {order_str} [!!] ---")
            
            self.order_pub.publish(String(data=order_str))

        errors_msg = Float64MultiArray()
        errors_msg.data = [errors.get(t, 0.0) for t in self.targets]
        self.errors_pub.publish(errors_msg)

        if self.show:
            cv2.line(frame, (center_x, 0), (center_x, h), (0, 255, 255), 1)
            for name, center in target_centers.items():
                if name == 'circle': color = (255, 0, 0)
                elif name == 'triangle': color = (0, 0, 255)
                elif name == 'rectangle': color = (255, 0, 255) 
                else: color = (255, 255, 0)
                cv2.line(frame, (center_x, h), center, color, 2)
            
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
