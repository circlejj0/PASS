#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv2
from ultralytics import YOLO

class DetectImage(Node):

    def __init__(self):
        super().__init__('detect_image_node')
        self.publisher_ = self.create_publisher(String, 'object_direction', 10)
        self.model = YOLO('best.pt')
        self.target_class = 0 # 원하는 도형의 class number 삽입
        self.camera = cv2.VideoCapture(2)
        if not self.camera.isOpened():
            self.get_logger().error("Can't open camera")
            return
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.camera.read()
        if not ret:
            self.get_logger().warning("Can't read frame")
            return
        results = self.model.predict(source=frame, conf=0.5, show=False, save=False)
        boxes = results[0].boxes

        direction_msg = "Can't found target"

        if boxes:
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == self.target_class:
                    x1, y1, x2, y2 = box.xyxy[0]
                    center_x = (x1 + x2) / 2
                    frame_width = frame.shape[1]

                    if center_x < frame_width / 3:
                        direction_msg = "left"
                    elif center_x > frame_width * 2 / 3:
                        direction_msg = "right"
                    else: 
                        direction_msg = "center"
                    break
        msg = String()
        msg.data = direction_msg
        self.publisher_.publish(msg)
        self.get_logger().info(f'Direction Publish: "{msg.data}"')

    def destroy_node(self):
        self.camera.release()
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectImage()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
