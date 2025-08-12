#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
# from mavros_msgs.msg import OverrideRCIn

import cv2
from ultralytics import YOLO

class DetectImage(Node):

    def __init__(self):
        super().__init__('detect_image_node')
        self.dir_publisher_ = self.create_publisher(String, 'object_direction', 10)
        self.pwm_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        self.model = YOLO('shape.pt')
        self.target_class = 0 # 원하는 도형의 class number 삽입
        self.camera = cv2.VideoCapture(2) # camera number 삽입
        if not self.camera.isOpened():
            self.get_logger().error("Can't open camera")
            return
        
        # pwm 설정 (추후 확인하고 변경해야 할 듯?)
        self.stop_pwm = 1500
        self.forward_pwm = 1550
        self.left_pwm_port = 1400
        self.left_pwm_stbd = 1550
        self.right_pwm_port = 1550
        self.right_pwm_stbd = 1400

        # 이것도 나중에 확인하고 바꿔야 할 듯
        self.port_channel = 0
        self.stbd_channel = 1

        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.camera.read()
        if not ret:
            self.get_logger().warning("Can't read frame")
            return
        results = self.model.predict(source=frame, conf=0.5, show=False, save=False)
        boxes = results[0].boxes

        direction_msg = "Can't found target"
        pwm_cmd = "stop"

        pwm_msg = OverrideRCIn()
        pwm_msg.channels = [1500] * 18

        target_found = False

        if boxes:
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == self.target_class:
                    target_found = True
                    x1, y1, x2, y2 = box.xyxy[0]
                    center_x = (x1 + x2) / 2
                    frame_width = frame.shape[1]

                    if center_x < frame_width / 3:
                        direction_msg = "left"
                        pwm_msg.channels[self.port_channel] = self.right_pwm_port
                        pwm_msg.channels[self.stbd_channel] = self.left_pwm_stbd
                    elif center_x > frame_width * 2 / 3:
                        direction_msg = "right"
                        pwm_msg.channels[self.port_channel] = self.right_pwm_port
                        pwm_msg.channels[self.stbd_channel] = self.right_pwm_stbd
                    else: 
                        direction_msg = "center"
                        pwm_msg.channels[self.port_channel] = self.forward_pwm
                        pwm_msg.channels[self.stbd_channel] = self.forward_pwm
                    break
        else:
            pwm_msg.channels[self.port_channel] = self.stop_pwm
            pwm_msg.channels[self.stbd_channel] = self.stop_pwm

        dir_msg = String()
        dir_msg.data = direction_msg
        self.dir_publisher_.publish(dir_msg)

        self.pwm_pub.publish(pwm_msg)

        self.get_logger().info(f'Direction: {dir_msg.data}, PWM Port: {pwm_msg.channels[self.port_channel]}, Stbd: {pwm_msg.channels[self.stbd_channel]}')

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
