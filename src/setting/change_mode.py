#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# RC 조종기로 모드 전환하는 코드

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import OverrideRCIn, RCIn, State
from mavros_msgs.srv import SetMode, CommandBool

class ThrusterControl(Node):
    def __init__(self):
        super().__init__('thruster_control')
        self.override_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.rc_sub = self.create_subscription(RCIn, '/mavros/rc/in', self.rc_cb, 10)

        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')

        self.current_mode = "MANUAL"
        self.switch_val = 1500

        # Arm 
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for arming service...")
        arm_req = CommandBool.Request()
        arm_req.value = True
        self.arm_client.call_async(arm_req)

        self.timer = self.create_timer(0.1, self.loop)

    def state_cb(self, msg):
        self.current_mode = msg.mode

    def rc_cb(self, msg: RCIn):
        self.switch_val = msg.channels[5]

    def set_mode(self, mode):
        if not self.mode_client.wait_for_service(timeout_sec=1.0):
            return
        req = SetMode.Request()
        req.custom_mode = mode
        self.mode_client.call_async(req)
        self.get_logger().info(f"🔄 Mode change request: {mode}")

    def loop(self):
        msg = OverrideRCIn()
        msg.channels = [65535]*18

        if self.switch_val < 1200:
            msg.channels[0] = 1500
            msg.channels[1] = 1500
            if self.current_mode != "HOLD":
                self.set_mode("HOLD")
            self.get_logger().info("🛑 Stop mode")

        elif self.switch_val < 1700:
            if self.current_mode != "MANUAL":
                self.set_mode("MANUAL")
            self.get_logger().info("🎮 RC Manual mode")

        else:
            msg.channels[0] = 1600
            msg.channels[1] = 1600
            if self.current_mode != "OFFBOARD":
                self.set_mode("OFFBOARD")
            self.get_logger().info("🤖 Computer Offboard mode")

        self.override_pub.publish(msg)

def main():
    rclpy.init()
    node = ThrusterControl()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
