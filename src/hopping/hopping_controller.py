#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pixhawk hopping code

import time
import math
from dronekit import connect, VehicleMode, mavutil
from pyproj import Transformer
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import transforms3d.euler as euler

# Pixhawk port number
connection_string = '/dev/ttyACM0'
print(f"Connecting to vehicle on: {connection_string}")
vehicle = connect(connection_string, wait_ready=True, timeout=60)

class RvizVisualizer(Node):
    def __init__(self):
        super().__init__('dronekit_rviz_visualizer')
        self.path_pub = self.create_publisher(Path, '/boat_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/boat_waypoints', 10)
        
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

    def publish_path(self, utm_x, utm_y, yaw_rad):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = utm_x
        pose.pose.position.y = utm_y
        
        # Yaw(rad)를 Quaternion으로 변환
        q = euler.euler2quat(0, 0, yaw_rad)
        pose.pose.orientation.w = q[0]
        pose.pose.orientation.x = q[1]
        pose.pose.orientation.y = q[2]
        pose.pose.orientation.z = q[3]

        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    def publish_waypoints(self, waypoints_utm, next_obj_index):
        marker_array = MarkerArray()
        for i, (wx, wy) in enumerate(waypoints_utm):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.scale.x = 2.0
            marker.scale.y = 2.0
            marker.scale.z = 2.0
            marker.color.a = 1.0

            if i == next_obj_index:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

def arm_and_guided(vehicle):
    print("Waiting for vehicle to become armable...")
    while not vehicle.is_armable:
        time.sleep(1)
    
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    print("Waiting for arming...")
    while not vehicle.armed:
        time.sleep(1)
    print("Vehicle armed and in GUIDED mode!")

def send_velocity_command(vehicle, velocity_x, yaw_rate_deg):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000011111000111,
        0, 0, 0, velocity_x, 0, 0, 0, 0, 0,
        0, yaw_rate_deg * (math.pi/180.0))
    vehicle.send_mavlink(msg)

def main(args=None):
    rclpy.init(args=args)
    visualizer = RvizVisualizer()

    try:
        arm_and_guided(vehicle)
        
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True)
        waypoints_lonlat = [
            (129.104375, 35.133902),
            (129.104595, 35.133731),
            (129.106874, 35.133931),
            (129.106810, 35.134896),
            (129.105754, 35.134853)
        ]
        
        origin_utm = transformer.transform(waypoints_lonlat[0][0], waypoints_lonlat[0][1])
        waypoints_local = [(x - origin_utm[0], y - origin_utm[1]) for x, y in [transformer.transform(lon, lat) for lon, lat in waypoints_lonlat]]

        next_obj = 0
        arrival_radius = 10.0
        target_speed = 0.1 # 1.5
        kp, kd, prev_heading_error = 30.0, 10.0, 0.0

        while next_obj < len(waypoints_local) and rclpy.ok():
            visualizer.publish_waypoints(waypoints_local, next_obj)
            
            lat = vehicle.location.global_relative_frame.lat
            lon = vehicle.location.global_relative_frame.lon
            if lat is None or lon is None:
                print("Waiting for GPS signal...")
                time.sleep(1)
                continue
            
            current_utm = transformer.transform(lon, lat)
            current_local = (current_utm[0] - origin_utm[0], current_utm[1] - origin_utm[1])
            
            goal_local = waypoints_local[next_obj]
            distance_to_goal = math.sqrt((goal_local[0] - current_local[0])**2 + (goal_local[1] - current_local[1])**2)
            
            print(f"Waypoint {next_obj}: Distance = {distance_to_goal:.1f} m")

            if distance_to_goal < arrival_radius:
                print(f"--- Waypoint {next_obj} Reached! ---")
                next_obj += 1 
                if next_obj >= len(waypoints_local):
                    break
                send_velocity_command(vehicle, 0, 0)
                time.sleep(10)
                continue
            
            desired_heading_rad = math.atan2(goal_local[0] - current_local[0], goal_local[1] - current_local[1])
            current_heading_rad = vehicle.attitude.yaw
            heading_error = desired_heading_rad - current_heading_rad
            while heading_error > math.pi: heading_error -= 2 * math.pi
            while heading_error < -math.pi: heading_error += 2 * math.pi
            
            turn_rate = kp * heading_error + kd * (heading_error - prev_heading_error)
            prev_heading_error = heading_error
            
            send_velocity_command(vehicle, target_speed, turn_rate)
            
            visualizer.publish_path(current_local[0], current_local[1], current_heading_rad)
            rclpy.spin_once(visualizer, timeout_sec=0)
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    finally:
        print("Stopping vehicle...")
        send_velocity_command(vehicle, 0, 0)
        vehicle.close()
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
