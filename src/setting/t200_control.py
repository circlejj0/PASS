#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
from dronekit import connect, VehicleMode, mavutil

try:
    import keyboard
except ImportError:
    print("Error: 'keyboard' module not found. Please install it using 'pip3 install keyboard'")
    exit()

# 연결해야할 것
# connection_string = 'udp:127.0.0.1:14550' --- 시뮬레이션할 때 씀
connection_string = '/dev/ttyACM0' # 픽스호크 포트

print(f"Connecting to vehicle on: {connection_string}")
vehicle = connect(connection_string, wait_ready=True, timeout=60)

# 전역 변수로 목표 속도와 회전 속도 관리
target_velocity_x = 0.0
target_yaw_rate = 0.0
running = True

def arm_and_guided():
    """기체를 Arm하고 Guided 모드로 변경합니다."""
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

def send_velocity_command(velocity_x, yaw_rate):
    """기체에 속도와 회전각속도 명령을 보냅니다."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000011111000111,
        0, 0, 0,
        velocity_x, 0, 0,
        0, 0, 0,
        0, yaw_rate * (3.1415926535/180.0)) # Yaw rate를 rad/s로 변환
    vehicle.send_mavlink(msg)

def velocity_sender():
    """별도 스레드에서 주기적으로 속도 명령을 전송합니다."""
    while running:
        send_velocity_command(target_velocity_x, target_yaw_rate)
        print(f"DEBUG: Sending Speed={target_velocity_x:.1f} m/s, Yaw Rate={target_yaw_rate:.1f} deg/s")
        time.sleep(0.5) # 0.5초 간격 (2Hz)

# --- 메인 코드 실행 ---
try:
    arm_and_guided()

    # 별도 스레드에서 velocity_sender 함수 시작
    sender_thread = threading.Thread(target=velocity_sender)
    sender_thread.start()

    print("\n--- Controls (Hold keys, no need to press Enter) ---")
    print("w: Speed up, s: Speed down, a: Turn Left, d: Turn Right, space: Stop, q: Quit")
    print("-----------------------------------------------------")

    while running:
        # 'a'나 'd' 키가 눌리지 않으면 회전은 멈춤
        if not (keyboard.is_pressed('a') or keyboard.is_pressed('d')):
            target_yaw_rate = 0.0

        if keyboard.is_pressed('w'):
            target_velocity_x += 0.2
        elif keyboard.is_pressed('s'):
            target_velocity_x -= 0.2
        elif keyboard.is_pressed('a'):
            target_yaw_rate = -30.0 # 반시계 방향 30 deg/s
        elif keyboard.is_pressed('d'):
            target_yaw_rate = 30.0  # 시계 방향 30 deg/s
        elif keyboard.is_pressed('space'):
            target_velocity_x = 0.0
            target_yaw_rate = 0.0
        elif keyboard.is_pressed('q'):
            running = False
            
        # 속도 제한 (최대 2m/s, 최저 0m/s)
        target_velocity_x = max(0.0, min(target_velocity_x, 2.0))
        time.sleep(0.1)

except Exception as e:
    print(f"An error occurred: {e}")
    running = False

finally:
    running = False
    if 'sender_thread' in locals() and sender_thread.is_alive():
        sender_thread.join()
    
    print("Stopping vehicle...")
    if vehicle.armed:
        send_velocity_command(0, 0)
        
    vehicle.close()
    print("Done.")
