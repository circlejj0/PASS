#!/usr/bin/env python3
#-*- coding:utf-8 -*-

from pymavlink import mavutil
import time

# Pixhwak 연결
master = mavutil.mavlink_connection('/dev/ttyACM0', baud = 57600)
master.wait_heartbeat()
print("Success to connect pixhwak")

# Parameter settings
params_to_set = {
    b"SYS_AUTOSTART" : 2100, # 프레임 종류 
    b"PWM_MAIN_FUNC1" : 104, # PWM setting 1
    b"PWM_MAIN_FUNC2" : 105, # PWM settung 2
    b"COM_RC_IN_MODE" : 4, # RC 유무
    b"CBRK_IO_SAFETY" : 22027, # Safety 스위치 여부
    b"COM_RC_LOSS_T" : 0, # RC failsafe
    b"COM_ARM_WO_GPS" : 1, # allow arming without GPS
    b"COM_POS_FS_DELAY" : 0, # GPS 위치 잃어버렸을 때 delay 없도록
    b"GPS_1_CONFIG" : 0, # GPS 비활성화
    b"SENS_MAG_MODE" : 0, # 나침반 사용 유무
    b"BAT1_SOURCE" : 0, # 배터리 monitor
    b"PWM_MAIN_MIN1" : 1100, # 최소 PWM
    b"PWM_MAIN_MAX1" : 1900 # 최대 PWM
}
print("Setting Parameter...waiting")
for pid, val in params_to_set.items():
    ptype = (mavutil.mavlink.MAV_PARAM_TYPE_REAL32 #------------------
             if isinstance(val, float)
             else mavutil.mavlink.MAV_PARAM_TYPE_INT32) #------------------
    master.mav.param_set_send( #------------------
        master.target_system,
        master.target_component,
        pid,
        val,
        ptype
    )
    time.sleep(0.2) 

print("Finish setting Parameter")
time.sleep(3)

# Unlock safety switch
print("Unlocking Safety switch")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
    0,
    81,
    0, 0, 0, 0, 0, 0
)
print("Finish unlocking safety switch")

# arming
print("Send to arm command")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1,
    0, 0, 0, 0, 0, 0
)

# Waiting to arm
timeout = time.time() + 15
armed_success = False
while time.time() < timeout:
    msg = master.recv_match(type='HEARTBEAT', blocking=False)
    if msg and (msg.base_mode & mavutil.MAV_MODE_FLAG_SAFETY_ARMED):
        armed_success = True
        print("Success to arm")
        break
    time.sleep(0.2)

if not armed_success:
    print("Fail to arm")
    exit()

print("Finish setting")
