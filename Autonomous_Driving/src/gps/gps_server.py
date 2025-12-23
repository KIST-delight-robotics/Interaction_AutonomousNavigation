#!/usr/bin/env python3
"""
FastAPI server that

▪ loads one-or-more GPS-path CSV files (lat,lon columns)
▪ exposes /paths (JSON)           – pre-recorded tracks
▪ exposes /gps   (SSE)            – live position
▪ serves  /      (HTML)           – Leaflet map that draws all of the above
"""

from __future__ import annotations
import argparse, asyncio, csv, json, os, random, threading, time
from typing import List, Tuple
from queue import Queue
import struct, math
import glob

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

import asyncio
import threading
import serial
import time
import math
from queue import Queue
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse

# GPS
import base64
import socket
import threading
import serial
import sys
from datetime import datetime
from queue import Queue
from pyubx2 import UBXReader, UBXMessage, SET
from pynmeagps.nmeamessage import NMEAMessage
from termcolor import colored

import folium
import csv
import argparse

# Sport mode high state
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# Sport mode control
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
from dataclasses import dataclass

# camera streaming
import cv2

# custom code
from controller import PriorityPD
#from states import KalmanFilter2D_ENU
from heading import HeadingOffsetEstimator
from utils import *

# --------------------------------
# model configuration
# --------------------------------
EXTENSION_DOCK = 'go2-edu' # or 'jetson'
REACH_TOL = 0.5 # threshold distance for reaching goal (meter)

if EXTENSION_DOCK == 'go2-edu':
    import d435i # realsense visualization

# ---------------------------------
# Live-GPS shared state
# ---------------------------------

#gps_pos: Tuple[float | None, float | None] = (37.598163, 127.045357)  # (lat, lon)
#gps_lat = 37.598163
#gps_lon = 127.045357
gps_lat = None
gps_lon = None
latlon_queue: "Queue[Tuple[float, float]]" = Queue(maxsize=1)

key = 0
flag = 0
buff = {}
angularVelocity = [0, 0, 0]
acceleration    = [0, 0, 0]
magnetometer    = [0, 0, 0]
angle_degree    = [0, 0, 0]
odom_pos        = [0, 0, 0]
odom_angle      = [0, 0, 0]
yaw_offset      = 0.0
yaw_offset_imu  = 0.0
global_heading = 0.0
go2_status = ''
go2_command = ''
pub_flag = [True, True]

# ---------------------------------
# state variable
# ---------------------------------
 
mission_active = False
selected_path_file = None
ALL_PATHS = {}


# --------------------------------
# camera
# --------------------------------
#current_frame = np.zeros((480,1280,3), np.uint8)


# GPS 
def imu_print(text):
    print(colored("IMU:\t", 'green'), text)

def ubx_print(text):
    print(colored("GPS:\t", 'green'), text)

def ntrip_print(text):
    print(colored("NTRIP:\t", 'green'), text)

def decode_mode(msg_mode):
    if msg_mode == 1:
        return 'stop walking'
    elif msg_mode == 2:
        return 'standing'
    elif msg_mode == 3:
        return 'walking'
    elif msg_mode == 5:
        return 'lie down'
    elif msg_mode == 6:
        return 'stand up'
    elif msg_mode == 7:
        return 'damp'
    else:
        return str(msg_mode)

def decode_gait(msg_mode):
    if msg_mode == 0:
        return 'idle'
    elif msg_mode == 1:
        return 'trot'
    elif msg_mode == 2:
        return 'run'
    elif msg_mode == 3:
        return 'for-climb'
    elif msg_mode == 4:
        return 'rev-climb'
    else:
        return f'gait({msg_mode})'

def decode_speed(msg_mode):
    if msg_mode == -1:
        return 'slow'
    elif msg_mode == 0:
        return 'normal'
    elif msg_mode == 1:
        return 'fast'
    else:
       return f'speed({msg_mode})'

# ─────────────────────────────────────────────────────────────
# Pre-recorded path loader
# ─────────────────────────────────────────────────────────────
def load_paths(files: list[str]) -> list[dict]:
    """Return [{'file': 'xyz.txt', 'coords': [[lat,lon], …]}, …]"""
    paths = []
    for f in files:
        coords = []
        try:
            with open(f, newline="") as fp:
                reader = csv.DictReader(fp)
                hdr = reader.fieldnames or []
                # detect columns (case-insensitive)
                lat_key = next((h for h in hdr if h.strip().lower() in
                                ("lat","latitude","y","gps_lat")), None)
                lon_key = next((h for h in hdr if h.strip().lower() in
                                ("lon","lng","longitude","x","gps_lon")), None)
                if not (lat_key and lon_key):
                    print(f"[load_paths] {f}: header {hdr} lacks lat/lon")
                    continue
                for row in reader:
                    try:
                        coords.append([float(row[lat_key]), float(row[lon_key])])
                    except ValueError:
                        pass            # skip bad rows
        except FileNotFoundError:
            print(f"[load_paths] {f}: not found")
        if coords:
            paths.append({"file": os.path.basename(f), "coords": coords})
            print(f"[load_paths] {f}: loaded {len(coords)} points")
    return paths


# ──────────────────────────────────────────────
# IMU Thread
# ──────────────────────────────────────────────
def checkSum(list_data, check_data):
    data = bytearray(list_data)
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for i in range(8):
            if (crc & 1) != 0:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    return hex(((crc & 0xff) << 8) + (crc >> 8)) == hex(check_data[0] << 8 | check_data[1])

def hex_to_ieee(raw_data):
    ieee_data = []
    raw_data.reverse()
    for i in range(0, len(raw_data), 4):
        data2str =hex(raw_data[i] | 0xff00)[4:6] + hex(raw_data[i + 1] | 0xff00)[4:6] + hex(raw_data[i + 2] | 0xff00)[4:6] + hex(raw_data[i + 3] | 0xff00)[4:6]
        ieee_data.append(struct.unpack('>f', bytes.fromhex(data2str))[0])
    ieee_data.reverse()
    return ieee_data

def handleSerialData(raw_data):
    global buff, key, magnetometer, angle_degree, acceleration, angularVelocity, pub_flag
    buff[key] = raw_data

    key += 1
    if buff[0] != 0xaa:
        key = 0
        return
    if key < 3:
        return
    if buff[1] != 0x55:
        key = 0
        return
    if key < buff[2] + 5:  # 根据数据长度位的判断, 来获取对应长度数据
        return

    else:
        data_buff = list(buff.values())  # 获取字典所以 value

        if buff[2] == 0x2c and pub_flag[0]:
            if checkSum(data_buff[2:47], data_buff[47:49]):
                data = hex_to_ieee(data_buff[7:47])
                angularVelocity = data[1:4]
                acceleration = data[4:7]
                magnetometer = data[7:10]
            else:
                print('校验失败')
            pub_flag[0] = False
        elif buff[2] == 0x14 and pub_flag[1]:
            if checkSum(data_buff[2:23], data_buff[23:25]):
                data = hex_to_ieee(data_buff[7:23])
                #angle_degree = data[1:4]
                angle_degree[0] = data[1]
                angle_degree[1] = data[2]
                angle_degree[2] = data[3]
            else:
                print('校验失败')
            pub_flag[1] = False
        else:
            print("该数据处理类没有提供该 " + str(buff[2]) + " 的解析")
            print("或数据错误")
            buff = {}
            key = 0

        buff = {}
        key = 0
        if pub_flag[0] == True or pub_flag[1] == True:
            return
        pub_flag[0] = pub_flag[1] = True
        acc_k = math.sqrt(acceleration[0] ** 2 + acceleration[1] ** 2 + acceleration[2] ** 2)
        """
        print('''
            acc (m/s²)：
                x: %.2f
                y: %.2f
                z: %.2f

            angular (rad/s)：
                x: %.2f
                y: %.2f
                z: %.2f

            欧拉角(°)：
                x: %.2f
                y: %.2f
                z: %.2f

            磁场：
                x: %.2f
                y: %.2f
                z: %.2f
''' % (acceleration[0] * -9.8 / acc_k, acceleration[1] * -9.8 / acc_k, acceleration[2] * -9.8 / acc_k,
       angularVelocity[0], angularVelocity[1], angularVelocity[2],
       angle_degree[0], angle_degree[1], angle_degree[2],
       magnetometer[0], magnetometer[1], magnetometer[2]
      ))
        """

def imu_thread(imu_port='/dev/ttyUSB0', imu_baud=921600):
    imu_ser = serial.Serial(imu_port, imu_baud, timeout=0.01)
    imu_print(f"Listening on {imu_ser.port}@{imu_ser.baudrate}")

    while(True):
        try:
            buff_count = imu_ser.inWaiting()
        except Exception as e:
            print("exception:" + str(e))
            print("imu 失去连接，接触不良，或断线")
            exit(0)
        else:
            if buff_count > 0:
                buff_data = imu_ser.read(buff_count)
                for i in range(0, buff_count):
                    handleSerialData(buff_data[i])
 

# ──────────────────────────────────────────────
# GPS Threads
# ──────────────────────────────────────────────
def make_gga(lat: float, lon: float) -> bytes:
    """위경도를 받아 NMEA GGA 문장(ASCII byte) 생성."""
    t = datetime.utcnow().strftime("%H%M%S.00")
    # 위도 DD → DDMM.MMMM
    lat_d = int(abs(lat))
    lat_m = (abs(lat) - lat_d) * 60
    lat_dir = 'N' if lat >= 0 else 'S'
    # 경도 DDD → DDDMM.MMMM
    lon_d = int(abs(lon))
    lon_m = (abs(lon) - lon_d) * 60
    lon_dir = 'E' if lon >= 0 else 'W'
    core = (f"GPGGA,{t},{lat_d:02d}{lat_m:07.4f},{lat_dir},"
            f"{lon_d:03d}{lon_m:07.4f},{lon_dir},1,12,1.0,0.0,M,0.0,M,,")
    chk = 0
    for c in core:
        chk ^= ord(c)
    sentence = f"${core}*{chk:02X}\r\n"
    return sentence.encode('ascii')

def ntrip_thread(caster, port, mountpoint, user, password, ser, latlon_queue: Queue):
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    req = (
        f"GET /{mountpoint} HTTP/1.1\r\n"
        f"Host: {caster}:{port}\r\n"
        "User-Agent: NTRIP PythonClient/1.0\r\n"
        "Accept: */*\r\n"
        "Connection: close\r\n"
        f"Authorization: Basic {auth}\r\n"
        "\r\n"
    ).encode('ascii')

    # ubx_thread 에서 위치가 들어올 때까지 대기
    lat, lon = latlon_queue.get(block=True)
    ntrip_print(f"Got approx pos: Lat={lat}, Lon={lon}")

    while True:
        try:
            ntrip_print(f"Connecting to {caster}:{port}/{mountpoint} …")
            sock = socket.create_connection((caster, port), timeout=10)
            sock.sendall(req)

            # 응답 헤더 스킵
            buf = b""
            while b"\r\n\r\n" not in buf:
                buf += sock.recv(1)
            sock.settimeout(None)

            # 최초 GGA 전송
            gga = make_gga(lat, lon)
            sock.sendall(gga)
            ntrip_print(f"Sent GGA: {gga.decode().strip()}")

            ntrip_print("RTCM stream started")
            while True:
                chunk = sock.recv(1024)
                if not chunk:
                    raise ConnectionError("NTRIP stream closed")
                ser.write(chunk)
                
        except Exception as e:
            ntrip_print(f"Error: {e}. Retrying in 5s…")
            time.sleep(5)

def ubx_thread(ser, latlon_queue: Queue):
    global gps_lat, gps_lon, odom_pos, odom_angle
    ubr = UBXReader(ser, protfilter=3)
    ubx_print(f"Listening on {ser.port}@{ser.baudrate}")
    start_time = time.time()
    while True:
        raw, msg = ubr.read()
        # UBX NAV-PVT
        if isinstance(msg, UBXMessage) and msg.identity == 'NAV-PVT':
            lat = msg.lat * 1e-7
            lon = msg.lon * 1e-7
            fixtype = msg.fixType
            #print(f"[{msg.iTOW} ms] fixType={fixtype}  Lat={lat:.7f} Lon={lon:.7f}")
            if fixtype >= 2 and latlon_queue.empty():
                latlon_queue.put((lat, lon))
                gps_lat, gps_lon = lat, lon

        # NMEA GGA (GNGGA, GPGGA 등)
        elif isinstance(msg, NMEAMessage) and msg.identity.endswith('GGA'):
            # pynmeagps는 msg.lat/msg.lon이 이미 float
            try:
                if latlon_queue.empty():
                    if isinstance(msg.lat, float):
                        latlon_queue.put((msg.lat, msg.lon))
                gps_lat = msg.lat
                gps_lon = msg.lon
                gps_hdop = msg.HDOP
               
                #pos_queue.put((gps_lat, gps_lon, odom_pos[0], odom_pos[1], odom_angle[2]))
                #print(f"[GGA] Lat={gps_lat:.7f}, Lon={gps_lon:.7f}, Hdop={msg.HDOP}, quality={msg.quality} (2:DGPS, 5:rtk float, 4: rtk fix(final quality))")
            except:
                if time.time() - start_time > 10.0:
                    ubx_print("bad signal.")
                    ubx_print(msg)
                    start_time = time.time()

        #ubx_print(f"{gps_lat}, {gps_lon}")



# ─────────────────────────────────────────────────────────────
# Unitree Go2 High State
# ─────────────────────────────────────────────────────────────
def HighStateHandler(msg: SportModeState_):
    global go2_status, odom_pos, odom_angle
    go2_mode = decode_mode(msg.mode)
    _odom_pos = msg.position
    _odom_angle = quat_to_euler_deg(msg.imu_state.quaternion, flip_yaw=True) 
    _odom_pos[1] = -_odom_pos[1] # flip y-axis for gps alignment
    odom_pos = _odom_pos
    odom_angle = _odom_angle
    gait_type = decode_gait(msg.gait_type)
    #speed_level = decode_speed(msg.speed_level)
  
    go2_status = f'{go2_mode},{gait_type}'

    time.sleep(1.0/100.0) # 100 hz

    # List os variables in go2 high states
    #print("Position: ", odom_pos, odom_angle)
    #print(msg.position, go2_status, msg.gait_type)
    #print(msg.imu_state.quaternion)
    #print("Velocity: ", msg.velocity)
    #print("Yaw velocity: ", msg.yaw_speed)
    #print("Foot position in body frame: ", msg.foot_position_body)
    #print("Foot velocity in body frame: ", msg.foot_speed_body)

# ─────────────────────────────────────────────────────────────
# Unitree Go2 Controller 
# https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/high_level/go2_sport_client.py
# ─────────────────────────────────────────────────────────────
def compute_yaw_offset(x0,y0,yaw0,lat0,lon0,x1,y1,yaw1,lat1,lon1):
    # 4) odom forward vector   (expected ≈ (distance, 0))
    odom_dx = x1 - x0
    odom_dy = y1 - y0
    yaw_from_odom = math.atan2(odom_dy, odom_dx)    # rad in odom frame

    # 5) global forward vector from GPS
    east, north  = haversine_xy(lat0, lon0, lat1, lon1)
    yaw_global   = math.atan2(east, north)          # rad, 0=north, CCW=left

    # 6) offset such that   yaw_global = yaw_from_odom + offset
    yaw_offset = math.degrees(normalize(yaw_global - yaw_from_odom))

    return yaw_offset

def calibrate_heading_gps(sport_client,
                          distance=2.0,
                          v_init=0.5,
                          sample_rate=20.0):
    """
    Drive +x 'distance' metres, compute yaw_offset = heading_global - yaw_odom.
    Returns yaw_offset [rad].
    """
    global odom_pos, odom_angle, gps_lat, gps_lon, go2_command, yaw_offset
    # 1) snapshot starting state
    x0, y0, yaw0 = odom_pos[0], odom_pos[1], odom_angle[2]
    lat0, lon0   = gps_lat, gps_lon

    # 2) command forward motion
    t_start = time.time()
    loop_dt = 1.0 / sample_rate

    while True:
        x, y = odom_pos[0], odom_pos[1]
        dx = x - x0
        dy = y - y0
        travelled = math.hypot(dx, dy)

        if travelled >= distance:
            break
        if time.time() - t_start > 10:   # safety timeout
            break

        sport_client.Move(v_init, 0.0, 0.0)

        time.sleep(loop_dt)

    # stop
    sport_client.Move(0.0, 0.0, 0.0)

    # 3) snapshot ending pose
    x1, y1, yaw1 = odom_pos[0], odom_pos[1], odom_angle[2]
    lat1, lon1   = gps_lat, gps_lon
    
    yaw_offset = compute_yaw_offset(x0,y0,yaw0,lat0,lon0,x1,y1,yaw1,lat1,lon1)

    go2_command =f"[calibration] offset (deg)        : {yaw_offset:.2f}"
    print(go2_command)


def calibrate_heading_imu():
    global yaw_offset
    global go2_message 
    start_time = time.time()
    cur_time = time.time()
    offset_list = []
    heading_list = []
    go2_message = 'calibrate heading by imu' 
    print(go2_message)
    while cur_time - start_time < 1.0:
        offset_list.append(odom_angle[2])
        heading_list.append(angle_degree[2])
        time.sleep(0.04) # 25 hz

        cur_time = time.time()

    odom_mean = sum(offset_list)/len(offset_list)
    init_heading = sum(heading_list)/len(heading_list)
    yaw_offset = odom_mean + init_heading

    go2_message = f'Finish calibrating global heading by odom:{odom_mean:.2f}, imu:{init_heading:.2f}, offset:{yaw_offset:.2f} (deg)'
    print(go2_message)


def goal_to_xy(lat_curr, lon_curr,
               lat_goal, lon_goal,
               heading_deg):
    """GPS ↦ 평면 (x,y) [m]; x=heading 방향, y=좌측 +"""
    # 1) 동-북 오프셋
    dlat  = math.radians(lat_goal - lat_curr)
    dlon  = math.radians(lon_goal - lon_curr)
    lat_avg = math.radians((lat_goal + lat_curr) * 0.5)

    north = EARTH_R * dlat                       # N (m)
    east  = EARTH_R * math.cos(lat_avg) * dlon   # E (m)

    # 2) heading 기준 회전
    psi = math.radians(heading_deg)
    x =  east * math.sin(psi) + north * math.cos(psi)
    y = -east * math.cos(psi) + north * math.sin(psi)
    return x, y


def control_thread(sport_client, rate=10.0, real_control=False):
    global gps_lat, gps_lon # gps
    global odom_pos, odom_angle # go2 odometry
    global global_heading, yaw_offset
    global angle_degree  # imu 
    global kf, go2_command # the others
    global mission_active
    ctrl = PriorityPD()

    is_not_stopped = False
    while True:
        if mission_active:
            path = LinearPath(ALL_PATHS[selected_path_file]['coords'], reach_tol=REACH_TOL) # 1.0 m for pedestrian mode
            is_not_stopped = True

            #calibrate_heading_imu()

            init_gps = False
            while not init_gps:
                if isinstance(gps_lon, float) and isinstance(gps_lat, float):
                    init_gps = True
                else:
                    if mission_active:
                        go2_message = 'wait for gps signal in control thread'
                        print(go2_message)
                        time.sleep(5.0)
                    else:
                        break

            #calibrate_heading_imu()

            if real_control:
                sport_client.StandUp()
                sport_client.BalanceStand()
                sport_client.SwitchGait(1)
                
                calibrate_heading_gps(sport_client) 
                
            lat_pre, lon_pre = gps_lat, gps_lon
            x_pre, y_pre = odom_pos[0], odom_pos[1]
            yaw_pre = odom_angle[2]
              
            REACH_GOAL = False
            while not REACH_GOAL:
                is_not_stopped = True
                goal, is_goal_updated = path.update(gps_lat, gps_lon)
                global_heading = math.degrees(normalize(math.radians(odom_angle[2] + yaw_offset)))
                if goal is None:
                    sport_client.Move(0,0,0)
                    sport_client.BalanceStand()
                    sport_client.SwitchGait(0)
                    REACH_GOAL = True
                    mission_active = False
                    is_not_stopped = False
                    
                    go2_command = f'Path Finished!'
                    print('[DEBUG] PATH FINISH')
                else:
                    lat0, lon0 = gps_lat, gps_lon
                    x0, y0 = odom_pos[0], odom_pos[1]
                    yaw0 = odom_angle[2]
                    
                     # update yaw_offset when distance is more than 1
                    if (x_pre - x0) ** 2 + (y_pre - y0) ** 2 > 2.0:
                        t_cur = time.time()
                        yaw_offset_new = compute_yaw_offset(x_pre,y_pre,yaw_pre,lat_pre,lon_pre,x0,y0,yaw0,lat0,lon0)
                        yaw_offset = yaw_offset * 0.9 + yaw_offset_new * 0.1
                        go2_message = 'update yaw_offset:{yaw_offset:.2f} yaw_off_new:{yaw_offset_new:.2f}'
                         # update values
                        x_pre, y_pre = x0, y0
                        lat_pre, lon_pre = lat0, lon0

                    t_cur = time.time()

                    dx, dy = goal_to_xy(lat0, lon0, goal[0], goal[1], global_heading)
                    x1, y1 = x0 + dx, y0 + dy

                    vx, vy, vyaw = ctrl.step(dx, dy)
                    go2_command = f'Subgoal [{path.idx}/{len(path.waypoints)}] (dx,dy)=({dx:.1f},{dy:.1f}) Move({vx:.1f},{vy:.1f},{vyaw:.1f})'
                    if real_control:
                        sport_client.Move(vx,vy,vyaw)

                    print('[DEBUG] CONTROL' + go2_command)
                    time.sleep(1/rate)

                if not mission_active:
                    if sport_client is not None:
                        if is_not_stopped:
                            sport_client.Move(0,0,0)
                            sport_client.BalanceStand()
                            sport_client.SwitchGait(0)
                            is_not_stopped = False
                       
                    REACH_GOAL = True # finish gps-based navigation
                    go2_command = 'Switch to inactive mission.'
                
                print(go2_command)

        else:
            REACH_GOAL = False
            if sport_client is not None:
                if is_not_stopped:
                    sport_client.Move(0,0,0)
                    sport_client.BalanceStand()
                    sport_client.SwitchGait(0)
                    is_not_stopped = False

            time.sleep(1/rate)


from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio, json, time, math

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")



# ────────────── 경로 불러오기 ──────────────
def load_all_paths(directory='paths'):
    paths = {}
    path_list = [i for i in glob.glob(f"{directory}/*.txt")]
    #path_list = [f.name for f in path_list]
    coords = load_paths(path_list)
    for i, fname in enumerate(path_list):
        #paths[fname] = {'file': fname, 'coords': coords[i]}
        paths[fname] = {'file': fname, 'coords': coords[i]['coords']}

    return paths

ALL_PATHS = load_all_paths()
selected_path_file = list(ALL_PATHS.keys())[0] if ALL_PATHS else None
# ────────────── API 엔드포인트 ──────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return open("map_view.html", "r", encoding="utf-8").read()

@app.get("/paths", response_class=JSONResponse)
async def paths():
    return list(ALL_PATHS.values())

@app.get("/set_path")
async def set_path(file: str):
    global selected_path_file
    if file in ALL_PATHS:
        selected_path_file = file
        return {"success": True, "selected": file}
    return {"success": False, "error": "File not found"}

@app.get("/selected_path", response_class=JSONResponse)
async def selected_path():
    if selected_path_file and selected_path_file in ALL_PATHS:
        return ALL_PATHS[selected_path_file]
    return {"coords": [], "file": None}

@app.get("/toggle_mission")
async def toggle_mission():
    global mission_active
    mission_active = not mission_active
    return {"active": mission_active}

@app.get("/mission_status")
async def mission_status():
    return {"active": mission_active}

@app.get("/gps")
async def gps_stream():
    async def event_gen():
        while True:
            await asyncio.sleep(0.1)
            head = angle_degree[2]
            payload = {
                "lat": gps_lat,
                "lon": gps_lon,
                "heading": odom_angle[2] + yaw_offset,
                "heading_odom": yaw_offset,
                "status": go2_status,
                "command": go2_command
            }
            yield f"data: {json.dumps(payload)}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")

if EXTENSION_DOCK == 'go2-edu':
    @app.get("/camera")
    async def camera_stream(fps: int = 10):
        """
        multipart/x-mixed-replace(PNG) 스트림.
        ex) <img src="/camera?fps=10">
        """
        fps = max(1, min(fps, 30))
        interval = 1.0 / fps

        async def gen():
            boundary = b'--frame\r\n'
            while True:
                frm = None if d435i.current_frame is None else d435i.current_frame.copy()
                if frm is None:
                    await asyncio.sleep(0.1)
                    continue
                # BGR→RGB·JPEG 인코딩
                #_, png = cv2.imencode('.png', cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                #frm = cv2.resize(frm, dsize=(640, 240), interpolation=cv2.INTER_AREA)
                frm = cv2.resize(frm, dsize=(400, 150), interpolation=cv2.INTER_AREA)
                _, png = cv2.imencode('.png', frm)

                png_bytes = png.tobytes()

                yield boundary
                yield b'Content-Type: image/png\r\n\r\n' + png_bytes + b'\r\n'
                await asyncio.sleep(interval)

        return StreamingResponse(gen(),
                media_type='multipart/x-mixed-replace; boundary=frame')


def sensor_thread():
    global latlon_queue

    # GPS Reader
    GPS_TYPE    = 'ublox-f9p'
    gps_port    = '/dev/ttyACM0'
    gps_baud    = 115200

    # NTRIP server
    caster_host = 'rts2.ngii.go.kr'
    caster_port = 2101
    mountpt     = 'VRS-RTCM32'
    user        = 'tackgeun90'
    password    = 'ngii'

    # CONTROLLER
    real_control = True
    CONTROL_RATE = 10.0

    ser = serial.Serial(gps_port, gps_baud, timeout=1.0)

    if GPS_TYPE == 'ublox-f9p':
        ser.write(UBXMessage('CFG','CFG-RATE', SET, measRate=100, navRate=1, timeref=0).serialize())
        time.sleep(0.1)
        ser.write(UBXMessage('CFG','CFG-MSG', SET, msgClass=0x01, msgID=0x07, rateUART1=1).serialize())
        time.sleep(0.1)

    # NTRIP → RTCM → Serial thread
    ntrip_th = threading.Thread(
        target=ntrip_thread,
        args=(caster_host, caster_port, mountpt, user, password, ser, latlon_queue),
        daemon=True
    )
    ntrip_th.start()

    # Serial → UBX 파싱 → 위치 출력 thread
    ubx_th = threading.Thread(
        target=ubx_thread,
        args=(ser, latlon_queue),
        daemon=True
    )
    ubx_th.start()

    """
    try:
        imu_port = '/dev/ttyUSB0' # USB serial port linux
        imu_baud = 921600 
        imu_th = threading.Thread(
            target=imu_thread,
            args=(imu_port, imu_baud),
            daemon=True
        )
        imu_th.start()
    except:
        print('failed to set-up 9-axis IMU')
    """

    ## Camera reader
    if EXTENSION_DOCK == 'go2-edu':
        camera_th=threading.Thread(
                target=d435i.realsense_thread,
                daemon = True
        )
        camera_th.start()

    ## init high state reader
    try:
        if EXTENSION_DOCK == 'go2-edu':
            network_interface = 'eth0'
        elif EXTENSION_DOCK == 'jetson':
            network_interface = 'eno1'

        ChannelFactoryInitialize(0, network_interface)
        sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        sub.Init(HighStateHandler, 10)
    except:
        print('reading go2 high states failed')
        real_control = False

    if real_control:
        sport_client = SportClient()  
        sport_client.SetTimeout(10.0)
        sport_client.Init()
    else:
        sport_client = None

    control_th = threading.Thread(
            target=control_thread,
            args=(sport_client, CONTROL_RATE, real_control),
            daemon=True
    )
    control_th.start()


@app.on_event('startup')
def start_sensor():
    sensor_thread()

