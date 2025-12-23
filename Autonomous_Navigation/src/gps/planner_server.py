#!/usr/bin/env python3
"""
FastAPI server that

▪ loads one-or-more GPS-path CSV files (lat,lon columns)
▪ exposes /paths (JSON)           – pre-recorded tracks
▪ exposes /gps   (SSE)            – live position
▪ serves  /      (HTML)           – Leaflet map that draws all of the above
"""

from __future__ import annotations
import argparse, asyncio, csv, json, random, threading, time
from typing import List, Tuple
from multiprocessing import Process, Queue, set_start_method
import struct

import os, sys, glob, ctypes

# tensorrt 
#import pycuda.driver as cuda
#import pycuda.autoinit

# 1) static-TLS 후보들을 선제 로드
for p in ("/lib/aarch64-linux-gnu/libstdc++.so.6",
          "/usr/lib/aarch64-linux-gnu/libgomp.so.1"):
    if os.path.exists(p):
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
import tensorrt as trt

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import json

import uvicorn

import serial
import time
import math


# GPS
import socket
import threading
from queue import Queue
from termcolor import colored
from pyubx2 import UBXReader, UBXMessage, SET

import folium
import argparse


from dataclasses import dataclass

# camera streaming
import cv2

# segmentation
import numpy as np

from PIL import Image
#import torchvision.transforms as T
#import torch.nn.functional as F
import torch


# custom code
from controller import PriorityPD
from heading import HeadingOffsetEstimator
from nav_utils import *

import periph.gps
import periph.d435i
import periph.go2

import segment



def print_info(headline, text, color='green'):
    print(colored(headline + "\t", color), text)


# ---------------------------------
# Live-GPS shared state
# ---------------------------------

key = 0
flag = 0
buff = {}
yaw_offset      = 0.0
yaw_offset_imu  = 0.0
global_heading = 0.0

ctrl_msg = ''
planner_msg = ''

pub_flag = [True, True]

# ---------------------------------
# state variable
# ---------------------------------
selected_path_file = None
ALL_PATHS = {}
goal_x, goal_y = None, None

MISSION_ACTIVE = False
REACH_GOAL = False
INIT_GPS = False

REAL_CONTROL = False

REACH_TOL = 1.0 # threshold distance for reaching goal (meter)

IS_NOT_STOPPED = False

queue_in, queue_out = Queue(), Queue()
# ---------------------------------
# Initialize segmentator
# ---------------------------------
USE_SEGMENTATION = True
#seg_image = torch.zeros(120,160,3)
seg_image = np.zeros((120,160,3),dtype=np.uint8)
# ---------------------------------
# Unitree Go2 Controller 
# ---------------------------------
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
    global planner_msg 
    global yaw_offset
    # 1) snapshot starting state
    x0, y0, yaw0 = periph.go2.pos[0], periph.go2.pos[1], periph.go2.angle[2]
    lat0, lon0   = periph.gps.lat, periph.gps.lon

    # 2) command forward motion
    t_start = time.time()
    loop_dt = 1.0 / sample_rate

    while True:
        x, y = periph.go2.pos[0], periph.go2.pos[1]
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
    x1, y1, yaw1 = periph.go2.pos[0], periph.go2.pos[1], periph.go2.angle[2]
    lat1, lon1   = periph.gps.lat, periph.gps.lon
   
    yaw_offset = compute_yaw_offset(x0,y0,yaw0,lat0,lon0,x1,y1,yaw1,lat1,lon1)

    planner_msg =f"offset (deg)        : {yaw_offset:.2f}"

    print_info("Calibration", planner_msg)

def smooth_stop(sport_client):
    global IS_NOT_STOPPED, REAL_CONTROL
    if REAL_CONTROL:
        sport_client.Move(0,0,0)
        sport_client.BalanceStand()
        sport_client.SwitchGait(0)
    IS_NOT_STOPPED = False

def control_thread(sport_client, rate=10.0):
    global goal_x, goal_y
    global INIT_GPS
    global REAL_CONTROL
    global ctrl_msg

    ctrl = PriorityPD()

    print_info("Control", 'start control thread')
    while not INIT_GPS:
        if isinstance(periph.gps.lon, float) and isinstance(periph.gps.lat, float):
            print_info("Control", f'GPS initialized {periph.gps.lon}, {periph.gps.lat}')
            INIT_GPS = True
        else:
            ctrl_msg = 'waiting for gps signal...'
            print_info("Control", ctrl_msg)
            time.sleep(5.0)

    while True:
        if MISSION_ACTIVE:
            break
        else:
            time.sleep(1.0)

    if REAL_CONTROL:
        sport_client.StandUp()
        sport_client.BalanceStand()
        sport_client.SwitchGait(1)
        
        IS_NOT_STOPPED = True
        calibrate_heading_gps(sport_client) 

    while True:
        if MISSION_ACTIVE and (goal_x is not None) and (goal_y is not None):
            if REACH_GOAL:
                smooth_stop(sport_client)
            else:
                x0, y0 = periph.go2.pos[0], periph.go2.pos[1]
                dx, dy = goal_x - x0 , goal_y - y0
                vx, vy, vyaw = ctrl.step(dx, dy)
                new_ctrl_msg = ''
                if REAL_CONTROL:
                    IS_NOT_STOPPED = True
                    sport_client.Move(vx,vy,vyaw)
                    new_ctrl_msg += 'Real-'
                new_ctrl_msg += f'Move({vx:.1f},{vy:.1f},{vyaw:.1f})'
                ctrl_msg = new_ctrl_msg
                print_info("Control", ctrl_msg)
                time.sleep(1/rate)
        else:
            if sport_client is not None:
                if IS_NOT_STOPPED:
                    smooth_stop(sport_client)
                    IS_NOT_STOPPED = False


def planner_thread(queue_in, queue_out, rate=1.0):
    global goal_x, goal_y
    global global_heading, yaw_offset
    global planner_msg # the others

    global seg_image
    global MISSION_ACTIVE, INIT_GPS, REACH_GOAL, REACH_TOL

    print_info("Planner", "queue_out wait for segmentation init.")
    queue_out.join() # wait for model initialization

    # debug code segment for planner
    DEBUG = True
    while DEBUG:
        dx, dy = 10.0, 0
        x0, y0 = periph.go2.pos[0], periph.go2.pos[1]
        delay_t, pre_time, t0 = time.time(), time.time(), time.time()
       
        msg_in = (periph.d435i.color_image.copy(), periph.d435i.depth_image.copy(), [dx, dy])
        queue_in.put(msg_in)
        print_info("Planner", "put rgb+depth.")
        trajs, waypoints, fear, seg_image = queue_out.get() # block until single element
        print_info("Planner", 'get seg+waypoints.')
        dt = time.time() - t0
        goal_x, goal_y = float(waypoints[0, 0, 0]) + x0, float(waypoints[0, 0, 1]) + y0
        print_info("Planner", waypoints[0, :, 0:1])
        t0 = time.time()

    while not INIT_GPS:
        time.sleep(1/rate)

    while True:
        if MISSION_ACTIVE:
            print_info("Planner", 'start new path')
            path = LinearPath(ALL_PATHS[selected_path_file]['coords'], reach_tol=REACH_TOL) # 1.0 m for pedestrian mode
            REACH_GOAL = False
            lat_pre, lon_pre = periph.gps.lat, periph.gps.lon
            x_pre, y_pre = periph.go2.pos[0], periph.go2.pos[1]
            yaw_pre = periph.go2.angle[2]
              
            while not REACH_GOAL and MISSION_ACTIVE:
                goal, is_goal_updated = path.update(periph.gps.lat, periph.gps.lon)
                global_heading = math.degrees(normalize(math.radians(periph.go2.angle[2] + yaw_offset)))
                print_info("Planner", 'not reach goal', goal, is_goal_updated, global_heading)
                if goal is None:
                    REACH_GOAL = True
                    MISSION_ACTIVE = False
                    
                    planner_msg = f'Path Finished!'
                    print_info("Planner", 'PATH FINISH', 'red')
                else:
                    lat0, lon0 = periph.gps.lat, periph.gps.lon
                    x0, y0 = periph.go2.pos[0], periph.go2.pos[1]
                    yaw0 = periph.go2.angle[2]
                    
                    # update yaw_offset when distance is more than 1
                    if (x_pre - x0) ** 2 + (y_pre - y0) ** 2 > 2.0:
                        t_cur = time.time()
                        yaw_offset_new = compute_yaw_offset(x_pre,y_pre,yaw_pre,lat_pre,lon_pre,x0,y0,yaw0,lat0,lon0)
                        yaw_offset = yaw_offset * 0.9 + yaw_offset_new * 0.1
                        planner_msg = 'update yaw_offset:{yaw_offset:.2f} yaw_off_new:{yaw_offset_new:.2f}'
                         # update values
                        x_pre, y_pre = x0, y0
                        lat_pre, lon_pre = lat0, lon0

                    t_cur = time.time()

                    dx, dy = goal_to_xy(lat0, lon0, goal[0], goal[1], global_heading)

                    # viplanner
                    msg_in = (periph.d435i.color_image.copy(), periph.d435i.depth_image.copy(), [dx, dy])
                    queue_in.put(msg_in)
                    print_info("Planner", "put rgb+depth.")
                    trajs, waypoints, fear, seg_image = queue_out.get() # block until single element
                    print_info("Planner", 'get seg+waypoints.')
                   
                    dt = time.time() - t0
                    goal_x, goal_y = float(waypoints[0, 0, 0]) + x0, float(waypoints[0, 0, 1]) + y0
                    planner_msg = f'sub-goal [{path.idx}/{len(path.waypoints)}] (dx,dy)=({goal_x:.1f},{goal_y:.1f}) '

        else:
            planner_msg = 'Switch to inactive mission.'
            REACH_GOAL = False
            goal_x, goal_y = 0.0, 0.0
            time.sleep(1/rate)


def sensor_thread():
    global REAL_CONTROL
    global queue_in, queue_out
    # --------------------------------
    # model configuration
    # --------------------------------
    EXTENSION_DOCK = 'go2-edu' # or 'jetson'
    CONTROL_RATE = 10.0
    USE_IMU = False
   
    # Serial information for GPS
    GPS_TYPE    = 'ublox-f9p'
    gps_port    = '/dev/ttyACM0'
    gps_baud    = 115200

    # NTRIP server information
    caster_host = 'rts2.ngii.go.kr'
    caster_port = 2101
    mountpt     = 'VRS-RTCM32'
    user        = 'tackgeun90'
    password    = 'ngii'

    # GPS Thread
    latlon_queue: Queue[Tuple[float, float]] = Queue(maxsize=1)
    ser = serial.Serial(gps_port, gps_baud, timeout=1.0)

    if GPS_TYPE == 'ublox-f9p':
        ser.write(UBXMessage('CFG','CFG-RATE', SET, measRate=100, navRate=1, timeref=0).serialize())
        time.sleep(0.1)
        ser.write(UBXMessage('CFG','CFG-MSG', SET, msgClass=0x01, msgID=0x07, rateUART1=1).serialize())
        time.sleep(0.1)

    # NTRIP -> RTCM -> Serial thread
    ntrip_th = threading.Thread(
        target=periph.gps.ntrip_thread,
        args=(caster_host, caster_port, mountpt, user, password, ser, latlon_queue),
        daemon=True
    )
    ntrip_th.start()

    # Serial -> UBX 파싱 -> 위치 출력 thread
    ubx_th = threading.Thread(
        target=periph.gps.ubx_thread,
        args=(ser, latlon_queue),
        daemon=True
    )
    ubx_th.start()

    if USE_IMU:
        imu_th = threading.Thread(
            target=periph.imu.thread,
            args=('/dev/ttyUSB0', 921600),
            daemon=True
        )
        imu_th.start()

    # Camera reader
    if EXTENSION_DOCK == 'go2-edu':
        camera_th=threading.Thread(
                target=periph.d435i.realsense_thread,
                daemon = True
        )
        camera_th.start()

    REAL_CONTROL = periph.go2.init_sport_client(REAL_CONTROL, EXTENSION_DOCK)

    control_th = threading.Thread(
            target=control_thread,
            args=(periph.go2.sport_client, CONTROL_RATE),
            daemon=True
    )
    control_th.start()

    queue_out.put('pop after segmentator initialization.')

    planner_th = threading.Thread(
            target=planner_thread,
            args=(queue_in, queue_out, CONTROL_RATE,),
            daemon=True
    )
    planner_th.start()

    
    #segment_th = Process(target=segment.segment_thread, args=(queue_in, queue_out, USE_SEGMENTATION, 10.0))
    segment_th = threading.Thread(
            target=segment.segment_thread,
            args=(queue_in, queue_out, USE_SEGMENTATION), daemon=True
    )
    segment_th.start()

   


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

ALL_PATHS = load_all_paths()
selected_path_file = list(ALL_PATHS.keys())[0] if ALL_PATHS else None

# API endpoint
@app.get("/", response_class=HTMLResponse)
async def index():
    return open("planner_view.html", "r", encoding="utf-8").read()

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
    global MISSION_ACTIVE
    MISSION_ACTIVE = not MISSION_ACTIVE
    return {"active": MISSION_ACTIVE}

@app.get("/toggle_control")
async def toggle_mission():
    global REAL_CONTROL
    REAL_CONTROL = not REAL_CONTROL
    if REAL_CONTROL:
        print_info("Server", 'change to real control')
    else:
        print_info("Server", 'freeze mode')
    return {"active": REAL_CONTROL}

@app.get("/mission_status")
async def mission_status():
    return {"active": MISSION_ACTIVE}

@app.get("/control_status")
async def mission_status():
    return {"active": REAL_CONTROL}

@app.get("/gps")
async def gps_stream():
    async def event_gen():
        while True:
            await asyncio.sleep(0.1)
            payload = {
                "lat": periph.gps.lat,
                "lon": periph.gps.lon,
                "heading": periph.go2.angle[2] + yaw_offset,
                "heading_odom": periph.go2.angle[2],
                "status": periph.go2.status,
                "planner": f"[PLAN]{planner_msg}",
                "command": f"[CTRL]{ctrl_msg}"
            }
            yield f"data: {json.dumps(payload)}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/camera")
async def camera_stream(fps: int = 5):
    """
    multipart/x-mixed-replace(PNG) stream.
    ex) <img src="/camera?fps=10">
    """
    fps = max(1, min(fps, 30))
    interval = 1.0 / fps

    async def gen():
        global seg_image
        boundary = b'--frame\r\n'
        while True:
            frm = None if periph.d435i.current_frame is None else periph.d435i.current_frame.copy()
            if frm is None:
                await asyncio.sleep(0.1)
                continue
            # PNG encoding 
            frm = cv2.resize(frm, dsize=(320, 120), interpolation=cv2.INTER_AREA)
            if USE_SEGMENTATION:
                #seg_img = seg_image.clone().cpu().numpy().astype(np.uint8)
                seg_img = cv2.resize(seg_image, dsize=(160, 120), interpolation=cv2.INTER_AREA)
                frm = np.hstack((frm, seg_img))

            _, png = cv2.imencode('.png', frm)

            png_bytes = png.tobytes()

            yield boundary
            yield b'Content-Type: image/png\r\n\r\n' + png_bytes + b'\r\n'
            await asyncio.sleep(interval)

    return StreamingResponse(gen(),
            media_type='multipart/x-mixed-replace; boundary=frame')

@app.on_event('startup')
def start_sensor():
    sensor_thread()

