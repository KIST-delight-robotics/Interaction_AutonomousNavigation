#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- ubx_thread / ntrip_thread 를 실행해 (lat, lon [, hdop, quality])를 큐로 받음
- 최신 좌표를 ROS2 NavSatFix(/gps/fix)로 퍼블리시
- yaw_offset / goal_to_xy 계산은 별도 모듈에서 lat/lon을 구독해서 사용
"""

import os
import threading
from queue import Queue, Empty
from typing import Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus

# 너의 코드: ubx_thread, ntrip_thread를 그대로 사용
from gps import ubx_thread, ntrip_thread

# ---- Queue payload 형식 ----
# 기본: (lat, lon)
# 확장: (lat, lon, hdop, quality)  -> 나중에 ubx_thread 수정 시 자동 활용됨
Payload = Tuple[float, float]

def quality_to_status(quality):
    """
    NMEA GGA fix quality 매핑(대략):
      0: invalid -> STATUS_NO_FIX
      1: GPS fix -> STATUS_FIX
      2: DGPS/SBAS -> STATUS_SBAS_FIX
      4,5: RTK fixed/float -> STATUS_GBAS_FIX (대략)
    """
    if quality is None:
        return NavSatStatus.STATUS_FIX
    try:
        q = int(quality)
    except Exception:
        return NavSatStatus.STATUS_FIX

    if q == 0:
        return NavSatStatus.STATUS_NO_FIX
    if q == 1:
        return NavSatStatus.STATUS_FIX
    if q == 2:
        return NavSatStatus.STATUS_SBAS_FIX
    if q in (4, 5):
        return NavSatStatus.STATUS_GBAS_FIX
    return NavSatStatus.STATUS_FIX

def hdop_to_covariance(hdop):
    """
    매우 러프한 근사. HDOP 없으면 UNKNOWN으로 둠.
    """
    if hdop is None:
        return [0.0]*9, NavSatFix.COVARIANCE_TYPE_UNKNOWN
    try:
        hd = float(hdop)
    except Exception:
        return [0.0]*9, NavSatFix.COVARIANCE_TYPE_UNKNOWN

    uere = 5.0  # 환경 따라 조정
    var_h = (hd * uere) ** 2
    var_v = (2.0 * hd * uere) ** 2  # 수직 오차 더 큼 가정
    cov = [0.0]*9
    cov[0] = var_h
    cov[4] = var_h
    cov[8] = var_v
    return cov, NavSatFix.COVARIANCE_TYPE_APPROXIMATED

class GPSFixPublisher(Node):
    def __init__(self, latlon_queue: "Queue[Payload]", frame_id: str = "gps"):
        super().__init__("gps_fix_publisher")
        self.pub = self.create_publisher(NavSatFix, "/gps/fix", qos_profile_sensor_data)
        self.latlon_queue = latlon_queue
        self.last_tuple = None  # 최신값 저장
        self.frame_id = frame_id

        # GNSS 서비스 플래그 
        self.service = (
            NavSatStatus.SERVICE_GPS
            | NavSatStatus.SERVICE_GLONASS
            | NavSatStatus.SERVICE_GALILEO
        )

        # 10 Hz 퍼블리시
        self.timer = self.create_timer(0.05, self.on_timer)

    def on_timer(self):
        # 큐에 여러 개가 쌓였으면 마지막 것만 반영
        while True:
            try:
                self.last_tuple = self.latlon_queue.get_nowait()
            except Empty:
                break

        if self.last_tuple is None:
            return

        # (lat, lon) 또는 (lat, lon, hdop, quality)
        lat = self.last_tuple[0]
        lon = self.last_tuple[1]
        hdop = self.last_tuple[2] if len(self.last_tuple) >= 3 else None
        quality = self.last_tuple[3] if len(self.last_tuple) >= 4 else None

        # 메시지 구성
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.status.status = quality_to_status(quality)
        msg.status.service = self.service

        msg.latitude = float(lat)
        msg.longitude = float(lon)
        msg.altitude = 0.0  # 고도 모르면 0.0 (나중에 값 얻으면 채워도 됨)

        cov, cov_type = hdop_to_covariance(hdop)
        msg.position_covariance = cov
        msg.position_covariance_type = cov_type

        self.pub.publish(msg)

def main():
    # ---- 시리얼 및 스레드 구동 (너의 main.py 흐름 그대로) ----
    import serial
    ser = serial.Serial('/dev/ttyACM0', baudrate=115200, timeout=1)

    latlon_queue: "Queue[Payload]" = Queue(maxsize=5)

    # UBX 파서 스레드
    threading.Thread(
        target=ubx_thread,
        args=(ser, latlon_queue),
        daemon=True
    ).start()

    # .env에서 NTRIP 정보
    from dotenv import load_dotenv
    load_dotenv()
    caster     = os.getenv('caster')
    port       = os.getenv('port')
    mountpoint = os.getenv('mountpoint')
    user       = os.getenv('user')
    password   = os.getenv('password')

    # NTRIP 스레드
    threading.Thread(
        target=ntrip_thread,
        args=(caster, port, mountpoint, user, password, ser, latlon_queue),
        daemon=True
    ).start()

    # ---- ROS2 노드 구동 ----
    rclpy.init()
    node = GPSFixPublisher(latlon_queue, frame_id="gps")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
