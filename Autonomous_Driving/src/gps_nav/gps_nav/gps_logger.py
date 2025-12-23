#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
단일 파일 GPS 로거

- u-blox 수신기를 10Hz (100ms)로 설정
- UBX NAV-PVT / NMEA GGA 수신
- NTRIP (RTCM) 보정 수신 (환경변수 설정 시)
- 0.75m 이상 이동할 때마다 위도/경도/품질/HDOP를 파일에 기록
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

import math
import time
import base64
import socket
from datetime import datetime
from queue import Queue
import threading

import serial
from dotenv import load_dotenv

from pyubx2 import UBXMessage, UBXReader, SET
from pynmeagps import NMEAMessage

load_dotenv()

# =============================== 전역 상태 ===============================

_latest_lat = None
_latest_lon = None
_latest_lock = threading.Lock()


def _set_latest_gps(lat: float, lon: float):
    global _latest_lat, _latest_lon
    with _latest_lock:
        _latest_lat = lat
        _latest_lon = lon


def get_gps_latlon():
    with _latest_lock:
        return _latest_lat, _latest_lon


# =============================== 유틸 함수들 ===============================

def _resolve_log_path(path_env: str) -> str:
    """디렉토리면 자동으로 파일 생성, 파일이면 그대로 반환"""
    if not path_env:
        path_env = "gps_points.txt"

    if os.path.isdir(path_env) or path_env.endswith(os.sep):
        os.makedirs(path_env, exist_ok=True)
        fname = f"gps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        return os.path.join(path_env, fname)
    else:
        os.makedirs(os.path.dirname(path_env) or ".", exist_ok=True)
        return path_env


def _ensure_header(path: str):
    """파일이 없거나 비어 있으면 헤더 추가"""
    need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    if need_header:
        with open(path, "w", encoding="utf-8") as f:
            f.write("time,lat,lon,quality,hdop\n")


def _append_row(path: str, gps_time_str: str,
                lat: float, lon: float, hdop, quality):
    """한 줄 추가 (즉시 flush)"""
    hdop_str = "" if hdop is None else f"{float(hdop):.2f}"
    qual_str = "" if quality is None else str(int(quality))
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{gps_time_str},{lat:.7f},{lon:.7f},{qual_str},{hdop_str}\n")
        f.flush()


def _dist_meter(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 점 사이의 거리 [m] (haversine, WGS84)"""
    R = 6378137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2.0) ** 2 \
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) \
        * math.sin(dlon / 2.0) ** 2

    c = 2 * math.asin(math.sqrt(a))
    return R * c


# =============================== GGA 생성 ===============================

def _deg_to_nmea_lat(lat):
    """위도(deg) -> (ddmm.mmmm, 'N'/'S')"""
    hemi = 'N' if lat >= 0 else 'S'
    lat = abs(lat)
    deg = int(lat)
    minutes = (lat - deg) * 60.0
    return f"{deg:02d}{minutes:07.4f}", hemi


def _deg_to_nmea_lon(lon):
    """경도(deg) -> (dddmm.mmmm, 'E'/'W')"""
    hemi = 'E' if lon >= 0 else 'W'
    lon = abs(lon)
    deg = int(lon)
    minutes = (lon - deg) * 60.0
    return f"{deg:03d}{minutes:07.4f}", hemi


def make_gga(lat: float, lon: float, quality: int = 1, hdop: float = 1.0,
             alt: float = 0.0, nsat: int = 8) -> bytes:
    """
    최소한의 GGA 문장 생성 (RTK 업링크용)
    - quality: 1=GPS fix, 4=RTK fixed 등
    """
    now = datetime.utcnow()
    t_str = now.strftime("%H%M%S.%f")[:9]  # hhmmss.sss
    lat_str, lat_hemi = _deg_to_nmea_lat(lat)
    lon_str, lon_hemi = _deg_to_nmea_lon(lon)

    # GGA 포맷:
    # $GPGGA,hhmmss.sss,lat,NS,lon,EW,quality,nsat,hdop,alt,M,geoid,M,,*CS
    body = f"GPGGA,{t_str},{lat_str},{lat_hemi},{lon_str},{lon_hemi}," \
           f"{quality},{nsat},{hdop:.1f},{alt:.1f},M,0.0,M,,"

    cs = 0
    for ch in body:
        cs ^= ord(ch)
    sentence = f"${body}*{cs:02X}\r\n"
    return sentence.encode("ascii")


# =============================== UBX 수신 스레드 ===============================

def ubx_thread(ser: serial.Serial, log_queue: Queue, init_queue: Queue):
    """
    GPS 수신(UBX + NMEA 파싱)
      - u-blox 10Hz 설정
      - NAV-PVT / NMEA GGA 파싱
      - log_queue 에 (time_str, lat, lon, quality, hdop) 10Hz로 푸시
      - init_queue 에 초기 (lat, lon) 한 번만 푸시 (NTRIP용)
    """
    try:
        # (1) 측정 주기 = 100 ms → 10 Hz
        msg_rate = UBXMessage('CFG', 'CFG-RATE', SET,
                              measRate=100,   # ms
                              navRate=1,
                              timeRef=0)      # 0=UTC
        ser.write(msg_rate.serialize())
        time.sleep(0.1)

        # (2) NAV-PVT 메시지를 UART1으로 1회당 한 번씩 출력 (즉, 10 Hz)
        msg_navpvt = UBXMessage('CFG', 'CFG-MSG', SET,
                                msgClass=0x01,  # NAV 클래스
                                msgID=0x07,     # PVT 메시지
                                rateUART1=1)
        ser.write(msg_navpvt.serialize())
        time.sleep(0.1)

        print("[GPS] UBX NAV-PVT 10 Hz 설정 완료")
    except Exception as e:
        print(f"[GPS] 10 Hz 설정 실패: {e}")

    ubr = UBXReader(ser, protfilter=3)
    print(f"[GPS] Listening on {ser.port}@{ser.baudrate}")
    pushed_init = False

    while True:
        try:
            raw, msg = ubr.read()
        except Exception as e:
            print(f"[GPS] read error: {e}")
            time.sleep(0.5)
            continue

        # UBX NAV-PVT
        if isinstance(msg, UBXMessage) and msg.identity == 'NAV-PVT':
            try:
                lat = msg.lat * 1e-7
                lon = msg.lon * 1e-7
            except Exception:
                continue

            _set_latest_gps(lat, lon)

            fixtype = getattr(msg, "fixType", 0)   # 0..5
            # carrsoln = getattr(msg, "carrSoln", 0)  # 필요하면 사용

            # 초기 위치 (fixType >= 2 일 때) 한 번만 NTRIP용으로 보냄
            if not pushed_init and fixtype >= 2:
                init_queue.put((lat, lon))
                pushed_init = True

            gps_time_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            quality = fixtype          # 단순히 fixType을 quality로 사용
            hdop = None                # 여기서는 없음
            log_queue.put((gps_time_str, lat, lon, quality, hdop))

        # NMEA GGA
        elif isinstance(msg, NMEAMessage) and msg.identity.endswith('GGA'):
            try:
                lat = float(msg.lat)
                lon = float(msg.lon)
            except Exception:
                continue

            _set_latest_gps(lat, lon)

            hdop = None
            quality = None
            try:
                hdop = float(msg.HDOP) if msg.HDOP not in ("", None) else None
            except Exception:
                pass
            try:
                quality = int(msg.quality) if msg.quality not in ("", None) else None
            except Exception:
                pass

            if not pushed_init and isinstance(lat, float) and isinstance(lon, float):
                init_queue.put((lat, lon))
                pushed_init = True

            gps_time_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            log_queue.put((gps_time_str, lat, lon, quality, hdop))


# =============================== NTRIP 스레드 ===============================

def ntrip_thread(caster, port, mountpoint, user, password,
                 ser: serial.Serial, init_queue: Queue):
    """
    NTRIP → RTCM 수신 → 시리얼로 전송, 1 Hz로 최신 GGA 업링크.
    """
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    req = (
        f"GET /{mountpoint} HTTP/1.1\r\n"
        f"Host: {caster}:{port}\r\n"
        "User-Agent: NTRIP PythonClient/1.0\r\n"
        "Accept: */*\r\n"
        "Connection: keep-alive\r\n"
        f"Authorization: Basic {auth}\r\n\r\n"
    ).encode('ascii')

    # 초기 대략 위치
    lat, lon = init_queue.get(block=True)
    print(f"[NTRIP] Got approx pos: {lat:.7f}, {lon:.7f}")

    while True:
        try:
            print(f"[NTRIP] Connecting {caster}:{port}/{mountpoint} …")
            sock = socket.create_connection((caster, int(port)), timeout=10)
            sock.sendall(req)

            buf = b""
            while b"\r\n\r\n" not in buf:
                b1 = sock.recv(1)
                if not b1:
                    raise ConnectionError("No NTRIP header")
                buf += b1
            sock.settimeout(1.0)

            last_gga = 0.0
            bytes_acc, last_t = 0, time.time()
            print("[NTRIP] RTCM stream started")

            while True:
                now = time.time()
                if now - last_gga >= 1.0:
                    lat_l, lon_l = get_gps_latlon()
                    if lat_l is not None and lon_l is not None:
                        lat, lon = lat_l, lon_l
                    gga = make_gga(lat, lon, quality=4, hdop=0.8)
                    sock.sendall(gga)
                    last_gga = now

                try:
                    chunk = sock.recv(1024)
                except socket.timeout:
                    chunk = b""
                if chunk:
                    ser.write(chunk)
                    bytes_acc += len(chunk)

                if now - last_t >= 2.0:
                    print(f"[NTRIP] RTCM {bytes_acc/(now-last_t):.0f} B/s   "
                          f"GGA=({lat:.5f},{lon:.5f})")
                    bytes_acc, last_t = 0, now

        except Exception as e:
            print(f"[NTRIP] Error: {e}. Retrying in 5s…")
            time.sleep(5)


# =============================== 메인 ===============================

def main():
    ser = serial.Serial('/dev/gps', baudrate=115200, timeout=1)

    # 큐 생성
    log_queue = Queue()   # (time, lat, lon, quality, hdop) 저장용
    init_queue = Queue()  # NTRIP용 초기 좌표 전달용

    # GPS 수신 스레드 (10Hz 설정 + 큐에 푸시)
    threading.Thread(
        target=ubx_thread,
        args=(ser, log_queue, init_queue),
        daemon=True
    ).start()

    # NTRIP 환경 변수
    caster = os.getenv('caster')
    port = os.getenv('port')
    mountpoint = os.getenv('mountpoint')
    user = os.getenv('user')
    password = os.getenv('password')

    # NTRIP 스레드 (환경 변수 모두 설정된 경우에만)
    if all([caster, port, mountpoint, user, password]):
        threading.Thread(
            target=ntrip_thread,
            args=(caster, port, mountpoint, user, password, ser, init_queue),
            daemon=True
        ).start()
    else:
        print("[NTRIP] 환경변수(caster/port/mountpoint/user/password) 미설정 → RTK 없이 진행")

    # 로그 파일 경로 자동 설정
    raw_path = os.getenv("GPS_LOG_PATH", "gps_points.txt")
    LOG_PATH = _resolve_log_path(raw_path)
    _ensure_header(LOG_PATH)
    print(f"[LOG] PATH 데이터 저장 파일: {LOG_PATH}")

    # 0.75 m 간격 저장 상태
    last_lat = None
    last_lon = None
    saved_count = 0
    min_step_m = 0.75

    try:
        while True:
            if not log_queue.empty():
                data = log_queue.get()

                # (time, lat, lon, quality, hdop)
                if not (isinstance(data, (tuple, list)) and len(data) >= 5):
                    continue

                gps_time_str = str(data[0])
                try:
                    lat = float(data[1])
                    lon = float(data[2])
                except Exception:
                    continue
                quality = int(data[3]) if data[3] is not None else None
                hdop = float(data[4]) if data[4] is not None else None

                # 첫 포인트는 무조건 저장
                if last_lat is None:
                    _append_row(LOG_PATH, gps_time_str, lat, lon, hdop, quality)
                    last_lat, last_lon = lat, lon
                    saved_count += 1
                    print(f"[PATH] 첫 포인트 저장 #{saved_count}: "
                          f"lat={lat:.7f}, lon={lon:.7f}")
                    continue

                # 이전 저장점과의 거리 계산
                dist = _dist_meter(last_lat, last_lon, lat, lon)

                if dist >= min_step_m:
                    _append_row(LOG_PATH, gps_time_str, lat, lon, hdop, quality)
                    last_lat, last_lon = lat, lon
                    saved_count += 1
                    print(f"[PATH] +{dist:.2f} m → 저장 #{saved_count}: "
                          f"lat={lat:.7f}, lon={lon:.7f}")
                # 0.75m 미만이면 저장하지 않음

            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("프로그램 종료")


if __name__ == '__main__':
    main()
