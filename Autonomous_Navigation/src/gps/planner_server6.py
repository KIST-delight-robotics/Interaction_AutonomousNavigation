#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ROS2 구독 기반 내비 스켈레톤 (GPS는 내부 스레드로 직접 구동, /cmd 퍼블리시 포함)

- GPS: ubx_thread / ntrip_thread (내부 스레드) → 최신 lat/lon 전역에 저장
- ROS2: /sportmodestate (unitree_go/SportModeState)만 구독해서 x,y,yaw 사용
- goal_to_xy, yaw_offset, planner/control 스레드에서 lat/lon 바로 사용
- RTK 상태(carrSoln, GGA quality, HDOP) 1초 주기로 로그 출력
- 제어 스레드에서 계산한 (vx,vy,vyaw)을 /cmd (geometry_msgs/Twist)로 퍼블리시

dependence:
  - rclpy, unitree_go
  - geometry_msgs.msg.Twist
  - pyubx2, pynmeagps
  - controller.PriorityPD
  - nav_utils (EARTH_R, haversine_xy, normalize, LinearPath, load_all_paths 등)
  - python-dotenv (선택)
"""



from __future__ import annotations
import os, math, time, threading, socket, base64
from datetime import datetime
from typing import Optional, Tuple
from queue import Queue

# ---------- 설정 ----------
SERIAL_PORT = os.getenv("GPS_SERIAL", "/dev/gps")
SERIAL_BAUD = int(os.getenv("GPS_BAUD", "115200"))
GO2_TOPIC   = os.getenv("GO2_TOPIC", "/sportmodestate")
CMD_TOPIC   = os.getenv("CMD_TOPIC", "/cmd_vel")
CMD_RATE_HZ = float(os.getenv("CMD_RATE_HZ", "20.0"))

# 초기 강제 직진 & 보정 파라미터 (환경변수로 조절 가능)
INIT_STRAIGHT_DIST  = float(os.getenv("INIT_STRAIGHT_DIST", "3.0"))   # m
INIT_STRAIGHT_V     = float(os.getenv("INIT_STRAIGHT_V", "0.5"))      # m/s
INIT_STRAIGHT_TMAX  = float(os.getenv("INIT_STRAIGHT_TMAX", "15.0"))  # s
INIT_STRAIGHT_KP_YAW= float(os.getenv("INIT_STRAIGHT_KP_YAW", "0.6"))

# ---------- ROS2 ----------
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from unitree_go.msg import SportModeState
from geometry_msgs.msg import Twist

# ---------- GPS 파서 ----------
from pyubx2 import UBXReader, UBXMessage
from pynmeagps.nmeamessage import NMEAMessage

# ---------- 프로젝트 종속 ----------
from termcolor import colored
from controller import PriorityPD
from nav_utils import *   # EARTH_R, haversine_xy, normalize, LinearPath, load_all_paths

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# =========================
# Logging
# =========================
def print_info(headline, text="", color='green'):
    try:
        print(colored(f"{headline}\t", color), text)
    except Exception:
        print(f"[{headline}] {text}")

# =========================
# 전역 상태
# =========================
# ---- GPS 최신값 / RTK 상태 ----
_gps_lock = threading.Lock()
_gps_lat: Optional[float] = None
_gps_lon: Optional[float] = None
_gps_ready = threading.Event()

_rtk_lock = threading.Lock()
_rtk_quality: Optional[int] = None  # NMEA GGA quality: 0/1/2/4/5
_rtk_hdop: Optional[float] = None
_rtk_carrsoln: int = 0              # UBX-NAV-PVT carrSoln: 0 none, 1 float, 2 fixed
_last_rtk_log = 0.0

# ---- GO2 odom ----
_go2_lock = threading.Lock()
_go2_x: float = 0.0
_go2_y: float = 0.0
_go2_yaw_deg: float = 0.0    # heading in degrees
_go2_ready = threading.Event()

# ---- 내비/미션 ----
yaw_offset: float = 0.0
global_heading: float = 0.0
HEADING_CALIBRATED: bool = False  

ctrl_msg = ''
planner_msg = ''

# ---- 경로 선택 ----
ALL_PATHS = load_all_paths()
selected_path_file = None
if ALL_PATHS:
    wanted = os.getenv("SELECTED_PATH") 
    if wanted and wanted in ALL_PATHS:
        selected_path_file = wanted
    else:
        selected_path_file = sorted(ALL_PATHS.keys())[0]

print_info("Planner", f"available paths: {', '.join(sorted(ALL_PATHS.keys()))}")
print_info("Planner", f"selected path: {selected_path_file}", "yellow")

goal_x: Optional[float] = None
goal_y: Optional[float] = None

MISSION_ACTIVE = False
REACH_GOAL = False
INIT_GPS = False
REACH_TOL = 2.0  # m

# ---- cmd 퍼블리시용 공유 변수 ----
_cmd_lock = threading.Lock()
_cmd_vx = 0.0
_cmd_vy = 0.0
_cmd_vyaw = 0.0

def set_cmd(vx: float, vy: float, vyaw: float):
    global _cmd_vx, _cmd_vy, _cmd_vyaw
    with _cmd_lock:
        _cmd_vx, _cmd_vy, _cmd_vyaw = float(vx), float(vy), float(vyaw)

def get_cmd() -> Tuple[float, float, float]:
    with _cmd_lock:
        return _cmd_vx, _cmd_vy, _cmd_vyaw
    
# ---- 디버그용: 목표 오차(dx,dy) 공유 ----  # <<< added
_err_lock = threading.Lock()
_err_dx = 0.0
_err_dy = 0.0

def set_err(dx: float, dy: float):  # <<< added
    global _err_dx, _err_dy
    with _err_lock:
        _err_dx, _err_dy = float(dx), float(dy)

def get_err() -> Tuple[float, float]:  # <<< added
    with _err_lock:
        return _err_dx, _err_dy


# =========================
# GPS I/O (내부 스레드)
# =========================
def _set_latest_gps(lat: float, lon: float):
    global _gps_lat, _gps_lon
    with _gps_lock:
        _gps_lat, _gps_lon = float(lat), float(lon)
    _gps_ready.set()

def get_gps_latlon() -> Tuple[Optional[float], Optional[float]]:
    with _gps_lock:
        return _gps_lat, _gps_lon

def wait_gps(timeout: Optional[float]=None) -> bool:
    return _gps_ready.wait(timeout)

def _set_rtk_state(quality: Optional[int]=None, hdop: Optional[float]=None, carrsoln: Optional[int]=None):
    global _rtk_quality, _rtk_hdop, _rtk_carrsoln
    with _rtk_lock:
        if quality is not None:
            _rtk_quality = int(quality)
        if hdop is not None:
            _rtk_hdop = float(hdop)
        if carrsoln is not None:
            _rtk_carrsoln = int(carrsoln)

def get_rtk_state() -> Tuple[Optional[int], Optional[float], int]:
    with _rtk_lock:
        return _rtk_quality, _rtk_hdop, _rtk_carrsoln

def _q_to_str(q):
    try:
        q = int(q)
    except:
        return "UNK"
    return {0:"NO FIX", 1:"GPS FIX", 2:"DGPS/SBAS", 4:"RTK FIXED", 5:"RTK FLOAT"}.get(q, f"UNK({q})")

def _carr_to_str(c):
    try:
        c = int(c)
    except:
        return "UNK"
    return {0:"NO CARRIER", 1:"RTK FLOAT", 2:"RTK FIXED"}.get(c, f"UNK({c})")

def make_gga(lat: float, lon: float) -> bytes:
    """위경도를 받아 NMEA GGA 문장(ASCII byte) 생성 (1 Hz 업링크용)."""
    t = datetime.utcnow().strftime("%H%M%S.00")
    # lat: DD -> DDMM.MMMM
    lat_d = int(abs(lat)); lat_m = (abs(lat) - lat_d) * 60; lat_dir = 'N' if lat >= 0 else 'S'
    # lon: DDD -> DDDMM.MMMM
    lon_d = int(abs(lon)); lon_m = (abs(lon) - lon_d) * 60; lon_dir = 'E' if lon >= 0 else 'W'
    core = (f"GPGGA,{t},{lat_d:02d}{lat_m:07.4f},{lat_dir},"
            f"{lon_d:03d}{lon_m:07.4f},{lon_dir},1,12,1.0,0.0,M,0.0,M,,")
    chk = 0
    for c in core:
        chk ^= ord(c)
    return f"${core}*{chk:02X}\r\n".encode('ascii')

def ubx_thread(ser, init_queue):
    """GPS 수신(UBX+NMEA 파싱) → 최신 lat/lon/RTK 상태 갱신, 초기 좌표 1회 init_queue에 푸시."""
    global _last_rtk_log
    ubr = UBXReader(ser, protfilter=3)
    print(f"[GPS] Listening on {ser.port}@{ser.baudrate}")
    pushed_init = False
    while True:
        raw, msg = ubr.read()

        # UBX NAV-PVT
        if isinstance(msg, UBXMessage) and msg.identity == 'NAV-PVT':
            lat = msg.lat * 1e-7
            lon = msg.lon * 1e-7
            _set_latest_gps(lat, lon)

            fixtype = getattr(msg, "fixType", 0)      # 0..5
            carrsoln = getattr(msg, "carrSoln", 0)    # 0 none, 1 float, 2 fixed
            _set_rtk_state(carrsoln=carrsoln)

            if not pushed_init and fixtype >= 2:
                init_queue.put((lat, lon))
                pushed_init = True

            now = time.time()
            if now - _last_rtk_log > 1.0:
                print(f"[UBX] fixType={fixtype}  carrSoln={carrsoln} ({_carr_to_str(carrsoln)})  "
                      f"Lat={lat:.7f} Lon={lon:.7f}")
                _last_rtk_log = now

        # NMEA GGA
        elif isinstance(msg, NMEAMessage) and msg.identity.endswith('GGA'):
            try:
                lat = float(msg.lat); lon = float(msg.lon)
                _set_latest_gps(lat, lon)
                hdop = float(msg.HDOP) if msg.HDOP not in ("", None) else None
                quality = int(msg.quality) if msg.quality not in ("", None) else None
                _set_rtk_state(quality=quality, hdop=hdop)

                if not pushed_init and isinstance(msg.lat, float) and isinstance(msg.lon, float):
                    init_queue.put((lat, lon))
                    pushed_init = True

                now = time.time()
                if now - _last_rtk_log > 1.0:
                    print(f"[GGA] quality={quality} ({_q_to_str(quality)}), HDOP={hdop}  "
                          f"Lat={lat:.7f} Lon={lon:.7f}")
                    _last_rtk_log = now
            except:
                pass

def ntrip_thread(caster, port, mountpoint, user, password, ser, init_queue):
    """NTRIP → RTCM 수신 → 시리얼로 전송, 1 Hz로 최신 GGA 업링크."""
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    req = (
        f"GET /{mountpoint} HTTP/1.1\r\n"
        f"Host: {caster}:{port}\r\n"
        "User-Agent: NTRIP PythonClient/1.0\r\n"
        "Accept: */*\r\n"
        "Connection: keep-alive\r\n"
        f"Authorization: Basic {auth}\r\n\r\n"
    ).encode('ascii')

    # 초기 대략 좌표(ubx_thread가 한 번만 넣어줌)
    lat, lon = init_queue.get(block=True)
    print(f"[NTRIP] Got approx pos: {lat:.7f}, {lon:.7f}")

    while True:
        try:
            print(f"[NTRIP] Connecting {caster}:{port}/{mountpoint} …")
            sock = socket.create_connection((caster, int(port)), timeout=10)
            sock.sendall(req)

            # 응답 헤더 스킵
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
                # 1 Hz GGA 업링크 (최신 좌표 사용)
                if now - last_gga >= 1.0:
                    lat_l, lon_l = get_gps_latlon()
                    if lat_l is not None and lon_l is not None:
                        lat, lon = lat_l, lon_l
                    sock.sendall(make_gga(lat, lon))
                    last_gga = now

                # RTCM 수신 및 시리얼로 전달
                try:
                    chunk = sock.recv(1024)
                except socket.timeout:
                    chunk = b""
                if chunk:
                    ser.write(chunk)
                    bytes_acc += len(chunk)

                if now - last_t >= 2.0:
                    print(f"[NTRIP] RTCM {bytes_acc/(now-last_t):.0f} B/s   GGA=({lat:.5f},{lon:.5f})")
                    bytes_acc, last_t = 0, now

        except Exception as e:
            print(f"[NTRIP] Error: {e}. Retrying in 5s…")
            time.sleep(5)

def start_gps_io(serial_port=SERIAL_PORT, serial_baud=SERIAL_BAUD):
    """시리얼 열고 ubx/ntrip 스레드 시작. .env에서 caster/port/mountpoint/user/password를 읽음."""
    import serial
    ser = serial.Serial(serial_port, baudrate=serial_baud, timeout=1)

    caster     = os.getenv('caster')
    port       = os.getenv('port')
    mountpoint = os.getenv('mountpoint')
    user       = os.getenv('user')
    password   = os.getenv('password')
    if not all([caster, port, mountpoint, user, password]):
        print_info("GPS", "NTRIP 환경변수(caster/port/mountpoint/user/password) 미설정 — RTK 없이 GPS만 구동", "yellow")

    q = Queue(maxsize=2)
    threading.Thread(target=ubx_thread, args=(ser, q), daemon=True).start()
    if all([caster, port, mountpoint, user, password]):
        threading.Thread(target=ntrip_thread, args=(caster, port, mountpoint, user, password, ser, q), daemon=True).start()

# =========================
# ROS2: Go2 오도메트리 구독 + /cmd 퍼블리시
# =========================
class Go2OdomSub(Node):
    def __init__(self, topic_name="/sportmodestate"):
        super().__init__("go2_odom_sub")
        self.sub = self.create_subscription(
            SportModeState, topic_name, self.cb, qos_profile_sensor_data
        )
    def cb(self, msg: SportModeState):
        x = float(msg.position[0]) if len(msg.position) > 0 else 0.0
        y = float(msg.position[1]) if len(msg.position) > 1 else 0.0
        yaw_rad = float(msg.imu_state.rpy[2]) if len(msg.imu_state.rpy) > 2 else 0.0  # rad
        yaw_deg = math.degrees(yaw_rad)  # deg
        global _go2_x, _go2_y, _go2_yaw_deg
        with _go2_lock:
            _go2_x = x; _go2_y = y; _go2_yaw_deg = yaw_deg
        _go2_ready.set()

class CmdPublisher(Node):
    def __init__(self, topic_name="/cmd_vel", rate_hz=20.0):
        super().__init__("cmd_publisher")
        self.pub = self.create_publisher(Twist, topic_name, 10)
        self.timer = self.create_timer(1.0 / rate_hz, self._on_timer)
    def _on_timer(self):
        vx, vy, vyaw = get_cmd()
        msg = Twist()
        msg.linear.x  = vx
        msg.linear.y  = vy
        msg.angular.z = vyaw
        self.pub.publish(msg)

def get_go2_xy_yawdeg() -> Tuple[float, float, float]:
    with _go2_lock:
        return _go2_x, _go2_y, _go2_yaw_deg

def wait_go2(timeout: Optional[float]=None) -> bool:
    return _go2_ready.wait(timeout)

def start_ros_subscribers(go2_topic=GO2_TOPIC, cmd_topic=CMD_TOPIC, pub_rate=CMD_RATE_HZ):
    def _spin():
        rclpy.init()
        go2_node = Go2OdomSub(go2_topic)
        cmd_node = CmdPublisher(topic_name=cmd_topic, rate_hz=pub_rate)
        execu = MultiThreadedExecutor()
        execu.add_node(go2_node)
        execu.add_node(cmd_node)
        try:
            execu.spin()
        except KeyboardInterrupt:
            pass
        finally:
            execu.shutdown()
            go2_node.destroy_node()
            cmd_node.destroy_node()
            rclpy.shutdown()
    threading.Thread(target=_spin, daemon=True).start()

# =========================
# 유틸 함수
# =========================
def goal_to_xy(lat_curr, lon_curr, lat_goal, lon_goal, heading_deg):
    """GPS ↦ 평면 (x,y) [m]; x=heading 방향, y=좌측 + (heading_deg는 북 기준)"""
    # 1) 동-북 오프셋
    dlat  = math.radians(lat_goal - lat_curr)
    dlon  = math.radians(lon_goal - lon_curr)
    lat_avg = math.radians((lat_goal + lat_curr) * 0.5)
    north = EARTH_R * dlat
    east  = EARTH_R * math.cos(lat_avg) * dlon
    # 2) 전진 방향에 맞춰 회전
    psi = math.radians(heading_deg)
    x =  east * math.sin(psi) + north * math.cos(psi)
    y = -east * math.cos(psi) + north * math.sin(psi)
    return x, y

def compute_yaw_offset(x0,y0,yaw0_deg,lat0,lon0,x1,y1,yaw1_deg,lat1,lon1):
    """odom 이동 벡터 vs GPS 전진 벡터로 오프셋(deg) 추정."""
    odom_dx = x1 - x0
    odom_dy = y1 - y0
    if abs(odom_dx) < 1e-6 and abs(odom_dy) < 1e-6:
        return 0.0
    yaw_from_odom = math.atan2(odom_dy, odom_dx)  # rad (동 기준 0°)
    east, north = haversine_xy(lat0, lon0, lat1, lon1)
    yaw_global  = math.atan2(east, north)         # rad (북 기준 0°)
    return math.degrees(normalize(yaw_global - yaw_from_odom))

def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

# =========================
# 제어 스레드
# =========================
def control_thread(rate=10.0):
    global goal_x, goal_y, INIT_GPS, ctrl_msg, HEADING_CALIBRATED, yaw_offset

    ctrl = PriorityPD()
    print_info("Control", "start control thread")

    # GPS / GO2 초기화 대기
    wait_gps(timeout=None)
    wait_go2(timeout=None)
    lat, lon = get_gps_latlon()
    x0, y0, yaw0 = get_go2_xy_yawdeg()
    print_info("Control", f"GPS init lat={lat}, lon={lon}")
    print_info("Control", f"GO2  init x={x0:.3f}, y={y0:.3f}, yaw(deg)={yaw0:.2f}")
    INIT_GPS = True

    # yaw angle 초기화
    lat0, lon0 = get_gps_latlon()
    x0, y0, yaw0 = get_go2_xy_yawdeg()
    if (lat0 is None) or (lon0 is None):
        print_info("Init", "GPS not ready; skip initial straight.", "yellow")
        return

    print_info("Init", f"Initial straight: {INIT_STRAIGHT_DIST:.1f} m @ {INIT_STRAIGHT_V:.2f} m/s")
    t0 = time.time()
    yaw_hold = yaw0  # 시작 yaw 유지(선택)

    # 미션 대기
    #while not MISSION_ACTIVE:
    #    time.sleep(0.2)

    period = 1.0 / rate
    while True:
        # ===== 초기 3 m 직진 + yaw_offset 보정 ===== 정지했다가 보정 후 다시 출발
        x, y, yaw = get_go2_xy_yawdeg()
        if not HEADING_CALIBRATED:
            # 루프 안
            yaw_err_deg = _wrap_deg(yaw_hold - yaw)        # deg
            yaw_err_rad = math.radians(yaw_err_deg)        # rad
            vyaw_cmd = INIT_STRAIGHT_KP_YAW * yaw_err_rad  # rad/s  

            # 클램프
            vyaw_cmd = max(min(vyaw_cmd, 0.6), -0.6)       # rad/s
            set_cmd(INIT_STRAIGHT_V, 0.0, vyaw_cmd)

            if math.hypot(x - x0, y - y0) >= INIT_STRAIGHT_DIST:
                x1, y1, yaw1 = get_go2_xy_yawdeg()
                lat1, lon1 = get_gps_latlon()
                if (lat1 is None) or (lon1 is None):
                    print_info("Init", "GPS lost; skip offset update.", "yellow")

                off = compute_yaw_offset(x0,y0,yaw0, lat0,lon0, x1,y1,yaw1, lat1,lon1)
                yaw_offset = off
                HEADING_CALIBRATED = True
                print_info("Init", f"Initial yaw_offset set to {yaw_offset:.2f} deg")
                # =======================================
                set_cmd(0.0, 0.0, 0.0)
        elif MISSION_ACTIVE and (goal_x is not None) and (goal_y is not None):
            #x, y, _ = get_go2_xy_yawdeg()
            dx, dy = goal_x, goal_y 
            vx, vy, vyaw = ctrl.step(dx, dy)
            ctrl_msg = f"Move({vx:.2f}, {vy:.2f}, {vyaw:.2f})"
            # print_info("Control", ctrl_msg)
            set_cmd(vx, vy, -vyaw)  # /cmd 퍼블리시 공유값 갱신
            # set_cmd(0.0, 0.0, 0.0)
        else:
            set_cmd(0.0, 0.0, 0.0)
        time.sleep(period)

# =========================
# 플래너 스레드
# =========================
def planner_thread(rate=1.0):
    global goal_x, goal_y
    global global_heading, yaw_offset, HEADING_CALIBRATED
    global planner_msg, MISSION_ACTIVE, INIT_GPS, REACH_GOAL, REACH_TOL
    global selected_path_file, ALL_PATHS

    # 초기화 대기
    while not (wait_gps(0.5) and wait_go2(0.5)):
        time.sleep(1.0 / rate)
    INIT_GPS = True

    if not ALL_PATHS or selected_path_file not in ALL_PATHS:
        print_info("Planner", "No path loaded. Planner idle.", "yellow")

    # 이미 초기 직진에서 보정했으면 스킵
    # if not HEADING_CALIBRATED:
    #     yaw_offset_try = calibrate_yaw_offset(min_move_m=2.0, timeout_s=30.0)
    #     if abs(yaw_offset_try) > 1e-3:
    #         yaw_offset = yaw_offset_try
    #         HEADING_CALIBRATED = True

    while True:
        if MISSION_ACTIVE and ALL_PATHS and selected_path_file in ALL_PATHS:
            print_info("Planner", f"start new path: {selected_path_file}")

            path = LinearPath(ALL_PATHS[selected_path_file]['coords'], reach_tol=REACH_TOL)
            REACH_GOAL = False

            # 초기 스냅샷
            lat_pre, lon_pre = get_gps_latlon()
            x_pre, y_pre, yaw_pre = get_go2_xy_yawdeg()

            while not REACH_GOAL and MISSION_ACTIVE:
                lat_cur, lon_cur = get_gps_latlon()
                if (lat_cur is None) or (lon_cur is None):
                    time.sleep(1.0 / rate)
                    continue

                goal, is_goal_updated = path.update(lat_cur, lon_cur)## _ 가 waypoint 


                # 현재 heading(도): 로봇 yaw + yaw_offset  → 북 기준
                _, _, yaw_deg = get_go2_xy_yawdeg()
                global_heading = math.degrees(
                    normalize(math.radians(yaw_deg + yaw_offset))
                )
                print_info("HEAD", f"yaw_deg={yaw_deg:.2f}, yaw_offset={yaw_offset:.2f}, global_heading={global_heading:.2f}")
                print_info("Planner", 'not reach goal', goal, is_goal_updated)
 
                if goal is None:
                    REACH_GOAL = True
                    MISSION_ACTIVE = False
                    planner_msg = "Path Finished!"
                    print_info("Planner", planner_msg, "red")
                    break

                # 주행 중에도 천천히 offset 미세보정 (이동량 충분할 때만)
                x0, y0, yaw0 = get_go2_xy_yawdeg()
                if (x_pre - x0) ** 2 + (y_pre - y0) ** 2 > 2.0:  # ≈1.41 m
                    yaw_offset_new = compute_yaw_offset(
                        x_pre, y_pre, yaw_pre, lat_pre, lon_pre,
                        x0, y0, yaw0, lat_cur, lon_cur
                    )
                    yaw_offset = yaw_offset * 0.9 + yaw_offset_new * 0.1
                    planner_msg = f"update yaw_offset:{yaw_offset:.2f} yaw_off_new:{yaw_offset_new:.2f}, global_heading={global_heading:.2f}"
                    print_info("Planner", planner_msg)
                    # 스냅샷 갱신
                    x_pre, y_pre, yaw_pre = x0, y0, yaw0
                    lat_pre, lon_pre = lat_cur, lon_cur

                # 현 위치 기준 (dx, dy) 계산 → 로컬 목표
                dx, dy = goal_to_xy(lat_cur, lon_cur, goal[0], goal[1], global_heading)
                set_err(dx, dy)

                # 로컬 절대 목표 (로봇 좌표계 기준 + 현재 pose)
                # x0, y0, _ = get_go2_xy_yawdeg()
                goal_x = dx #+ x0
                goal_y = dy #+ y0

                planner_msg = f"sub-goal (goal_x,goal_y)=({goal_x:.2f},{goal_y:.2f})"
                print_info("sub-goal", planner_msg)
                time.sleep(1.0 / rate)
        else:
            planner_msg = "Switch to inactive mission."
            REACH_GOAL = False
            goal_x, goal_y = None, None
            time.sleep(1.0 / rate)

# =========================
# 스레드 시작
# =========================
def sensor_thread(go2_topic=GO2_TOPIC, cmd_topic=CMD_TOPIC, cmd_rate=CMD_RATE_HZ):
    # 1) ROS2: Go2 오도메트리 구독 + /cmd 퍼블리셔 시작
    start_ros_subscribers(go2_topic=go2_topic, cmd_topic=cmd_topic, pub_rate=cmd_rate)

    # 2) GPS I/O 시작 (내부 스레드)
    start_gps_io(SERIAL_PORT, SERIAL_BAUD)

    # 3) 제어/플래너 스레드
    threading.Thread(target=control_thread, args=(10.0,), daemon=True).start()
    threading.Thread(target=planner_thread, args=(1.0,), daemon=True).start()

# =========================
# main
# =========================
if __name__ == "__main__":
    print_info("Server", "starting threads")
    # 토픽명이 '/lf/sportmodestate'라면 인자 바꿔줘: sensor_thread(go2_topic="/lf/sportmodestate")
    sensor_thread(go2_topic=GO2_TOPIC, cmd_topic=CMD_TOPIC, cmd_rate=CMD_RATE_HZ)

    # 데모: 둘 다 준비되면 미션 온
    if wait_gps(timeout=60.0) and wait_go2(timeout=60.0):
        print_info("Server", "Sensors ready. Activating mission in 2s…")
        time.sleep(2.0)
        MISSION_ACTIVE = True
    else:
        print_info("Server", "Sensors not ready. Mission inactive.", "yellow")

    # 유지 루프 + RTK 상태 요약 출력
    try:
        while True:
            lat, lon = get_gps_latlon()
            x, y, yaw = get_go2_xy_yawdeg()
            yaw_rad = math.radians(yaw)
            err_dx, err_dy = get_err()
            vx, vy, vyaw = get_cmd()
            print_info("DBG", f"err(dx,dy)=({err_dx:.2f},{err_dy:.2f}) | "
                            f"cmd(vx,vy,vyaw)=({vx:.2f},{vy:.2f},{vyaw:.2f}) | "
                            f"yaw(deg)={yaw:.3f}")

            if lat is not None and lon is not None:
                q, hd, carr = get_rtk_state()
                rtk_str = f"RTK: carrSoln={carr}({_carr_to_str(carr)}), quality={q}({_q_to_str(q)}), HDOP={hd}"
                # print_info("GPS", f"lat={lat:.7f}, lon={lon:.7f} | {rtk_str}")
            print_info("GO2", f"x={x:.3f}, y={y:.3f}, yaw={yaw_rad:.2f}")
            print_info("---------------------------loop------------------------------")
            time.sleep(2.0)
    except KeyboardInterrupt:
        print_info("Server", "Stopped.")
