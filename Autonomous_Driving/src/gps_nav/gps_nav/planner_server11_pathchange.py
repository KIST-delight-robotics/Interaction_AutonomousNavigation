
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
import csv
from pathlib import Path
from geometry_msgs.msg import Twist, Point
import redis
import serial
import json

# ===== CSV 로그 경로 설정 =====
_default_log_name = f"log_{int(time.time())}.csv"
_env_log = os.getenv("LOG_CSV", _default_log_name)

if os.path.isabs(_env_log):
    LOG_PATH = Path(_env_log)
else:
    base_dir = Path(__file__).resolve().parent     
    log_dir = base_dir / "log"                    
    log_dir.mkdir(parents=True, exist_ok=True)   
    LOG_PATH = log_dir / _env_log

SERIAL_PORT = os.getenv("GPS_SERIAL", "/dev/gps")
SERIAL_BAUD = int(os.getenv("GPS_BAUD", "115200"))
GO2_TOPIC   = os.getenv("GO2_TOPIC", "/sportmodestate")
CMD_TOPIC   = os.getenv("CMD_TOPIC", "/cmd_vel")
CMD_RATE_HZ = float(os.getenv("CMD_RATE_HZ", "20.0"))
CMD_MON_TOPIC = os.getenv("CMD_MON_TOPIC", "/cmd")
DXDY_TOPIC = os.getenv("DXDY_TOPIC", "/dxdy")
REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
REDIS_KEY   = "gps_state"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


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
from std_msgs.msg import String

# ---------- GPS 파서 ----------
from pyubx2 import UBXReader, UBXMessage, SET
from pynmeagps.nmeamessage import NMEAMessage

from termcolor import colored
from .controller import PriorityPD
from .nav_utils import *   # EARTH_R, haversine_xy, normalize, LinearPath, load_all_paths

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

# ---- nav, mission ----
yaw_offset: float = 0.0
global_heading: float = 0.0
HEADING_CALIBRATED: bool = False  

ctrl_msg = ''
planner_msg = ''
insert_dash_flag = False 

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

DEST_TO_PATH = {
    # "사용자가 말할 목적지 라벨": "ALL_PATHS의 키(경로 파일 이름)"
    # 예: "gate": "paths/gate_to_center.txt",
    #     "dorm": "paths/center_to_dorm.txt",
         "L1":  "paths/center_to_L1.txt",
}

# ★ 인터랙션에서 들어오는 목적지 상태
DEST_REQUESTED = False
DEST_LABEL = None

goal_x: Optional[float] = None
goal_y: Optional[float] = None

MISSION_ACTIVE = False
REACH_GOAL = False
INIT_GPS = False
REACH_TOL = 5.0  # m
print_info("reach tol", f"reach_tol={REACH_TOL}")

# ---- cmd publish  ----
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
    
_cmdin_lock = threading.Lock()
_cmdin_lx: Optional[float] = None
_cmdin_ly: Optional[float] = None
_cmdin_az: Optional[float] = None  # angular.z

def _set_cmd_in(lx: float, ly: float, az: float):
    global _cmdin_lx, _cmdin_ly, _cmdin_az
    with _cmdin_lock:
        _cmdin_lx, _cmdin_ly, _cmdin_az = float(lx), float(ly), float(az)

def get_cmd_in() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    with _cmdin_lock:
        return _cmdin_lx, _cmdin_ly, _cmdin_az
    
# ---- 디버그용: 목표 오차(dx,dy) 공유 ---- 
_err_lock = threading.Lock()
_err_dx = 0.0
_err_dy = 0.0

def set_err(dx: float, dy: float):  
    global _err_dx, _err_dy
    with _err_lock:
        _err_dx, _err_dy = float(dx), float(dy)

def get_err() -> Tuple[float, float]:  
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

def publish_gps(lat, lon):
    """Redis에 현재 lat/lon 상태 전송"""
    try:
        data = {"gps": {"lat": lat, "lon": lon}, "timestamp": time.time()}
        r.set(REDIS_KEY, json.dumps(data))
    except Exception as e:
        print(f"[Redis] Publish error: {e}")

def set_rtk_state(quality: Optional[int]=None, hdop: Optional[float]=None, carrsoln: Optional[int]=None):
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


# ==============================
# 키 입력 스레드
# ==============================
def key_listener():
    global insert_dash_flag
    print("[Key] Press Enter to add '-' row, or 'q' + Enter to quit.")
    while True:
        try:
            key = input().strip()
            if key == "":
                insert_dash_flag = True
            elif key.lower() == "q":
                print("[Key] Quit signal detected.")
                os._exit(0)
        except EOFError:
            break
    
# ====================================================================================================================

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
    try:
        # (1) 측정 주기 = 100 ms → 10 Hz
        msg_rate = UBXMessage('CFG', 'CFG-RATE', SET,
                              measRate=100,   # ms 단위
                              navRate=1,
                              timeRef=0)      # 0=UTC
        ser.write(msg_rate.serialize())
        time.sleep(0.1)

        # (2) NAV-PVT 메시지를 UART1으로 1회당 한 번씩 출력 (즉, 10 Hz)
        msg_navpvt = UBXMessage('CFG', 'CFG-MSG', SET,
                                msgClass=0x01,  # NAV 클래스
                                msgID=0x07,     # PVT 메시지
                                rateUART1=1)
                                # rateUSB=1,      # USB 포트도 쓸 경우
                                # rateI2C=0,
                                # rateSPI=0)
        ser.write(msg_navpvt.serialize())
        time.sleep(0.1)

        print("[GPS] UBX NAV-PVT 10 Hz 설정 완료")
    except Exception as e:
        print(f"[GPS] 10 Hz 설정 실패: {e}")
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
            publish_gps(lat,lon)

            fixtype = getattr(msg, "fixType", 0)      # 0..5
            carrsoln = getattr(msg, "carrSoln", 0)    # 0 none, 1 float, 2 fixed
            set_rtk_state(carrsoln=carrsoln)

            if not pushed_init and fixtype >= 2:
                init_queue.put((lat, lon))
                pushed_init = True

            now = time.time()
            # if now - _last_rtk_log > 1.0:
            #     print(f"[UBX] fixType={fixtype}  carrSoln={carrsoln} ({_carr_to_str(carrsoln)})  "
            #           f"Lat={lat:.7f} Lon={lon:.7f}")
            _last_rtk_log = now

        # NMEA GGA
        elif isinstance(msg, NMEAMessage) and msg.identity.endswith('GGA'):
            try:
                lat = float(msg.lat); lon = float(msg.lon)
                _set_latest_gps(lat, lon)
                hdop = float(msg.HDOP) if msg.HDOP not in ("", None) else None
                quality = int(msg.quality) if msg.quality not in ("", None) else None
                set_rtk_state(quality=quality, hdop=hdop)
                publish_gps(lat,lon)
                
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
                # 1 Hz GGA 업링크 
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
# ROS2: Topic Sub/Pub
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
        self.pub_dxdy = self.create_publisher(Point, DXDY_TOPIC, 10)

        self.timer = self.create_timer(1.0 / rate_hz, self._on_timer)
    def _on_timer(self):
        vx, vy, vyaw = get_cmd()

        signal_vz = -10.0 if not HEADING_CALIBRATED else 0.0
        if REACH_GOAL:

            signal_vz = -100.0  # ★ 미션 완료 시 정지 신호

        msg = Twist()
        msg.linear.x  = vx
        msg.linear.y  = vy
        msg.angular.z = vyaw
        msg.linear.z  = signal_vz   # <<< 신호로만 씀

        self.pub.publish(msg)

        err_dx, err_dy = get_err()
        p = Point()
        p.x = float(err_dx)
        p.y = float(err_dy)
        p.z = 0.0
        self.pub_dxdy.publish(p)

class CmdMonitor(Node):
    def __init__(self, topic_name="/cmd"):
        super().__init__("cmd_monitor")
        self.sub = self.create_subscription(Twist, topic_name, self.cb, 10)
    def cb(self, msg: Twist):
        _set_cmd_in(msg.linear.x, msg.linear.y, msg.angular.z)  # ← z 사용

class DestinationSub(Node):
    """
    /destination (std_msgs/String) 으로 목적지 라벨을 받아서
    DEST_REQUESTED / DEST_LABEL 전역 플래그 세팅
    """
    def __init__(self, topic_name="/destination"):
        super().__init__("destination_sub")
        self.sub = self.create_subscription(
            String, topic_name, self.cb, 10
        )

    def cb(self, msg: String):
        global DEST_REQUESTED, DEST_LABEL
        dest = msg.data.strip().lower()
        DEST_LABEL = dest
        DEST_REQUESTED = True
        print_info("Dest", f"destination requested: '{dest}'", "yellow")

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
        cmdmon_node = CmdMonitor(topic_name=CMD_MON_TOPIC)
        dest_node  = DestinationSub("/destination") 

        execu = MultiThreadedExecutor()
        execu.add_node(go2_node)
        execu.add_node(cmd_node)
        execu.add_node(cmdmon_node)
        execu.add_node(dest_node)
        try:
            execu.spin()
        except KeyboardInterrupt:
            pass
        finally:
            execu.shutdown()
            go2_node.destroy_node()
            cmd_node.destroy_node()
            cmdmon_node.destroy_node()
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
    # lat_avg = lat_curr * math.pi /180
    north = EARTH_R * dlat
    west  = - EARTH_R * math.cos(lat_avg) * dlon
    # 2) 전진 방향에 맞춰 회전
    psi = math.radians(heading_deg)
    x =  west * math.sin(psi) + north * math.cos(psi)
    y = -west * math.cos(psi) + north  * math.sin(psi)
    return x,y,north,west

def compute_yaw_offset(x0,y0,yaw0_deg,lat0,lon0,x1,y1,yaw1_deg,lat1,lon1):
    """odom 이동 벡터 vs GPS 전진 벡터로 오프셋(deg) 추정."""
    odom_dx = x1 - x0
    odom_dy = y1 - y0
    if abs(odom_dx) < 1e-6 and abs(odom_dy) < 1e-6:
        return 0.0
    yaw_from_odom = math.atan2(odom_dy, odom_dx)  # rad (동 기준 0°)
    west, north = haversine_xy(lat0, lon0, lat1, lon1)
    yaw_global  = math.atan2(west, north)         # rad (북 기준 0°)
    return math.degrees(normalize(yaw_global - yaw_from_odom))

def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

# =========================
# 제어 스레드
# =========================
def control_thread(rate=10.0):
    global insert_dash_flag
    global goal_x, goal_y, INIT_GPS, ctrl_msg, HEADING_CALIBRATED, yaw_offset

    # From Planner Thread, Start
    global global_heading
    global planner_msg, MISSION_ACTIVE, INIT_GPS, REACH_GOAL, REACH_TOL
    global selected_path_file, ALL_PATHS
    global DEST_REQUESTED, DEST_LABEL, DEST_TO_PATH 

    # ===== CSV 로깅 세팅 =====
    log_path = Path(LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, mode="a", newline="")
    log_writer = csv.writer(log_file)
    log_headers = ["idx",
            "ratio",
            "lat","lon",
            "goal_lat", "goal_lon",
            "dX", "dY",
            "dx", "dy",
            "dist_m",
            "dX_curr", "dY_curr",
            "dx_curr","dy_curr",
            "GO2_X0","GO2_Y0","GO2_yaw0",
            "GO2_X1","GO2_Y1","GO2_yaw1",
            "yaw_offset",
            "global_heading_deg",
            "heading_err_rad",
            "cmd",          # "vx|vy|vyaw" 
            "quality",       # NMEA GGA quality (0/1/2/4/5)
            "cmd_in_lx","cmd_in_ly","cmd_in_az"]
    if log_path.stat().st_size == 0:
        log_writer.writerow(log_headers)
        log_file.flush()
        
        if insert_dash_flag:
            insert_dash_flag = False
            log_writer.writerow(['-' for _ in log_headers])
            log_file.flush()
            print("[LOG] Added '-' row")

    print_info("Planner", f"start new path: {selected_path_file}")

    path = LinearPath(ALL_PATHS[selected_path_file]['coords'], reach_tol=REACH_TOL)
    REACH_GOAL = False
    # From Planner Thread, End
    # new params to control
    max_vx = 0.8 #0.7
    max_vy = 0.0
    max_vyaw = 45*(math.pi/180)
    Kx = 0.6
    Ky = 0.0
    Kw = 1.0
    heading_margin = 45*(math.pi/180)
    print_info("params", f"max_vx={max_vx},max_vy={max_vy},max_vyaw={max_vyaw}, Kx={Kx},Ky={Ky},Kyaw={Kw},heading_margin={heading_margin}")

    # GPS Filter
    gps_count = 0
    gps_tol = 0.1 * max_vx * 2
    gps_tolSQ = gps_tol*gps_tol

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

    # yaw angle initialization
    lat0, lon0 = get_gps_latlon()
    x0, y0, yaw0 = get_go2_xy_yawdeg()
    if (lat0 is None) or (lon0 is None):
        print_info("Init", "GPS not ready; skip initial straight.", "yellow")
        return
    
    ###########################추가한 부분###############################

    try:
        goal,idx,R = path.reset_to_nearest(lat0,lon0)
        if goal is not None:
            print_info( "Planner", f"reset_to_nearest : idx = {idx}, R={R:.2f}, goal=({goal[0]:.7f}, {goal[1]:.7f})")
        else:
            print_info("Planner","reset_to_nearest: path done (no goal)", "yellow")
    
    except Exception as e:
        print_info("Planner", f"reset_to_nearest failed: {e}", "red")


    ###########################추가한 부분###############################

    print_info("Init", f"Initial straight: {INIT_STRAIGHT_DIST:.1f} m @ {INIT_STRAIGHT_V:.2f} m/s", "red")
    t0 = time.time()
    yaw_hold = yaw0  # 시작 yaw 유지(선택)

    # 미션 대기
    #while not MISSION_ACTIVE:
    #    time.sleep(0.2)
    prev_lat, prev_lon = lat0, lon0
    prev_x, prev_y, prev_yaw = x0, y0, yaw0
    snap_inited = True
    MOV_THRESH_M = 16.0

    period = 1.0 / rate
    try:
        while True:
            # ===== 초기 3 m 직진 + yaw_offset 보정 ===== 
            x, y, yaw = get_go2_xy_yawdeg()
            lat, lon = get_gps_latlon()
            # dx, dy = goal_to_xy(lat0, lon0, lat, lon, global_heading)
            # print_info("Distance", f"({dx:.2f}, {dy:.2f})", "yellow")

            if not HEADING_CALIBRATED:
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
                    
                    else:

                        off = compute_yaw_offset(x0,y0,yaw0, lat0,lon0, x1,y1,yaw1, lat1,lon1)
                        yaw_offset = off
                        HEADING_CALIBRATED = True
                        MISSION_ACTIVE= True
                        print_info("Init", f"Initial yaw_offset set to {yaw_offset:.2f} deg")
                        # =======================================
                        set_cmd(0.0, 0.0, 0.0)
            elif MISSION_ACTIVE: #and (goal_x is not None) and (goal_y is not None):

#========================== path change =========================

                if DEST_REQUESTED and DEST_LABEL is not None:
                    DEST_REQUESTED = False
                    dest = DEST_LABEL.lower()
                    if dest not in DEST_TO_PATH:
                        print_info("Planner", f"unknown destination '{dest}' (DEST_TO_PATH에 없음)", "red")
                    else:
                        new_path_file = DEST_TO_PATH[dest]
                        if new_path_file not in ALL_PATHS:
                            print_info("Planner", f"DEST_TO_PATH['{dest}']={new_path_file} 가 ALL_PATHS에 없음", "red")
                        else:
                            selected_path_file = new_path_file
                            print_info("Planner", f"switch path -> {selected_path_file}", "yellow")
                            # 새 LinearPath 생성
                            path = LinearPath(
                                ALL_PATHS[selected_path_file]['coords'],
                                reach_tol=REACH_TOL
                            )
                            REACH_GOAL = False

                            # 현재 위치에서 가장 가까운 웨이포인트로 idx 스냅
                            cur_lat_snap, cur_lon_snap = get_gps_latlon()
                            if (cur_lat_snap is not None) and (cur_lon_snap is not None):
                                try:
                                    goal_snap, idx_snap, R_snap = path.reset_to_nearest(cur_lat_snap, cur_lon_snap)
                                    if goal_snap is not None:
                                        print_info(
                                            "Planner",
                                            f"reset_to_nearest (dest={dest}): idx={idx_snap}, R={R_snap:.2f}, "
                                            f"goal=({goal_snap[0]:.7f},{goal_snap[1]:.7f})"
                                        )
                                    else:
                                        print_info("Planner", f"reset_to_nearest: path done for dest={dest}", "yellow")
                                except Exception as e:
                                    print_info("Planner", f"reset_to_nearest error for dest={dest}: {e}", "red")

                cur_x, cur_y, cur_yaw = get_go2_xy_yawdeg()
                cur_lat, cur_lon = get_gps_latlon()
                
                
                # read current heading
                _, _, yaw_deg = get_go2_xy_yawdeg()
                global_heading = math.degrees(
                    normalize(math.radians(yaw_deg + yaw_offset))
                )
                
                # GPS Filter
                # if gps_count == 0 :
                #     lat_prev = lat
                #     lon_prev = lon
                #     gps_count = 1
                # else :
                #     dX_gps  = EARTH_R * math.radians(lat - lat_prev)
                #     dY_gps  = - EARTH_R * math.cos(math.radians(lat)) * math.radians(lon - lon_prev)
                #     if dX_gps*dX_gps + dY_gps*dY_gps > gps_tolSQ :#* gps_count * gps_count :
                #         lat = lat_prev
                #         lon = lon_prev
                #         gps_count += 1
                #     else :
                #         lat_prev = lat
                #         lon_prev = lon
                #         gps_count = 1

                #현재 거리차 구하기
                dX_curr  = EARTH_R * math.radians(lat - lat0)
                dY_curr  = - EARTH_R * math.cos(math.radians(lat)) * math.radians(lon - lon0)
                print_info("distance", f"dX_curr={dX_curr:.6f}, dY_curr={dY_curr:.6f}")
                dx_curr = cur_x - prev_x
                dy_curr = cur_y - prev_y
                print_info("distance", f"dx_curr={dx_curr:.6f}, dy_curr={dy_curr:.6f}")

                # Read goal
                goal, idx, R = path.update(lat, lon)## _ 가 waypoint 
                # if reach goal            
                if goal is None:
                    REACH_GOAL = True
                    MISSION_ACTIVE = False
                    planner_msg = "Path Finished!"
                    print_info("Planner", planner_msg, "red")
                    
                    set_cmd(0.0, 0.0, 0.0)
                    time.sleep(0.05)
                    continue  


                print_info("goal",f"goal[0]={goal[0]:.2f},goal[1]={goal[1]:.2f},idx={idx}")

                # yaw_offset calibration
                if snap_inited:
                    moved = math.hypot(cur_x - prev_x, cur_y - prev_y)
                    if moved > MOV_THRESH_M and (cur_lat is not None) and (cur_lon is not None):
                        yaw_offset_new = compute_yaw_offset(
                            prev_x, prev_y, prev_yaw, prev_lat, prev_lon,
                            cur_x,  cur_y,  cur_yaw, cur_lat, cur_lon
                        )
                        # yaw_offset update 
                        yaw_offset = 0.9 * yaw_offset + 0.1 * yaw_offset_new
                        planner_msg = f"update yaw_offset:{yaw_offset:.2f} yaw_off_new:{yaw_offset_new:.2f}"
                        print_info("Planner", planner_msg)

                        prev_x, prev_y, prev_yaw = cur_x, cur_y, cur_yaw
                        prev_lat, prev_lon = cur_lat, cur_lon
                else:
                    prev_x, prev_y, prev_yaw = cur_x, cur_y, cur_yaw
                    prev_lat, prev_lon = cur_lat, cur_lon
                    snap_inited = True


                # Calcuate error
                dX  = EARTH_R * math.radians(goal[0] - lat)
                dY  = - EARTH_R * math.cos(math.radians(lat)) * math.radians(goal[1] - lon)
                dth = math.radians(global_heading)
                print_info("dX,dY", f"dX={dX:.2f}, dY={dY:.2f}, global_heading={global_heading:.2f}")

                # to robot frame
                dx = math.cos(dth) * dX + math.sin(dth) * dY
                dy = -math.sin(dth) * dX + math.cos(dth) * dY
                heading_err = math.atan2(dy,dx)
                print_info("dx,dy", f"dx={dx:.2f}, dy={dy:.2f}, heading_err={math.degrees(heading_err):.2f}")
                # dx,dy,dX,dY = goal_to_xy(lat, lon, goal[0], goal[1], global_heading)
                # heading_err = math.atan2(dy,dx) 
                # print_info("ddx,ddy", f"dx={dx:.6f}, dy={dy:.6f}, heading_err={math.degrees(heading_err):.2f}")
                # print_info("dX,dY", f"dX={dX:.6f}, dY={dY:.6f}, heading_err={math.degrees(heading_err):.2f}")
                 
                set_err(dx,dy)
                
                # vx = Kx * dx
                vy = Ky * dy      
                #vy = 0    
                vyaw = Kw * heading_err
                if vyaw < -max_vyaw :
                    vyaw = -max_vyaw
                elif vyaw > max_vyaw :
                    vyaw = max_vyaw
                # ctrl_msg = f"Init_Move({vx:.2f}, {vy:.2f}, {vyaw:.2f})"

                if math.fabs(heading_err) > heading_margin :
                    vx = 0
                else:
                    # 곡률 기반 속도
                    #vx = max_vx * R
                # vyaw 기반 속도
                    vx = max_vx * (1 - math.fabs(vyaw)/max_vyaw)
                    '''
                    # 둘 중 최소값
                    if max_vx * R < vx :
                        vx = max_vx * R
                    # 둘 다 고려
                    vx = max_vx * (1 - math.abs(vyaw)/max_vyaw) * R
                    '''
                # if vx < -max_vx :
                #     vx = -max_vx
                # elif vx > max_vx :
                #     vx = max_vx
                # if vy < -max_vy :
                #     vy = -max_vy
                # elif vy > max_vy :
                #     vy = max_vy

                            # ===== CSV 로깅 =====
                try:
                    # 거리(동-북 평면 오프셋) 크기
                    dist_m = math.hypot(dX, dY)

                    q, _hd, _carr = get_rtk_state()  
                    q_val = "" if q is None else int(q)
                    cmd_str = f"{vx:.3f}|{vy:.3f}|{vyaw:.3f}"

                    # dwa에서의 vx,vy,vyaw값 추출
                    cmd_in_lx, cmd_in_ly, cmd_in_az= get_cmd_in()
                    # goal은 (위도, 경도) 순서로 저장
                    log_writer.writerow([
                        int(idx) if idx is not None else "",
                        float(R),
                        float(lat), float(lon),
                        float(goal[0]) if goal is not None else "",
                        float(goal[1]) if goal is not None else "",
                        float(dX), float(dY),
                        float(dx), float(dy),
                        float(dist_m),
                        float(dX_curr), float(dY_curr),
                        float(dx_curr),float(dy_curr),
                        float(prev_x),float(prev_y),float(prev_yaw),
                        float(cur_x),float(cur_y),float(cur_yaw),
                        float(yaw_offset),
                        float(global_heading),
                        float(heading_err),
                        cmd_str,
                        q_val,
                        "" if cmd_in_lx is None else float(cmd_in_lx),
                        "" if cmd_in_ly is None else float(cmd_in_ly),
                        "" if cmd_in_az is None else float(cmd_in_az),
                    ])
                    log_file.flush()
                except Exception as e:
                    pass

                # Send command
                ctrl_msg = f"Move({vx:.2f}, {vy:.2f}, {vyaw:.2f})"
                print_info("Control", ctrl_msg)
                set_cmd(vx, vy, vyaw)  # /cmd 퍼블리시 공유값 갱신
                # set_cmd(0.0, 0.0, 0.0)
                # set_cmd(0.8, vy, vyaw)
            else:
                set_cmd(0.0, 0.0, 0.0)
            time.sleep(period)

    finally:
        try:
            log_file.close()
        except: 
            pass

# =========================
# 스레드 시작
# =========================
def sensor_thread(go2_topic=GO2_TOPIC, cmd_topic=CMD_TOPIC, cmd_rate=CMD_RATE_HZ):
    start_ros_subscribers(go2_topic=go2_topic, cmd_topic=cmd_topic, pub_rate=cmd_rate)

    start_gps_io(SERIAL_PORT, SERIAL_BAUD)

    threading.Thread(target=control_thread, args=(10.0,), daemon=True).start()
    threading.Thread(target=key_listener, daemon=True).start()
    #threading.Thread(target=planner_thread, args=(1.0,), daemon=True).start()

# =========================
# main
# =========================
def main():
    global MISSION_ACTIVE
    print_info("Server", "starting threads")
    sensor_thread(go2_topic=GO2_TOPIC, cmd_topic=CMD_TOPIC, cmd_rate=CMD_RATE_HZ)

    if wait_gps(timeout=60.0) and wait_go2(timeout=60.0):
        print_info("Server", "Sensors ready. Activating mission in 2s…")
        time.sleep(2.0)
        MISSION_ACTIVE = True
    else:
        print_info("Server", "Sensors not ready. Mission inactive.", "yellow")

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


if __name__ == "__main__":
    main()

