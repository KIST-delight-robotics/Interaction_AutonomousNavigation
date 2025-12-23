from queue import Queue
import base64
from datetime import datetime
import socket
import time

from pyubx2 import UBXReader, UBXMessage, SET
from pynmeagps.nmeamessage import NMEAMessage

lon, lat = None, None
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
    print(f"Got approx pos: Lat={lat}, Lon={lon}")

    while True:
        try:
            print(f"Connecting to {caster}:{port}/{mountpoint} …")
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
            print(f"Sent GGA: {gga.decode().strip()}")

            print("RTCM stream started")
            while True:
                chunk = sock.recv(1024)
                if not chunk:
                    raise ConnectionError("NTRIP stream closed")
                ser.write(chunk)
                
        except Exception as e:
            print(f"Error: {e}. Retrying in 5s…")
            time.sleep(5)

def ubx_thread(ser, latlon_queue: Queue):
    global lat, lon
    ubr = UBXReader(ser, protfilter=3)
    print(f"Listening on {ser.port}@{ser.baudrate}")
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

        # NMEA GGA (GNGGA, GPGGA 등)
        elif isinstance(msg, NMEAMessage) and msg.identity.endswith('GGA'):
            # pynmeagps는 msg.lat/msg.lon이 이미 float
            try:
                if latlon_queue.empty():
                    if isinstance(msg.lat, float):
                        latlon_queue.put((msg.lat, msg.lon))
                lat = msg.lat
                lon = msg.lon
                gps_hdop = msg.HDOP
               
                #pos_queue.put((gps_lat, gps_lon, odom_pos[0], odom_pos[1], odom_angle[2]))
                # print(f"[GGA] Lat={gps_lat:.7f}, Lon={gps_lon:.7f}, Hdop={msg.HDOP}, quality={msg.quality} (2:DGPS, 5:rtk float, 4: rtk fix(final quality))")
            except:
                if time.time() - start_time > 10.0:
                    print("bad signal.")
                    print(msg)
                    start_time = time.time()


