# main.py
import serial
import threading
from queue import Queue
from gps import ubx_thread, ntrip_thread
from dotenv import load_dotenv
import os, time
from datetime import datetime

load_dotenv()

# -------------------------------
# 로그 파일 경로 자동 처리
# -------------------------------
def _resolve_log_path(path_env: str) -> str:
    """디렉토리면 자동으로 파일 생성, 파일이면 그대로 반환"""
    if not path_env:
        path_env = "gps_points.txt"

    # 디렉토리만 주어진 경우
    if os.path.isdir(path_env) or path_env.endswith(os.sep):
        os.makedirs(path_env, exist_ok=True)
        fname = f"gps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        return os.path.join(path_env, fname)
    else:
        # 파일 경로면 상위 폴더 생성
        os.makedirs(os.path.dirname(path_env) or ".", exist_ok=True)
        return path_env

# -------------------------------
# 헤더 및 로그 작성 함수
# -------------------------------
def _ensure_header(path: str):
    """파일이 없거나 비어 있으면 헤더 추가"""
    need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    if need_header:
        with open(path, "w", encoding="utf-8") as f:
            f.write("time,lat,lon,hdop,quality\n")

def _append_row(path: str, t: float, lat: float, lon: float, hdop, quality):
    """한 줄 추가 (즉시 flush)"""
    hdop_str = "" if hdop is None else f"{float(hdop):.2f}"
    qual_str = "" if quality is None else str(int(quality))
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{t:.6f},{lat:.7f},{lon:.7f},{hdop_str},{qual_str}\n")
        f.flush()

# -------------------------------
# 메인 함수
# -------------------------------
def main():

    ser = serial.Serial('/dev/gps', baudrate=115200, timeout=1)

    # 큐 생성
    latlon_queue = Queue()

    # GPS 수신 스레드
    threading.Thread(target=ubx_thread, args=(ser, latlon_queue), daemon=True).start()

    # NTRIP 환경 변수
    caster = os.getenv('caster')
    port = os.getenv('port')
    mountpoint = os.getenv('mountpoint')
    user = os.getenv('user')
    password = os.getenv('password')

    # NTRIP 스레드
    if all([caster, port, mountpoint, user, password]):
        threading.Thread(
            target=ntrip_thread,
            args=(caster, port, mountpoint, user, password, ser, latlon_queue),
            daemon=True
        ).start()
    else:
        print("[NTRIP] 환경변수(caster/port/mountpoint/user/password) 미설정 → RTK 없이 진행")

    # 로그 파일 경로 자동 설정
    raw_path = os.getenv("GPS_LOG_PATH", "gps_points.txt")
    LOG_PATH = _resolve_log_path(raw_path)
    _ensure_header(LOG_PATH)
    print(f"[LOG] GPS 데이터 저장 파일: {LOG_PATH}")

    # 메인 루프: 콘솔 + 파일 저장
    try:
        while True:
            if not latlon_queue.empty():
                data = latlon_queue.get()

                # (lat, lon) or (lat, lon, hdop, quality)
                if isinstance(data, (tuple, list)) and len(data) >= 2:
                    lat, lon = float(data[0]), float(data[1])
                    hdop = float(data[2]) if len(data) >= 3 and data[2] is not None else None
                    quality = int(data[3]) if len(data) >= 4 and data[3] is not None else None
                else:
                    time.sleep(0.05)
                    continue

                ts = time.time()
                print(f"현재 위도: {lat:.7f}, 경도: {lon:.7f}, hdop: {hdop}, quality: {quality}")
                _append_row(LOG_PATH, ts, lat, lon, hdop, quality)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("프로그램 종료")

if __name__ == '__main__':
    main()
