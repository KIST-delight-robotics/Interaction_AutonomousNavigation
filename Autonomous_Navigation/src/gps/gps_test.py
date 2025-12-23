# main.py
import serial
import threading
from queue import Queue
from gps import ubx_thread, ntrip_thread
from dotenv import load_dotenv
import os
load_dotenv()

def main():
    # 시리얼 포트 및 보드레이트 설정 (환경에 맞게 수정)
    ser = serial.Serial('/dev/gps', baudrate=115200, timeout=1)

    # 위경도 좌표 주고받을 큐 생성
    latlon_queue = Queue()

    # ubx_thread 실행 (GPS 수신 및 파싱)
    threading.Thread(target=ubx_thread, args=(ser, latlon_queue), daemon=True).start()

    # NTRIP 경로 및 계정 정보 설정 (환경에 맞게 수정)
    caster = os.getenv('caster')
    port = os.getenv('port')
    mountpoint = os.getenv('mountpoint')
    user = os.getenv('user')
    password = os.getenv('password')
    
    # ntrip_thread 실행 (RTK 보정 데이터 수신 전송)
    threading.Thread(target=ntrip_thread, args=(caster, port, mountpoint, user, password, ser, latlon_queue), daemon=True).start()

    # 메인 루프: 위경도 데이터 출력
    import time
    try:
        while True:
            if not latlon_queue.empty():
                lat, lon = latlon_queue.get()
                print(f"현재 위도: {lat:.7f}, 경도: {lon:.7f}")
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("프로그램 종료")

if __name__ == '__main__':
    main()






