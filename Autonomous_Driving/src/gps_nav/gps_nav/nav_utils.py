import math, os
from typing import List, Tuple, Optional
import glob

import csv
import torch
EARTH_R = 6_378_137.0  # WGS-84 semi-major axis (metres)

def quat_to_euler_deg(quat: List[float], flip_yaw: bool=False) -> Tuple[float, float, float]:
    """
    Quaternion  ➜  Euler angles (roll, pitch, yaw) in *degrees*.
    Rotation order = XYZ (roll-pitch-yaw), same as ROS / ENU.


    Args
    ----
    qw, qx, qy, qz : float
        Quaternion components (w, x, y, z)


    Returns
    -------
    roll_deg , pitch_deg , yaw_deg  : float
        Angles in degrees
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    # 1) roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # 2) pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:               # use 90° if out of range
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # 3) yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # convert to degrees
    roll_deg  = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg   = -math.degrees(yaw) if flip_yaw else math.degree(yaw)

    return roll_deg, pitch_deg, yaw_deg

def haversine_xy(lat1, lon1, lat2, lon2):
    """
    Equirectangular ENU approximation – returns (east, north) in metres.
    Good to ~1 cm at 1 m scale.
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    latc = math.radians((lat1 + lat2) * 0.5)
    # latc = lat1*math.pi/180
    north = EARTH_R * dlat
    west  = - EARTH_R * math.cos(latc) * dlon
    return west, north


def normalize(a):
    """[-pi, pi]"""
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


def haversine_m(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two (lat, lon) pairs in **metres**.
    """
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    phi1  = math.radians(lat1)
    phi2  = math.radians(lat2)

    a = (math.sin(d_lat / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lon / 2) ** 2)
    return 2 * EARTH_R * math.asin(math.sqrt(a))

class LinearPath: ### gps 경로 중에서 무조건 처음부터 시작하는 게 아니라 자신과 가까운 점부터 시작하도록
    """
    Maintains a *linear* directed path (start → … → end) of GPS waypoints.

        path = LinearPath(waypoints, reach_tol=2.0)

    • call path.goal        → current goal (lat, lon) or None
    • call path.update(lat, lon) each cycle → goal may auto-advance
    """

    def __init__(self,
                 waypoints: List[Tuple[float, float]],
                 reach_tol: float = 2.0):
        if not waypoints:
            raise ValueError("waypoints list is empty")
        self.waypoints = waypoints
        self.reach_tol = reach_tol          # metres
        self.idx = 0                        # current goal index
        self.ratio = [1.0] * len(waypoints)
        for i in range(2,len(self.ratio)-1):
            dX_prev  = EARTH_R * math.radians(self.waypoints[i][0] - self.waypoints[i-1][0])
            dY_prev  = - EARTH_R * math.cos(math.radians(self.waypoints[i][0])) * math.radians(self.waypoints[i][1] - self.waypoints[i-1][1])
            dX_next  = EARTH_R * math.radians(self.waypoints[i+1][0] - self.waypoints[i][0])
            dY_next  = - EARTH_R * math.cos(math.radians(self.waypoints[i][0])) * math.radians(self.waypoints[i+1][1] - self.waypoints[i][1])
            th = math.atan2(dY_next,dX_next) - math.atan2(dY_prev,dX_prev)
            while th > math.pi:
                th = th - 2*math.pi
            while th < -math.pi:
                th = th + 2*math.pi
            self.ratio[i] = 1.0 - 0.75*math.fabs(th)/(0.5*math.pi)
            if self.ratio[i] < 0.25 :
                self.ratio[i] = 0.25
        self.ratio[-1] = 0.25

            
    # ------------------------------------------------------------------ #
    @property
    def goal(self) -> Optional[Tuple[float, float]]:
        """Return (lat, lon) of current goal, or None if finished."""
        return None if self.idx >= len(self.waypoints) else self.waypoints[self.idx]
    
    
    @property
    def done(self) -> bool:
        """True when all waypoints are reached."""
        return self.idx >= len(self.waypoints)

    # ------------------------------------------------------------------ #

    # def update(self, lat: float, lon: float) -> Optional[Tuple[float, float]]:
    #     """
    #     Advance goal index while the robot is within reach_tol of it.
    #     Return the *new* goal (or None if finished).
    #     """
    #     idx_old = self.idx
    #     while not self.done:
    #         g_lat, g_lon = self.waypoints[self.idx]
    #         dist = haversine_m(lat, lon, g_lat, g_lon)
    #         if dist < self.reach_tol*self.ratio[self.idx]:
    #             self.idx += 1      # reached → move to next
    #         else:
    #             break              # current goal still ahead
    #     if self.idx == idx_old:
    #         is_goal_updated = False
    #     else:
    #         is_goal_updated = True
    #     return self.goal, self.idx, self.ratio[self.idx]

    def update(self, lat: float, lon: float, lookahead: int = 5):
        """
        Look-ahead 윈도우(기본 50개)에서 reach_tol*ratio 안에 들어온 웨이포인트가 있으면
        그 중 '가장 먼' 인덱스로 점프하고, 그 다음(+1)을 새로운 goal로 설정.
        아무 것도 충족하지 않으면 현재 goal 유지.
        반환: (new_goal or None, idx, R)  # R은 현재 goal의 ratio (done이면 0.25)
        """
        if self.done:
            return None, self.idx, 0.25

        n = len(self.waypoints)
        idx_start = self.idx
        idx_end = min(n - 1, idx_start + max(1, int(lookahead)))

        reached_k = None
        for k in range(idx_start, idx_end + 1):
            g_lat, g_lon = self.waypoints[k]
            dist = haversine_m(lat, lon, g_lat, g_lon)
            if dist < self.reach_tol * self.ratio[k]:
                reached_k = k  # 조건을 만족하는 가장 마지막 k를 계속 갱신
        if reached_k is not None:
            # 해당 지점을 '지남'으로 간주 → 다음 인덱스가 새로운 goal
            self.idx = reached_k + 1
            if self.idx >= n:
                return None, self.idx, 0.25
            return self.waypoints[self.idx], self.idx, self.ratio[self.idx]

        # look-ahead 내에 들어온 점이 없으면 기존 while(한 점만 체크) 대신 그대로 유지
        # (= 현재 goal을 계속 추종)
        return self.goal, self.idx, self.ratio[self.idx]
    
##################################추가함수################################333

    def reset_to_nearest(self, lat: float, lon: float):
        """
        현재 (lat, lon)에 가장 가까운 웨이포인트를 찾아서
        그 인덱스를 새로운 시작점으로 설정한다.
        반환: (goal, idx, R)
        """
        n = len(self.waypoints)
        if n == 0:
            raise ValueError("waypoints list is empty")

        best_idx = 0
        best_dist = float("inf")

        for i, (g_lat, g_lon) in enumerate(self.waypoints):
            d = haversine_m(lat, lon, g_lat, g_lon)
            if d < best_dist:
                best_dist = d
                best_idx = i

        self.idx = best_idx
        # 방금 설정한 goal과 ratio를 같이 리턴 (편의용)
        return self.goal, self.idx, self.ratio[self.idx]    
##################################추가함수################################333

# ------------------------------------------
# Image Processing
# ------------------------------------------

def _concat_image(images, stitch_type, stitch_mask, height):
    if stitch_type == 'back(45) left(90) front(90) right(90) back(45)':
        images = images[:, :, :, height*3//4:height*5//4]
        _, new_h, new_w = images[0].size()
        new_images = [images[1][:, :, new_w//2:], images[0], images[3], images[2], images[1][:, :, 0:new_w//2]]
        
        if stitch_mask:
            new_masks = [masks[1][:, :, new_w//2:], masks[0], masks[3], masks[2], masks[1][:, :, 0:new_w//2]]
            concat_msk = torch.cat(new_masks, dim=2)

    elif stitch_type == 'left(90) front(180) right(90)':
        images = images[:, :, :, height//2:height*3//2]
        _, new_h, new_w = images[0].size()           
        new_images = [images[0][:, :, 0:new_w//2], images[3], images[2][:, :, new_w//2:]]

    elif stitch_type == 'left(180) front(180) right(180)':
        images = images[:, :, :, height//2:height*3//2]
        _, new_h, new_w = images[0].size()           
        new_images = [images[0], images[3], images[2]]

    elif stitch_type == 'front(180)':
        new_images = [images[3]]
        if stitch_mask:
            new_masks = [masks[3]]
            concat_msk = torch.cat(new_masks, dim=2)

    elif stitch_type == 'back(90) front(180) back(90)':
        images = images[:, :, :, height//2:3*height//2]
        new_images = [images[1][:, :, height//2:], images[3], images[1][:, :, :height//2]]
        if stitch_mask:
            new_masks = [masks[3]]
            concat_msk = torch.cat(new_masks, dim=2)

    if stitch_mask:
        concat_img = torch.cat(new_images, dim=2)
        concat = torch.cat([concat_img, concat_msk], dim=1)
    else:
        concat = torch.cat(new_images, dim=2)
    return concat.contiguous()

def imu_print(text):
    print(colored("IMU:\t", 'green'), text)

def ubx_print(text):
    print(colored("GPS:\t", 'green'), text)

def ntrip_print(text):
    print(colored("NTRIP:\t", 'green'), text)


# ─────────────────────────────────────────────────────────────
# Pre-recorded path loader
# ─────────────────────────────────────────────────────────────
# load all paths in directory './paths' 
def load_all_paths(directory='paths'):
    paths = {}
    path_list = [i for i in glob.glob(f"{directory}/*.txt")]
    coords = load_paths(path_list)
    for i, fname in enumerate(path_list):
        paths[fname] = {'file': fname, 'coords': coords[i]['coords']}
    return paths

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




