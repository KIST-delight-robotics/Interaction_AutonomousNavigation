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

class LinearPath:
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
    def update(self, lat: float, lon: float) -> Optional[Tuple[float, float]]:
        """
        Advance goal index while the robot is within reach_tol of it.
        Return the *new* goal (or None if finished).
        """
        idx_old = self.idx
        while not self.done:
            g_lat, g_lon = self.waypoints[self.idx]
            dist = haversine_m(lat, lon, g_lat, g_lon)
            if dist < self.reach_tol:
                self.idx += 1      # reached → move to next
            else:
                break              # current goal still ahead
        if self.idx == idx_old:
            is_goal_updated = False
        else:
            is_goal_updated = True
        return self.goal, self.idx


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




