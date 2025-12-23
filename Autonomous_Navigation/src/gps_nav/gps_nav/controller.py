import math
import time

class PriorityPD:
    """
    Priority controller:
        1) Align heading (yaw) first
        2) Move forward (x) next
        3) Correct sideways (y) last

    Coordinate frame:
        +x : robot forward
        +y : robot left
        yaw error = atan2(dy, dx)  (left is positive)

    Output:
        vx   [m/s] forward
        vy   [m/s] left   (0 until close in x)
        vyaw [rad/s]      (always active)
    """

    def __init__(self,
                 # gains
                 kp_x=0.6, kd_x=0.05,
                 kp_y=0.6, kd_y=0.05,
                 kp_yaw=2.0, kd_yaw=0.2,#줄여도 됨
                 # thresholds for stage switching
                 yaw_tol_deg=30.0,      # heading must be inside this (deg)
                 dx_tol=0.30,           # then x must be inside this (m)
                 # saturations
                 v_x_max=0.8, v_y_max=0.4, w_max=0.785): #45deg
        # store gains
        self.kp_x, self.kd_x = kp_x, kd_x
        self.kp_y, self.kd_y = kp_y, kd_y
        self.kp_yaw, self.kd_yaw = kp_yaw, kd_yaw
        # limits
        self.v_x_max, self.v_y_max, self.w_max = v_x_max, v_y_max, w_max
        # thresholds
        self.yaw_tol = math.radians(yaw_tol_deg)
        self.dx_tol = dx_tol
        # previous values for derivatives
        self.prev_dx = 0.0
        self.prev_dy = 0.0
        self.prev_yaw_err = 0.0
        self.prev_t = None

    def step(self, dx, dy):
        """
        Args:
            dx, dy : position error in robot frame (meters)
        Returns:
            (vx, vy, vyaw)
        """
        now = time.time()
        dt = 1e-3 if self.prev_t is None else max(now - self.prev_t, 1e-3)

        # --- yaw PD ------------------------------------------------------
        yaw_err = math.atan2(dy, dx)               # rad, left is +
        yaw_err_dot = (yaw_err - self.prev_yaw_err) / dt
        vyaw = self.kp_yaw * yaw_err + self.kd_yaw * yaw_err_dot
        vyaw = max(-self.w_max, min(self.w_max, vyaw))

        # --- default outputs --------------------------------------------
        vx = 0.0
        vy = 0.0

        # --- stage 1: heading -------------------------------------------
        heading_ok = abs(yaw_err) <= self.yaw_tol

        # --- stage 2: forward -------------------------------------------
        if heading_ok:
            dx_dot = (dx - self.prev_dx) / dt
            vx = self.kp_x * dx + self.kd_x * dx_dot
            vx = max(-self.v_x_max, min(self.v_x_max, vx))

        # --- stage 3: sideways ------------------------------------------
        if heading_ok and abs(dx) <= self.dx_tol:
            dy_dot = (dy - self.prev_dy) / dt
            vy = self.kp_y * dy + self.kd_y * dy_dot
            vy = max(-self.v_y_max, min(self.v_y_max, vy))

        # --- save state --------------------------------------------------
        self.prev_dx = dx
        self.prev_dy = dy
        self.prev_yaw_err = yaw_err
        self.prev_t = now

        return vx, vy, vyaw

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
    global odom_pos, odom_angle, gps_lat, gps_lon, go2_command, yaw_offset
    # 1) snapshot starting state
    x0, y0, yaw0 = odom_pos[0], odom_pos[1], odom_angle[2]
    lat0, lon0   = gps_lat, gps_lon

    # 2) command forward motion
    t_start = time.time()
    loop_dt = 1.0 / sample_rate

    while True:
        x, y = odom_pos[0], odom_pos[1]
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
    x1, y1, yaw1 = odom_pos[0], odom_pos[1], odom_angle[2]
    lat1, lon1   = gps_lat, gps_lon
    
    yaw_offset = compute_yaw_offset(x0,y0,yaw0,lat0,lon0,x1,y1,yaw1,lat1,lon1)

    go2_command =f"[calibration] offset (deg)        : {yaw_offset:.2f}"
    print(go2_command)


def calibrate_heading_imu():
    global yaw_offset
    global go2_message 
    start_time = time.time()
    cur_time = time.time()
    offset_list = []
    heading_list = []
    go2_message = 'calibrate heading by imu' 
    print(go2_message)
    while cur_time - start_time < 1.0:
        offset_list.append(odom_angle[2])
        heading_list.append(angle_degree[2])
        time.sleep(0.04) # 25 hz

        cur_time = time.time()

    odom_mean = sum(offset_list)/len(offset_list)
    init_heading = sum(heading_list)/len(heading_list)
    yaw_offset = odom_mean + init_heading

    go2_message = f'Finish calibrating global heading by odom:{odom_mean:.2f}, imu:{init_heading:.2f}, offset:{yaw_offset:.2f} (deg)'
    print(go2_message)


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
