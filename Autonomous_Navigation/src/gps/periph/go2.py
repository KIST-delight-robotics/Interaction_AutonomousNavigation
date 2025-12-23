import time

# Sport mode high state
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# Sport mode control
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)

from nav_utils import *

pos        = [0, 0, 0]
angle      = [0, 0, 0]
status = ''

def decode_mode(msg_mode):
    if msg_mode == 1:
        return 'stop walking'
    elif msg_mode == 2:
        return 'standing'
    elif msg_mode == 3:
        return 'walking'
    elif msg_mode == 5:
        return 'lie down'
    elif msg_mode == 6:
        return 'stand up'
    elif msg_mode == 7:
        return 'damp'
    else:
        return str(msg_mode)

def decode_gait(msg_mode):
    if msg_mode == 0:
        return 'idle'
    elif msg_mode == 1:
        return 'trot'
    elif msg_mode == 2:
        return 'run'
    elif msg_mode == 3:
        return 'for-climb'
    elif msg_mode == 4:
        return 'rev-climb'
    else:
        return f'gait({msg_mode})'

def decode_speed(msg_mode):
    if msg_mode == -1:
        return 'slow'
    elif msg_mode == 0:
        return 'normal'
    elif msg_mode == 1:
        return 'fast'
    else:
       return f'speed({msg_mode})'


# ─────────────────────────────────────────────────────────────
# Unitree Go2 High State
# ─────────────────────────────────────────────────────────────
def HighStateHandler(msg: SportModeState_):
    global status, pos, angle
    go2_mode = decode_mode(msg.mode)
    _odom_pos = msg.position
    _odom_angle = quat_to_euler_deg(msg.imu_state.quaternion, flip_yaw=True) 
    _odom_pos[1] = -_odom_pos[1] # flip y-axis for gps alignment
    pos = _odom_pos
    angle = _odom_angle
    gait_type = decode_gait(msg.gait_type)
  
    status = f'{go2_mode},{gait_type}'

    time.sleep(1.0/100.0) # 100 hz

    # List os variables in go2 high states
    #print("Position: ", odom_pos, odom_angle)
    #print(msg.position, go2_status, msg.gait_type)
    #print(msg.imu_state.quaternion)
    #print("Velocity: ", msg.velocity)
    #print("Yaw velocity: ", msg.yaw_speed)
    #print("Foot position in body frame: ", msg.foot_position_body)
    #print("Foot velocity in body frame: ", msg.foot_speed_body)


def init_sport_client(real_control, EXTENSION_DOCK): 
    global sport_client

    ## init high state reader
    try:
        if EXTENSION_DOCK == 'go2-edu':
            network_interface = 'eth0'
        elif EXTENSION_DOCK == 'jetson':
            network_interface = 'eno1'

        ChannelFactoryInitialize(0, network_interface)
        sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        sub.Init(HighStateHandler, 10)

    except:
        print('reading go2 high states failed')
        real_control = False

    if real_control:
        sport_client = SportClient()  
        sport_client.SetTimeout(10.0)
        sport_client.Init()
    else:
        sport_client = None

    return real_control
