#ifndef _SK_ROBOT_LIB_SERVO_DYNAMIXEL_H_
#define _SK_ROBOT_LIB_SERVO_DYNAMIXEL_H_

#include "rclcpp/rclcpp.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "dynamixel_sdk/dynamixel_sdk.h"                                   // Uses DYNAMIXEL SDK library

#include "sk_robot_lib/skServo.h"

#define DYXEL_DEFAULT_DEVICENAME                        "/dev/ttyUSB0"      // Check which port is being used on your controller
#define DYXEL_MODE_IDLE                                 (1)
#define DYXEL_MODE_POSITION                             (3)
#define DYXEL_MODE_CURRENT                              (0)
#define DYXEL_MODE_VELOCITY                             (1)
#define DYXEL_TORQUE_ENABLE                             (1)
#define DYXEL_TORQUE_DISABLE                            (0)
#define DYXEL_PROTOCOL_VERSION_XM                       (2.0)                 // See which protocol version is used in the Dynamixel
#define DYXEL_PROTOCOL_VERSION_AX                       (1.0)                 // See which protocol version is used in the Dynamixel
#define DYXEL_BAUDRATE_XM                               (4000000)                    // was 57600
#define DYXEL_BAUDRATE_AX                               (1000000)                    // was 57600

// Dynamixel models
#define DYXEL_MODEL_XM430_W210                          (4321)
#define DYXEL_MODEL_XM430_W350                          (4335)
#define DYXEL_MODEL_XC430_W150                          (4315)
#define DYXEL_MODEL_AX12_A                              (1201)
#define DYXEL_MODEL_AX12_W                              (1202)

// Control table address for XM430 & XC430(not support current control)
#define DYXEL_XM430_ADDR_TORQUE_ENABLE                  64
#define DYXEL_XM430_ADDR_GOAL_CURRENT                   102
#define DYXEL_XM430_ADDR_GOAL_VELOCITY                  104
#define DYXEL_XM430_ADDR_GOAL_POSITION                  116
#define DYXEL_XM430_ADDR_PRESENT_CURRENT                 126
#define DYXEL_XM430_ADDR_PRESENT_VELOCITY                128
#define DYXEL_XM430_ADDR_PRESENT_POSITION                132
#define DYXEL_XM430_ADDR_DRIVE_MODE                     10
#define DYXEL_XM430_ADDR_OPERATING_MODE                 11
#define DYXEL_XM430_ADDR_VELOCITY_I_GAIN                76
#define DYXEL_XM430_ADDR_VELOCITY_P_GAIN                78
#define DYXEL_XM430_ADDR_CURRENT_LIMIT                  38
#define DYXEL_XM430_ADDR_VELOCITY_LIMIT                 44

// Speifications of XM430
//#define DYXEL_XM430_W210_MAX_PULSE                           (4095)
//#define DYXEL_XM430_W210_MAX_DEG                             (360.0)
//#define DYXEL_XM430_W210_MAX_RAD                             (2.0*M_PI)
//#define DYXEL_XM430_W210_MAX_VELOCITY_PULSE                  (1023)
//#define DYXEL_XM430_W210_MAX_CURRENT_PULSE                   (1193)

//#define DYXEL_XM430_W350_MAX_PULSE                           (4095)
//#define DYXEL_XM430_W350_MAX_DEG                             (360.0)
//#define DYXEL_XM430_W350_MAX_RAD                             (2.0*M_PI)
//#define DYXEL_XM430_W350_MAX_VELOCITY_PULSE                  (1023)
//#define DYXEL_XM430_W350_MAX_CURRENT_PULSE                   (1193)
//#define DYXEL_XM430_W350_PULSE_PER_REV                       (4096)
//#define DYXEL_XM430_W350_TORQUE_TO_CURRENT_A                 ((1.50-0.9)/(2.45-1.40)) // 3.8Nm for 2.1A when V = 11.1
//#define DYXEL_XM430_W350_TORQUE_TO_CURRENT_B                 (0.9-DYXEL_XM430_W350_TORQUE_TO_CURRENT_A*1.40) // 3.8Nm for 2.1A when V = 11.1

// Params for XH430W350
#define DYXEL_XM430_W350_DEG_PER_PULSE                       (0.088) // deg/pulse
#define DYXEL_XM430_W350_VELOCITY_PER_PULSE                  (0.229*2.0*M_PI/60.0) // 0.229 rev/min
#define DYXEK_XM430_W350_CURRENT_PER_PULSE                   (2.69/1000.0) // 2.69 mA per pulse
#define DYXEL_XM430_W350_TORQUE_PER_CURRENT                  (3.1/1.2) // 3.1Nm for 1.2A when V = 11.1

// Params for XH430W210
#define DYXEL_XM430_W210_DEG_PER_PULSE                       (0.088) // deg/pulse
#define DYXEL_XM430_W210_VELOCITY_PER_PULSE                  (0.229*2.0*M_PI/60.0) // 0.229 rev/min
#define DYXEK_XM430_W210_CURRENT_PER_PULSE                   (2.69/1000.0) // 2.69 mA per pulse
#define DYXEL_XM430_W210_TORQUE_PER_CURRENT                  (2.2/1.2) // 2.2Nm for 1.2A when V = 11.1

// Params for XC430T150
#define DYXEL_XC430_T150_DEG_PER_PULSE                       (0.088) // deg/pulse
#define DYXEL_XC430_T150_VELOCITY_PER_PULSE                  (0.229*2.0*M_PI/60.0) // 0.229 rev/min

// Control Params for XM430
#define DYXEL_XM430_VELOCITY_I_GAIN                     1920       // defualt = 1920 <16,383
#define DYXEL_XM430_VELOCITY_P_GAIN                     100     // default = 100 < 16,383
#define DYXEL_XM430_CURRENT_LIMIT                       1193    // defualt = max = 1193 
#define DYXEL_XM430_VELOCITY_LIMIT                      1000     // default = 200, max = 1023

// Control table address for AX12W
#define DYXEL_AX12_ADDR_TORQUE_ENABLE                  24
#define DYXEL_AX12_ADDR_GOAL_POSITION                  30
#define DYXEL_AX12_ADDR_GOAL_VELOCITY                  32
#define DYXEL_AX12_ADDR_PRESENT_POSITION               36
#define DYXEL_AX12_ADDR_PRESENT_VELOCITY               38
#define DYXEL_AX12_ADDR_PRESENT_LOAD                   40

#define DYXEL_AX12_W_DEG_PER_PULSE                          (0.29) // deg/pulse
#define DYXEL_AX12_W_RADSEC_PER_PULSE                       ((470.0*2.0*M_PI/60.0)/1023.0) //470 rpm, 1024 bit
#define DYXEL_AX12_W_TORQUE_PER_PULSE                       (0.2/1024.0) // 0.2Nm for 1023 pulse 12V, 1.4A

using namespace std;

class skServoDynamixel : public skServo
{
private:

    dynamixel::PortHandler *p_portHandler;
    dynamixel::PacketHandler *p_packetHandler;
    double m_protocol;
    int m_baudrate;
    int m_type;

    int m_dxl_comm_result;
    uint8_t m_dxl_error;

    // addresses
    uint16_t m_add_drive_mode;
    uint16_t m_add_operating_mode;
    uint16_t m_add_torque_enable;
    uint16_t m_add_goal_current;
    uint16_t m_add_goal_velocity;
    uint16_t m_add_goal_position;
    uint16_t m_add_goal_present;
    uint16_t m_add_present_current;
    uint16_t m_add_present_velocity;
    uint16_t m_add_present_position;
    uint16_t m_add_velocity_gain_P;
    uint16_t m_add_velocity_gain_I;

    // Ratio
    //double m_ration_torque_to_pulse;

public:
    skServoDynamixel();
    skServoDynamixel(const int& address, const int& param);
    ~skServoDynamixel();

    bool initialize(std::string port_name, int mode, int number_of_motor/* = 1*/);
    void closeServo();

    bool sendCmd(const int& j, const double& cmd);
    void setMode(const int& j, const int& mode);

    double getVelocity(const int& j) const;
    double getPosition(const int& j) const;
    double getCurrent(const int& j) const;

    void setting(const int& address, const int& param);

    void setGain(const int& mode=SK_ROBOT_LIB_SERVO_GAIN_VEL, const double& P=DYXEL_XM430_VELOCITY_P_GAIN, const double& I=DYXEL_XM430_VELOCITY_I_GAIN, const double& D=0.0);

private:
    void setAddress430();
    void setParam350();
    void setParam210();
    void setParam150();
    void setAddressAX12();
    void setParamW();

    int32_t deg2pulse(const double& deg) const;
    int32_t rad2pulse(const double& rad) const;
    double pulse2deg(const int& pulse) const;
    double pulse2rad(const int& pulse) const;

    int32_t vel2pulse(const double& vel) const;
    double pulse2vel(const int& pulse) const;
    int16_t tor2pulse(const double& tor) const;
    double pulse2tor(const int& pulse) const;

    int tor2current(const double& tor) const;
    double current2tor(const int& current) const;

};

#endif
