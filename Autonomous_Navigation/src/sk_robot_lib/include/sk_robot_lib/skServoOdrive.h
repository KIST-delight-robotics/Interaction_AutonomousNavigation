#ifndef _SK_ROBOT_LIB_SERVO_ODRIVE_H_
#define _SK_ROBOT_LIB_SERVO_ODRIVE_H_

#include "rclcpp/rclcpp.hpp"

#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include <signal.h>

// Serial Communication
#include <errno.h>
#include <fcntl.h> 
#include <termios.h>
#include <unistd.h>

#include "sk_robot_lib/skServo.h"

// Parameters
#define MAX_PACKET_SIZE   (100)
#define SK_ROBOT_LIB_ODRIVE_DEFAULT_PORT        "/dev/ttyACM0"
#define CONTROL_MODE_VELOCITY_CONTROL (2)
#define AXIS_STATE_CLOSED_LOOP_CONTROL (8)

/*typedef unsigned char 	    BYTE;
typedef unsigned int 	    WORD;

typedef struct {
	BYTE byLow;
	BYTE byHigh;
} IByte;*/

using namespace std;

#define SK_ROBOT_LIB_ODRIVE_MODE_IDLE                       (0)
#define SK_ROBOT_LIB_ODRIVE_MODE_VEL                        (2)
#define SK_ROBOT_LIB_ODRIVE_STATE_CLOSED_LOOP_CONTROL       (8)
#define SK_ROBOT_LIB_ODRIVE_VELOVITY_P_GAIN                 (1.0)
#define SK_ROBOT_LIB_ODRIVE_VELOVITY_I_GAIN                 (1.0)
#define SK_ROBOT_LIB_ODRIVE_VELOVITY_D_GAIN                 (1.0)

using namespace std;

class skServoOdrive : public skServo
{
private:
    int m_fd;
    struct termios m_tty;
  	BYTE m_byD[MAX_PACKET_SIZE], m_byInData[MAX_PACKET_SIZE];

public:
    skServoOdrive();
    ~skServoOdrive();

    bool initialize(std::string port_name, int mode, int number_of_motor/* = 1*/);
    void closeServo();

    bool sendCmd(const int& j, const double& cmd);
    void setMode(const int& j, const int& mode);

    double getVelocity(const int& j) const;
    double getPosition(const int& j) const;
    double getCurrent(const int& j) const;

    void setting(const int& address, const int& param);

    void setGain(const int& mode=SK_ROBOT_LIB_ODRIVE_MODE_VEL, const double& P=SK_ROBOT_LIB_ODRIVE_VELOVITY_P_GAIN, const double& I=SK_ROBOT_LIB_ODRIVE_VELOVITY_I_GAIN, const double& D=SK_ROBOT_LIB_ODRIVE_VELOVITY_D_GAIN);

private:

    int sendData(const int &n, const bool& response = false);
    int readParam(const char* param);
    void setParam(const int& motor, const char* param, const char* value = NULL);
    int vel2rpm(const double& v);
    IByte Int2Byte(const int &nIn);

    int32_t deg2pulse(const double& deg) const;
    int32_t rad2pulse(const double& rad) const;
    double pulse2deg(const int& pulse) const;
    double pulse2rad(const int& pulse) const;

    int32_t vel2pulse(const double& vel) const;
    double pulse2vel(const int& pulse) const;
    int16_t tor2pulse(const double& tor) const;
};

#endif
