#ifndef _SK_ROBOT_LIB_SERVO_WHEELTEC_H_
#define _SK_ROBOT_LIB_SERVO_WHEELTEC_H_

#include "rclcpp/rclcpp.hpp"

#include <stdlib.h>
#include <string.h>

// Serial Communication
#include <errno.h>
#include <fcntl.h> 
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

#include "sk_robot_lib/skServo.h"

#define MAX_PACKET_SIZE_WHEELTEC        (11)
#define FRAME_HEADER                    (0x7B)
#define FRAME_TAIL                      (0x7D)

// Port = /dev/ttyCH343USB0

using namespace std;

class skServoWheelTec : public skServo
{
private:
    int m_fd;
    struct termios m_tty;
  	BYTE m_byD[MAX_PACKET_SIZE_WHEELTEC], m_byInData[MAX_PACKET_SIZE_WHEELTEC];

public:
    skServoWheelTec();
    ~skServoWheelTec();

    bool initialize(std::string port_name, int mode, int number_of_motor/* = 1*/);
    void closeServo();

    bool sendCmd(double *cmd);
    bool sendCmd(const int& j, const double& cmd);
    void setMode(const int& j, const int& mode);

    double getVelocity(const int& j) const;
    double getPosition(const int& j) const;
    double getCurrent(const int& j) const;

    void setGain(const int& mode=SK_ROBOT_LIB_SERVO_GAIN_VEL, const double& P=0.0, const double& I=0.0, const double& D=0.0);

private:
    BYTE check_sum();
};

#endif
