#ifndef _SK_ROBOT_LIB_SERVO_MD400T_H_
#define _SK_ROBOT_LIB_SERVO_MD400T_H_

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

// Parameters
#define MOTOR_CONTROLLER_MACHINE_ID     183
#define USER_MACHINE_ID                 184
#define ID                              1
#define PID_VEL_CMD                     130
#define PID_VEL_CMD_2                   131
#define PID_PNT_VEL_CMD                 207
#define PID_PNT_MAIN_TOTAL_DATA_NUM     24 

#define PID_MAIN_DATA                   193

#define ENABLE                          1  
#define RETURN_PNT_MAIN_DATA            2    

#define MAX_PACKET_SIZE_MD400T          16

#define MDH180_GEAR_RATIO               (4.33)

/*typedef unsigned char 	BYTE;
typedef unsigned int 	WORD;

typedef struct {
	BYTE byLow;
	BYTE byHigh;
} IByte;*/

using namespace std;

class skServoMD400T : public skServo
{
private:

    int m_fd;
    struct termios m_tty;
  	BYTE m_byD[MAX_PACKET_SIZE_MD400T], m_byInData[MAX_PACKET_SIZE_MD400T];
    double m_gear_ratio;
    double m_motor_angle[2];

public:
    skServoMD400T();
    ~skServoMD400T();

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
    int sendData(const int& n, const int read_response = 0);
    IByte vel2Byte(const double &v);
    double Byte2Pos(const BYTE* data);
};

#endif
