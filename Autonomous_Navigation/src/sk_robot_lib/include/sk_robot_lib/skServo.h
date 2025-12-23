#ifndef _SK_ROBOT_LIB_SERVO_H_
#define _SK_ROBOT_LIB_SERVO_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "sk_robot_lib/skRobotCommon.h"
#include "sk_robot_lib/skRobotBasicFunctions.h"

#define SK_ROBOT_LIB_SERVO_GAIN_DEFAULT     (0)
#define SK_ROBOT_LIB_SERVO_GAIN_POS         (1)
#define SK_ROBOT_LIB_SERVO_GAIN_VEL         (2)
#define SK_ROBOT_LIB_SERVO_GAIN_TOR         (3)

typedef unsigned char 	BYTE;
typedef unsigned int 	WORD;

typedef struct {
	BYTE byLow;
	BYTE byHigh;
} IByte;

using namespace std;

class skServo
{
protected:
    int m_number_of_motor;
    std::vector<int> m_mode_current;

    // Unit convertion, angle
    double m_rad2pulse;
    double m_deg2pulse;
    double m_pulse2rad;
    double m_pulse2deg;

    // Unit convertion, velocity, rad/sec
    double m_vel2pulse;
    double m_pulse2vel;

    // Unit convertion, torque, Nm
    double m_pulse2tor;
    double m_tor2pulse;

    // for ROS Log
    rclcpp::Node* p_node;

    // for Debugging
    std::vector<int> m_pulse;

public:
    skServo();
    ~skServo();

    virtual bool initialize(std::string port_name, int mode, int number_of_motor = 1);
    virtual void closeServo();

    virtual bool sendCmd(double *cmd);
    virtual bool sendCmd(const int& j, const double& cmd) = 0;
    void setMode(int *mode);
    virtual void setMode(const int& j, const int& mode) = 0;

    bool getVelocity(double *vel) const;
    bool getPosition(double *pos) const;
    bool getCurrent(double *cur) const;
    virtual double getVelocity(const int& j) const = 0;
    virtual double getPosition(const int& j) const = 0;
    virtual double getCurrent(const int& j) const = 0;

    bool getState(int *state) const;
    int getState(const int& j) const;

    int getNumberOfMotor() const;

    virtual void setGain(const int& mode=SK_ROBOT_LIB_SERVO_GAIN_DEFAULT, const double& P=0.0, const double& D=0.0, const double& I=0.0) = 0;

    void activateLog(rclcpp::Node* node);

    // for Debugging
    int getPulse(unsigned int idx) const;
};

#endif
