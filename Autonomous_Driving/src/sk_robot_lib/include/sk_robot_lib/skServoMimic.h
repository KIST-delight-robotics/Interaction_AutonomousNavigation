#ifndef _SK_ROBOT_LIB_SERVO_MIMIC_H_
#define _SK_ROBOT_LIB_SERVO_MIMIC_H_

#include "rclcpp/rclcpp.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sk_robot_lib/skServo.h"

#define SK_SERVO_MIMIC_PERIOD               (20)

using namespace std;

class skServoMimic : public skServo
{
private:
    double* p_v;
    int m_count;

public:
    skServoMimic();
    ~skServoMimic();

    bool initialize(std::string port_name, int mode, int number_of_motor = 1);
    void closeServo();

    bool sendCmd(const int& j, const double& cmd);
    void setMode(const int& j, const int& mode);

    double getVelocity(const int& j) const;
    double getPosition(const int& j) const;
    double getCurrent(const int& j) const;

    void setGain(const int& mode=SK_ROBOT_LIB_SERVO_GAIN_DEFAULT, const double& P=0.0, const double& D=0.0, const double& I=0.0);
};

#endif
