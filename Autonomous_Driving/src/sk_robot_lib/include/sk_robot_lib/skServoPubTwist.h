#ifndef _SK_ROBOT_LIB_SERVO_PUB_TWIST_H_
#define _SK_ROBOT_LIB_SERVO_PUB_TWIST_H_

#include "rclcpp/rclcpp.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "geometry_msgs/msg/twist.hpp"

#include "sk_robot_lib/skServo.h"

using namespace std;

class skServoPubTwist : public skServo
{
private:
    geometry_msgs::msg::Twist m_cmd;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr p_pubCmd;

public:
    skServoPubTwist();
    ~skServoPubTwist();

    bool initialize(std::string port_name, int mode, int number_of_motor = 1);
    void closeServo();

    bool sendCmd(double *cmd);
};

#endif
