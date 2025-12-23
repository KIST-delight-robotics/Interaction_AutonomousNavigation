#ifndef _SK_ROBOT_LIB_SUB_ROBOT_STATE_H_
#define _SK_ROBOT_LIB_SUB_ROBOT_STATE_H_

#include "rclcpp/rclcpp.hpp"
#include "sk_robot_msgs/msg/robot_state.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubRobotState : public skSubMsg<sk_robot_msgs::msg::RobotState>
{
public:
    skSubRobotState();

    ~skSubRobotState();

    double getTime() const;
    double getTimeFrom(builtin_interfaces::msg::Time start) const;
};

#endif // _SK_ROBOT_LIB_SUB_ROBOT_STATE_H_
