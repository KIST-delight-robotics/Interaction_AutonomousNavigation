#ifndef _SK_MOBILE_LIB_SUB_TWIST_H_
#define _SK_MOBILE_LIB_SUB_TWIST_H_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubTwist : public skSubMsg<geometry_msgs::msg::Twist>
{
public:
    skSubTwist();

    ~skSubTwist();
};

#endif // _SK_MOBILE_LIB_SUB_TWIST_H_
