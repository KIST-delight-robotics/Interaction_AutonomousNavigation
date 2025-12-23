#ifndef _SK_MOBILE_LIB_SUB_QUATERNIONSTAMPED_H_
#define _SK_MOBILE_LIB_SUB_QUATERNIONSTAMPED_H_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/quaternion_stamped.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubQuaternionStamped : public skSubMsg<geometry_msgs::msg::QuaternionStamped>
{
public:
    skSubQuaternionStamped();

    ~skSubQuaternionStamped();

    double getRoll() const;
    double getPitch() const;
    double getYaw() const;
};

#endif // _SK_MOBILE_LIB_SUB_QUATERNIONSTAMPED_H_
