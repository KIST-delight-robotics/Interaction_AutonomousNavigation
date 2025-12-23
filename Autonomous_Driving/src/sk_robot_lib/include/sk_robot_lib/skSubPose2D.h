#ifndef _SK_ROBOT_LIB_SUB_POSE2D_H_
#define _SK_ROBOT_LIB_SUB_POSE2D_H_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubPose2D : public skSubMsg<geometry_msgs::msg::Pose2D>
{
public:
    skSubPose2D();

    ~skSubPose2D();
};

#endif // _SK_ROBOT_LIB_SUB_POSE2D_H_
