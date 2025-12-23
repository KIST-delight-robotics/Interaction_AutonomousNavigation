#ifndef _SK_ROBOT_LIB_SUB_F64MA_H_
#define _SK_ROBOT_LIB_SUB_F64MA_H_

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubF64MA : public skSubMsg<std_msgs::msg::Float64MultiArray>
{
public:
    skSubF64MA();

    ~skSubF64MA();
};

#endif // _SK_ROBOT_LIB_SUB_F64MA_H_
