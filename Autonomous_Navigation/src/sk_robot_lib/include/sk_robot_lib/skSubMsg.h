#ifndef _SK_ROBOT_LIB_SUB_MSG_H_
#define _SK_ROBOT_LIB_SUB_MSG_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/quaternion_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

#include "aruco_opencv_msgs/msg/aruco_detection.hpp"

#include "sk_robot_msgs/msg/robot_state.hpp"

#include "sk_robot_lib/skRobotBasicFunctions.h"
#include "sk_robot_lib/skAvgFilter.h"

//#include "message_memory_strategy.hpp"
//#include <memory>

//using namespace std;
//using namespace sk_robot_msgs;
using std::placeholders::_1;

template<class cMsg>
class skSubMsg
{
protected:
    class rclcpp::Subscription< cMsg >::SharedPtr p_sub;

    bool m_have_msg;
    cMsg m_msg;

    bool m_avg_filter;

    rclcpp::Node* p_node;

    virtual void msgCallback(const class cMsg::SharedPtr msg);

    virtual void saveMsgFilter();

public:
    skSubMsg();

    ~skSubMsg();

    void initialize(rclcpp::Node* node, const std::string param, const std::string name);
    virtual void initialize(rclcpp::Node* node, const std::string name);

    virtual void activateAvgFilter(const int length = 10);

    bool haveMsg() const;
    void resetMsg();
    cMsg getMsg() const;
};

template class skSubMsg<std_msgs::msg::Float64MultiArray>;
template class skSubMsg<geometry_msgs::msg::Twist>;
template class skSubMsg<geometry_msgs::msg::Pose2D>;
template class skSubMsg<geometry_msgs::msg::QuaternionStamped>;
template class skSubMsg<sensor_msgs::msg::Imu>;
template class skSubMsg<sensor_msgs::msg::Joy>;
template class skSubMsg<sensor_msgs::msg::LaserScan>;
template class skSubMsg<sensor_msgs::msg::NavSatFix>;
template class skSubMsg<sensor_msgs::msg::MagneticField>;
template class skSubMsg<nav_msgs::msg::OccupancyGrid>;
template class skSubMsg<aruco_opencv_msgs::msg::ArucoDetection>;
template class skSubMsg<sk_robot_msgs::msg::RobotState>;

#endif