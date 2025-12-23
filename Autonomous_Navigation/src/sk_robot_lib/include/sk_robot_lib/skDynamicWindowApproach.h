#ifndef _SK_ROBOT_LIB_DYNAMIC_WINDOW_APPROACH_H_
#define _SK_ROBOT_LIB_DYNAMIC_WINDOW_APPROACH_H_

#include "rclcpp/rclcpp.hpp"

#include <iostream>
#include <fstream>

#include "sensor_msgs/msg/laser_scan.hpp"

#include "sk_robot_lib/skRobotCommon.h"
#include "sk_robot_lib/skDynamicObstacleAvoidance.h"

#define DWA_DEFAULT_RESOLUTION          (10)
#define DWA_ACTIVATE_DIST_MAP           (1)

using namespace std;

class skDynamicWindowApproach : public skDynamicObstacleAvoidance
{
private:
    sensor_msgs::msg::LaserScan m_scan;

    /*double m_radius;
    double m_margin;

    int m_resolution_x;
    int m_resolution_y;
    int m_resolution_th;*/

    double distTwist(const geometry_msgs::msg::Twist vd, const geometry_msgs::msg::Twist v) const;
    double cost(const geometry_msgs::msg::Twist v_d, const geometry_msgs::msg::Twist v, const bool& log = false) const;

public:
    skDynamicWindowApproach();

    ~skDynamicWindowApproach();

    //virtual void setParams(const skDynamicObstacleAvoidanceParam& param);
    geometry_msgs::msg::Twist getTwist(const geometry_msgs::msg::Twist v_d, const geometry_msgs::msg::Twist v_c) const;
};

#endif