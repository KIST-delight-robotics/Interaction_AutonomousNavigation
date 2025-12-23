#ifndef _SK_ROBOT_LIB_DYNAMIC_OBSTACLE_AVOIDANCE_H_
#define _SK_ROBOT_LIB_DYNAMIC_OBSTACLE_AVOIDANCE_H_

#include "rclcpp/rclcpp.hpp"

#include <iostream>
#include <fstream>

#include "geometry_msgs/msg/twist.hpp"

#include "sk_robot_lib/skRobotCommon.h"
#include "sk_robot_lib/skSubLaserScan.h"
#include "sk_robot_lib/skSubOccupancyGrid.h"

using namespace std;

struct skDynamicObstacleAvoidanceParam
{
    double radius;
    double margin;

    double max_vel_x;
    double max_vel_y;
    double max_vel_th;

    double acc;
    double dt;

    double step_size_x;
    double step_size_y;
    double step_size_th;

    int resolution_x;
    int resolution_y;
    int resolution_th;
};

class skRobot;

class skDynamicObstacleAvoidance
{
protected:
    rclcpp::Node* p_node;
    skRobot* p_robot;
    skSubLaserScan* p_scan;
    skSubOccupancyGrid* p_grid;

    skDynamicObstacleAvoidanceParam m_param;

public:
    skDynamicObstacleAvoidance();

    ~skDynamicObstacleAvoidance();

    void activate(rclcpp::Node* node);
    void activate(skRobot* robot);
    void activate(skSubLaserScan* scan);
    void activate(skSubOccupancyGrid* grid);

    virtual void setParams(const skDynamicObstacleAvoidanceParam& param);
    virtual geometry_msgs::msg::Twist getTwist(const geometry_msgs::msg::Twist v_d, const geometry_msgs::msg::Twist v_c) const;
};

#endif