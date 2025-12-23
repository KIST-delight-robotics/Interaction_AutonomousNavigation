#include "sk_robot_lib/skDynamicObstacleAvoidance.h"

skDynamicObstacleAvoidance::skDynamicObstacleAvoidance() : p_robot(NULL), p_scan(NULL), p_grid(NULL)
{}

skDynamicObstacleAvoidance::~skDynamicObstacleAvoidance()
{}

void skDynamicObstacleAvoidance::activate(rclcpp::Node* node)
{
    this->p_node = node;
}

void skDynamicObstacleAvoidance::activate(skRobot* robot)
{
    this->p_robot = robot;
}
void skDynamicObstacleAvoidance::activate(skSubLaserScan* scan)
{
    this->p_scan = scan;
}

void skDynamicObstacleAvoidance::activate(skSubOccupancyGrid* grid)
{
    this->p_grid = grid;
}

void skDynamicObstacleAvoidance::setParams(const skDynamicObstacleAvoidanceParam& param)
{
    this->m_param = param;
    this->m_param.step_size_x = this->m_param.max_vel_x * this->m_param.dt / this->m_param.acc;
    this->m_param.step_size_y = this->m_param.max_vel_y * this->m_param.dt / this->m_param.acc;
    this->m_param.step_size_th = this->m_param.max_vel_th * this->m_param.dt / this->m_param.acc;
}

geometry_msgs::msg::Twist skDynamicObstacleAvoidance::getTwist(const geometry_msgs::msg::Twist v_d, const geometry_msgs::msg::Twist v_c) const
{
    return (v_d);
}
