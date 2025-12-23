#include "sk_robot_lib/skKinematicsMobilePass.h"

skKinematicsMobilePass::skKinematicsMobilePass()
{
    this->m_number_of_motor = 3;
    this->m_angular_vel_first = false;

    this->m_motor_velocity.resize(3,0.0);
    this->m_motor_position.resize(3,0.0);
}

skKinematicsMobilePass::~skKinematicsMobilePass()
{
}

bool skKinematicsMobilePass::initialize()
{
    return (true);
}

bool skKinematicsMobilePass::setCmd(const geometry_msgs::msg::Twist& cmd)
{
    //ROS_INFO("[DEBUG][2WDD] m_wheel_dist = %.3f, m_wheel_radius = %.3f.", this->m_wheel_dist, this->m_wheel_radius);
    //this->m_motor_velocity[0] = this->m_wheel_dir[0]*(cmd.linear.x + this->m_wheel_side[0] * this->m_wheel_dist*cmd.angular.z*0.5) / this->m_wheel_radius;
    //this->m_motor_velocity[1] = this->m_wheel_dir[1]*(cmd.linear.x + this->m_wheel_side[1] * this->m_wheel_dist*cmd.angular.z*0.5) / this->m_wheel_radius;
    this->m_motor_velocity[0] = cmd.linear.x;
    this->m_motor_velocity[1] = cmd.linear.y;
    this->m_motor_velocity[2] = cmd.angular.z;

    return (true);
}

geometry_msgs::msg::Twist skKinematicsMobilePass::getTwist(const double *position, const bool init/* = false*/)
{
    geometry_msgs::msg::Twist odom;

    odom.linear.x = position[0];
    odom.linear.y = 0.0;
    odom.linear.z = 0.0;
    odom.angular.x = 0.0;
    odom.angular.y = 0.0;
    odom.angular.z = position[1];

    return (odom);
}
