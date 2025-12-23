#include "sk_robot_lib/skKinematicsMobile.h"

skKinematicsMobile::skKinematicsMobile() : m_number_of_motor(0), m_angular_vel_first(false)
{}

skKinematicsMobile::~skKinematicsMobile()
{
}

bool skKinematicsMobile::initialize(const string name)
{
    return (false);
}

bool skKinematicsMobile::setCmd(const geometry_msgs::msg::Twist& cmd)
{
    this->m_twist = cmd;

    // Add Kinematics Solver for each robot

    return (false);
}

double skKinematicsMobile::getVel(const int& j) const
{
    if( j < this->m_number_of_motor )
        return (this->m_motor_velocity[j]);
    
    return (0.0);
}

bool skKinematicsMobile::getVel(double *vel) const
{
    for( unsigned int j = 0; j < this->m_number_of_motor; j++ )
        vel[j] = this->m_motor_velocity[j];

    return (true);
}

void skKinematicsMobile::initOdom(const double* position)
{
    for( unsigned int j = 0; j < this->m_number_of_motor; j++ )
        this->m_motor_position[j] = position[j];
    
    this->m_pose_x = 0.0;
    this->m_pose_y = 0.0;
    this->m_pose_th = 0.0;
}

geometry_msgs::msg::Twist skKinematicsMobile::getOdom(const double*position)
{
    const geometry_msgs::msg::Twist twist(this->getTwist(position));
    const double th(this->m_odom.angular.z + 0.5*twist.angular.z);

    this->m_odom.linear.x += twist.linear.x * cos(th) - twist.linear.y * sin(th);
    this->m_odom.linear.y += twist.linear.x * sin(th) + twist.linear.y * cos(th);
    this->m_odom.angular.z += twist.angular.z;

    return (this->m_odom);
}

void skKinematicsMobile::updatePose()
{
    const double dth(this->m_odom.angular.z);
    const double dx(this->m_odom.linear.x*cos(dth)-this->m_odom.linear.x*sin(dth));
    const double dy(this->m_odom.linear.x*sin(dth)+this->m_odom.linear.x*cos(dth));
}

void skKinematicsMobile::setAngularFirstMode(const double& maxAngularVelocity)
{}

void skKinematicsMobile::releaseAngularFirstMode()
{}
