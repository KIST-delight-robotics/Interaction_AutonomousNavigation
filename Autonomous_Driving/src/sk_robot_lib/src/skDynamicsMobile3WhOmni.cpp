#include "sk_robot_lib/skDynamicsMobile3WhOmni.h"

skDynamicsMobile3WhOmni::skDynamicsMobile3WhOmni()
{
    this->m_number_of_motor = 3;
    this->m_motor_velocity.resize(3,0.0);
}

skDynamicsMobile3WhOmni::skDynamicsMobile3WhOmni(const string name)
{
    this->m_number_of_motor = 3;
    this->m_motor_velocity.resize(3,0.0);
    this->skKinematicsMobile3WhOmni::initialize(name);

    if( name.compare("Kibot") == 0 || name.compare("KIBOT") == 0  || name.compare("kibot") == 0 )
    {
        this->m_m = KIBOT_MASS;
        this->m_I = KIBOT_INERTIA;
        this->m_Kp = KIBOT_KP;
    }
}

skDynamicsMobile3WhOmni::~skDynamicsMobile3WhOmni()
{
}

void skDynamicsMobile3WhOmni::setGain(const double& Kp)
{
    this->m_Kp = Kp;
}

void skDynamicsMobile3WhOmni::setCurVel(const geometry_msgs::msg::Twist& vel)
{
    this->m_vel = vel;
}

void skDynamicsMobile3WhOmni::setCurVel(const double* vel)
{
    this->m_vel.linear.x = 0;
    this->m_vel.linear.y = 0;
    this->m_vel.angular.z = 0;

    for( unsigned int j = 0; j < 3; j++ )
    {
        this->m_vel.linear.x += this->m_J[0][j]*vel[j];
        this->m_vel.linear.y += this->m_J[1][j]*vel[j];
        this->m_vel.angular.z += this->m_J[2][j]*vel[j];
    }
}

double skDynamicsMobile3WhOmni::ifCollide() const
{
    if( fabs(this->m_vel.angular.z) < KIBIT_COLLIDE_MIN_ANGULAR_VELOCITY )
        return (-3.0*M_PI);

    const double r(sqrt(this->m_vel.linear.x*this->m_vel.linear.x+this->m_vel.linear.y*this->m_vel.linear.y)/fabs(this->m_vel.angular.z));

    if( r < KIBOT_BASE_RADIUS-KIBOT_COLLIDE_DIST_MARGIN || KIBOT_BASE_RADIUS+KIBOT_COLLIDE_DIST_MARGIN < r )
        return (-3.0*M_PI);

    double theta(atan2(this->m_vel.linear.x/this->m_vel.angular.z,-1.0*this->m_vel.linear.y/this->m_vel.angular.z));
    while( theta < 0.0 )
        theta += 2.0*M_PI;
    
    return (theta);
}

bool skDynamicsMobile3WhOmni::setCmd(const geometry_msgs::msg::Twist& cmd)
{
    const double fx(this->m_m*this->m_Kp*(cmd.linear.x - this->m_vel.linear.x));
    const double fy(this->m_m*this->m_Kp*(cmd.linear.y - this->m_vel.linear.y));
    const double tz(this->m_I*this->m_Kp*(cmd.angular.z - this->m_vel.angular.z));

    for( unsigned int j = 0; j < 3; j++ )
    {
        this->m_motor_torque[j] = this->m_J[0][j]*fx + this->m_J[1][j]*fy + this->m_J[2][j]*tz;
        //ROS_INFO("%.3f", this->m_J[0][j]*fx + this->m_J[1][j]*fy + this->m_J[2][j]*tz);
    }
    //ROS_INFO("[DEBUT] this->m_motor_torque = (%.3f, %.3f, %.3f).", this->m_motor_torque[0], this->m_motor_torque[1], this->m_motor_torque[2]);

    return (true);
}

double skDynamicsMobile3WhOmni::getTor(const int& j) const
{
    if( j < this->m_number_of_motor )
        return (this->m_motor_torque[j]);
    
    return (0.0);
}

bool skDynamicsMobile3WhOmni::getTor(double *tor) const
{
    for( unsigned int j = 0; j < this->m_number_of_motor; j++ )
        tor[j] = this->m_motor_torque[j];

    return (true);
}

geometry_msgs::msg::Twist skDynamicsMobile3WhOmni::getOdom(const double *position)
{
    geometry_msgs::msg::Twist odom;
    odom.linear.x = 0.0;
    odom.linear.y = 0.0;
    odom.linear.z = 0.0;
    odom.angular.x = 0.0;
    odom.angular.y = 0.0;
    odom.angular.z = 0.0;

    double diff[3];
    for( unsigned int j = 0; j < 3; j++ )
    {
        diff[j] = (position[j] - this->m_motor_position[j])*this->m_wheel_radius;

        odom.linear.x += this->m_J[0][j]*diff[j];
        odom.linear.y += this->m_J[1][j]*diff[j];
        odom.angular.z += this->m_J[2][j]*diff[j];

        this->m_motor_position[j] = position[j];
    }

    return (odom);
}
