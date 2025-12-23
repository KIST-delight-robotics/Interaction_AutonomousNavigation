#include "sk_robot_lib/skSubJoyMobile.h"
#include "stdio.h"

skSubJoyMobile::skSubJoyMobile() : m_twist(), m_event(0)
{
}

skSubJoyMobile::~skSubJoyMobile()
{
}

void skSubJoyMobile::updateJoy()
{
    this->updateTwist();
}

void skSubJoyMobile::updateTwist()
{
    this->zeroTwist();

    if( this->LB() )
    {
        // Manual mode
        this->m_state = SK_MOBILE_ROBOT_MANUAL;

        // check linear speed
        this->m_twist.linear.x = 0.0;
        this->m_twist.linear.y = 0.0;
        if( this->UP() )
        {
            this->m_twist.linear.x = 1.0;
        }
        else if( this->DOWN() )
        {
            this->m_twist.linear.x = -1.0;
        }
        else if( this->LEFT() )
        {
            this->m_twist.linear.y = 1.0;
        }
        else if( this->RIGHT() )
        {
            this->m_twist.linear.y = -1.0;
        }
        else
        {
            this->m_twist.linear.x = this->LEFT_Y();
            this->m_twist.linear.y = this->LEFT_X();
        }

        // check angular speed
        if( this->X() )
        {
            this->m_twist.angular.z = 1.0;
        }
        else if( this->B() )
        {
            this->m_twist.angular.z = -1.0;
        }
        else
        {
            this->m_twist.angular.z = this->RIGHT_X();
        }
    }
    else
    {
        this->m_state = SK_MOBILE_ROBOT_IDLE;
    }
}

geometry_msgs::msg::Twist skSubJoyMobile::getTwist() const
{
    return (this->m_twist);
}

unsigned int skSubJoyMobile::getEvent()
{
    const unsigned int ret(this->m_event);
    this->m_event = 0;

    return (ret);
}

void skSubJoyMobile::zeroTwist()
{
    this->m_twist.linear.x = 0.0;
    this->m_twist.linear.y = 0.0;
    this->m_twist.linear.z = 0.0;
    this->m_twist.angular.x = 0.0;
    this->m_twist.angular.y = 0.0;
    this->m_twist.angular.z = 0.0;
}
