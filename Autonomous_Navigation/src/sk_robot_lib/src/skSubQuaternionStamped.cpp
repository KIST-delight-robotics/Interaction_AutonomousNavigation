#include "sk_robot_lib/skSubQuaternionStamped.h"
//#include "skSubMsg.cpp"

skSubQuaternionStamped::skSubQuaternionStamped()
{
}

skSubQuaternionStamped::~skSubQuaternionStamped()
{
}

double skSubQuaternionStamped::getRoll() const
{
    if( this->haveMsg() )
    {
        return roll(this->m_msg.quaternion);
    }

    return (0.0);
}

double skSubQuaternionStamped::getPitch() const
{
    if( this->haveMsg() )
    {
        return pitch(this->m_msg.quaternion);
    }

    return (0.0);
}

double skSubQuaternionStamped::getYaw() const
{
    if( this->haveMsg() )
    {
        return yaw(this->m_msg.quaternion);
    }

    return (0.0);
}

