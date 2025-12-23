#include "sk_robot_lib/skSubNavSatFix.h"
//#include "skSubMsg.cpp"

skSubNavSatFix::skSubNavSatFix()
{
}

skSubNavSatFix::~skSubNavSatFix()
{
}

int skSubNavSatFix::getState() const
{
    return (this->m_msg.status.status);
}

bool skSubNavSatFix::valid() const
{
    return (this->m_msg.status.status >= 0);
}

double skSubNavSatFix::getLatitude() const // Degree
{
    return (this->m_msg.latitude);
}

double skSubNavSatFix::getLongitude() const // Degree
{
    return (this->m_msg.longitude);
}

double skSubNavSatFix::getAltitude() const // meter
{
    return (this->m_msg.altitude);
}