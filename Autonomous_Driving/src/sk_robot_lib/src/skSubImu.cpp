#include "sk_robot_lib/skSubImu.h"
//#include "skSubMsg.cpp"

skSubImu::skSubImu() : m_yaw_offset(0.0), m_drift(0.0)
{
}

skSubImu::~skSubImu()
{
}

void skSubImu::setDrift(const double& drift)
{
    this->m_drift = drift;
}

double skSubImu::getRoll(bool init/* = false*/)
{
    if( this->haveMsg() )
    {
        const double x(this->m_msg.orientation.x);
        const double y(this->m_msg.orientation.y);
        const double z(this->m_msg.orientation.z);
        const double w(this->m_msg.orientation.w);

        const double time(toSec(this->m_msg.header.stamp));

        double roll(atan2(2.0*(w*x+y*z), 1.0-2.0*(x*x+y*y)));
        /*if( init )
        {
            this->m_time_start = time;
        }
        else
        {
            yaw -= this->m_drift * (time - this->m_time_start);
        }*/

        return (roll);
    }

    return (0.0);
}

double skSubImu::getPitch(bool init/* = false*/)
{
    if( this->haveMsg() )
    {
        const double x(this->m_msg.orientation.x);
        const double y(this->m_msg.orientation.y);
        const double z(this->m_msg.orientation.z);
        const double w(this->m_msg.orientation.w);

        const double time(toSec(this->m_msg.header.stamp));

        double pitch(2.0*atan2(sqrt(1.0+2.0*(w*y-x*z)), sqrt(1.0-2.0*(w*y-x*z)))-0.5*M_PI);
        /*if( init )
        {
            this->m_time_start = time;
        }
        else
        {
            pitch -= this->m_drift * (time - this->m_time_start);
        }*/

        return (pitch);
    }

    return (0.0);
}

double skSubImu::getYaw(bool init/* = false*/)
{
    if( this->haveMsg() )
    {
        const double x(this->m_msg.orientation.x);
        const double y(this->m_msg.orientation.y);
        const double z(this->m_msg.orientation.z);
        const double w(this->m_msg.orientation.w);

        //const double time(toSec(this->m_msg.header.stamp));

        double yaw(atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z))+this->m_yaw_offset);
        trimPIPI(yaw);
        /*if( init )
        {
            this->m_time_start = time;
        }
        else
        {
            yaw -= this->m_drift * (time - this->m_time_start);
        }*/

        return (yaw);
    }

    return (0.0);
}

bool skSubImu::setZeroYaw(const double offset)
{
    if( this->haveMsg() )
    {
        const double x(this->m_msg.orientation.x);
        const double y(this->m_msg.orientation.y);
        const double z(this->m_msg.orientation.z);
        const double w(this->m_msg.orientation.w);

        const double yaw(atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z)));

        this->m_yaw_offset = offset - yaw;

        return (true);
    }

    return (false);
}

