#include "sk_robot_lib/skSubMag.h"

skSubMag::skSubMag() : m_offset(0.0), p_x(NULL), p_y(NULL)
{
}

skSubMag::~skSubMag()
{
    if( this->m_avg_filter )
    {
        delete this->p_x;
        delete this->p_y;
    }
}

void skSubMag::setOffset(const double offset)
{
    this->m_offset = offset;
}

void skSubMag::activateAvgFilter(const int length/* = 10*/)
{
    this->m_avg_filter = true;

    this->p_x = new skAvgFilter(length);
    this->p_y = new skAvgFilter(length);
}

void skSubMag::saveMsgFilter()
{
    this->p_x->addData(this->m_msg.magnetic_field.x);
    this->p_y->addData(this->m_msg.magnetic_field.y);
}

#if 0
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
#endif

double skSubMag::getYaw()
{
    if( this->haveMsg() )
    {
        double yaw;
        if( this->m_avg_filter )
            yaw = atan2(this->p_y->getAvg(), this->p_x->getAvg());
        else
            yaw = atan2(this->m_msg.magnetic_field.y, this->m_msg.magnetic_field.x);
        yaw += this->m_offset;
        trim(yaw);

        return (yaw);
    }

    return (0.0);
}

