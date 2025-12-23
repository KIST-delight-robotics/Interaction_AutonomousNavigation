#include "sk_robot_lib/skSubArucoDetection.h"

skSubArucoDetection::skSubArucoDetection()
{
}

skSubArucoDetection::~skSubArucoDetection()
{
}

bool skSubArucoDetection::hasID(const int& id) const
{
    for( unsigned int j = 0; j < this->m_msg.markers.size(); j++ )
    {
        if( this->m_msg.markers[j].marker_id == id )
            return (true);
    }
    return (false);
}

geometry_msgs::msg::Pose skSubArucoDetection::getPose(const int& id) const
{
    for( unsigned int j = 0; j < this->m_msg.markers.size(); j++ )
    {
        if( this->m_msg.markers[j].marker_id == id )
            return (this->m_msg.markers[j].pose);
    }
    return (geometry_msgs::msg::Pose());
}

double skSubArucoDetection::getTime() const
{
    double time = toMSec(this->m_msg.header.stamp);
    int trim(time/100.0);
    return (time - ((double)trim)*100,0);
}

double skSubArucoDetection::getTimeFrom(builtin_interfaces::msg::Time start) const
{
    return (toMSecBtw(this->m_msg.header.stamp, start));
}

unsigned int skSubArucoDetection::sizeOfDetections() const
{
    return (this->m_msg.markers.size());
}

