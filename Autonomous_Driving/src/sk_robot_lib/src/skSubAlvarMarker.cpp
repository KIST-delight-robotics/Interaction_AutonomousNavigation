#include "ros/ros.h"
#include "sk_mobile_lib/skSubAlvarMarker.h"
#include "stdio.h"

void skSubAlvarMarker::initialize(ros::NodeHandle n, const int id/* = 0*/, const string name/* = "pose_est"*/)
{
    m_id = id;
    m_count = 0;
    m_subPose = n.subscribe(name, 2, &skSubAlvarMarker::poseCallback, this);
}

void skSubAlvarMarker::poseCallback(const ar_track_alvar_msgs::AlvarMarkers& msg)
{
    m_valid_pose = false;
    
    for( int j = 0; j < msg.markers.size(); j++ )
    {
        if( msg.markers[j].id == m_id )
        {
            this->m_pose = msg.markers[j].pose.pose;
            this->m_count++;
            this->m_valid_pose = true;
        }
    }
    //ROS_INFO("[DEBUG/SUB_POSE] m_pose_est = (%.2f, %.2f, %.1f).", m_pose.x, m_pose.y, m_pose.theta*RAD2DEG);
}

geometry_msgs::Pose skSubAlvarMarker::getPose()
{
    if( m_valid_pose )
        return (m_pose);
    else
    {
        m_pose.position.x = 0.0;
        m_pose.position.y = 0.0;
        m_pose.position.z = 0.0;
        m_pose.orientation.x = 0.0;
        m_pose.orientation.y = 0.0;
        m_pose.orientation.z = 0.0;
        m_pose.orientation.w = 1.0;
        return (m_pose);
    }
}

bool skSubAlvarMarker::isValidPose()
{
    return (m_valid_pose);
}

unsigned int skSubAlvarMarker::getCount() const
{
    return (m_count);
}
