#include "ros/ros.h"
#include "sk_mobile_lib/skSubPoseStamped.h"
#include "stdio.h"

void skSubPoseStamped::initialize(ros::NodeHandle n, const string name/* = "pose_est"*/)
{
    m_subPose = n.subscribe(name, 2, &skSubPoseStamped::poseCallback, this);
}

void skSubPoseStamped::poseCallback(const geometry_msgs::PoseStamped& msg)
{
    this->m_msg = msg;
    this->m_valid_pose = true;
    //ROS_INFO("[DEBUG/SUB_POSE] m_pose_est = (%.2f, %.2f, %.1f).", m_pose.x, m_pose.y, m_pose.theta*RAD2DEG);
}

geometry_msgs::Pose skSubPoseStamped::getPose()
{
    if( m_valid_pose )
        return (m_msg.pose);
    else
    {
        geometry_msgs::Pose pose;
        pose.position.x = 0.0;
        pose.position.y = 0.0;
        pose.position.z = 0.0;
        pose.orientation.x = 0.0;
        pose.orientation.y = 0.0;
        pose.orientation.z = 0.0;
        pose.orientation.w = 1.0;
        return (pose);
    }
    this->m_valid_pose = false;
}

bool skSubPoseStamped::isValidPose()
{
    return (m_valid_pose);
}
