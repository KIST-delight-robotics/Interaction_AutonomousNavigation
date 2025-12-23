#include "ros/ros.h"
#include "sk_mobile_lib/skGazeboPoseExt.h"
#include "stdio.h"

void skGazeboPoseExt::initialize(ros::NodeHandle n, const string name, const string robot)
{
    m_robot_name = robot;
    m_subState = n.subscribe(name, 2, &skGazeboPoseExt::stateCallback, this);
}

void skGazeboPoseExt::addNoise(const double& dev)
{
    m_add_noise = true;
    m_dist = std::normal_distribution<double>(0.0,dev);
}

void skGazeboPoseExt::stateCallback(const gazebo_msgs::ModelStates& msg)
{
    int idx = -1;
    for( int j = 0; j < msg.name.size(); j++ )
        if( msg.name[j].compare(m_robot_name) == 0 )
            idx = j;
    if( idx > -1 )
    {
        this->m_pose.x = msg.pose[idx].position.x;
        this->m_pose.y = msg.pose[idx].position.y;
        double siny_cosp = 2.0 * (msg.pose[idx].orientation.w * msg.pose[idx].orientation.z + msg.pose[idx].orientation.x * msg.pose[idx].orientation.y);
	    double cosy_cosp = 1.0 - 2.0 * (msg.pose[idx].orientation.y * msg.pose[idx].orientation.y + msg.pose[idx].orientation.z * msg.pose[idx].orientation.z);  
    	this->m_pose.theta = atan2(siny_cosp, cosy_cosp);
        this->m_valid_pose = true;
        if( m_add_noise )
        {
            this->m_pose.x += m_dist(m_generator);
            this->m_pose.y += m_dist(m_generator);
            this->m_pose.theta += 1.0*DEG2RAD*m_dist(m_generator);
        }
        //ROS_INFO("Get pose of [%.2f, %.2f, %.2f].", g_pose_x, g_pose_y, g_pose_th*RAD2DEG);
    }
}

geometry_msgs::Pose2D skGazeboPoseExt::getPose()
{
    if( m_valid_pose )
        return (m_pose);
    else
    {
        geometry_msgs::Pose2D pose;
        pose.x = 0.0;
        pose.y = 0.0;
        pose.theta = 0.0;
        return (pose);
    }
}

bool skGazeboPoseExt::isValidPose()
{
    return (m_valid_pose);
}
