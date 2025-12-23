#ifndef _SK_MOBILE_LIB_SUB_POSE_STAMPED_H_
#define _SK_MOBILE_LIB_SUB_POSE_STAMPED_H_

#include "ros/ros.h"
//#include <stdio.h>
#include <iostream>
#include <fstream>
//#include <vector>
//#include <ctime>
#include <stdio.h>
//#include <cmath>
#include <string.h>
#include "geometry_msgs/PoseStamped.h"

#define RAD2DEG (180.0/M_PI)
#define DEG2RAD (M_PI/180.0)

using namespace std;

class skSubPoseStamped
{
private:
    ros::Subscriber m_subPose;
    geometry_msgs::PoseStamped m_msg;
    bool m_valid_pose;

public:
    skSubPoseStamped() : m_valid_pose(false)
    {};

    ~skSubPoseStamped()
    {
    };

    void initialize(ros::NodeHandle n, const string name = "pose_est");
    void poseCallback(const geometry_msgs::PoseStamped& msg);
    geometry_msgs::Pose getPose();
    bool isValidPose();
};

#endif // _SK_MOBILE_LIB_SUB_POSE_STAMPED_H_
