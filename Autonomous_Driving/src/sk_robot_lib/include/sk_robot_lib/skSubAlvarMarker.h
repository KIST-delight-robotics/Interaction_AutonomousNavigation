#ifndef _SK_MOBILE_LIB_SUB_ALVAR_MARKER_H_
#define _SK_MOBILE_LIB_SUB_ALVAR_MARKER_H_

#include "ros/ros.h"
//#include <stdio.h>
#include <iostream>
#include <fstream>
//#include <vector>
//#include <ctime>
#include <stdio.h>
//#include <cmath>
#include <string.h>
#include "ar_track_alvar_msgs/AlvarMarkers.h"

#define RAD2DEG (180.0/M_PI)
#define DEG2RAD (M_PI/180.0)

using namespace std;

class skSubAlvarMarker
{
private:
    ros::Subscriber m_subPose;
    geometry_msgs::Pose m_pose;
    bool m_valid_pose;
    int m_id;
    unsigned int m_count;

public:
    skSubAlvarMarker() : m_valid_pose(false)
    {};

    ~skSubAlvarMarker()
    {
    };

    void initialize(ros::NodeHandle n, const int id = 0, const string name = "pose_est");
    void poseCallback(const ar_track_alvar_msgs::AlvarMarkers& msg);
    geometry_msgs::Pose getPose();
    bool isValidPose();
    unsigned int getCount() const;
};

#endif // _SK_MOBILE_LIB_SUB_ALVAR_MARKER_H_
