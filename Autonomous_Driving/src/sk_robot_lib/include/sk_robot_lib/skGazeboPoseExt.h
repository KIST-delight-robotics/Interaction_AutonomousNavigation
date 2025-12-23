#ifndef _SK_MOBILE_LIB_GAZEBO_POSE_EXT_H_
#define _SK_MOBILE_LIB_GAZEBO_POSE_EXT_H_

#include "ros/ros.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <random>
#include <string.h>

#include "gazebo_msgs/ModelStates.h"
#include "geometry_msgs/Pose2D.h"

#define RAD2DEG (180.0/M_PI)
#define DEG2RAD (M_PI/180.0)

using namespace std;

class skGazeboPoseExt
{
private:
    ros::Subscriber m_subState;
    geometry_msgs::Pose2D m_pose;
    string m_robot_name;
    bool m_valid_pose;

    // Add Gaussian Noise
    bool m_add_noise;
    std::default_random_engine m_generator;
    std::normal_distribution<double> m_dist;

public:
    skGazeboPoseExt() : m_valid_pose(false), m_add_noise(false)
    {};

    ~skGazeboPoseExt()
    {
    };

    void initialize(ros::NodeHandle n, const string name, const string robot);
    void addNoise(const double& dev);
    void stateCallback(const gazebo_msgs::ModelStates& msg);
    geometry_msgs::Pose2D getPose();
    bool isValidPose();
};

#endif // _SK_MOBILE_LIB_SUB_POSE2D_H_
