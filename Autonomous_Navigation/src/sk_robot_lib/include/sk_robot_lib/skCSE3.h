#ifndef _SK_ROBOT_LIB_C_SE3_H_
#define _SK_ROBOT_LIB_C_SE3_H_

#include "rclcpp/rclcpp.hpp"

#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include <iostream>
#include <fstream>

#include "geometry_msgs/msg/pose.hpp"

using namespace std;

// Quaternion Operations
geometry_msgs::msg::Quaternion inverse(const geometry_msgs::msg::Quaternion q);
geometry_msgs::msg::Quaternion operator *(const geometry_msgs::msg::Quaternion r, const geometry_msgs::msg::Quaternion s);

geometry_msgs::msg::Quaternion rollToQuat(const double& a); // rotate by X
geometry_msgs::msg::Quaternion pitchToQuat(const double& a); // rotate by Y
geometry_msgs::msg::Quaternion yawToQuat(const double& a); // rotate by Z

// Point Operations
geometry_msgs::msg::Point operator *(const double a, const geometry_msgs::msg::Point p);
geometry_msgs::msg::Point operator +(const geometry_msgs::msg::Point a, const geometry_msgs::msg::Point b);

// Quaternion and Point Operations
geometry_msgs::msg::Point operator *(const geometry_msgs::msg::Quaternion q, const geometry_msgs::msg::Point p);

// Pose Operation
geometry_msgs::msg::Pose inverse(const geometry_msgs::msg::Pose T);
geometry_msgs::msg::Pose operator *(const geometry_msgs::msg::Pose T1, const geometry_msgs::msg::Pose T2);

geometry_msgs::msg::Pose rollToPose(const double& a); // rotate by X
geometry_msgs::msg::Pose pitchToPose(const double& a); // rotate by Y
geometry_msgs::msg::Pose yawToPose(const double& a); // rotate by Z

// Print Pose
void print(rclcpp::Node *p_node, geometry_msgs::msg::Pose T, std::string name);

// Save Pose as file
void savePoseAsParam(const geometry_msgs::msg::Pose T, const char* filename);

// Evaluate the pose of Target
double evaluatePose(const geometry_msgs::msg::Pose T);

/*
// Class SE(3)
class CSE3
{
private:
    geometry_msgs::msg::Quaternion q;
    geometry_msgs::msg::Point p;

public:

    CSE3();
    CSE3(geometry_msgs::msg::Quaternion iq, geometry_msgs::msg::Point ip);
    ~CSE3();

    CSE3 getInv() const;
    geometry_msgs::msg::Point getPoint() const;
    geometry_msgs::msg::Quaternion getQuaternion() const;
    geometry_msgs::msg::Pose getPose() const;

    friend CSE3 operator *(CSE3 Ta, CSE3 Tb);

    void saveAsParam(const char* filename) const;
};

CSE3 operator *(CSE3 Ta, CSE3 Tb);
*/
#endif