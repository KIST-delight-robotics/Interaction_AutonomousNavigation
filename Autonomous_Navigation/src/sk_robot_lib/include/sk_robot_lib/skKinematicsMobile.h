#ifndef _SK_ROBOT_LIB_KINEMATICS_MOBILE_H_
#define _SK_ROBOT_LIB_KINEMATICS_MOBILE_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "sk_robot_lib/skRobotCommon.h"
#include "sk_robot_lib/skRobotBasicFunctions.h"

// Mechanical parts parameteres for Mobile Robots
#define OMNI_WHEEL_RADIUS                                 (0.058*0.5)
#define ROBOTIS_WHEEL_RADIUS                              (0.066*0.5)

using namespace std;

class skKinematicsMobile
{
protected:
    int m_number_of_motor;

    geometry_msgs::msg::Twist m_twist;
    geometry_msgs::msg::Twist m_odom;
    nav_msgs::msg::Odometry m_pose;
    double m_pose_x, m_pose_y, m_pose_th;

    std::vector<double> m_motor_velocity;
    std::vector<double> m_motor_position;

    // Angular velocity first mode
    double m_max_angular_vel;
    bool m_angular_vel_first;

public:
    skKinematicsMobile();
    ~skKinematicsMobile();

    virtual bool initialize(const string name);

    virtual bool setCmd(const geometry_msgs::msg::Twist& cmd);
    double getVel(const int& j) const;
    bool getVel(double *vel) const;

    void initOdom(const double* position);
    geometry_msgs::msg::Twist getOdom(const double*position);
    virtual geometry_msgs::msg::Twist getTwist(const double* position, const bool init = false) = 0;
    void updatePose();

    virtual void setAngularFirstMode(const double& maxAngularVelocity);
    virtual void releaseAngularFirstMode();
};

#endif //_SK_ROBOT_LIB_KINEMATICS_MOBILE_H_
