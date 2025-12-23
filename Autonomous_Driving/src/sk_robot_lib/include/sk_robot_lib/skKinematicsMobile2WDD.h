#ifndef _SK_ROBOT_LIB_KINEMATICS_MOBILE_2WDD_H_
#define _SK_ROBOT_LIB_KINEMATICS_MOBILE_2WDD_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "sk_robot_lib/skKinematicsMobile.h"

using namespace std;

// Metabot Kinematic Parameters
#define METABOT_WHEEL_DIST                                   (0.140)
#define METABOT_WHEEL_RADIUS                                 (0.066*0.5)
#define METABOT_MAX_LINEAR                                   (0.3)
#define METABOT_MAX_ANGULAR                                  (90.0*DEG2RAD)
#define METABOT_DEFAULT_LINEAR                               (0.1)
#define METABOT_DEFAULT_ANGULAR                              (30.0*DEG2RAD)
#define METABOT_ACC                                          (1.0)
#define METABOT_DT                                           (0.05)

// FishBot Kinematic Parameters
#define FISHBOT_WHEEL_DIST                                   (0.104*0.5)
//#define FISHBOT_WHEEL_RADIUS                                 (0.058*0.5)

// Kibot2WDD Kinematic Parameters
#define KIBOT_WHEEL_DIST                                    (0.058+0.015)
//#define KIBOT_WHEEL_RADIUS                                  (0.058*0.5)

// VacuumBot Kinematic Parameters
#define VACUUMBOT_WHEEL_DIST                                    (0.058+0.015)

class skKinematicsMobile2WDD : public skKinematicsMobile
{
protected:
    double m_wheel_radius;
    double m_wheel_dist;
    double m_wheel_dir[2];
    double m_wheel_side[2];
    double m_J[2][2];
    double m_iJ[2][2];

public:
    skKinematicsMobile2WDD();
    skKinematicsMobile2WDD(const string name);
    skKinematicsMobile2WDD(const double wheel_radius, const double wheel_dist, const int direction = 0);
    ~skKinematicsMobile2WDD();

    bool initialize(const double wheel_radius, const double wheel_dist, const int direction);
    //bool initialize(const string name);

    bool setCmd(const geometry_msgs::msg::Twist& cmd);

    geometry_msgs::msg::Twist getTwist(const double* position, const bool init = false);

    void setAngularFirstMode(const double& maxAngularVelocity);
    void releaseAngularFirstMode();
};

#endif //_SK_ROBOT_LIB_KINEMATICS_MOBILE_2WDD_H_
