#ifndef _SK_ROBOT_LIB_KINEMATICS_MOBILE_OMNI_H_
#define _SK_ROBOT_LIB_KINEMATICS_MOBILE_OMNI_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "sk_robot_lib/skKinematicsMobile.h"

using namespace std;

// Kibot Kinematic Parameters
//#define KIBOT_WHEEL_ARM                                    (0.058+0.015)
#define KIBOT_WHEEL_ARM                                    (0.125)
//#define KIBOT_WHEEL_RADIUS                                 (0.058*0.5)
#define KIBOT_WHEEL_OFFSET                                 (60.0*DEG2RAD)
#define KIBOT_WHEEL_STEP                                   (120.0*DEG2RAD)
#define KIBOT_MAX_LINEAR                                   (0.3)
#define KIBOT_MAX_ANGULAR                                  (90.0*DEG2RAD)
#define KIBOT_DEFAULT_LINEAR                               (0.1)
#define KIBOT_DEFAULT_ANGULAR                              (30.0*DEG2RAD)
#define KIBOT_ACC                                          (0.5)
#define KIBOT_DT                                           (0.05)

class skKinematicsMobile3WhOmni : public skKinematicsMobile
{
protected:
    double m_wheel_radius;

    // for Kibot and 3wheel omnis
    double m_iJ[3][3];
    double m_J[3][3];

public:
    skKinematicsMobile3WhOmni();
    skKinematicsMobile3WhOmni(const string name);
    ~skKinematicsMobile3WhOmni();

    virtual bool initialize(const string name);

    bool setCmd(const geometry_msgs::msg::Twist& cmd);

    geometry_msgs::msg::Twist getTwist(const double* position, const bool init = false);
};

#endif //_SK_ROBOT_LIB_KINEMATICS_MOBILE_OMNI_H_
