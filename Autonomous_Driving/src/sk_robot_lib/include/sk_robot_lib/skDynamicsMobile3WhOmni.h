#ifndef _SK_ROBOT_LIB_DYANMICS_MOBILE_OMNI_H_
#define _SK_ROBOT_LIB_DYANMICS_MOBILE_OMNI_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "sk_robot_lib/skKinematicsMobile3WhOmni.h"

using namespace std;

// Kibot Kinematic Parameters
#define KIBOT_MASS                                         (0.2)
#define KIBOT_INERTIA                                      (0.01)
#define KIBOT_KP                                           (100.0)
#define KIBOT_BASE_RADIUS                                  (0.1)
#define KIBOT_COLLIDE_DIST_MARGIN                          (0.02)
#define KIBIT_COLLIDE_MIN_ANGULAR_VELOCITY                 (1.0*DEG2RAD)

class skDynamicsMobile3WhOmni : public skKinematicsMobile3WhOmni
{
protected:
    double m_motor_torque[3];

    // dynamic parameters
    double m_m;
    double m_I;
    double m_Kp;

    // Current state(velocity) of robot
    geometry_msgs::msg::Twist m_vel;

public:
    skDynamicsMobile3WhOmni();
    skDynamicsMobile3WhOmni(const string name);
    ~skDynamicsMobile3WhOmni();

    void setGain(const double& Kp);

    void setCurVel(const geometry_msgs::msg::Twist& vel);
    void setCurVel(const double* vel);
    double ifCollide() const;

    bool setCmd(const geometry_msgs::msg::Twist& cmd);

    double getTor(const int& j) const;
    bool getTor(double *tor) const;

    geometry_msgs::msg::Twist getOdom(const double*position);
};

#endif //_SK_ROBOT_LIB_DYANMICS_MOBILE_OMNI_H_
