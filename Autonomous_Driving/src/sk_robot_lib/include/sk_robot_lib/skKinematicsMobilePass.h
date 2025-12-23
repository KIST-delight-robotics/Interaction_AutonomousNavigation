#ifndef _SK_ROBOT_LIB_KINEMATICS_MOBILE_PASS_H_
#define _SK_ROBOT_LIB_KINEMATICS_MOBILE_PASS_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "sk_robot_lib/skKinematicsMobile.h"

using namespace std;

class skKinematicsMobilePass : public skKinematicsMobile
{
public:
    skKinematicsMobilePass();
    ~skKinematicsMobilePass();

    bool initialize();
    //bool initialize(const string name);

    bool setCmd(const geometry_msgs::msg::Twist& cmd);

    geometry_msgs::msg::Twist getTwist(const double* position, const bool init = false);
};

#endif //_SK_ROBOT_LIB_KINEMATICS_MOBILE_2WDD_H_
