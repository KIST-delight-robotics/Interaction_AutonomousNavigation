#ifndef _SK_ROBOT_LIB_SUB_JOY_MOBILE_H_
#define _SK_ROBOT_LIB_SUB_JOY_MOBILE_H_

#include "geometry_msgs/msg/twist.hpp"
#include "sk_robot_lib/skSubJoy.h"

using namespace std;

class skSubJoyMobile : public skSubJoy
{
protected:
    geometry_msgs::msg::Twist m_twist;
    unsigned int m_event;

    void zeroTwist();
    void updateJoy();
    void updateTwist();

public:
    skSubJoyMobile();

    ~skSubJoyMobile();

    geometry_msgs::msg::Twist getTwist() const;

    unsigned int getEvent();

};

#endif // SK_ROBOT_LIB_SUB_JOY_MOBILE_H_
