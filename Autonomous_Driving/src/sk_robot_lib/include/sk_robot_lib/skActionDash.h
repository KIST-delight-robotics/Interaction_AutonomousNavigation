#ifndef _SK_ROBOT_LIB_ACTION_DASH_H_
#define _SK_ROBOT_LIB_ACTION_DASH_H_

#include "rclcpp/rclcpp.hpp"

#include "sk_robot_lib/skAction.h"
#include "sk_robot_lib/skRobot.h"

using namespace std;

class skActionDash : public skAction
{
protected:
    // Params
    double m_linear_speed;
    double m_duration;
    double m_heading_ratio;
    double m_heading_margin;

    // Variables
    double m_heading_desire;
    //double m_dash_time_start;
    double m_dash_time_finish;

public:
    skActionDash();

    ~skActionDash();

    bool activate(skRobot* robot, const sActionData& data);

    bool update();
};

#endif