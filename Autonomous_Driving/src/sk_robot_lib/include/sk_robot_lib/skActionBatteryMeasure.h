#ifndef _SK_ROBOT_LIB_ACTION_BATTERY_MEASURE_H_
#define _SK_ROBOT_LIB_ACTION_BATTERY_MEASURE_H_

#include "rclcpp/rclcpp.hpp"

#include "sk_robot_lib/skAction.h"
#include "sk_robot_lib/skRobot.h"

using namespace std;

class skActionBatteryMeasure : public skAction
{
protected:
    // Params
    double m_rotating_speed;
    double m_rotating_time;
    double m_rest_time;

    // Variables
    bool m_rotating;
    double m_time_switch;

public:
    skActionBatteryMeasure();

    ~skActionBatteryMeasure();

    bool activate(skRobot* robot, const sActionData& data);

    bool update();
};

#endif