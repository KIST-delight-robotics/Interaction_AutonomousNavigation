#ifndef _SK_ROBOT_LIB_ACTION_WALL_FOLLOW_H_
#define _SK_ROBOT_LIB_ACTION_WALL_FOLLOW_H_

#include "rclcpp/rclcpp.hpp"

#include "sk_robot_lib/skAction.h"
#include "sk_robot_lib/skRobot.h"

#define SK_ACTION_WALL_FOLLOW_LEFT                      (0)
#define SK_ACTION_WALL_FOLLOW_RIGHT                     (1)
#define SK_ACTION_WALL_FOLLOW_CRUISE_TIME               (6.0)

using namespace std;

class skActionWallFollow : public skAction
{
protected:
    // Params
    double m_linear_speed;
    double m_side_dist;
    double m_front_dist;
    double m_front_margin;
    int m_side;

public:
    skActionWallFollow();

    ~skActionWallFollow();

    bool activate(skRobot* robot, const sActionData& data);

    bool update();
};

#endif