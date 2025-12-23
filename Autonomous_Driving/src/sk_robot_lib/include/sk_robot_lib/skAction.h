#ifndef _SK_ROBOT_LIB_ACTION_H_
#define _SK_ROBOT_LIB_ACTION_H_

#include "rclcpp/rclcpp.hpp"

#include <iostream>
#include <fstream>

#include "sk_robot_lib/skRobotCommon.h"

#define SK_ACTION_TYPE_NONE             (0)
#define SK_ACTION_TYPE_DASH             (1)
#define SK_ACTION_TYPE_WALL_FOLLOW      (2)

#define SK_ACTION_STATE_IDLE          (0)
#define SK_ACTION_STATE_INIT          (2)
#define SK_ACTION_STATE_START         (4)
#define SK_ACTION_STATE_RUNNING       (6)
#define SK_ACTION_STATE_WAIT          (8)
#define SK_ACTION_STATE_FINISH        (10)
#define SK_ACTION_STATE_DONE          (12)

using namespace std;
using namespace std::chrono_literals;

struct sActionData
{
    double linear_velocity;
    double angular_velocity;

    double control_period;
    double sensor_valid_time;

    double duration_time;
    double rest_time;

    double ratio;
    double margin;
    
    double dist_front;
    double dist_side;
    int side;

    double goal_x;
    double goal_y;
    double goal_margin;

    std::vector<double> goals_x;
    std::vector<double> goals_y;
    std::vector<double> goals_margin;

    double heading_offset;

    bool target;
    double target_x;
    double target_y;
    double target_margin;
    double target_threshold;
};

class skRobot;

class skAction
{
protected:
    skRobot* p_robot;
    int m_state;

public:
    skAction();

    ~skAction();

    bool isRunning() const;
    bool isDone() const;
    void setDone();

    virtual bool activate(skRobot* robot, const sActionData& data);
    virtual bool update();
};

#endif