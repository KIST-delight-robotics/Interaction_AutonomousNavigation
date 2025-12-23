#ifndef _SK_ROBOT_LIB_ACTION_DRIVE_GPS_H_
#define _SK_ROBOT_LIB_ACTION_DRIVE_GPS_H_

#include "rclcpp/rclcpp.hpp"

#include "sk_robot_lib/skAction.h"
#include "sk_robot_lib/skRobot.h"

#define SK_EARTH_RADIUS         (6378000.0)  // Radius of Earth in meter

using namespace std;

class skActionDriveGPS : public skAction
{
protected:
    // Params
    std::vector<double> m_goal_latitude;
    std::vector<double> m_goal_longitude;
    std::vector<double> m_goal_margin;

    // Kinematic Params
    double m_linear_speed;
    double m_angular_speed;
    double m_acc;
    double m_dt;
    double m_heading_margin;

    // GPS Params
    double m_heading_offset;
    double m_gps_valid_time;

    // Track Target
    bool m_track_target;
    double m_target_x;
    double m_target_y;
    double m_target_margin;
    double m_target_threshold;

public:
    skActionDriveGPS();

    ~skActionDriveGPS();

    bool activate(skRobot* robot, const sActionData& data);

    bool update();
};

#endif