#ifndef _SK_ROBOT_LIB_C_DW_CONTROLLER_H_
#define _SK_ROBOT_LIB_C_DW_CONTROLLER_H_

#include "rclcpp/rclcpp.hpp"

#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>

//#include "sensor_msgs/LaserScan.h"
//#include "sensor_msgs/PointCloud.h"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "sk_robot_lib/skRobotCommon.h"
#include "sk_robot_lib/skRobotBasicFunctions.h"

#define DWCON_GEOMETRY_NONE                 (0)
#define DWCON_GEOMETRY_CIRCLE               (1)
#define DWCON_GEOMETRY_BOX                  (2)

#define DWCON_KINEMATICS_NONE               (0)
#define DWCON_KINEMATICS_2WDD               (2)
#define DWCON_KINEMATICS_OMNI               (3)

class CDWController
{
private:
    double m_margin;
    double m_margin_x, m_margin_y;
    int m_geomtry;
    double m_r;
    double m_x, m_y;
    int m_kinematics;
    double m_linear_speed, m_linear_acc, m_angular_speed, m_angular_acc, m_dt;
    double m_clsq;
    int m_N;
    //double m_c_length, m_c_lengthSQ;

public:
    CDWController();

    ~CDWController ();

    void setMargin(const double &m1, const double &m2 = -1.0);
    bool setAsCircle(const double &r);
    bool setAsBox(const double &x, const double &y);
    void setAs2WDD();
    void setAsOMNI();
    void setLinearParams(const double& speed, const double& acc);
    void setAngularParams(const double& speed, const double& acc);
    void setControlPeriod(const double& dt);
    void setResolution(const int& N);

    geometry_msgs::msg::Twist updateCmd(geometry_msgs::msg::Twist& cmd_c, geometry_msgs::msg::Twist& cmd_d, sensor_msgs::msg::LaserScan* scan = NULL, double* p_dist = NULL);
    geometry_msgs::msg::Twist projectCmd(geometry_msgs::msg::Twist& cmd_c, geometry_msgs::msg::Twist& cmd_d);

private:

    double costVel(geometry_msgs::msg::Twist& t, const double& x, const double& y, const double& w);
    geometry_msgs::msg::Pose2D expPose(const double& vx, const double& vy, const double& w);

    // To delete
    bool pointOnLeft(const double& x, const double &y, const double &margin);
    bool pointOnRight(const double& x, const double &y, const double &margin);
    bool pointOnFront(const double& x, const double &y, const double &margin, const int& dir);
    bool pointOnBack(const double& x, const double &y, const double &margin, const int& dir);
    sLine getLine(const std::vector<double>& x, const std::vector<double>& y);
    double dist2Line(const sLine& line);
    double feedbackLinearVel(const double& e);
    double feedbackAngularVel(const double& e);
};

#endif // _RMI_MB_KINEMATICS_H_
