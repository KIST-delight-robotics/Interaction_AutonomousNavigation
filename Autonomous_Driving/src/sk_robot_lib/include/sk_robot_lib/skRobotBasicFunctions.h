#ifndef _SK_MOBILE_ROBOT_BASIC_FUNCTIONS_H_
#define _SK_MOBILE_ROBOT_BASIC_FUNCTIONS_H_

#include "rclcpp/rclcpp.hpp"
#include "builtin_interfaces/msg/time.hpp"

#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/vector3.hpp"

#include "sk_robot_lib/skRobotCommon.h"

double dist2stop(const double &v, const double& maxV, const double& acc, const double& dt, const double alpha = 2.0);
double getCmdFromError(const double& error, const double& maxV, const double& acc, const double& dt, const double alpha = 2.0);

double toSec(builtin_interfaces::msg::Time stamp);
double toMSec(builtin_interfaces::msg::Time stamp);
double toMSecBtw(builtin_interfaces::msg::Time end, builtin_interfaces::msg::Time start);

void trim(double& th);
void trimPIPI(double& th);
void trim2PI0(double& th);

void projectToRange(int& x, const int& range);

double distSQ(const sPoint2D& p, const sPoint2D& q);

double roll(const geometry_msgs::msg::Quaternion msg);
double pitch(const geometry_msgs::msg::Quaternion msg);
double yaw(const geometry_msgs::msg::Quaternion msg);

geometry_msgs::msg::Quaternion quat_from_yaw(const double& yaw);

double operator*(const geometry_msgs::msg::Vector3& x, const geometry_msgs::msg::Vector3& y);
geometry_msgs::msg::Vector3 operator+(const geometry_msgs::msg::Vector3& x, const geometry_msgs::msg::Vector3& y);
geometry_msgs::msg::Vector3 operator-(const geometry_msgs::msg::Vector3& x, const geometry_msgs::msg::Vector3& y);
geometry_msgs::msg::Vector3 operator*(const double& a, const geometry_msgs::msg::Vector3& x);

struct Axb2
{
    double a;
    double b;
    double c;
    double d;
    double b0;
    double b1;
    double x0;
    double x1;

    bool solve();
};

#endif // _SK_MOBILE_ROBOT_BASIC_FUNCTIONS_H_
