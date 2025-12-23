#ifndef _SK_ROBOT_LIB_ROBOT_H_
#define _SK_ROBOT_LIB_ROBOT_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

//#include "std_msgs/msg/float64_multi_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"

#include "sk_robot_msgs/srv/robot_cmd.hpp"
#include "sk_robot_msgs/msg/robot_state.hpp"

#include "sk_robot_lib/skRobotCommon.h"
#include "sk_robot_lib/skRobotBasicFunctions.h"

#include "sk_robot_lib/skSubJoyMobile.h"
#include "sk_robot_lib/skSubTwist.h"
#include "sk_robot_lib/skSubImu.h"
#include "sk_robot_lib/skSubLaserScan.h"
#include "sk_robot_lib/skSubNavSatFix.h"
#include "sk_robot_lib/skSubQuaternionStamped.h"
#include "sk_robot_lib/skSubPose2D.h"
#include "sk_robot_lib/skSubMag.h"

#include "sk_robot_lib/skServo.h"
#include "sk_robot_lib/skKinematicsMobile.h"

#include "sk_robot_lib/skDynamicObstacleAvoidance.h"
#include "sk_robot_lib/skAction.h"

#define SK_MOBILE_ROBOT_SRV_DEFAULT_NAME            ("cmd")

using namespace std;
using namespace std::chrono_literals;

class skRobot : public rclcpp::Node
{
protected:
    int m_state;

    // Sub joy
    skSubJoyMobile* p_joy;

    // Sub twist command
    skSubTwist* p_cmd;
    geometry_msgs::msg::Twist m_cmd_d;
    geometry_msgs::msg::Twist m_cmd_c;

    // Sub IMU
    skSubImu* p_imu;
    sensor_msgs::msg::Imu m_imu;
    double m_heading_time_start;
    double m_heading_drift;
    double getYaw(bool init = false);

    // Sub LaserScanner
    skSubLaserScan* p_scan;
    sensor_msgs::msg::LaserScan m_scan;

    // Sub GPS Fix
    skSubNavSatFix* p_gps;
    sensor_msgs::msg::NavSatFix m_gps;

    // Sub QuaternionStamped
    skSubQuaternionStamped* p_quatStm;
    geometry_msgs::msg::QuaternionStamped m_quatStm;

    // Sub Magnetic Fiels
    skSubMag* p_mag;
    sensor_msgs::msg::MagneticField m_mag;

    // Sub Target Pose
    skSubPose2D* p_target;

    // Control Servo
    skServo* p_servo;
    double* p_motor_pos;
    double* p_motor_vel;
    double* p_motor_cur;
    double m_max_motor_vel;

    // Read State of Motor
    double* p_current_pos;
    double* p_current_vel;
    double* p_current_cur;

    // Kinematics of mobile base
    skKinematicsMobile* p_solver;
    double m_max_linear_vel;
    double m_max_side_vel;
    double m_max_angular_vel;
    double m_linear_acc;
    double m_side_acc;
    double m_angular_acc;
    double m_acc;

    // Local Collision Avoidance
    bool m_activate_collision_avoidance;
    double m_collision_radius;
    double m_collision_margin;
    void updateDesiredTwistByScan();
    sPoint2D expetedPointFromTwist(const geometry_msgs::msg::Twist& cmd) const;
    bool ifCollide(sPoint2D p) const;
    double minDistOfScan(sPoint2D p) const;

    // Odom Publisher
    nav_msgs::msg::Odometry m_odom;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr p_pubOdom;

    std::shared_ptr<tf2_ros::TransformBroadcaster> p_tf_broadcaster;
    geometry_msgs::msg::TransformStamped m_tf;

    // Current Publisher
    //std_msgs::msg::Float64MultiArray m_cur;
    //rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr p_pubCur;

    // State Publisher
    sk_robot_msgs::msg::RobotState m_robot_state;
    rclcpp::Publisher<sk_robot_msgs::msg::RobotState>::SharedPtr p_pubRobotState;

    // periodic controller
    double m_control_dt;
    rclcpp::TimerBase::SharedPtr p_timer;

    // Dynamic Obstacle Avoidance
    skDynamicObstacleAvoidance* p_avoidance;

    // Service to Operate Robot
    rclcpp::Service<sk_robot_msgs::srv::RobotCmd>::SharedPtr p_service;

    // Actions;
    skAction* p_action;
    sActionData m_action_data;

    void updateCurrentFromDesiredTwist();

public:
    skRobot();
    ~skRobot();

    virtual void controller();
    virtual bool updateState();

    bool activateOdom(std::string name = "odom");
    bool activateCollisionAvoidance(const double& radius, const double& margin = 0.0);

    void srvCall(const std::shared_ptr<sk_robot_msgs::srv::RobotCmd::Request> request, std::shared_ptr<sk_robot_msgs::srv::RobotCmd::Response> response);
    virtual void srvAct(const std::shared_ptr<sk_robot_msgs::srv::RobotCmd::Request> request, std::shared_ptr<sk_robot_msgs::srv::RobotCmd::Response> response);

    friend class skAction;
    friend class skActionDash;
    friend class skActionWallFollow;
    friend class skActionBatteryMeasure;
    friend class skActionDriveGPS;

    friend class skDynamicWindowApproach;
};

#endif
