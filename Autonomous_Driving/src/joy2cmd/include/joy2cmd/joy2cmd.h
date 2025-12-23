#ifndef _C_JOY_2_CMD_HEADER_H_
#define _C_JOY_2_CMD_HEADER_H_

#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"

#include "sk_robot_lib/skSubJoyMobile.h"
#include "sk_robot_lib/skSubTwist.h"
#include "sk_robot_lib/skSubLaserScan.h"
#include "sk_robot_lib/skSubOccupancyGrid.h"
#include "sk_robot_lib/skDynamicObstacleAvoidance.h"
#include "sk_robot_lib/skDynamicWindowApproach.h"
#include "sk_robot_lib/skSubOccupancyGrid.h"

#define SK_DEFAULT_JOY_MSG                              ("joy")

#define JOY2CMD_PUBLISH_TF_4_DEBUGGING                  (1)

using namespace std;
using namespace std::chrono_literals;

class CJoy2Cmd : public rclcpp::Node
{
private:
    skSubJoyMobile* p_subJoy;
    skSubTwist* p_subCmd;

    double m_max_linear_vel;
    double m_max_side_vel;
    double m_max_angular_vel;
    double m_acc;
    double m_dt;

    geometry_msgs::msg::Twist m_cmd_d;
    geometry_msgs::msg::Twist m_cmd;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr p_pubCmd;

    // Collision avoidance
    skSubLaserScan* p_subScan;
    skSubOccupancyGrid* p_subGrid;
    skDynamicObstacleAvoidance* p_avoidance;

    rclcpp::TimerBase::SharedPtr p_timer;

#if JOY2CMD_PUBLISH_TF_4_DEBUGGING
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_d;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_c;
    geometry_msgs::msg::TransformStamped m_t_d;
    geometry_msgs::msg::TransformStamped m_t_c;
#endif

public:
    CJoy2Cmd(): Node("joy2cmd")
    {
        // sub joypad input
        std::string joy_msg(this->declare_parameter("joy_msg", ""));
        if( joy_msg.length() > 0 )
        {
            this->p_subJoy = new skSubJoyMobile();
            this->p_subJoy->initialize(this, "joy_msg", SK_DEFAULT_JOY_MSG);
        }

        // sub twist cmd input
        std::string cmd_msg(this->declare_parameter("cmd_msg", ""));
        if( cmd_msg.length() > 0 )
        {
            this->p_subCmd = new skSubTwist();
            this->p_subCmd->initialize(this, cmd_msg);
        }
        else
            this->p_subCmd = NULL;

        // read velocity limits
        this->m_max_linear_vel = this->declare_parameter("max_linear_vel", 1.0);
        this->m_max_side_vel = this->declare_parameter("max_side_vel", 0.0);
        this->m_max_angular_vel = this->declare_parameter("max_angular_vel", 30.0)*DEG2RAD;
        this->m_acc = this->declare_parameter("acc", 1.0);
        this->m_dt = this->declare_parameter("dt", 0.05);
        
        // initialize cmd
        std::string scan_msg(this->declare_parameter("scan_msg", ""));
        if( scan_msg.length() > 0 )
        {
            this->p_subScan = new skSubLaserScan();
            this->p_subScan->initialize(this, scan_msg);
            RCLCPP_INFO(this->get_logger(), "[DEBUG][Joy2Cmd] Subscribe scan of %s.", scan_msg.c_str());
        }
        else
            this->p_subScan = NULL;
        std::string grid_msg(this->declare_parameter("grid_msg", ""));
        if( grid_msg.length() > 0 )
        {
            this->p_subGrid = new skSubOccupancyGrid();
            this->p_subGrid->initialize(this, grid_msg);
            RCLCPP_INFO(this->get_logger(), "[DEBUG][Joy2Cmd] Subscribe grid of %s.", grid_msg.c_str());
        }
        else
            this->p_subGrid = NULL;

        // Collision Avoidance
        this->p_avoidance = NULL;
#if 1
        if( this->p_subScan || this->p_subGrid )
        {
            this->p_avoidance = new skDynamicWindowApproach();
            skDynamicObstacleAvoidanceParam param;
            param.radius = this->declare_parameter("collision_radius", 0.3);
            param.margin = this->declare_parameter("collision_marign", 0.1);

            param.max_vel_x = this->m_max_linear_vel;
            param.max_vel_y = this->m_max_side_vel;
            param.max_vel_th = this->m_max_angular_vel;

            param.acc = this->m_acc;
            param.dt = this->m_dt;

            param.resolution_x = this->declare_parameter("resolution_x", 4);
            param.resolution_y = this->declare_parameter("resolution_y", 0);
            param.resolution_th = this->declare_parameter("resolution_th", 4);

            this->p_avoidance->setParams(param);

            if( this->p_subScan )
                this->p_avoidance->activate(this->p_subScan);
            else
                this->p_avoidance->activate(this->p_subGrid);
            this->p_avoidance->activate(this);
            RCLCPP_INFO(this->get_logger(), "[DEBUG][Robot] Activate Collision Avoidance Algorithm with radius = %.3f, margin = %.3f.", param.radius, param.margin);
        }
#endif

        // Pub cmd vel
        this->p_pubCmd = create_publisher<geometry_msgs::msg::Twist>("cmd", 10);

#if JOY2CMD_PUBLISH_TF_4_DEBUGGING
    this-> tf_broadcaster_d = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    this->tf_broadcaster_c = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    this->m_t_d.header.frame_id = "map";
    this->m_t_d.child_frame_id = "desired";
    this->m_t_d.transform.translation.x = 0.0;
    this->m_t_d.transform.translation.y = 0.0;
    this->m_t_d.transform.translation.z = 0.0;
    this->m_t_d.transform.rotation.x = 0.0;
    this->m_t_d.transform.rotation.y = 0.0;
    this->m_t_d.transform.rotation.z = 0.0;
    this->m_t_d.transform.rotation.w = 1.0;
    this->m_t_c.header.frame_id = "map";
    this->m_t_c.child_frame_id = "current";
    this->m_t_c.transform.translation.x = 0.0;
    this->m_t_c.transform.translation.y = 0.0;
    this->m_t_c.transform.translation.z = 0.0;
    this->m_t_c.transform.rotation.x = 0.0;
    this->m_t_c.transform.rotation.y = 0.0;
    this->m_t_c.transform.rotation.z = 0.0;
    this->m_t_c.transform.rotation.w = 1.0;
#endif

    // make loop
        this->p_timer = this->create_wall_timer(std::chrono::duration<double>(this->m_dt), std::bind(&CJoy2Cmd::loop, this));
    }

    ~CJoy2Cmd()
    {
    }

    void loop()
    {
        //RCLCPP_INFO(this->get_logger(), "[DEBUG][Joy2Cmd] loop() is running.");
        // Read desired cmd
        bool followCmd(true);
        if( this->p_subJoy )
        {
            if( this->p_subJoy->getState() == SK_MOBILE_ROBOT_MANUAL ) // if joy is valid
            {
                this->m_cmd_d = this->p_subJoy->getTwist();
                this->m_cmd_d.linear.x *= this->m_max_linear_vel;
                this->m_cmd_d.linear.y *= this->m_max_side_vel;
                this->m_cmd_d.angular.z *= this->m_max_angular_vel;
                followCmd = false;
            }
        }
        if( followCmd && this->p_subCmd )
        {
            this->m_cmd_d = this->p_subCmd->getMsg();
        }
        //RCLCPP_INFO(this->get_logger(), "[DEBUG][Joy2Cmd] twist_d = [%.2f, %.2f, %.1f].", this->m_cmd_d.linear.x, this->m_cmd_d.linear.y, this->m_cmd_d.angular.z*RAD2DEG);
# if 1 // for debugging
        this->m_cmd_d.linear.x = this->m_max_linear_vel;
        this->m_cmd_d.linear.y = 0.0;
        //this->m_cmd_d.angular.z = this->m_max_angular_vel*0.1;
#endif
        // Check collision
        if( this->p_avoidance )
        {
            //RCLCPP_INFO(this->get_logger(), "[DEBUG][Joy2Cmd] Start DWA.");
            this->m_cmd = this->p_avoidance->getTwist(this->m_cmd_d, this->m_cmd);
            //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] After  twist_d = [%.2f, %.2f, %.1f].", this->m_cmd_d.linear.x, this->m_cmd_d.linear.y, this->m_cmd_d.angular.z*RAD2DEG);
#if JOY2CMD_PUBLISH_TF_4_DEBUGGING
            //double dx(dist2stop(this->m_cmd_d.linear.x, this->m_max_linear_vel, 1.0, 0.05));
            //double dy(dist2stop(this->m_cmd_d.linear.y, this->m_max_side_vel, 1.0, 0.05));
            //double th(dist2stop(this->m_cmd_d.angular.z, this->m_max_angular_vel, 1.0, 0.05));
            const double t(2.0);
            double dx(this->m_cmd_d.linear.x*t);
            double dy(this->m_cmd_d.linear.y*t);
            double th(this->m_cmd_d.angular.z*t);
            this->m_t_d.transform.translation.x = dx*cos(th) - dy*sin(th);
            this->m_t_d.transform.translation.y = dx*sin(th) + dy*cos(th);
            this->m_t_d.transform.rotation.x = 0.0;
            this->m_t_d.transform.rotation.y = 0.0;
            this->m_t_d.transform.rotation.z = sin(0.5*th);
            this->m_t_d.transform.rotation.w = cos(0.5*th);
            this->m_t_d.header.stamp = this->get_clock()->now();
            this->tf_broadcaster_d->sendTransform(this->m_t_d);
            RCLCPP_INFO(this->get_logger(), "[DEBUG][joy2cmd] pose_d = [%.2f, %.2f, %.1f].", dx, dy, th*RAD2DEG);
            //dx = dist2stop(this->m_cmd.linear.x, this->m_max_linear_vel, 1.0, 0.05);
            //dy = dist2stop(this->m_cmd.linear.y, this->m_max_side_vel, 1.0, 0.05);
            //th = dist2stop(this->m_cmd.angular.z, this->m_max_angular_vel, 1.0, 0.05);
            dx = (this->m_cmd.linear.x*t);
            dy = (this->m_cmd.linear.y*t);
            th = (this->m_cmd.angular.z*t);
            this->m_t_c.transform.translation.x = dx*cos(th) - dy*sin(th);
            this->m_t_c.transform.translation.y = dx*sin(th) + dy*cos(th);
            this->m_t_c.transform.rotation.x = 0.0;
            this->m_t_c.transform.rotation.y = 0.0;
            this->m_t_c.transform.rotation.z = sin(0.5*th);
            this->m_t_c.transform.rotation.w = cos(0.5*th);
            this->m_t_c.header.stamp = this->get_clock()->now();
            this->tf_broadcaster_c->sendTransform(this->m_t_c);
            RCLCPP_INFO(this->get_logger(), "[DEBUG][joy2cmd] pose_c = [%.2f, %.2f, %.1f].", dx, dy, th*RAD2DEG);
#endif
        }
        else
        {
            this->m_cmd = this->m_cmd_d;
        }

        // publish Data
        this->p_pubCmd->publish(this->m_cmd);
    }
};

#endif
