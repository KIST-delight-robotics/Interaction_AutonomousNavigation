#include "sk_robot_lib/skActionDash.h"
#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include <signal.h>
//#include <boost/algorithm/string.hpp>

using namespace std;
using namespace std::chrono_literals;

skActionDash::skActionDash()
{}

skActionDash::~skActionDash()
{}

bool skActionDash::activate(skRobot* robot, const sActionData& data)
{
    this->p_robot = robot;
    this->m_state = SK_ACTION_STATE_DONE;

    if( !this->p_robot )
        return (false);

    if( !this->p_robot->p_imu )
        return (false);

    this->m_linear_speed = data.linear_velocity;
    this->m_duration = data.duration_time;
    this->m_heading_ratio = data.ratio;
    this->m_heading_margin = data.margin;

    this->m_dash_time_finish = toSec(this->p_robot->get_clock()->now()) + this->m_duration;
    this->m_state = SK_ACTION_STATE_INIT;

    return (true);
}

bool skActionDash::update()
{
    if( !this->p_robot )
        return (false);

    // check run time
    if( this->m_duration > 0.0 )
    {
        if( toSec(this->p_robot->get_clock()->now()) > this->m_dash_time_finish )
        {
            this->m_state = SK_ACTION_STATE_DONE;
            return (true);
        }
    }

    // Check to start or finish dashmode
    if( this->m_state == SK_ACTION_STATE_INIT )
    {
        if( this->p_robot->p_imu->haveMsg() )
        {
            //this->p_robot->p_solver->setAngularFirstMode(this->m_max_motor_vel);
            //this->m_dash_time_start = toSec(now());
            this->m_dash_time_finish = toSec(this->p_robot->get_clock()->now()) + this->m_duration;
            this->m_heading_desire = this->p_robot->getYaw(true);
            this->m_state = SK_ACTION_STATE_RUNNING;
        }
    }

    // update command based on mode
    else if( this->m_state == SK_ACTION_STATE_RUNNING )
    {
        this->p_robot->m_cmd_d.linear.x = this->m_linear_speed;

        double yaw = this->p_robot->p_imu->getYaw();
        double error = this->m_heading_desire - yaw;
        trim(error);
        if( fabs(error) < this->m_heading_margin )
        {
            this->p_robot->m_cmd_d.angular.z = 0.0;
        }
        else
        {
            //RCLCPP_INFO(this->get_logger(), "[DEBUG][Robot][DASH mode] error = %.2f, m_gain_heading_ratio = %.1f, m_max_angular_vel = %.2f", error*RAD2DEG, this->m_gain_heading_ratio, this->m_max_angular_vel*RAD2DEG);
            this->p_robot->m_cmd_d.angular.z = getCmdFromError(error, this->m_heading_ratio*this->p_robot->m_max_angular_vel, this->p_robot->m_acc, this->p_robot->m_control_dt, 4.0);            
        }
        //RCLCPP_INFO(this->get_logger(), "[DEBUG][Robot][DASH mode] v = %.2f, w = %.2f, yaw = (%.3f / %.3f).", this->m_cmd_d.linear.x, this->m_cmd_d.angular.z*RAD2DEG, yaw*RAD2DEG, this->m_heading_d*RAD2DEG);

        return (true);
    }
    // End dash mode
    //this->p_robot->p_solver->releaseAngularFirstMode();
    //this->p_robot->m_cmd_d.linear.x = 0.0;
    //this->m_state = SK_MOBILE_ROBOT_IDLE;


    return (false);
}
