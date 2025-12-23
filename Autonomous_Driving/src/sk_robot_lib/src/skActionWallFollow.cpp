#include "sk_robot_lib/skActionWallFollow.h"
#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include <signal.h>
//#include <boost/algorithm/string.hpp>

using namespace std;
using namespace std::chrono_literals;

skActionWallFollow::skActionWallFollow()
{}

skActionWallFollow::~skActionWallFollow()
{}

bool skActionWallFollow::activate(skRobot* robot, const sActionData& data)
{
    this->p_robot = robot;
    this->m_state = SK_ACTION_STATE_DONE;

    if( !this->p_robot )
        return (false);

    if( !this->p_robot->p_scan )
        return (false);

    this->m_linear_speed = data.linear_velocity;
    this->m_side_dist = data.dist_side;
    this->m_front_dist = data.dist_front;
    this->m_front_margin = data.margin;
    this->m_side = data.side;

    this->m_state = SK_ACTION_STATE_INIT;

    return (true);
}

bool skActionWallFollow::update()
{
    if( !this->p_robot )
        return (false);

    if( !this->p_robot->p_scan )
        return (false);

    // Check to start or finish dashmode
    if( this->m_state == SK_ACTION_STATE_INIT )
    {
        this->m_state = SK_ACTION_STATE_RUNNING;
    }

    // update command based on mode
    else if( this->m_state == SK_ACTION_STATE_RUNNING )
    {
        this->p_robot->m_cmd_d.linear.x = 0.0;
        this->p_robot->m_cmd_d.angular.z = 0.0;
        if( this->p_robot->p_scan->haveMsg() )
        {
            this->p_robot->m_scan = this->p_robot->p_scan->getMsg();

            // linear speed
            const sLine lineF(this->p_robot->p_scan->getLineFromLaserScan(0.0));
            if( lineF.valid )
            {
                const double errorF(lineF.dist - this->m_front_dist);
                //RCLCPP_INFO(this->get_logger(), "[DEBUG][Mobile2WDD] lineFollowMode() (error, maxV, acc, dt) = (%.3f, %.3f, %.3f, %.3f).", errorF, this->m_line_follow_speed, SK_ACC, this->m_control_dt);
                if( fabs(errorF) < this->m_front_margin )
                {
                    this->m_state = SK_ACTION_STATE_DONE;
                    return (true);
                }
                this->p_robot->m_cmd_d.linear.x = getCmdFromError(errorF, this->m_linear_speed, this->p_robot->m_acc, this->p_robot->m_control_dt, 4.0);
                //RCLCPP_INFO(this->get_logger(), "[DEBUG][Mobile2WDD] lineFollowMode() ret = %.3f.", getCmdFromError(errorF, this->m_line_follow_speed, SK_ACC, this->m_control_dt, 4.0));
            }
            else
            {
                this->p_robot->m_cmd_d.linear.x = this->m_linear_speed;
            }
            RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][SKActionWallFollow] (distF/goal, v) = (%.3f, %.3f, %.3f).", lineF.dist, this->m_front_dist, this->p_robot->m_cmd_d.linear.x);

            // angular speed
            sLine lineS;
            lineS.valid = false;
            double sign(0.0);
            if( this->m_side == SK_ACTION_WALL_FOLLOW_LEFT )
            {
                lineS = this->p_robot->p_scan->getLineFromLaserScan(90.0*DEG2RAD);
                sign = -1.0;
            }
            else if( this->m_side == SK_ACTION_WALL_FOLLOW_RIGHT )
            {
                lineS = this->p_robot->p_scan->getLineFromLaserScan(270.0*DEG2RAD);
                sign = 1.0;
            }

            if( !lineS.valid )
            {
                this->p_robot->m_cmd_d.linear.x = 0.0;
                return (false);
            }
            while( lineS.angle < -0.5*M_PI )
                lineS.angle += M_PI;
            while( lineS.angle > 0.5*M_PI )
                lineS.angle -= M_PI;

            double theta_error_dist(sign*atan2(this->m_side_dist-lineS.dist, this->p_robot->m_cmd_d.linear.x*SK_ACTION_WALL_FOLLOW_CRUISE_TIME));
            //this->p_robot->m_cmd_d.angular.z = getCmdFromError(theta_error_dist-lineS.angle, this->p_robot->m_max_angular_vel, this->p_robot->m_acc, this->p_robot->m_control_dt, 4.0);
            double w_dist = getCmdFromError(theta_error_dist, this->p_robot->m_max_angular_vel, this->p_robot->m_acc, this->p_robot->m_control_dt, 4.0);
            w_dist = MIN(w_dist,this->p_robot->m_max_angular_vel);
            w_dist = MAX(w_dist,-this->p_robot->m_max_angular_vel);
            double w_angle = getCmdFromError(lineS.angle, this->p_robot->m_max_angular_vel, this->p_robot->m_acc, this->p_robot->m_control_dt, 4.0);
            w_angle = MIN(w_angle,this->p_robot->m_max_angular_vel);
            w_angle = MAX(w_angle,-this->p_robot->m_max_angular_vel);
            const double ratio(MIN(1.0,fabs(this->m_side_dist-lineS.dist)/this->m_front_margin));
            this->p_robot->m_cmd_d.angular.z = ratio*w_dist + (1.0-ratio)*w_angle;
            RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][SKActionWallFollow] lineS.(dist/goal,angle) = (%.3f/%.3f, %.2f).", lineS.dist, this->m_side_dist, lineS.angle*RAD2DEG);
            RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][SKActionWallFollow] theta_error_dist = %.2f.", theta_error_dist*RAD2DEG);
            RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][SKActionWallFollow] w = %.2f.", this->p_robot->m_cmd_d.angular.z*RAD2DEG);

            return (true);
        }
    }
    // End dash mode
    //this->p_robot->p_solver->releaseAngularFirstMode();
    //this->p_robot->m_cmd_d.linear.x = 0.0;
    //this->m_state = SK_MOBILE_ROBOT_IDLE;


    return (false);
}
