#include "sk_robot_lib/skActionDriveGPS.h"
#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include <signal.h>
//#include <boost/algorithm/string.hpp>

using namespace std;
using namespace std::chrono_literals;

skActionDriveGPS::skActionDriveGPS() : m_acc(1.0), m_dt(1.0), m_heading_margin(45.0*DEG2RAD)
{}

skActionDriveGPS::~skActionDriveGPS()
{}

bool skActionDriveGPS::activate(skRobot* robot, const sActionData& data)
{
    this->p_robot = robot;
    this->m_state = SK_ACTION_STATE_DONE;

    // Check if have necessary sensors
    if( !this->p_robot )
        return (false);

    if( !this->p_robot->p_gps )
        return (false);

    if( !this->p_robot->p_mag && !this->p_robot->p_imu )
        return (false);

    // Set Kinematic Params
    this->m_linear_speed = data.linear_velocity;
    this->m_angular_speed = data.angular_velocity;
    this->m_gps_valid_time = data.sensor_valid_time;

    this->m_heading_offset = data.heading_offset;

    // Set Goals
    this->m_goal_latitude.clear();
    this->m_goal_longitude.clear();
    this->m_goal_margin.clear();
    if( data.goals_x.size() > 0 && data.goals_y.size() > 0 )
    {
        if( data.goals_x.size() != data.goals_y.size() )
            return (false);
        if( data.goals_margin.size() < 1 )
            return (false);

        const unsigned int n(data.goals_x.size());

        this->m_goal_latitude.resize(n);
        this->m_goal_longitude.resize(n);
        this->m_goal_margin.resize(n);

        for( unsigned int j = 0; j < n; j++ )
        {
            this->m_goal_latitude[n-1-j] = data.goals_x[j];
            this->m_goal_longitude[n-1-j] = data.goals_y[j];
            if( data.goals_margin.size() == n )
            {
                this->m_goal_margin[n-1-j] = data.goals_margin[j];
            }
            else
            {
                this->m_goal_margin[n-1-j] = data.goals_margin[0];
            }
        }
        if( data.goals_margin.size() == 2)
        {
            this->m_goal_margin[0] = data.goals_margin[1];
        }
    }
    else
    {
        this->m_goal_latitude.resize(1);
        this->m_goal_longitude.resize(1);
        this->m_goal_margin.resize(1);

        this->m_goal_latitude[0] = data.goal_x;
        this->m_goal_longitude[0] = data.goal_y;
        this->m_goal_margin[0] = data.goal_margin;
    }

    // Set Track Target
    if( data.target && this->p_robot->p_target )
    {
        this->m_target_x = data.target_x;
        this->m_target_y = data.target_y;
        this->m_target_margin = data.target_margin;
        this->m_target_threshold = data.target_threshold;
        this->m_track_target = true;
    }
    else
    {
        this->m_track_target = false;
    }

    // Done
    this->m_state = SK_ACTION_STATE_INIT;

    return (true);
}

bool skActionDriveGPS::update()
{
    if( !this->p_robot )
        return (false);

    if( !this->p_robot->p_gps )
        return (false);

    if( !this->p_robot->p_mag && !this->p_robot->p_imu )
        return (false);

    this->p_robot->m_cmd_d.linear.x = 0.0;
    this->p_robot->m_cmd_d.angular.z = 0.0;

    if( this->m_state == SK_ACTION_STATE_INIT || this->m_state == SK_ACTION_STATE_RUNNING || this->m_state == SK_ACTION_STATE_WAIT )
    {
        this->m_state = SK_ACTION_STATE_WAIT;

        double dx(0.0), dy(0.0);
        double v_gps(0.0), w_gps(0.0), ratio_gps(1.0);
        double v_t(0.0), w_t(0.0), ratio_t(0.0);

        // Check if GPS Signal is Valid
        bool validGPS(this->p_robot->p_gps->haveMsg());
        if( validGPS )
            validGPS &= this->p_robot->p_gps->valid();
        if( validGPS )
            validGPS &= toSec(this->p_robot->now()) < toSec(this->p_robot->p_gps->getMsg().header.stamp) + this->m_gps_valid_time;
        //if( validGPS )
        //    validGPS &= this->p_robot->p_quatStm->haveMsg();
        if( validGPS && this->p_robot->p_imu )
            validGPS &= this->p_robot->p_imu->haveMsg();
        //if( validGPS )
        //    validGPS &= toSec(this->p_robot->now()) < toSec(this->p_robot->p_quatStm->getMsg().header.stamp) + this->m_gps_valid_time;

        // Calculate control by GPS
        if( validGPS )
        {
            // Calculate error;
            dx = SK_EARTH_RADIUS*(this->m_goal_latitude.back() - this->p_robot->p_gps->getLatitude())*DEG2RAD;
            dy = -1.0*SK_EARTH_RADIUS*(this->m_goal_longitude.back() - this->p_robot->p_gps->getLongitude())*DEG2RAD;

            // Check if reach goal
            if( dx*dx+dy*dy < this->m_goal_margin.back()*this->m_goal_margin.back() )
            {
                if( this->m_goal_latitude.size() == 1 )
                {
                    this->m_state = SK_ACTION_STATE_DONE;

                    return (true);
                }
                if( this->m_goal_margin.size() == this->m_goal_latitude.size() )
                {
                    this->m_goal_margin.pop_back();
                }
                this->m_goal_latitude.pop_back();
                this->m_goal_longitude.pop_back();
            }

            // Calculate proper heading and angular velocity
            const double heading_d(atan2(dy,dx));
            //const double heading_c(this->p_robot->p_quatStm->getYaw()+this->m_heading_offset);
            double heading_c;
            if( this->p_robot->p_mag )
                heading_c = this->p_robot->p_mag->getYaw();
            else if( this->p_robot->p_imu )
                heading_c = this->p_robot->p_imu->getYaw();
            else
                heading_c = heading_d;

            double heading_error(heading_d - heading_c);
            trim(heading_error);
            w_gps = getCmdFromError(heading_error, this->m_angular_speed, this->m_acc, this->m_dt);

            // Forward velocity
            if( fabs(heading_error) < this->m_heading_margin )
            {
                v_gps = getCmdFromError(sqrt(dx*dx+dy*dy), this->m_linear_speed, this->m_acc, this->m_dt);
            }
            // Print Logs
            RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][ActionGPS] Error = [%.3f, %.3f].", dx, dy);
            RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][ActionGPS] m_heading_[d,c,e] = [%.1f, %.1f, %.1f].", heading_d*RAD2DEG, heading_c*RAD2DEG, heading_error*RAD2DEG);
        }
        else
        {
            ratio_gps = 0.0;
            RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][ActionGPS] Sensor Singal is not valid.");
        }

        // Calculate control to track target
        //if( this->m_track_target )
        /*if( 0 )
	{
            geometry_msgs::msg::Pose2D pose(this->p_robot->p_target->getMsg());
            double ex(pose.x - this->m_target_x);
            double ey(pose.y - this->m_target_y);

            const double e(sqrt(ex*ex+ey*ey));

            if( e > this->m_target_threshold )
            {
                // Calculate proper heading and angular velocity
                double heading_error(atan2(ey,ex));
                trim(heading_error);
                w_t = getCmdFromError(heading_error, this->m_angular_speed, this->m_acc, this->m_dt);
                
                // Forward velocity
                if( fabs(heading_error) > this->m_heading_margin )
                {
                    v_t = getCmdFromError(e, this->m_linear_speed, this->m_acc, this->m_dt) * cos(heading_error);
                }

                ratio_t = (e-this->m_target_threshold) / (this->m_target_margin - this->m_target_threshold);
            }
        }

        const double sum(ratio_gps + ratio_t);
        if( sum > 0.0 )
        {
            this->p_robot->m_cmd_d.linear.x = (ratio_gps*v_gps + ratio_t*v_t) / sum;
            this->p_robot->m_cmd_d.angular.z = (ratio_gps*w_gps + ratio_t*w_t) / sum;
        }*/
        this->p_robot->m_cmd_d.linear.x = v_gps;
        this->p_robot->m_cmd_d.angular.z = w_gps;
        // Now the signal is valid
        this->m_state = SK_ACTION_STATE_RUNNING;

        return (true);
    }

    // End driving
    //this->p_robot->p_solver->releaseAngularFirstMode();
    //this->p_robot->m_cmd_d.linear.x = 0.0;
    //this->m_state = SK_MOBILE_ROBOT_IDLE;

    return (false);
}
