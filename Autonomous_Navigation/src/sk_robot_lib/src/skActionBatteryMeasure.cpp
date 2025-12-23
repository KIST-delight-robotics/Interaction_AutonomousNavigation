#include "sk_robot_lib/skActionBatteryMeasure.h"
#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include <signal.h>
//#include <boost/algorithm/string.hpp>

using namespace std;
using namespace std::chrono_literals;

skActionBatteryMeasure::skActionBatteryMeasure()
{}

skActionBatteryMeasure::~skActionBatteryMeasure()
{}

bool skActionBatteryMeasure::activate(skRobot* robot, const sActionData& data)
{
    this->p_robot = robot;
    this->m_state = SK_ACTION_STATE_DONE;

    if( !this->p_robot )
        return (false);

    this->m_rotating_speed = data.angular_velocity;
    this->m_rotating_time = data.duration_time;
    this->m_rest_time = data.rest_time;

    this->m_rotating = false;
    this->m_time_switch = toSec(this->p_robot->get_clock()->now()) + 1.0;//this->m_rest_time;

    this->m_state = SK_ACTION_STATE_INIT;

    return (true);
}

bool skActionBatteryMeasure::update()
{
    if( !this->p_robot )
        return (false);

    // Check to start or finish dashmode
    if( this->m_state == SK_ACTION_STATE_INIT )
    {
        this->m_state = SK_ACTION_STATE_RUNNING;
    }

    // update command based on mode
    else if( this->m_state == SK_ACTION_STATE_RUNNING )
    {
        // check switching
        if( toSec(this->p_robot->get_clock()->now()) > this->m_time_switch )
        {
            if( this->m_rotating )
            {
                // Save Log
               	ofstream save;
                char filename[256];
                std::time_t ct = std::time(0);
                string st(ctime(&ct));
                st.pop_back();
              	sprintf(filename, "%s/skData/Battery/%s.txt", getenv("HOME"), st.c_str());
              	save.open(filename);
          		save << "1";
              	save.close();

                this->m_time_switch = toSec(this->p_robot->get_clock()->now()) + this->m_rest_time;
                this->m_rotating_speed *= -1.0; // change rotating direction
                this->m_rotating = false;
            }
            else
            {
                this->m_time_switch = toSec(this->p_robot->get_clock()->now()) + this->m_rotating_time;
                this->m_rotating = true;
            }
        }

        this->p_robot->m_cmd_d.linear.x = 0.0;
        if( this->m_rotating )
        {
            this->p_robot->m_cmd_d.angular.z = this->m_rotating_speed;            
        }
        else
        {
            this->p_robot->m_cmd_d.angular.z = 0.0;            
        }

        return (true);
    }

    return (false);
}
