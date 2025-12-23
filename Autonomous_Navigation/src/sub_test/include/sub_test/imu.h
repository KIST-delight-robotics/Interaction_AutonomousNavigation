#ifndef _C_SUB_IMU_HEADER_H_
#define _C_SUB_IMU_HEADER_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "sk_robot_lib/skSubImu.h"

using namespace std;
using namespace std::chrono_literals;

class CSubImu : public rclcpp::Node
{
private:
    skSubImu* p_sub;

    rclcpp::TimerBase::SharedPtr p_timer;

public:
    CSubImu(): Node("sug_mag")
    {
        // build subsribers
        this->p_sub = new skSubImu();
        this->p_sub->initialize(this, "data");
        //this->p_sub->activateAvgFilter(10);

        // Read Offset
        //this->p_subMag->setOffset(0.0);

        // make loop
        this->p_timer = this->create_wall_timer(std::chrono::duration<double>(0.5), std::bind(&CSubImu::loop, this));
    }

    ~CSubImu()
    {
    }

    void loop()
    {
        RCLCPP_INFO(this->get_logger(), "[SubImu] Current Heading = %.2f.", this->p_sub->getYaw()*RAD2DEG);
    }
};

#endif
