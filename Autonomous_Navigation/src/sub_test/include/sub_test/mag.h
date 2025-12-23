#ifndef _C_SUB_MAG_HEADER_H_
#define _C_SUB_MAG_HEADER_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "sk_robot_lib/skSubMag.h"

using namespace std;
using namespace std::chrono_literals;

class CSubMag : public rclcpp::Node
{
private:
    skSubMag* p_subMag;

    rclcpp::TimerBase::SharedPtr p_timer;

public:
    CSubMag(): Node("sug_mag")
    {
        // build subsribers
        this->p_subMag = new skSubMag();
        this->p_subMag->initialize(this, "mag");
        this->p_subMag->activateAvgFilter(10);

        // Read Offset
        this->p_subMag->setOffset(0.0);

        // make loop
        this->p_timer = this->create_wall_timer(std::chrono::duration<double>(0.5), std::bind(&CSubMag::loop, this));
    }

    ~CSubMag()
    {
    }

    void loop()
    {
        RCLCPP_INFO(this->get_logger(), "[SubMag] Current Heading = %.2f.", this->p_subMag->getYaw()*RAD2DEG);
    }
};

#endif
