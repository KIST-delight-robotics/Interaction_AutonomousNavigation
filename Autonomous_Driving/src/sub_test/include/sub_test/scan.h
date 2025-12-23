#ifndef _C_SUB_SCAN_HEADER_H_
#define _C_SUB_SCAN_HEADER_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "sk_robot_lib/skSubLaserScan.h"

using namespace std;
using namespace std::chrono_literals;

class CSubScan : public rclcpp::Node
{
private:
    skSubLaserScan* p_sub;

    rclcpp::TimerBase::SharedPtr p_timer;

public:
    CSubScan(): Node("sub_scan")
    {
        // build subsribers
        this->p_sub = new skSubLaserScan();
        this->p_sub->initialize(this, "scan");
        //this->p_sub->activateAvgFilter(10);

        // Read Offset
        //this->p_subMag->setOffset(0.0);

        // make loop
        this->p_timer = this->create_wall_timer(std::chrono::duration<double>(0.5), std::bind(&CSubScan::loop, this));
    }

    ~CSubScan()
    {
    }

    void loop()
    {
        if( this->p_sub->haveMsg() )
        {
            float range(100.0), th(-M_PI);
            sPoint2D p;
            int opt(-1);
            for( int j = 0; j < this->p_sub->getLength(); j++ )
            {
                if( range > this->p_sub->getRange(j) && this->p_sub->getRange(j) > 0.0 )
                {
                    range = this->p_sub->getRange(j);
                    th = this->p_sub->getAngle(j);
                    p = this->p_sub->getPoint(j);
                    opt = j;
                    //RCLCPP_INFO(this->get_logger(), "[SubScan] %.3f <  %.3f < %.3f].", scan.range_min, scan.ranges[j], scan.range_max);
                }
            }
            RCLCPP_INFO(this->get_logger(), "[SubScan] Min Range = %.2f at %d, %.1f, p = [%.3f, %.3f].", range, opt, th*RAD2DEG, p.x, p.y);
        }
    }
};

#endif
