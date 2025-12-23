#ifndef _C_SUB_SCAN_2_GRID_HEADER_H_
#define _C_SUB_SCAN_2_GRID_HEADER_H_

#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

#include "sk_robot_lib/skSubLaserScan.h"

using namespace std;
using namespace std::chrono_literals;

class CScan2Grid : public rclcpp::Node
{
private:
    // Sub scan
    skSubLaserScan* p_sub;

    // Pub grid
    nav_msgs::msg::OccupancyGrid m_grid;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr p_pub;

    rclcpp::TimerBase::SharedPtr p_timer;

public:
    CScan2Grid(): Node("scan2grid")
    {
        // build subsribers
        this->p_sub = new skSubLaserScan();
        this->p_sub->initialize(this, "scan");
        //this->p_sub->activateAvgFilter(10);

        // Pub
        this->p_pub = create_publisher<nav_msgs::msg::OccupancyGrid>("grid", 2);
        this->m_grid.header.frame_id = "map";
        this->m_grid.info.resolution = 0.02;
        this->m_grid.info.width = 300;
        this->m_grid.info.height = 200;
        this->m_grid.info.origin.position.x = 0.02;
        this->m_grid.info.origin.position.y = -0.5*this->m_grid.info.resolution*((double)this->m_grid.info.height);
        this->m_grid.info.origin.position.z = 0.2;
        this->m_grid.info.origin.orientation.x = 0.0;
        this->m_grid.info.origin.orientation.y = 0.0;
        this->m_grid.info.origin.orientation.z = 0.0;
        this->m_grid.info.origin.orientation.w = 1.0;
        this->m_grid.data.resize(this->m_grid.info.width*this->m_grid.info.height);

        // make loop
        this->p_timer = this->create_wall_timer(std::chrono::duration<double>(0.5), std::bind(&CScan2Grid::loop, this));
    }

    ~CScan2Grid()
    {
    }

    void loop()
    {
        if( this->p_sub->haveMsg() )
        {
            for( int j = 0; j < this->m_grid.data.size(); j++ )
                this->m_grid.data[j] = 0;
            
            sPoint2D p;
            int x, y;
            for( int j = 0; j < this->p_sub->getLength(); j++ )
            {
                p = this->p_sub->getPoint(j);
                x = round((p.x-this->m_grid.info.origin.position.x)/this->m_grid.info.resolution);
                y = round((p.y-this->m_grid.info.origin.position.y)/this->m_grid.info.resolution);
                if( -1 < x && x < this->m_grid.info.width && -1 < y && y < this->m_grid.info.height )
                    this->m_grid.data[y*this->m_grid.info.width+x] = 100;
            }
            this->m_grid.header.stamp = this->get_clock()->now();
            this->p_pub->publish(this->m_grid);
        }
    }
};

#endif
