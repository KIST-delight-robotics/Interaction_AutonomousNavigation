#include "ros/ros.h"
#include "sk_mobile_lib/skCLaserScannerStorage.h"
#include "stdio.h"

CLaserScannerStorage::CLaserScannerStorage(ros::NodeHandle n, const string name/* = "laser_scan"*/)
{
    this->initialize(n,name);
}

bool CLaserScannerStorage::initialize(ros::NodeHandle n, const string name/* = "laser_scan"*/ )
{
    m_recent_idx = -1;
    m_sub = n.subscribe(name, 2, &CLaserScannerStorage::callback, this);
    m_name = name;

    //ROS_INFO("[DEBUG|LSS] initialize() : m_recent_idx = %d.", m_recent_idx);
    return (true);
}

void CLaserScannerStorage::callback(const sensor_msgs::LaserScan& msg)
{
    //ROS_INFO("[DEBUG|LSS] callback() : [%s] I got a msg. m_recent_idx = %d.", m_name.c_str(), m_recent_idx);
    //ROS_INFO("[DEBUG|LSS] callback() : I got a msg. m_recent_idx = %d.", m_recent_idx);
    this->m_recent_idx = (this->m_recent_idx+1)%2;
    m_sensor_data[this->m_recent_idx] = msg;
}

bool CLaserScannerStorage::hasData()
{
    //ROS_INFO("[DEBUG|LSS] hasData() : [%s] m_recent_idx = %d.", m_name.c_str(), this->m_recent_idx);
    return (-1 < m_recent_idx && m_recent_idx < 2 );
}

sensor_msgs::LaserScan& CLaserScannerStorage::getData()
{
    return (m_sensor_data[m_recent_idx]);
}
