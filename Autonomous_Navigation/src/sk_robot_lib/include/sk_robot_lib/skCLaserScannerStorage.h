#ifndef _SK_MOBILE_LIB_C_LASER_SCANNER_STORAGE_H_
#define _SK_MOBILE_LIB_C_LASER_SCANNER_STORAGE_H_

#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>

#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Pose2D.h"
#include "sk_mobile_lib/skMobileRobotCommon.h"
//#include "sk_mobile_lib/skCCollisionMap.h"

using namespace std;

class CLaserScannerStorage
{
private:
    // variables to subscribe
    int m_recent_idx;
    ros::Subscriber m_sub; 
    sensor_msgs::LaserScan m_sensor_data[2]; // [idx 0,1]
    // For debugging
    string m_name;

public:

    CLaserScannerStorage()
    {
    }

    CLaserScannerStorage(ros::NodeHandle n, const string name = "laser_scan");

    ~CLaserScannerStorage ()
    {
    }

    bool initialize(ros::NodeHandle n, const string name = "laser_scan" );
    void callback(const sensor_msgs::LaserScan& msg);

    bool hasData();
    sensor_msgs::LaserScan& getData();
};

#endif