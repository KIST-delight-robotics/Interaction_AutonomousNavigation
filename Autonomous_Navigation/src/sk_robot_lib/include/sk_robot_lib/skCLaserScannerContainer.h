#ifndef _SK_MOBILE_LIB_C_LASER_SCANNER_CONTAINER_H_
#define _SK_MOBILE_LIB_C_LASER_SCANNER_CONTAINER_H_

#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>

#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Pose2D.h"
#include "sk_mobile_lib/skMobileRobotCommon.h"
#include "sk_mobile_lib/skCCollisionMap.h"
#include "sk_mobile_lib/skCLaserScannerStorage.h"

using namespace std;

class CLaserScannerContainer
{
private:
    // sensor data and information
    std::vector<CLaserScannerStorage*> p_data;
    std::vector<geometry_msgs::Pose2D> m_pose;
    std::vector<double> m_dir;
    std::vector<int> m_margin_begin;
    std::vector<int> m_margin_end;
    double m_range_eps;
    int m_step;

    // map to build
    CCollisionMap* p_map;

    // For speed up
    std::vector<bool> m_cal_ready;
    std::vector< std::vector<double> > m_sin;
    std::vector< std::vector<double> > m_cos;
    std::vector< double > m_atan2;

    // To measure Front Distance
    double m_front_dist;
    bool m_measure_dist;

    bool set_cal(const int &idx);
    bool set_atan2(const int &resolution);
    double myAtan2(const double &y, const double &x);

public:
    CLaserScannerContainer(CCollisionMap* map = NULL);

    ~CLaserScannerContainer();

    void setMap(CCollisionMap* map);
    int addLaserScanner(ros::NodeHandle n, const geometry_msgs::Pose2D& pose, const double& dir, const string name = "laser_scan", const int margin_begin = 0, const int margin_end = 0);

    bool buildMap(const double& vx, const double& vy, const double& w);

    bool buildScan(sensor_msgs::LaserScan* scan);

    bool buildScanFromMap(sensor_msgs::LaserScan* scan);

    void setStep(const int &step);

    // Measure Front Distance
    void turnOnDistSensor();
    void turnOffDistSensor();
    double getDistMeasure();
};

#endif