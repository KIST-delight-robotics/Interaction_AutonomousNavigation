#ifndef _SK_MOBILE_LIB_C_COLLISION_MAP_H_
#define _SK_MOBILE_LIB_C_COLLISION_MAP_H_

#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include "sensor_msgs/PointCloud.h"
#include "geometry_msgs/Point32.h"
#include "sk_mobile_lib/skMobileRobotCommon.h"

class CCollisionMap
{
private:
    // variables for range
    double m_map_margin; // min dist between objects or data
    double m_map_marginSQ; // min dist between objects or data
    double m_collision_margin; // margin for collision
    double m_range; // distance to consider
    double m_rangeSQ;
    double m_height; // height of point cloud
    double m_filter;
    double m_filterSQ;

    // Save map as
    sensor_msgs::PointCloud m_pointCloud;

    friend class CLaserScannerContainer;

public:

    CCollisionMap(const double& map_margin = 0.05, const double& collision_margin = 0.1, const double& range = 5.0, const double filter = 0.35, const double& height = 0.3 );

    ~CCollisionMap() {};

    void setMapMargin(const double& margin);
    void setCollisionMargin(const double& margin);
    void setFilter(const double& filter);
    void setRange(const double& range);
    void setHeight(const double& height);
    void setFrame(const std::string& frame);

    bool isInRegion(const double& x, const double& y) const;
    void clear();
    void reserve(const int& n);
    int size();

    bool add_obs(const double& x, const double& y);
    double dist2obs(const double& x, const double& y) const;
    double dist2obs(const double& x, const double& y, const double& th, const double& margin_x, const double& margin_y) const;
    double dist2obsOnFront(const double& x, const double& y, const double& th, const double& margin_x, const double& margin_y) const;
    double dist2obsOnBack(const double& x, const double& y, const double& th, const double& margin_x, const double& margin_y) const;
    sensor_msgs::PointCloud& getPC();

private:
    double dist2PointSQ(const int& j, const double& x, const double& y) const;
    double dist2PointSQ(const int& j, const double& x, const double& y, const double& th, const double& margin_x, const double& margin_y) const;
};

#endif