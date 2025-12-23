#ifndef _SK_ROBOT_LIB_SUB_OCCUPANCY_GRID_H_
#define _SK_ROBOT_LIB_SUB_OCCUPANCY_GRID_H_

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubOccupancyGrid : public skSubMsg<nav_msgs::msg::OccupancyGrid>
{
private:
    double m_resolution;
    int m_width;
    int m_height;
    double m_x;
    double m_y;

    std::vector<double> m_distSQ;
    bool m_distMap;

public:
    skSubOccupancyGrid();

    ~skSubOccupancyGrid();

    double getDist(const sPoint2D p, const double& minDist = -1.0);
    bool buildDistMap(const double& maxDist);
    double getVoronoi(const double& x, const double& y, int& flag);
    void resetDistMap();

//private:
    int x2i(const double& x) const;
    int y2i(const double& y) const;
    bool isFeasible(const int& x, const int& y) const;
    bool isObstacle(const int& x, const int& y) const;
};

#endif // _SK_ROBOT_LIB_SUB_OCCUPANCY_GRID_H_
