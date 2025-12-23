#ifndef _SK_ROBOT_LIB_SUB_LASERSCAN_H_
#define _SK_ROBOT_LIB_SUB_LASERSCAN_H_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubLaserScan : public skSubMsg<sensor_msgs::msg::LaserScan>
{
private:
    double m_offset_x;
    double m_offset_y;
    sPoint2D m_offset;

public:
    skSubLaserScan();

    ~skSubLaserScan();

    void setOffset(const double&x, const double& y);
    sLine getLineFromLaserScan(const double& anchor, const int& range = 10);
    sPoint2D getPoint(int k) const;
    double getRange(int k) const;
    double getAngle(int k) const;
    int getLength() const;
    double getOffsetX() const;
    double getOffsetY() const;

    double getDist(const sPoint2D p, const double& minDist = -1.0) const;
};

#endif // _SK_ROBOT_LIB_SUB_LASERSCAN_H_
