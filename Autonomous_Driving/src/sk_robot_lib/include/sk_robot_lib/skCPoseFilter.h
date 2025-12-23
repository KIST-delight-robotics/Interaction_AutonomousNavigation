#ifndef _SK_MOBILE_LIB_C_POSE_FILTER_H_
#define _SK_MOBILE_LIB_C_POSE_FILTER_H_

#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include "geometry_msgs/Pose2D.h"

#define SK_MOBILE_LIB_POSE_NOT_DEFINE (0)
#define SK_MOBILE_LIB_POSE_LEFT_RIGHT (1)
#define SK_MOBILE_LIB_POSE_FRONT_BACK (2)

class CPoseFilter
{
private:
    std::vector<double> m_x1, m_y1, m_x2, m_y2;
    int m_recent;
    int m_filter_size;
    bool m_align;
    double m_marginSQ;

public:

    CPoseFilter(const int& filter_size = 10, const int& align = SK_MOBILE_LIB_POSE_LEFT_RIGHT, const double& margin = 0.3);

    ~CPoseFilter() {};

    void setAlign(const int& align);
    void setMarign(const double& margin);
    void addData(const double& x1, const double& y1, const double& x2, const double& y2); // left-right, front-back
    geometry_msgs::Pose2D addDataAndGetPose(const double& x1, const double& y1, const double& x2, const double& y2);
    geometry_msgs::Pose2D getPose();

private:
    bool validData(const double& x1i, const double& y1i, const double& x2i, const double& y2i);
};

#endif