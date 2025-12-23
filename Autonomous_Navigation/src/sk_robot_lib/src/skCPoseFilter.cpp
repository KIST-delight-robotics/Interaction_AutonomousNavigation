#include "ros/ros.h"
#include "sk_mobile_lib/skCPoseFilter.h"

/*
class CCPoseFilter
{
private:
    double m_x1, m_y1, m_x2, m_y2;
    int m_recent;
    int m_filter_size;
    bool m_align;
    double m_margin;

public:

    CCPoseFilter(const int& filter_size = 10, const int& align_type = SK_MOBILE_LIB_POSE_LEFT_RIGHT, const double& margin = 0.3);

    ~CCPoseFilter() {};

    void setAlign(const int& align_type);
    void setMarign(const double& margin_type);
    void addData(const double& x1, const double& y1, const double& x2, const double& y2);
    geometry_msgs::Pose2D addDataAndGetPose(const double& x1, const double& y1, const double& x2, const double& y2);
    geometry_msgs::Pose2D getPose();    
};
*/
CPoseFilter::CPoseFilter(const int& filter_size/* = 10*/, const int& align_type/* = SK_MOBILE_LIB_POSE_LEFT_RIGHT*/, const double& margin/* = 0.3*/)
: m_filter_size(filter_size), m_align(align_type), m_marginSQ(margin*margin), m_recent(-1)
{
    m_x1.clear();
    m_y1.clear();
    m_x2.clear();
    m_y2.clear();
}

void CPoseFilter::setAlign(const int& align)
{
    m_align = align;
}

void CPoseFilter::setMarign(const double& margin)
{
    m_marginSQ = margin*margin;
}

void CPoseFilter::addData(const double& x1, const double& y1, const double& x2, const double& y2)
{
    if( !this->validData(x1,y1,x2,y2) )
        return;
    if( m_x1.size() < m_filter_size )
    {
        m_recent = m_x1.size();
        m_x1.push_back(x1);
        m_y1.push_back(y1);
        m_x2.push_back(x2);
        m_y2.push_back(y2);
    }
    else if( -1 < m_recent && m_recent < m_filter_size )
    {
        m_recent = (m_recent+1)%m_filter_size;
        m_x1[m_recent] = x1;
        m_y1[m_recent] = y1;
        m_x2[m_recent] = x2;
        m_y2[m_recent] = y2;
    }
    else
    {
        ROS_ERROR("m_recent = %d, m_filter_size = %d, m_x1.size() = %d.", m_recent, m_filter_size, m_x1.size());
    }
}

geometry_msgs::Pose2D CPoseFilter::addDataAndGetPose(const double& x1, const double& y1, const double& x2, const double& y2)
{
    this->addData(x1,y1,x2,y2);
    return (this->getPose());
}

geometry_msgs::Pose2D CPoseFilter::getPose()
{
    geometry_msgs::Pose2D pose;
    double x1(0.0), y1(0.0), x2(0.0), y2(0.0);

    for( int j = 0; j < m_x1.size(); j++ )
    {
        x1 += m_x1[j];
        y1 += m_y1[j];
        x2 += m_x2[j];
        y2 += m_y2[j];
    }
    x1 /= (double)m_x1.size();
    y1 /= (double)m_y1.size();
    x2 /= (double)m_x2.size();
    y2 /= (double)m_y2.size();

    pose.x = 0.5*(x1+x2);
    pose.y = 0.5*(y1+y2);
    switch(m_align)
    {
        case SK_MOBILE_LIB_POSE_LEFT_RIGHT :
            pose.theta = atan2(y2-y1, x2-x1) + M_PI*0.5;
            break;
        case SK_MOBILE_LIB_POSE_FRONT_BACK :
            pose.theta = atan2(y1-y2,x1-x2);
            break;
        default :
            pose.theta = 0.0;
    }

    return (pose);
}

bool CPoseFilter::validData(const double& x1i, const double& y1i, const double& x2i, const double& y2i)
{
    return (true);
    if( m_x1.size() < m_filter_size )
        return true;

    double x1(0.0), y1(0.0), x2(0.0), y2(0.0);

    for( int j = 0; j < m_x1.size(); j++ )
    {
        x1 += m_x1[j];
        y1 += m_y1[j];
        x2 += m_x2[j];
        y2 += m_y2[j];
    }
    x1 /= (double)m_x1.size();
    y1 /= (double)m_y1.size();
    x2 /= (double)m_x2.size();
    y2 /= (double)m_y2.size();

    if( (x1-x1i)*(x1-x1i)+(y1-y1i)*(y1-y1i) > m_marginSQ )
        return (false);
    if( (x2-x2i)*(x2-x2i)+(y2-y2i)*(y2-y2i) > m_marginSQ )
        return (false);
    return (true);
}

