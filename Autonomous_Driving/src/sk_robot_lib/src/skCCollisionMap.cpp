#include "ros/ros.h"
#include "sk_mobile_lib/skCCollisionMap.h"
#include "stdio.h"

CCollisionMap::CCollisionMap(const double& map_margin/* = 0.05*/, const double& collision_margin/* = 0.1*/, const double& range/* = 5.0*/, const double filter/* = 0.35*/, const double& height/* = 0.3*/ )
{
    m_map_margin = map_margin;
    m_map_marginSQ = map_margin*m_map_margin;
    m_collision_margin = collision_margin;
    setRange(range);
    m_height = height;
    m_filter = filter;
    m_filterSQ = m_filter*m_filter;
    m_pointCloud.header.frame_id = "body";
}

void CCollisionMap::setMapMargin(const double& margin)
{
    m_map_margin = margin;
    m_map_marginSQ = m_map_margin*m_map_margin;
    setRange(m_range);
}

void CCollisionMap::setCollisionMargin(const double& margin)
{
    m_collision_margin = margin;
    setRange(m_range);
}

void CCollisionMap::setFilter(const double& filter)
{
    m_filter = filter;
    m_filterSQ = m_filter*m_filter;
}

void CCollisionMap::setRange(const double& range)
{
    m_range = m_map_margin + m_collision_margin + range;
    m_rangeSQ = m_range*m_range;
}

void CCollisionMap::setHeight(const double& height)
{
    m_height = height;
}

void CCollisionMap::setFrame(const std::string& frame)
{
    m_pointCloud.header.frame_id = frame;
    ROS_INFO("[DEBUG/CCollisionMap] new frame_id = %s", m_pointCloud.header.frame_id.c_str());
}

bool CCollisionMap::isInRegion(const double& x, const double& y) const
{
    return ( (x*x + y*y < m_rangeSQ) && (x*x + y*y > m_filterSQ) );
}

void CCollisionMap::clear()
{
    m_pointCloud.points.clear();
    m_pointCloud.channels.clear();
}

void CCollisionMap::reserve(const int& n)
{
    m_pointCloud.points.reserve(n);
}

int CCollisionMap::size()
{
    return (m_pointCloud.points.size());
}

bool CCollisionMap::add_obs(const double& x, const double& y)
{
    if( isInRegion(x,y) )
    {
        geometry_msgs::Point32 p;
        p.x = x;
        p.y = y;
        p.z = m_height;
        if( m_pointCloud.points.size() > 0 )
        {
            if( this->dist2PointSQ(m_pointCloud.points.size()-1,x,y) < m_map_marginSQ )
                return (false);
        }
        m_pointCloud.points.push_back(p);
        return (true);
    }
    return (false);
}

double CCollisionMap::dist2obs(const double& x, const double& y) const
{
    double min_distSQ(std::numeric_limits<double>::infinity());
    double distSQ;
    const double m_eps(m_map_marginSQ*0.0001);

    //ROS_INFO("[DEBUG|CMap] dist2obs() : m_pointCloud.points.size() = %d.", m_pointCloud.points.size());
    for( int j = 0; j < m_pointCloud.points.size(); j++ )
    {
        min_distSQ = MIN( this->dist2PointSQ(j,x,y), min_distSQ );
        if( min_distSQ < m_eps )
            break;
    }

    //ROS_INFO("[DEBUG|CMap] dist2obs() : min_distSQ = %.2f.", min_distSQ);
    return (sqrt(min_distSQ));
}

double CCollisionMap::dist2obs(const double& x, const double& y, const double& th, const double& margin_x, const double& margin_y) const
{
    double min_distSQ(std::numeric_limits<double>::infinity());
    double distSQ;
    const double m_eps(m_map_marginSQ*0.0001);

    for( int j = 0; j < m_pointCloud.points.size(); j++ )
    {
        min_distSQ = MIN( this->dist2PointSQ(j,x,y,th,margin_x,margin_y), min_distSQ );
        if( min_distSQ < 0.0 )
            return(min_distSQ);
    }

    return (sqrt(min_distSQ));
}

double CCollisionMap::dist2obsOnFront(const double& x, const double& y, const double& th, const double& margin_x, const double& margin_y) const
{
    double min_dist(std::numeric_limits<double>::infinity());
    double dx, dy;
    const double m_eps(m_map_margin*0.0001);

    for( int j = 0; j < m_pointCloud.points.size(); j++ )
    {
        dx = fabs((m_pointCloud.points[j].x-x)*cos(th) + (m_pointCloud.points[j].y-y)*sin(th));
        dy = fabs((m_pointCloud.points[j].x-x)*sin(th) - (m_pointCloud.points[j].y-y)*cos(th));
        if( dy < margin_y )
            min_dist = MIN(min_dist, dx-margin_x);
        if( min_dist < 0.0 )
            return(min_dist);
    }

    return (min_dist);
}

double CCollisionMap::dist2obsOnBack(const double& x, const double& y, const double& th, const double& margin_x, const double& margin_y) const
{
    double min_dist(std::numeric_limits<double>::infinity());
    double dx, dy;
    const double m_eps(m_map_margin*0.0001);

    for( int j = 0; j < m_pointCloud.points.size(); j++ )
    {
	if( m_pointCloud.points[j].x > -margin_x )
		continue;
        dx = fabs((m_pointCloud.points[j].x-x)*cos(th) + (m_pointCloud.points[j].y-y)*sin(th));
        dy = fabs((m_pointCloud.points[j].x-x)*sin(th) - (m_pointCloud.points[j].y-y)*cos(th));
        if( dy < margin_y )
            min_dist = MIN(min_dist, dx-margin_x);
        if( min_dist < 0.0 )
            return(min_dist);
    }

    return (min_dist);
}

sensor_msgs::PointCloud& CCollisionMap::getPC()
{
    return (m_pointCloud);
}

double CCollisionMap::dist2PointSQ(const int& j, const double& x, const double& y) const
{
    if( -1 < j && j < m_pointCloud.points.size() )
    {
        return( (m_pointCloud.points[j].x-x)*(m_pointCloud.points[j].x-x) + (m_pointCloud.points[j].y-y)*(m_pointCloud.points[j].y-y) );
    }
    return (0.0);
}

double CCollisionMap::dist2PointSQ(const int& j, const double& x, const double& y, const double& th, const double& margin_x, const double& margin_y) const
{
    if( -1 < j && j < m_pointCloud.points.size() )
    {
        const double dx = fabs((m_pointCloud.points[j].x-x)*cos(th) + (m_pointCloud.points[j].y-y)*sin(th));
        const double dy = fabs((m_pointCloud.points[j].x-x)*sin(th) - (m_pointCloud.points[j].y-y)*cos(th));
#if 1
        if( dx > margin_x )
        {
            return ((dx-margin_x)*(dx-margin_x)+dy*dy);
        }
        return (dy*dy);
    }
    return (std::numeric_limits<double>::infinity());
#else
        if( dx > margin_x )
        {
            if( dy > margin_y )
            {
                return ((dx-margin_x)*(dx-margin_x)+(dy-margin_y)*(dy-margin_y));
            }
            else
            {
                return((dx-margin_x)*(dx-margin_x));
            }            
        }
        else
        {
            if( dy > margin_y )
            {
                return ((dy-margin_y)*(dy-margin_y));
            }
            else
            {
                return MIN(dx-margin_x,dy-margin_y);
            }
            
        }
        
    }
    return (-MAX(margin_x,margin_y));
#endif
}
