#include "ros/ros.h"
#include "sk_mobile_lib/skCLaserScannerContainer.h"
#include "stdio.h"

CLaserScannerContainer::CLaserScannerContainer(CCollisionMap* map/* = NULL*/)
: p_map(map), m_range_eps(0.01), m_margin_begin(0), m_margin_end(0), m_step(1), m_measure_dist(false), m_front_dist(-1.0)
{
    p_data.clear();
    m_pose.clear();
    m_dir.clear();
    m_cal_ready.clear();
    m_sin.clear();
    m_cos.clear();
    m_margin_begin.clear();
    m_margin_end.clear();
}

CLaserScannerContainer::~CLaserScannerContainer()
{
    for( int j = 0; j < p_data.size(); j++ )
        delete p_data[j];
    p_data.clear();
    m_pose.clear();
    m_dir.clear();
    m_cal_ready.clear();
    for( int j = 0; j < m_sin.size(); j++ )
        m_sin[j].clear();
    m_sin.clear();
    for( int j = 0; j < m_cos.size(); j++ )
        m_cos[j].clear();
    m_cos.clear();
    m_margin_begin.clear();
    m_margin_end.clear();
}

void CLaserScannerContainer::setMap(CCollisionMap* map)
{
    p_map = map;
}

int CLaserScannerContainer::addLaserScanner(ros::NodeHandle n, const geometry_msgs::Pose2D& pose, const double& dir, const string name/* = "laser_scan"*/, const int margin_begin/* = 0*/, const int margin_end/* = 0*/)
{
    p_data.push_back(NULL);
    p_data.back() = new CLaserScannerStorage;
    p_data.back()->initialize(n,name);
    m_pose.push_back(pose);
    if( dir < 0.0 )
        m_dir.push_back(-1.0);
    else
        m_dir.push_back(1.0);

    m_margin_begin.push_back(margin_begin);
    m_margin_end.push_back(margin_end);

    m_cal_ready.push_back(false);
    std::vector<double> temp;
    temp.clear();
    m_sin.push_back(temp);
    m_cos.push_back(temp);

    return (p_data.size());        
}

bool CLaserScannerContainer::buildMap(const double& vx, const double& vy, const double& w)
{
    int s, idx, j;
    double x, y, th;

    if( !p_map )
        return (false);

    // initialize m_cpointCloud
    p_map->clear();
    sensor_msgs::LaserScan data;
    //m_obsMap.updateRegion(vx, vy, w);
    //ROS_INFO("[DEBUG|LSC] buildMap() : m_data.size() = %d, m_pose.size() = %d, m_dir.size() = %d.", m_data.size(), m_pose.size(), m_dir.size());
    //ros::Time start(ros::Time::now());
    double numberOfMeasure(0.0);
    if( m_measure_dist )
        m_front_dist = 0.0;
    else
        m_front_dist = -1.0;
    for( s = 0; s < p_data.size(); s++ )
    {
        //ROS_INFO("[DEBUG|LSC] buildMap() : Trying to grab Data s = %d.",s);
        if( !p_data[s]->hasData() )
        {
            //ROS_INFO("[DEBUG|LSC] buildMap() : I FAILED to get data of laser scanner %d.", s);
            continue;
        }
        // Fist pre calculate sin and cos
        if( !m_cal_ready[s] )
        {
            set_cal(s);
        }
        if( !m_cal_ready[s] )
        {
            continue;
        }
        //ROS_INFO("[DEBUG|LSC] buildMap() : I got data of laser scanner %d.", s);
        data = p_data[s]->getData();
        //p_map->reserve(p_map->size()+data.ranges.size());

        // build map
        //ROS_INFO("[DEBUG|LSC] buildMap() : data.ranges.size() = %d.", data.ranges.size());
        //ROS_INFO("[DEBUG] data.angle_min = %.1f", data.angle_min);
        if( m_step < 0 )
        {
            m_step = 1;
        }
        for( j = m_margin_begin[s]; j + m_margin_end[s] < data.ranges.size(); j += m_step )
        {
            if( data.ranges[j] < data.range_min + m_range_eps )
                continue;
            if( data.ranges[j] > data.range_max - m_range_eps )
                continue;

            //th = m_pose[s].theta + m_dir[s]*(data.angle_min + ((double)j)*data.angle_increment);                
            //x = m_pose[s].x + data.ranges[j]*cos(th);
            //y = m_pose[s].y + data.ranges[j]*sin(th);
            x = m_pose[s].x + data.ranges[j]*m_cos[s][j];
            y = m_pose[s].y + data.ranges[j]*m_sin[s][j];
            p_map->add_obs(x,y);
            if( m_measure_dist && fabs(y) < 0.05 && x > 0.0 )
            {
                m_front_dist += x;
                numberOfMeasure += 1.0;
            }
        }
    }
    if( m_measure_dist && numberOfMeasure > 1.0 )
        m_front_dist /= numberOfMeasure;
    //ros::Time finish(ros::Time::now());
    //ROS_INFO("[DEBUG] Map Computing time = %.9f",(finish-start).toSec());
    //ROS_INFO("[DEBUG|LSC] buildMap() : p_map->size() = %d.", p_map->size());

    return (true);
}

bool CLaserScannerContainer::buildScan(sensor_msgs::LaserScan* scan)
{
    int s, idx, j, k, i_min, i_max;
    double x0, y0, th0, r0, x1, y1, th1, r1, r, it;

    if( !scan )
        return (false);

    if( m_atan2.size() < 1 )
        set_atan2(400);

    // initialize scan
    scan->ranges.clear();
    scan->ranges.resize((int)ceil((scan->angle_max-scan->angle_min)/scan->angle_increment),scan->range_max+0.001);
    scan->intensities.clear();
    scan->intensities.resize(scan->ranges.size(),0);

    // read data and build
    sensor_msgs::LaserScan data;
    for( s = 0; s < p_data.size(); s++ )
    {
        if( !p_data[s]->hasData() )
        {
            continue;
        }
        data = p_data[s]->getData();

        // update information
        scan->time_increment = data.time_increment;

        // build scan
        if( m_step < 0 )
        {
            m_step = 1;
        }
        for( j = m_margin_begin[s]; j + 1 + m_margin_end[s] < data.ranges.size() ; j+= m_step )
        //for( j = m_margin_begin; j + m_margin_end < data.ranges.size(); j++ )
        {
            if( data.ranges[j] < data.range_min + m_range_eps )
                continue;
            if( data.ranges[j] > data.range_max - m_range_eps )
                continue;
            if( data.ranges[j+1] < data.range_min + m_range_eps )
                continue;
            if( data.ranges[j+1] > data.range_max - m_range_eps )
                continue;

            th0 = m_pose[s].theta + m_dir[s]*(data.angle_min + ((double)j)*data.angle_increment);                
            x0 = m_pose[s].x + data.ranges[j]*cos(th0);
            y0 = m_pose[s].y + data.ranges[j]*sin(th0);
            th1 = m_pose[s].theta + m_dir[s]*(data.angle_min + ((double)(j+1))*data.angle_increment);                
            x1 = m_pose[s].x + data.ranges[j+1]*cos(th1);
            y1 = m_pose[s].y + data.ranges[j+1]*sin(th1);

            // convert to scan
            r0 = sqrt(x0*x0+y0*y0);
            if( r0 < scan->range_min )
                continue;
            if( r0 > scan->range_max )
                continue;
            th0 = this->myAtan2(y0,x0);
            r1 = sqrt(x1*x1+y1*y1);
            if( r1 < scan->range_min )
                continue;
            if( r1 > scan->range_max )
                continue;
            th1 = this->myAtan2(y1,x1);
            r = MAX(r0, r1);
            //r = MIN(r0, r1);
            //r = 0.5*(r0 + r1);
            it = MIN(scan->intensities[j],scan->intensities[j+1]);

            if( data.angle_increment > 0.0 )
            {
                i_min = (int)floor((th0-scan->angle_min)/scan->angle_increment);
                i_max = (int)ceil((th1-scan->angle_min)/scan->angle_increment);
            }
            else
            {
                r = r0;
                r0 = r1;
                r1 = r;
                i_min = (int)floor((th1-scan->angle_min)/scan->angle_increment);
                i_max = (int)ceil((th0-scan->angle_min)/scan->angle_increment);
            }

            if( abs(i_max-i_min) < abs(i_min+(int)(scan->ranges.size())-i_max) )
            {
                for( k = 0; k < i_max - i_min + 1; k++ )
                {
                    if( i_min+k < 0 )
                        continue;
                    scan->ranges[i_min+k] = MIN(scan->ranges[i_min+k],(((double)(i_max-i_min-k))*r0+((double)k)*r1)/((double)(i_max-i_min)));
                    scan->intensities[i_min+k] = it;
                }
            }
            else
            {
                int d(i_min+(int)(scan->ranges.size())-i_max);
                for( k = 0; k < d + 1; k++ )
                {
                    idx = (i_max+k)%(int)(scan->ranges.size());
                    scan->ranges[idx] = MIN(scan->ranges[idx],(((double)(d-k))*r0+((double)k)*r1)/((double)d));
                    scan->intensities[idx] = it;
                }
            }
        }
    }
    scan->header.stamp = ros::Time::now();
    //ROS_INFO("[DEBUG|LSC] buildMap() : p_map->size() = %d.", p_map->size());

    return (true);
}

bool CLaserScannerContainer::buildScanFromMap(sensor_msgs::LaserScan* scan)
{
    int s, idx, j, k;
    double x, y, th, r;

    if( !scan )
        return (false);
    
    if( m_atan2.size() < 1 )
        set_atan2(200);

    // initialize scan
    scan->ranges.clear();
    scan->ranges.resize((int)ceil((scan->angle_max-scan->angle_min)/scan->angle_increment),scan->range_max+0.001);
    scan->intensities.clear();

    // read data and build
    for( j = 0; j < p_map->m_pointCloud.points.size(); j++ )
    {
        x = p_map->m_pointCloud.points[j].x;
        y = p_map->m_pointCloud.points[j].y;
        // convert to scan
        r = sqrt(x*x+y*y);
        if( r < scan->range_min )
            continue;
        if( r > scan->range_max )
            continue;
        th = this->myAtan2(y,x);
        k = (int)floor((th-scan->angle_min)/scan->angle_increment);
        if( -1 < k && k < scan->ranges.size() )
            scan->ranges[k] = MIN(scan->ranges[k],r);
        k = (int)ceil((th-scan->angle_min)/scan->angle_increment);
        if( -1 < k && k < scan->ranges.size() )
            scan->ranges[k] = MIN(scan->ranges[k],r);
    }
    //ROS_INFO("[DEBUG|LSC] buildMap() : p_map->size() = %d.", p_map->size());

    return (true);
}

bool CLaserScannerContainer::set_cal(const int &idx)
{
    if( idx< 0 || idx >= m_sin.size() || idx >= m_cos.size() || idx >= m_cal_ready.size() )
        return (false);

    if( !p_data[idx]->hasData() )
        return (false);

    sensor_msgs::LaserScan data(p_data[idx]->getData());
    double th;

    m_cos[idx].resize(data.ranges.size());
    m_sin[idx].resize(data.ranges.size());
    for( int j = 0; j < data.ranges.size(); j++ )
    {
        th = m_pose[idx].theta + m_dir[idx]*(data.angle_min + ((double)j)*data.angle_increment);  
        m_cos[idx][j] = cos(th);
        m_sin[idx][j] = sin(th);
    }

    m_cal_ready[idx] = true;

    return (true);
}

void CLaserScannerContainer::setStep(const int &step)
{
    m_step = step;
}

bool CLaserScannerContainer::set_atan2(const int &resolution)
{
    const double r((double)resolution);

    m_atan2.resize(resolution+1);

    for( int j = 0; j < m_atan2.size(); j++ )
    {
        m_atan2[j] = atan2((double)j,r);
    }

    return (true);
}

double CLaserScannerContainer::myAtan2(const double &y, const double &x)
{
    double th;
    if( y < 0.0 )
    {
        th = -(this->myAtan2(-y,x));
    }
    // now y >= 0.0
    else if( x < 0.0 )
    {
        th = M_PI-(this->myAtan2(y,-x));
    }
    // now x >= 0 && y >= 0
    else if( y > x )
    {
        th = 0.5*M_PI - this->myAtan2(x,y);
    }
    // now x >= y
    else
    {
        th = m_atan2[(int)round(((double)(m_atan2.size()-1))*y/x)];
    }
    while( th > M_PI )
        th -= 2.0*M_PI;
    while( th < -M_PI )
        th += 2.0*M_PI;

    return (th);
}

void CLaserScannerContainer::turnOnDistSensor()
{
    m_measure_dist = true;
}

void CLaserScannerContainer::turnOffDistSensor()
{
    m_measure_dist = false;
    m_front_dist = -1.0;
}

double CLaserScannerContainer::getDistMeasure()
{
    return (m_front_dist);
}
