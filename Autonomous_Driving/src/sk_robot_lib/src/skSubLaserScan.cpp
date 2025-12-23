#include "sk_robot_lib/skSubLaserScan.h"

skSubLaserScan::skSubLaserScan() : m_offset_x(0.0), m_offset_y(0.0)
{
    this->m_offset.x = 0.0;
    this->m_offset.y = 0.0;
}

skSubLaserScan::~skSubLaserScan()
{
}

void skSubLaserScan::setOffset(const double&x, const double& y)
{
    this->m_offset_x = x;
    this->m_offset_y = y;
    this->m_offset.x = x;
    this->m_offset.y = y;
}

sLine skSubLaserScan::getLineFromLaserScan(const double& anchor, const int& range/* = 10*/)
{
    sLine line;
    line.valid = false;

    const int angle0(round((anchor-this->m_msg.angle_min)/this->m_msg.angle_increment));
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][SubLaserScan] getLineFromLaserScan() anchor = %.3f, angle0 = %d.", anchor, angle0);
    int k;

    std::vector<sPoint2D> points;
    points.clear();
    sPoint2D point;

    // find point inside the range
    for( int j = -range + 1; j < range; j++ )
    {
        k = angle0 + j;
        while( k > this->m_msg.ranges.size() )
            k -= this->m_msg.ranges.size();
        while( k < 0 )
            k += this->m_msg.ranges.size();
        if( this->m_msg.range_min < this->m_msg.ranges[k] && this->m_msg.ranges[k] < this->m_msg.range_max )
        {
            point.x = this->m_msg.ranges[k] * cos(this->m_msg.angle_min+(double)k*this->m_msg.angle_increment);
            point.y = this->m_msg.ranges[k] * sin(this->m_msg.angle_min+(double)k*this->m_msg.angle_increment);
            points.push_back(point);
        }
    }
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][SubLaserScan] getLineFromLaserScan() points.size() = %u.", points.size());
    
    // If we have enough data or points, find line    
    if( points.size() > range/2 )
    {
        // find average of points
        double mx(0.0), my(0.0);
        for( int j = 0; j < points.size(); j++ )
        {
            mx += points[j].x;
            my += points[j].y;
        }
        mx /= (double)points.size();
        my /= (double)points.size();

        // find covariance matrix
        double a(0.0), b(0.0), c(0.0), d(0.0); // Matrix[a b; c d]
        for( int j = 0; j < points.size(); j++ )
        {
            a += (points[j].x - mx)*(points[j].x - mx);
            b += (points[j].x - mx)*(points[j].y - my);
            d += (points[j].y - my)*(points[j].y - my);
        }
        a /= points.size();
        b /= points.size();
        c = b;
        d /= points.size();

        // get eigen values lambda1 > lambda2
        const double lambda1((a+d+sqrt((a+d)*(a+d)-4.0*(a*d-b*c)))/2.0);
        const double lambda2((a+d-sqrt((a+d)*(a+d)-4.0*(a*d-b*c)))/2.0);
        //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][SubLaserScan] getLineFromLaserScan() lambda ratio = %.3f.", lambda1/lambda2);

        // check the points form line by eigen value ratio
        if( lambda1 > 50.0*lambda2 )
        {
            // Find eigen vector if lambda1
            line.dist = sqrt(mx*mx+my*my);
            line.angle = atan2(lambda1-a-c,b+d-lambda1);
            line.valid = true;
        }
    }

    return (line);
}

sPoint2D skSubLaserScan::getPoint(int k) const
{
    sPoint2D p;
    projectToRange(k, this->m_msg.ranges.size());
    p.x = this->m_offset_x + this->m_msg.ranges[k]*cos(((double)k)*this->m_msg.angle_increment+this->m_msg.angle_min);
    p.y = this->m_offset_y + this->m_msg.ranges[k]*sin(((double)k)*this->m_msg.angle_increment+this->m_msg.angle_min);
    //p.x = this->m_msg.ranges[k]*cos(((double)k)*this->m_msg.angle_increment+this->m_msg.angle_min);
    //p.y = this->m_msg.ranges[k]*sin(((double)k)*this->m_msg.angle_increment+this->m_msg.angle_min);

    return (p);
}

double skSubLaserScan::getRange(int k) const
{
    return (this->m_msg.ranges[k]);
}

double skSubLaserScan::getAngle(int k) const
{
    return ((double)k*this->m_msg.angle_increment+this->m_msg.angle_min);
}

int skSubLaserScan::getLength() const
{
    return (this->m_msg.ranges.size());
}

double skSubLaserScan::getOffsetX() const
{
    return (this->m_offset_x);
}

double skSubLaserScan::getOffsetY() const
{
    return (this->m_offset_y);
}

double skSubLaserScan::getDist(const sPoint2D p, const double& minDist/* = 0.0*/) const
{
    if( !this->haveMsg() )
        return (-1.0);
    
    double retSQ(this->m_msg.range_max*this->m_msg.range_max);

    int start(0), end(this->m_msg.ranges.size());

    //start = (2*this->m_msg.ranges.size())/6;
    //end = (4*this->m_msg.ranges.size())/6;

    /*if( minDist > 0.0 )
    {
        const double r(sqrt(distSQ(p,this->m_offset)));
        if( r > minDist )
        {
            const double dth(asin(minDist/r));
            const double th(atan2(p.y-this->m_offset.y,p.x-this->m_offset.x));
            start = MAX(start, floor((th-dth-this->m_msg.angle_min)/this->m_msg.angle_increment));
            end = MIN(end, ceil((th+dth-this->m_msg.angle_min)/this->m_msg.angle_increment)+1);
        }
    }*/

    sPoint2D q;
    const double minDistSQ(minDist*minDist);
    double pqSQ;
    for( int j = start; j < end; j++ )
    {
        if( this->m_msg.ranges[j] > 1.0e-6 )
        {
            q = this->getPoint(j);
            pqSQ = distSQ(p,q);

            if( pqSQ < retSQ )
            {
                retSQ = pqSQ;
                if( retSQ < minDistSQ )
                    return (sqrt(retSQ));
            }
        }
    }

    return (sqrt(retSQ));
}
