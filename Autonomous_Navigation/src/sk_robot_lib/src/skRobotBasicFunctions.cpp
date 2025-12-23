#include "sk_robot_lib/skRobotBasicFunctions.h"

double dist2stop(const double &v, const double& maxV, const double& acc, const double& dt, const double alpha/* = 2.0*/)
{
    // acc = time from 0 to maxV -> a = maxV/acc
    // time to stop = v / a = acc * fabs(v) / maxV
    // distance to travel until stop = 0.5 * v * time to stop = 0.5 * acc * v *fabs(v)/ maxV
    if( fabs(v) < 1.0e-8 )
        return (0.0);

    return (0.5*acc*v*fabs(v)/maxV + alpha*v*dt);
}

double getCmdFromError(const double& error, const double& maxV, const double& acc, const double& dt, const double alpha/*= 2.0*/)
{
    double v(0.0);
    // stop dist
    const double stop_dist(0.5*maxV*acc + alpha*maxV*dt);

    if( error > stop_dist )
        v = maxV;
    else if( error < -stop_dist )
        v = -maxV;
    else
    {
        /*const double a(0.5*acc/maxV), b(alpha*dt), c(-fabs(error));
        v = MAX((-b+sqrt(b*b-4.0*a*c))/(2.0*a),maxV);
        if( error < 0.0 )
            v *= -1.0;*/
        v = error/(0.5*acc+alpha*dt);
    }

    return (v);
}

double toSec(builtin_interfaces::msg::Time stamp)
{
    return ((double)(stamp.sec) + (double)(stamp.nanosec)*1.0e-9);
}

double toMSec(builtin_interfaces::msg::Time stamp)
{
    return ((double)(stamp.sec)*1.0e3 + (double)(stamp.nanosec)*1.0e-6);
}

double toMSecBtw(builtin_interfaces::msg::Time end, builtin_interfaces::msg::Time start)
{
    return ((double)(end.sec)*1.0e3 + (double)(end.nanosec)*1.0e-6 - (double)(start.sec)*1.0e3 - (double)(start.nanosec)*1.0e-6);
}

void trim(double& th)
{
    while( th > M_PI )
        th -= 2.0*M_PI;
    while( th < -M_PI )
        th += 2.0*M_PI;
}

void trimPIPI(double& th)
{
    while( th > M_PI )
        th -= 2.0*M_PI;
    while( th < -M_PI )
        th += 2.0*M_PI;
}

void trim2PI0(double& th)
{
    while( th > 2.0*M_PI )
        th -= 2.0*M_PI;
    while( th < 0.0 )
        th += 2.0*M_PI;
}

void projectToRange(int& x, const int& range)
{
    while( x > range )
        x -= range;
    while( x < 0 )
        x += range;
}

double distSQ(const sPoint2D& p, const sPoint2D& q)
{
    return ((p.x-q.x)*(p.x-q.x)+(p.y-q.y)*(p.y-q.y));
}

double roll(const geometry_msgs::msg::Quaternion msg)
{
    const double x(msg.x);
    const double y(msg.y);
    const double z(msg.z);
    const double w(msg.w);

    return (atan2(2.0*(w*x+y*z), 1.0-2.0*(x*x+y*y)));
}

double pitch(const geometry_msgs::msg::Quaternion msg)
{
    const double x(msg.x);
    const double y(msg.y);
    const double z(msg.z);
    const double w(msg.w);

    return (2.0*atan2(sqrt(1.0+2.0*(w*y-x*z)), sqrt(1.0-2.0*(w*y-x*z)))-0.5*M_PI);
}

double yaw(const geometry_msgs::msg::Quaternion msg)
{
    const double x(msg.x);
    const double y(msg.y);
    const double z(msg.z);
    const double w(msg.w);

    return (atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z)));
}

geometry_msgs::msg::Quaternion quat_from_yaw(const double& yaw)
{
    geometry_msgs::msg::Quaternion msg;

    msg.w = cos(0.5*yaw);
    msg.x = 0.0;
    msg.y = 0.0;
    msg.z = sin(0.5*yaw);

    return (msg);
}


double operator*(const geometry_msgs::msg::Vector3& x, const geometry_msgs::msg::Vector3& y)
{
    return (x.x*y.x+x.y*y.y+x.z*y.z);
}

geometry_msgs::msg::Vector3 operator+(const geometry_msgs::msg::Vector3& x, const geometry_msgs::msg::Vector3& y)
{
    geometry_msgs::msg::Vector3 z;

    z.x = x.x + y.x;
    z.y = x.y + y.y;
    z.z = x.z + y.z;

    return (z);
}

geometry_msgs::msg::Vector3 operator-(const geometry_msgs::msg::Vector3& x, const geometry_msgs::msg::Vector3& y)
{
    geometry_msgs::msg::Vector3 z;

    z.x = x.x - y.x;
    z.y = x.y - y.y;
    z.z = x.z - y.z;

    return (z);
}

geometry_msgs::msg::Vector3 operator*(const double& a, const geometry_msgs::msg::Vector3& x)
{
    geometry_msgs::msg::Vector3 y;

    y.x = a*x.x;
    y.y = a*x.y;
    y.z = a*x.z;

    return (y);
}

bool Axb2::solve()
{
    const double D(a*d-b*c);

    if( fabs(D) <1.0e-8 )
        return (false);
    
    x0 = (d*b0-b*b1)/D;
    x1 = (-c*b0+a*b1)/D;

    return (true);
}
