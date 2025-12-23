#include "sk_robot_lib/skCSE3.h"

// Quaternion Operations
geometry_msgs::msg::Quaternion inverse(const geometry_msgs::msg::Quaternion q)
{
    geometry_msgs::msg::Quaternion iq(q);

    iq.w = q.w;
    iq.x = -1.0*q.x;
    iq.y = -1.0*q.y;
    iq.z = -1.0*q.z;

    return (iq);
}

geometry_msgs::msg::Quaternion operator *(const geometry_msgs::msg::Quaternion r, const geometry_msgs::msg::Quaternion s)
{
    geometry_msgs::msg::Quaternion t;
    t.w = r.w*s.w - r.x*s.x - r.y*s.y - r.z*s.z;
    t.x = r.w*s.x + r.x*s.w - r.y*s.z + r.z*s.y;
    t.y = r.w*s.y + r.x*s.z + r.y*s.w - r.z*s.x;
    t.z = r.w*s.z - r.x*s.y + r.y*s.x + r.z*s.w;

    return (t);
}

geometry_msgs::msg::Quaternion rollToQuat(const double& a) // rotate by X
{
    geometry_msgs::msg::Quaternion t;

    t.w = cos(0.5*a);
    t.x = sin(0.5*a);
    t.y = 0.0;
    t.z = 0.0;

    return (t);
}

geometry_msgs::msg::Quaternion pitchToQuat(const double& a) // rotate by Y
{
    geometry_msgs::msg::Quaternion t;

    t.w = cos(0.5*a);
    t.x = 0.0;
    t.y = sin(0.5*a);
    t.z = 0.0;

    return (t);
}

geometry_msgs::msg::Quaternion yawToQuat(const double& a) // rotate by Z
{
    geometry_msgs::msg::Quaternion t;

    t.w = cos(0.5*a);
    t.x = 0.0;
    t.y = 0.0;
    t.z = sin(0.5*a);

    return (t);
}

// Point Operations
geometry_msgs::msg::Point operator *(const double a, const geometry_msgs::msg::Point p)
{
    geometry_msgs::msg::Point y;

    y.x = a * p.x;
    y.y = a * p.y;
    y.z = a * p.z;

    return (y);
}

geometry_msgs::msg::Point operator +(const geometry_msgs::msg::Point a, const geometry_msgs::msg::Point b)
{
    geometry_msgs::msg::Point y;

    y.x = a.x + b.x;
    y.y = a.y + b.y;
    y.z = a.z + b.z;

    return (y);
}

// Quaternion and Point Operations
geometry_msgs::msg::Point operator *(const geometry_msgs::msg::Quaternion q, const geometry_msgs::msg::Point p)
{
    geometry_msgs::msg::Point r;

    //const double r00(1.0 - 2.0*q.y*q.y - 2.0*q.z*q.z);
    const double r00(q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z);
    const double r01(2.0*q.x*q.y - 2.0*q.w*q.z);
    const double r02(2.0*q.x*q.z + 2.0*q.w*q.y);

    const double r10(2.0*q.x*q.y + 2.0*q.w*q.z);
    //const double r11(1.0 - 2.0*q.x*q.x - 2.0*q.z*q.z);
    const double r11(q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z);
    const double r12(2.0*q.y*q.z - 2.0*q.w*q.x);

    const double r20(2.0*q.x*q.z - 2.0*q.w*q.y);
    const double r21(2.0*q.y*q.z + 2.0*q.w*q.x);
    //const double r22(1.0 - 2.0*q.x*q.x - 2.0*q.y*q.y);
    const double r22(q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z);

    r.x = r00 * p.x + r01 * p.y + r02 * p.z;
    r.y = r10 * p.x + r11 * p.y + r12 * p.z;
    r.z = r20 * p.x + r21 * p.y + r22 * p.z;

    return (r);
}

// Pose Operation
geometry_msgs::msg::Pose inverse(const geometry_msgs::msg::Pose T)
{
    geometry_msgs::msg::Pose Ti;

    Ti.orientation = inverse(T.orientation);
    Ti.position = -1.0*(Ti.orientation*T.position);

    return (Ti);
}

geometry_msgs::msg::Pose operator *(const geometry_msgs::msg::Pose T1, const geometry_msgs::msg::Pose T2)
{
    geometry_msgs::msg::Pose T;

    T.orientation = T1. orientation * T2.orientation;
    T.position = T1.orientation * T2.position + T1.position;

    return (T);
}

geometry_msgs::msg::Pose rollToPose(const double& a) // rotate by X
{
    geometry_msgs::msg::Pose T;

    T.orientation = rollToQuat(a);
    T.position.x = 0.0;    
    T.position.y = 0.0;    
    T.position.z = 0.0;    

    return (T);
}
geometry_msgs::msg::Pose pitchToPose(const double& a) // rotate by Y
{
    geometry_msgs::msg::Pose T;

    T.orientation = pitchToQuat(a);
    T.position.x = 0.0;    
    T.position.y = 0.0;    
    T.position.z = 0.0;    

    return (T);
}

geometry_msgs::msg::Pose yawToPose(const double& a) // rotate by Z
{
    geometry_msgs::msg::Pose T;

    T.orientation = yawToQuat(a);
    T.position.x = 0.0;    
    T.position.y = 0.0;    
    T.position.z = 0.0;    

    return (T);
}

// Print Pose
void print(rclcpp::Node *p_node, geometry_msgs::msg::Pose T, std::string name)
{
    RCLCPP_INFO(p_node->get_logger(), "[Pose] name = %s.", name.c_str());
    RCLCPP_INFO(p_node->get_logger(), "       position x : %.6f.", T.position.x);
    RCLCPP_INFO(p_node->get_logger(), "       position y : %.6f.", T.position.y);
    RCLCPP_INFO(p_node->get_logger(), "       position z : %.6f.", T.position.z);
    RCLCPP_INFO(p_node->get_logger(), "       orientation x : %.6f.", T.orientation.x);
    RCLCPP_INFO(p_node->get_logger(), "       orientation y : %.6f.", T.orientation.y);
    RCLCPP_INFO(p_node->get_logger(), "       orientation z : %.6f.", T.orientation.z);
    RCLCPP_INFO(p_node->get_logger(), "       orientation w : %.6f.", T.orientation.w);
}

// Save Pose as file
void savePoseAsParam(const geometry_msgs::msg::Pose T, const char* filename)
{
    ofstream save;
    save.open(filename);

    save << "/**:" << "\n";
    save << "  ros__parameters:" << "\n";
    save << "    point_x : " << T.position.x << "\n";
    save << "    point_y : " << T.position.y << "\n";
    save << "    point_z : " << T.position.z << "\n";

    save << "    orientation_x : " << T.orientation.x << "\n";
    save << "    orientation_y : " << T.orientation.y << "\n";
    save << "    orientation_z : " << T.orientation.z << "\n";
    save << "    orientation_w : " << T.orientation.w << "\n";

    save.close();
}

// Evaluate the pose of Target
double evaluatePose(const geometry_msgs::msg::Pose T)
{
    // cost by distance
    // In case of D455 proper range is 0.6 ~ 6m
    const double p_dist(fabs(T.position.z-0.6)/6.0);

    // cost by angle(off-center)
    const double p_angle(sqrt(T.position.x*T.position.x+T.position.y*T.position.y)/sqrt(T.position.x*T.position.x+T.position.y*T.position.y+T.position.z*T.position.z));

    // cost by tile
    const double p_tilt(1.0-(T.orientation.w*T.orientation.w - T.orientation.x*T.orientation.x - T.orientation.y*T.orientation.y + T.orientation.z*T.orientation.z));

    return (1.0*p_dist + 1.0*p_angle + 2.0*p_tilt);
}

/*
// Class SE(3)
CSE3::CSE3() : q(), p()
{
}

CSE3::CSE3(geometry_msgs::msg::Quaternion iq, geometry_msgs::msg::Point ip)
: q(iq), p(ip)
{
}

CSE3::~CSE3()
{
}

CSE3 CSE3::getInv() const
{
    CSE3 inv;

    inv.q = invq(this->q);
    inv.p = -1.0*(inv.q*this->p);

    return (inv);
}

geometry_msgs::msg::Point CSE3::getPoint() const
{
    return (this->p);
}

geometry_msgs::msg::Quaternion CSE3::getQuaternion() const
{
    return (this->q);
}

geometry_msgs::msg::Pose CSE3::getPose() const
{
    geometry_msgs::msg::Pose pose;
    pose.position = this->p;
    pose.orientation = this->q;

    return (pose);
}

void CSE3::saveAsParam(const char* filename) const
{
    ofstream save;
    save.open(filename);

    save << "/**:" << "\n";
    save << "  ros__parameters:" << "\n";
    save << "    point_x : " << this->p.x << "\n";
    save << "    point_y : " << this->p.y << "\n";
    save << "    point_z : " << this->p.z << "\n";

    save << "    orientation_x : " << this->q.x << "\n";
    save << "    orientation_y : " << this->q.y << "\n";
    save << "    orientation_z : " << this->q.z << "\n";
    save << "    orientation_w : " << this->q.w << "\n";

    save.close();
}

// SE(3) Operation
CSE3 operator *(CSE3 Ta, CSE3 Tb)
{
    CSE3 T;

    T.q = Ta.q * Tb.q;
    T.p = Ta.q * Tb.p + Ta.p;

    return (T);
}
*/
