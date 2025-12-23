#include "sk_robot_lib/skDynamicWindowApproach.h"
#include "sk_robot_lib/skRobot.h"

skDynamicWindowApproach::skDynamicWindowApproach()// : m_resolution_x(DWA_DEFAULT_RESOLUTION), m_resolution_y(0), m_resolution_th(DWA_DEFAULT_RESOLUTION)
{}

skDynamicWindowApproach::~skDynamicWindowApproach()
{}

/*void skDynamicWindowApproach::setParams(const skDynamicObstacleAvoidanceParam& param)
{
    this->m_radius = param.radius;
    this->m_margin = param.margin;
    this->m_resolution_x = param.resolution_x;
    this->m_resolution_y = param.resolution_y;
    this->m_resolution_th = param.resolution_th;
}*/

double skDynamicWindowApproach::distTwist(const geometry_msgs::msg::Twist vd, const geometry_msgs::msg::Twist v) const
{
    const double rd(sqrt(vd.linear.x*vd.linear.x+vd.linear.y+vd.linear.y));
    const double r(sqrt(v.linear.x*v.linear.x+v.linear.y+v.linear.y));
    //const double cos_linear((v1.linear.x*v2.linear.x+v1.linear.y+v2.linear.y)/(r1*r2));

    double dist(0.0);
#if 1
    dist += fabs(vd.linear.x-v.linear.x) / this->m_param.max_vel_x;
    dist += fabs(vd.linear.y-v.linear.y) / MAX(this->m_param.max_vel_x, this->m_param.max_vel_y);
    dist += 0.01*fabs(vd.angular.z-v.angular.z) / this->m_param.max_vel_th;
#else
    // dist by linear magniture
    if( r < rd )
        dist += (rd - r) / MAX(this->p_robot->m_max_linear_vel, this->p_robot->m_max_side_vel);

    // dist by linear direction
    if( r > 0.0 )
        dist += (rd-(vd.linear.x*v.linear.x+vd.linear.y+v.linear.y)/r) / MAX(this->p_robot->m_max_linear_vel, this->p_robot->m_max_side_vel);
    else
        dist += rd / MAX(this->p_robot->m_max_linear_vel, this->p_robot->m_max_side_vel);
    
    // dist by side
    //dist += fabs(vd.linear.y - v.linear.y);

    // classical dist
    //dist = sqrt((vd.linear.x-v.linear.x)*(vd.linear.x-v.linear.x)+(vd.linear.y-v.linear.y)*(vd.linear.y-v.linear.y));

    // dist by angulal velocity
    //dist += fabs(vd.angular.z - v.angular.z) * (this->m_radius + this->m_margin);
    dist += fabs(vd.angular.z - v.angular.z) / (this->p_robot->m_max_angular_vel);
#endif
    return (dist);
}

double skDynamicWindowApproach::cost(const geometry_msgs::msg::Twist v_d, const geometry_msgs::msg::Twist v, const bool& log/* = false*/) const
{
    double cost(this->distTwist(v_d,v));

    const double x(dist2stop(v.linear.x, this->m_param.max_vel_x, this->m_param.acc, this->m_param.dt));
    const double y(dist2stop(v.linear.y, this->m_param.max_vel_x, this->m_param.acc, this->m_param.dt));
    const double th(dist2stop(v.angular.z, this->m_param.max_vel_th, this->m_param.acc, this->m_param.dt));
    //RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][DWA] Params = [%.2f, %.2f, %.2f, %.2f].", this->p_robot->m_max_linear_vel, this->p_robot->m_max_side_vel, this->p_robot->m_acc, this->p_robot->m_control_dt);
    //RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][DWA] X = [%.2f, %.2f, %.2f].", x, y, th*RAD2DEG);

    sPoint2D p;
    p.x = x*cos(th) - y*sin(th);
    p.y = x*sin(th) + y*cos(th);

    double dist;
    if( this->p_robot )
    {
        if( this->p_robot->p_scan )
            dist = this->p_robot->p_scan->getDist(p,this->m_param.radius);
        //else if( this->p_robot->p_grid )
        //    dist = this->p_robot->p_grid->getDist(p,this->m_param.radius);
    }
    else if( this->p_scan )
        dist = this->p_scan->getDist(p,this->m_param.radius);
    else if( this->p_grid )
    {
        dist = this->p_grid->getDist(p,this->m_param.radius);
        //if( this->p_node )
        //    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][DWA] Collision Check by Grid, dist = %.3f.", dist);
    }
    else
        dist = this->m_param.radius + 2.0*this->m_param.margin;

    if( log )
        RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][DWA] p = [%.2f, %.2f], dist = %.4f / %.4f.", p.x, p.y, dist, this->m_param.radius+this->m_param.margin);

    if( !(dist > this->m_param.radius) )
    {
        if( log )
            RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][DWA] v : [%.2f, %.2f] - > [%.2f, %.2f], p = [%.3f, %.3f], dist = %.3f, cost = [%.4f], collide.", v_d.linear.x, v_d.angular.z*RAD2DEG, v.linear.x, v.angular.z*RAD2DEG, p.x, p.y, dist, this->distTwist(v_d,v));
        return (std::numeric_limits<double>::infinity());
    }

    if( dist < this->m_param.radius + this->m_param.margin)
        cost += 10.0*pow(1.0-(dist-this->m_param.radius)/this->m_param.margin, 2.0);
        //cost += tan(M_PI*0.5*(1.0-(dist-this->m_radius)/this->m_margin));
        //cost *= (1.0+tan(M_PI*0.5*(1.0-(dist-this->m_radius)/this->m_margin)));
    if( log )
        RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][DWA] v : [%.2f, %.2f] - > [%.2f, %.2f], p = [%.3f, %.3f], dist = %.3f > %.2f + %.2f, cost = [%.4f, %.4f].", v_d.linear.x, v_d.angular.z*RAD2DEG, v.linear.x, v.angular.z*RAD2DEG, p.x, p.y, dist, this->m_param.radius, this->m_param.margin, this->distTwist(v_d,v), cost);
    return (cost);
}

geometry_msgs::msg::Twist skDynamicWindowApproach::getTwist(const geometry_msgs::msg::Twist v_d, const geometry_msgs::msg::Twist v_c) const
{
    /*if( !this->p_robot )
        return (v_d);
    
    if( !this->p_robot->p_scan )
        return (v_d);
    */

    //if( this->p_node )
    //    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][DWA] Resolution = [%d, %d, %d].", this->m_param.resolution_x, this->m_param.resolution_y, this->m_param.resolution_th);
    // Build dist map

#if 1 // vorronoi like method
    const double t(2.0);
    const double th(v_d.angular.z*t*0.5);
    const double x(MAX(0.5,v_d.linear.x*t*cos(th)-v_d.linear.y*t*sin(th)));
    const double y(v_d.linear.x*t*sin(th)+v_d.linear.y*t*cos(th));
    int flag;
    geometry_msgs::msg::Twist ret;
    ret.linear.x = 0.0;
    ret.linear.y = 0.0;
    ret.linear.z = 0.0;
    ret.angular.x = 0.0;
    ret.angular.y = 0.0;
    ret.angular.z = 0.0;

    // Rotate in position motion
    if( fabs(v_d.linear.x) < 0.05 || fabs(v_d.angular.z) > 30.0*RAD2DEG )
    {
    	//RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][DWA] v_d = %.4f", v_d.linear.x);
        ret.angular.z = v_d.angular.z;
        return (ret);
    }

    //double yg = this->p_grid->getVoronoi(x, y, flag);
    double yg = this->p_grid->getVoronoi(2.0, 0.0, flag);
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][DWA] (x,y)) = (%.2f, %.2f) = (%d, %d), Flag = %d, y = %.2f.", x, y, this->p_grid->x2i(x), this->p_grid->y2i(y), flag, yg);
    if( flag > 0 )
    {
        ret.linear.x = v_d.linear.x;
        ret.angular.z = atan2(yg,x) / t;
    }
    else if( flag == 0 )
    {
        ret.linear.x = 0.0;
        ret.angular.z = v_d.angular.z;
    }

    return (ret);    
#endif

#if DWA_ACTIVATE_DIST_MAP
    if( !this->p_robot && !this->p_scan && this->p_grid )
        this->p_grid->buildDistMap(dist2stop(this->m_param.max_vel_x, this->m_param.max_vel_x, this->m_param.acc, this->m_param.dt)+this->m_param.radius*2.0+this->m_param.margin*2.0);
#endif

    geometry_msgs::msg::Twist v_o, v;
    if( v_c.linear.x > 0.0 )
        v_o.linear.x = MAX(v_c.linear.x - this->m_param.step_size_x, 0.0);
    else
        v_o.linear.x = MIN(v_c.linear.x + this->m_param.step_size_x, 0.0);
    if( v_c.linear.y > 0.0 )
        v_o.linear.y = MAX(v_c.linear.y - this->m_param.step_size_y, 0.0);
    else
        v_o.linear.y = MIN(v_c.linear.y + this->m_param.step_size_x, 0.0);
    if( v_c.angular.z > 0.0 )
        v_o.angular.z = MAX(v_c.angular.z - this->m_param.step_size_th, 0.0);
    else
        v_o.angular.z = MIN(v_c.angular.z + this->m_param.step_size_th, 0.0);
    //double cost_o(std::numeric_limits<double>::infinity()), cost;
    double cost_o(this->cost(v_d,v_o)), cost;

    //RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][DWA] Resolution = [%d, %d, %d].", this->m_resolution_x, this->m_resolution_y, this->m_resolution_th);
    for(int x = -this->m_param.resolution_x; x <= this->m_param.resolution_x; x++ )
    {
        v.linear.x = v_c.linear.x + ((double)x/(double)this->m_param.resolution_x)*this->m_param.step_size_x;
        if( fabs(v.linear.x) > this->m_param.max_vel_x )
            continue;
        //if( fabs(v.linear.x) > fabs(v_d.linear.x) )
        //    continue;
        for(int y = -this->m_param.resolution_y; y <= this->m_param.resolution_y; y++ )
        {
            if( this->m_param.resolution_y > 0 )
                v.linear.y = v_c.linear.y + ((double)y/(double)this->m_param.resolution_y)*this->m_param.step_size_y;
            else
                v.linear.y = v_c.linear.y;
            for(int th = -this->m_param.resolution_th; th <= this->m_param.resolution_th; th++ )
            {
                v.angular.z = v_c.angular.z + ((double)th/(double)this->m_param.resolution_th)*this->m_param.step_size_th;
                if( fabs(v.angular.z) > this->m_param.max_vel_th )
                    continue;
                //if( this->p_node )
                //    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][DWA] Get Cost of the case = [%d, %d, %d].", x, y, th);
                //RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][DWA] v = [%.2f, %.2f, %.2f].", v.linear.x, v.linear.y, v.angular.z);
                cost = this->cost(v_d,v);
                //RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][DWA] cost / cost_o = [%.4f, %.4f].", cost, cost_o);
                if( cost < cost_o )
                {
                    cost_o = cost;
                    v_o = v;
                }
            }
        }
    }
    //this->cost(v_d, v_o, true);
    //RCLCPP_INFO(this->p_robot->get_logger(), "[DEBUG][DWA] v : [%.2f, %.2f] - > [%.2f, %.2f], cost = [%.4f, %.4f].", v_d.linear.x, v_d.angular.z*RAD2DEG, v_o.linear.x, v_o.angular.z*RAD2DEG, this->distTwist(v_d,v_o), cost_o);
#if DWA_ACTIVATE_DIST_MAP
    if( !this->p_robot && !this->p_scan && this->p_grid )
        this->p_grid->resetDistMap();
#endif

    return (v_o);
}
