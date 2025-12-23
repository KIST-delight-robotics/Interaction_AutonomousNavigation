#include "sk_robot_lib/skCDWController.h"

CDWController::CDWController() : m_geomtry(DWCON_GEOMETRY_NONE), m_kinematics(DWCON_KINEMATICS_NONE)

{
}

CDWController::~CDWController()
{
}

void CDWController::setMargin(const double &m1, const double &m2/* = -1.0*/)
{
    this->m_margin = m1;
    this->m_margin_x = m1;
    if( m2 > 0.0 )
    {
        this->m_margin_y = m2;
    }
    else
        this->m_margin_y = m1;
}

bool CDWController::setAsCircle(const double &r)
{
    if( r > 0.0 )
    {
        this->m_geomtry = DWCON_GEOMETRY_CIRCLE;
        this->m_r = r;

        return (true);
    }

    return (false);
}

bool CDWController::setAsBox(const double &x, const double &y)
{
    if( MIN(x,y) > 0.0 )
    {
        this->m_geomtry = DWCON_GEOMETRY_BOX;
        this->m_x = x;
        this->m_y = y;

        return (true);
    }

    return (false);
}

void CDWController::setAs2WDD()
{
    this->m_kinematics = DWCON_KINEMATICS_2WDD;
}

void CDWController::setAsOMNI()
{
    this->m_kinematics = DWCON_KINEMATICS_OMNI;
}

void CDWController::setLinearParams(const double& speed, const double& acc)
{
    this->m_linear_speed = speed;
    this->m_linear_acc = acc;
    this->m_clsq = dist2stop(this->m_linear_speed, this->m_linear_speed, this->m_linear_acc, this->m_dt);
    this->m_clsq *= this->m_clsq;
}

void CDWController::setAngularParams(const double& speed, const double& acc)
{
    this->m_angular_speed = speed;
    this->m_angular_acc = acc;
}

void CDWController::setControlPeriod(const double& dt)
{
    this->m_dt = dt;
    this->m_clsq = dist2stop(this->m_linear_speed, this->m_linear_speed, this->m_linear_acc, this->m_dt);
    this->m_clsq *= this->m_clsq;
}

void CDWController::setResolution(const int& N)
{
    this->m_N = N;
}

geometry_msgs::msg::Twist CDWController::updateCmd(geometry_msgs::msg::Twist& cmd_c, geometry_msgs::msg::Twist& cmd_d, sensor_msgs::msg::LaserScan* scan/* = NULL*/, double* p_dist/* = NULL*/)
{
    geometry_msgs::msg::Twist cmd(cmd_c);

    const double dv(this->m_dt*this->m_linear_speed/this->m_linear_acc/((double)this->m_N));
    const double dw(this->m_dt*this->m_angular_speed/this->m_angular_acc/((double)this->m_N));
    double vx, vy, w, x, y, th, dist;

    for( int j = -this->m_N; j <= this->m_N; j++ ) // search in x
    {
        vx = cmd_c.linear.x + ((double)j)*dv;
        if( vx > this->m_linear_speed )
            continue;
        if( vx < -this->m_linear_speed )
            continue;
        for( int k = (this->m_kinematics==DWCON_KINEMATICS_2WDD ? 0 : -this->m_N); k <= (this->m_kinematics==DWCON_KINEMATICS_2WDD ? 0 : this->m_N); k++ )
        {
            vy = (this->m_kinematics==DWCON_KINEMATICS_2WDD ? 0.0 : cmd_c.linear.y + ((double)k)*dv);
            if( vy > this->m_linear_speed )
                continue;
            if( vy < -this->m_linear_speed )
                continue;
            for( int a = -this->m_N; a <= this->m_N; a++ )
            {
                w = cmd_c.angular.z + ((double)a)*dw;
                if( w > m_angular_speed )
                    continue;
                if( w < -m_angular_speed )
                    continue;
            
            }
        }
        
    }
    return (cmd);
}

double CDWController::costVel(geometry_msgs::msg::Twist& t, const double& x, const double& y, const double& w)
{
    return ((t.linear.x-x)*(t.linear.x-x) + (t.linear.y-y)*(t.linear.y-y) + this->m_clsq*(t.angular.z-w)*(t.angular.z-w));
}

geometry_msgs::msg::Pose2D CDWController::expPose(const double& vx, const double& vy, const double& w)
{
    //double x(dist2stop)
}

#if 0
geometry_msgs::Twist CDWController::updateCmd(geometry_msgs::Twist& cmd_c, geometry_msgs::Twist& cmd_d, double* p_dist/* = NULL*/)
{
    geometry_msgs::Twist cmd(cmd_c);//, cmd_t(cmd_d);
    int j, k, a;

    if( cmd_c.linear.x > 0.0 )
        cmd.linear.x = MAX(cmd_c.linear.x - m_dt*m_linear_speed/m_linear_acc,0.0);
    else
        cmd.linear.x = MIN(cmd_c.linear.x + m_dt*m_linear_speed/m_linear_acc,0.0);
    {
    if( this->m_2wdd )
        cmd.linear.y = 0.0;
    }
    else
    {
        if( cmd_c.linear.y > 0.0 )
            cmd.linear.y = MAX(cmd_c.linear.y - m_dt*m_linear_speed/m_linear_acc,0.0);
        else
            cmd.linear.y = MIN(cmd_c.linear.y + m_dt*m_linear_speed/m_linear_acc,0.0);
    }
    if( m_on_rail )
    {
        cmd.angular.z = 0.0;
    }
    else
    {
        if( cmd_c.angular.z > 0.0 )
            cmd.angular.z = MAX(cmd_c.angular.z - m_dt*m_angular_speed/m_angular_acc,0.0);
        else
            cmd.angular.z = MIN(cmd_c.angular.z + m_dt*m_angular_speed/m_angular_acc,0.0);
    }

    // find the optimal one
    const double dv(m_dt*m_linear_speed/m_linear_acc/((double)m_N)), dw(m_dt*m_angular_speed/m_angular_acc/((double)m_N));
    double vx, vy, w, x, y, th, dist;
    if( m_2wdd )
    {
        if( m_on_rail )
        {
            th = 0.0;
            x = this->dist2stop(cmd.linear.x);
            y = 0.0;
        }
        else
        {
            th = this->angle2stop(cmd.angular.z);
            x = this->dist2stop(cmd.linear.x*cos(0.5*th));
            y = this->dist2stop(cmd.linear.x*sin(0.5*th));
        }
    }
    else
    {
        x = this->dist2stop(cmd.linear.x);
        y = this->dist2stop(cmd.linear.y);
    }
    double optCost(this->costVel(cmd_d,cmd.linear.x,cmd.linear.y,cmd.angular.z)), cost, optDist, maxDist(0.0);
    double dist0(p_map->dist2obs(0.0,0.0));
    if( m_2wdd )
    {
        optDist = p_map->dist2obs(x,y,th,m_margin_x,m_margin_y);
        dist0 = p_map->dist2obs(0.0,0.0,0.0,m_margin_x,m_margin_y);
    }
    else
    {
        optDist = p_map->dist2obs(x,y);
        dist0 = p_map->dist2obs(0.0,0.0);
    }

    for( j = -m_N; j <= m_N; j++ )
    {
        vx = cmd_c.linear.x + ((double)j)*dv;
        if( vx > m_linear_speed )
            continue;
        if( vx < -m_linear_speed )
            continue;
        for( k = (m_2wdd ? 0 : -m_N); k <= (m_2wdd ? 0 : m_N); k++ )
        {
            vy = (m_2wdd ? 0.0 : cmd_c.linear.y + ((double)k)*dv);
            if( vy > m_linear_speed )
                continue;
            if( vy < -m_linear_speed )
                continue;
            //for( a = -m_N; a <= m_N; a++ )
            for( a = (m_on_rail ? 0 : -m_N); a <= (m_on_rail ? 0 : m_N); a++ )
            {
                //w = cmd_c.angular.z + ((double)a)*dw;
                w = (m_on_rail ? 0.0 : cmd_c.angular.z + ((double)a)*dw);
                if( w > m_angular_speed )
                    continue;
                if( w < -m_angular_speed )
                    continue;
                //if( m_max_chr_speed > 0.0 && (sqrt(vx*vx+vy*vy) + m_wheel_arm*fabs(w)) > m_max_chr_speed )
                //if( m_max_chr_speed > 0.0 && vx*vx+vy*vy > (m_max_chr_speed-m_wheel_arm*fabs(w))*(m_max_chr_speed-m_wheel_arm*fabs(w)) )
                //    continue;

                // calculate cost
                cost = costVel(cmd_d, vx, vy, w);

                // check collision
                if( m_2wdd )
                {
                    if( m_on_rail )
                    {
                        th = 0.0;
                        x = this->dist2stop(vx);
                        y = 0.0;
                        dist = p_map->dist2obsOnFront(x,y,th,m_margin_x,m_margin_y);                    
                    }
                    else
                    {
                        th = this->angle2stop(w);
                        x = this->dist2stop(vx*cos(0.5*th));
                        y = this->dist2stop(vx*sin(0.5*th));
                        dist = p_map->dist2obs(x,y,th,m_margin_x,m_margin_y);                    
                    }
                }
                else
                {
                    x = this->dist2stop(vx);
                    y = this->dist2stop(vy);
                    dist = p_map->dist2obs(x,y);
                    th = this->angle2stop(w);
                    dist = MIN(dist, p_map->dist2obs(x*cos(th)-y*sin(th),x*sin(th)+y*cos(th)));
                }                
                maxDist = MAX(maxDist, dist);

                if( dist0 < m_margin )
                {
                    if( optDist < m_margin )
                    {
                        if( dist > optDist )
                        {
                            optCost = cost;
                            optDist = dist;
                            cmd.linear.x = vx;
                            cmd.linear.y = vy;
                            cmd.angular.z = w;
                        }
                        else if( !(dist < optDist) && cost < optCost )
                        {
                            optCost = cost;
                            optDist = dist;
                            cmd.linear.x = vx;
                            cmd.linear.y = vy;
                            cmd.angular.z = w;
                        }
                    }
                    else if( !(dist < optDist) && cost < optCost )
                    {
                        optCost = cost;
                        optDist = dist;
                        cmd.linear.x = vx;
                        cmd.linear.y = vy;
                        cmd.angular.z = w;
                    }
                }
                else if( dist > m_margin && cost < optCost)
                {
                    optCost = cost;
                    optDist = dist;
                    cmd.linear.x = vx;
                    cmd.linear.y = vy;
                    cmd.angular.z = w;
                }
            }
        }
    }

    if( p_dist )
    {
        *p_dist = optDist;
    }
    if( m_2wdd )
    {
        cmd.linear.y = 0.0;
    }

    //ROS_INFO("cmd = [%.2f, %.2f, %.2f], cmd_d = [%.2f, %.2f, %.2f], optCost = %.2f, optiDist = %.3f > %.3f, maxDist = %.3f.",
    //        cmd.linear.x, cmd.linear.y, cmd.angular.z*RAD2DEG, cmd_d.linear.x, cmd_d.linear.y, cmd_d.angular.z*RAD2DEG,
    //        optCost, optDist, m_margin, maxDist);
    return (cmd);
}


geometry_msgs::Twist CDWController::projectCmd(geometry_msgs::Twist& cmd_c, geometry_msgs::Twist& cmd_d)
{
    geometry_msgs::Twist cmd(cmd_d);//, cmd_t(cmd_d);

    cmd.linear.x = MIN(cmd.linear.x,m_linear_speed);
    cmd.linear.x = MAX(cmd.linear.x,-m_linear_speed);
    cmd.linear.x = MIN(cmd.linear.x,cmd_c.linear.x+m_dt*m_linear_speed/m_linear_acc);
    cmd.linear.x = MAX(cmd.linear.x,cmd_c.linear.x-m_dt*m_linear_speed/m_linear_acc);
    
    cmd.linear.y = MIN(cmd.linear.y,m_linear_speed);
    cmd.linear.y = MAX(cmd.linear.y,-m_linear_speed);
    cmd.linear.y = MIN(cmd.linear.y,cmd_c.linear.y+m_dt*m_linear_speed/m_linear_acc);
    cmd.linear.y = MAX(cmd.linear.y,cmd_c.linear.y-m_dt*m_linear_speed/m_linear_acc);

    cmd.angular.z = MIN(cmd.angular.z,m_angular_speed);
    cmd.angular.z = MAX(cmd.angular.z,-m_angular_speed);
    cmd.angular.z = MIN(cmd.angular.z,cmd_c.angular.z+m_dt*m_angular_speed/m_angular_acc);
    cmd.angular.z = MAX(cmd.angular.z,cmd_c.angular.z-m_dt*m_angular_speed/m_angular_acc);
    return (cmd);
}


double CDWController::costVel(geometry_msgs::Twist& t, const double& x, const double& y, const double& w)
{
    return ((t.linear.x-x)*(t.linear.x-x) + (t.linear.y-y)*(t.linear.y-y) + m_c_lengthSQ*(t.angular.z-w)*(t.angular.z-w));
    //return ((t.linear.x-x)*(t.linear.x-x) + (t.linear.y-y)*(t.linear.y-y) + dist2stop(m_linear_speed)*dist2stop(m_linear_speed)*(t.angular.z-w)*(t.angular.z-w));
}

bool CDWController::pointOnLeft(const double& x, const double &y, const double &margin)
{
    const double minX(-MAX(margin,m_margin)), maxX(this->dist2stop(this->m_linear_speed)+margin);
    if( minX < x && x < maxX && margin-0.2 < y && y < margin+0.3 )
        return (true);
    //if( this->getFreeDist() < x && x < maxX && -margin < y && y < margin+0.3 )
    //    return (true);
    return (false);
}

bool CDWController::pointOnRight(const double& x, const double &y, const double &margin)
{
    const double minX(-MAX(margin,m_margin)), maxX(this->dist2stop(this->m_linear_speed)+margin);
    if( minX < x && x < maxX && -margin-0.3 < y && y < -margin+0.2 )
        return (true);
    //if( this->getFreeDist() < x && x < maxX && -margin-0.3 < y && y < margin )
    //    return (true);
    return (false);
}

bool CDWController::pointOnFront(const double& x, const double &y, const double &margin, const int& dir)
{
    const double minX(MAX(0.0,MIN(margin,this->dist2stop(this->m_linear_speed))-0.2)), maxX(margin+this->dist2stop(this->m_linear_speed)+0.3);
    double minY(-margin-0.3), maxY(margin+0.3);
    if( dir == SK_MOBILE_DW_CONTROLLER_FOLLOW_LEFT )
        maxY = 0.0;
    else if( dir == SK_MOBILE_DW_CONTROLLER_FOLLOW_RIGHT )
        minY = 0.0;
    
    if( minX < x && x < maxX && minY < y && y < maxY )
        return (true);
    return (false);
}

bool CDWController::pointOnBack(const double& x, const double &y, const double &margin, const int& dir)
{
    const double minX(-MAX(margin,this->dist2stop(this->m_linear_speed))-0.2), maxX(0.0);
    double minY(-margin-0.3), maxY(margin+0.3);
    if( dir == SK_MOBILE_DW_CONTROLLER_FOLLOW_LEFT )
        maxY = 0.0;
    else if( dir == SK_MOBILE_DW_CONTROLLER_FOLLOW_RIGHT )
        minY = 0.0;
    
    if( minX < x && x < maxX && minY < y && y < maxY )
        return (true);
    return (false);
}

sLine CDWController::getLine(const std::vector<double>& x, const std::vector<double>& y)
{
    const int n(x.size());
    sLine line;

    double meanX(0.0), meanY(0.0), meanXX(0.0), meanXY(0.0), meanYY(0.0);

    for( int j = 0; j < n; j++ )
    {
        meanX += x[j];
        meanY += y[j];
    }
    meanX /= (double)n;
    meanY /= (double)n;

    line.cx = meanX;
    line.cy = meanY;

    for( int j = 0; j < n; j++ )
    {
        meanXX += (x[j]-meanX)*(x[j]-meanX);
        meanXY += (x[j]-meanX)*(y[j]-meanY);
        meanYY += (y[j]-meanY)*(y[j]-meanY);
    }

    meanXX /= (double)(n-1);
    meanXY /= (double)(n-1);
    meanYY /= (double)(n-1);

    /*
    A = [meanXX meanXY
         meanXY meanYY]
    (meanXX-x)*(meanYY-x) - meanXY^2 = 0
    x^2 - (meanXX+meanYY)x + meanXX*meanYY-meanXY^2 = 0
    */
    const double b(-meanXX-meanYY), c(meanXX*meanYY-meanXY*meanXY);
    const double e((-b+sqrt(b*b-4.0*c))*0.5); 
    /*
    An = en; -> (A-eI)n = 0;
    (meanXX-e)Xnx + meanXY*ny = 0
    */
    line.tx = -meanXY;
    line.ty = meanXX-e;
    if( line.tx < 0.0 )
    {
        line.tx *= -1.0;
        line.ty *= -1.0;
    }
    const double l(sqrt(line.tx*line.tx+line.ty*line.ty));
    line.tx /= l;
    line.ty /= l;

    return (line);
}

double CDWController::dist2Line(const sLine& line)
{
    return fabs(line.cx*line.ty-line.cy*line.tx);
}

double CDWController::feedbackLinearVel(const double& e)
{
    const double alpha(2.0);
    const double stop_dist(0.5*m_linear_speed*m_linear_acc + alpha*m_linear_speed*m_dt);
    double v;
    if( e > stop_dist )
    {
        v = m_linear_speed;
    }
    else if( e < -stop_dist )
    {
        v = -m_linear_speed;
    }
    else
    {
        const double a(0.5*m_linear_acc/m_linear_speed), b(alpha*m_dt), c(-fabs(e));
        v = (-b+sqrt(b*b-4.0*a*c))/(2.0*a);
        if( e < 0.0 )
            v *= -1.0;
    }
    return (v);
}

double CDWController::feedbackAngularVel(const double& e)
{
    const double alpha(2.0);
    const double stop_dist(0.5*m_angular_speed*m_angular_acc + alpha*m_angular_speed*m_dt);
    double v;
    if( e > stop_dist )
    {
        v = m_angular_speed;
    }
    else if( e < -stop_dist )
    {
        v = -m_angular_speed;
    }
    else
    {
        const double a(0.5*m_angular_acc/m_angular_speed), b(alpha*m_dt), c(-fabs(e));
        v = (-b+sqrt(b*b-4.0*a*c))/(2.0*a);
        if( e < 0.0 )
            v *= -1.0;
    }
    return (v);
}
#endif