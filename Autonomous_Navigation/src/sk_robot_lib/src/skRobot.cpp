#include "sk_robot_lib/skRobot.h"

skRobot::skRobot() : Node("robot_controller"), p_joy(NULL), p_cmd(NULL), p_gps(NULL), p_quatStm(NULL), p_servo(NULL), p_solver(NULL), p_motor_vel(NULL),
                                               p_pubOdom(NULL), p_pubRobotState(NULL), p_avoidance(NULL), p_action(NULL), p_tf_broadcaster(NULL),
                                               m_state(SK_MOBILE_ROBOT_IDLE), m_activate_collision_avoidance(false)
{
    //this->m_pubOdom = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
    // Build Service server
    std::string srv_name(this->declare_parameter("srv_name", SK_MOBILE_ROBOT_SRV_DEFAULT_NAME));
    this->p_service = this->create_service<sk_robot_msgs::srv::RobotCmd>(srv_name, std::bind(&skRobot::srvCall, this, std::placeholders::_1, std::placeholders::_2));
}

skRobot::~skRobot()
{
}

void skRobot::controller()
{
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] Running.");
    // TODO: Update odom
    if( !this->updateState() )
    {
        // Get Joy
        bool followCmd(true);
        if( this->p_joy )
        {
            //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] I have joy sub.");
            if( this->p_joy->getState() ) // if joy is valid
            {
                this->m_cmd_d = this->p_joy->getTwist();
                this->m_cmd_d.linear.x *= this->m_max_linear_vel;
                this->m_cmd_d.linear.y *= this->m_max_side_vel;
                this->m_cmd_d.angular.z *= this->m_max_angular_vel;
                followCmd = false;
                //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] Got Joy Cmd.");
                //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] Got Joy Cmd = [%.2f, %.2f, %.2f].", this->m_cmd_d.linear.x, this->m_cmd_d.linear.y, this->m_cmd_d.angular.z*RAD2DEG);
            }
            // Stop action with RB+A
            else if( this->p_action && this->p_joy->RB() && this->p_joy->A() )
            {
                delete this->p_action;
                this->p_action = NULL;
            }
        }

        // Excute action service
        if( followCmd && this->p_action )
        {
            //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] I have an action to do.");
            if( this->p_action->isDone() )
            {
                RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] This action is done.");
                delete this->p_action;
                this->p_action = NULL;
                this->m_state = SK_MOBILE_ROBOT_IDLE;
            }
            else if( this->p_action->update() )
            {
                //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] This action is updated.");
                this->m_state = SK_MOBILE_ROBOT_ACTION_RUNNING;
                followCmd = false;
            }
        }

        // Get Cmd if needed
        if( followCmd && this->p_cmd )
        {
            this->m_cmd_d = this->p_cmd->getMsg();
        }
    }
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] cmd_d = [%.2f, %.2f, %.1f].", this->m_cmd_d.linear.x, this->m_cmd_d.linear.y, this->m_cmd_d.angular.z);

    // Check collision Avoidance
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] Before twist_d = [%.2f, %.2f, %.1f].", this->m_cmd_d.linear.x, this->m_cmd_d.linear.y, this->m_cmd_d.angular.z*RAD2DEG);
    /*if( this->m_activate_collision_avoidance && this->p_scan )
    {
        if( this->p_scan->haveMsg() )
        {
            this->m_scan = this->p_scan->getMsg();
            this->updateDesiredTwistByScan();
        }
    }*/

    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] Before twist_d = [%.2f, %.2f, %.1f].", this->m_cmd_d.linear.x, this->m_cmd_d.linear.y, this->m_cmd_d.angular.z*RAD2DEG);
    if( this->p_avoidance )
    {
        this->m_cmd_d = this->p_avoidance->getTwist(this->m_cmd_d, this->m_cmd_c);
        //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] After  twist_d = [%.2f, %.2f, %.1f].", this->m_cmd_d.linear.x, this->m_cmd_d.linear.y, this->m_cmd_d.angular.z*RAD2DEG);
    }
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] After  twist_d = [%.2f, %.2f, %.1f].", this->m_cmd_d.linear.x, this->m_cmd_d.linear.y, this->m_cmd_d.angular.z*RAD2DEG);

    // Update vel and acc constratints, copy cmd_d to cmd_c
    this->updateCurrentFromDesiredTwist();
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] cmd_c = [%.2f, %.2f, %.1f].", this->m_cmd_c.linear.x, this->m_cmd_c.linear.y, this->m_cmd_c.angular.z*RAD2DEG);

    if( this->p_solver )
    {
        this->p_solver->setCmd(this->m_cmd_c);
        this->p_solver->getVel(this->p_motor_vel);
        //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] wheel_vel_desired = [%.2f, %.2f].", this->p_motor_vel[0]*RAD2DEG, this->p_motor_vel[1]*RAD2DEG);
    }

    if( this->p_servo && this->p_solver )
    {
        // Publish Odom
        if( this->p_pubOdom || this->p_tf_broadcaster )
        {
            this->p_servo->getPosition(this->p_motor_pos);
            geometry_msgs::msg::Twist twist(this->p_solver->getTwist(this->p_motor_pos));
            //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] twist = [%.2f, %.2f, %.1f].", twist.linear.x, twist.linear.y, twist.angular.z*RAD2DEG);
            const double th(twist.angular.z+yaw(this->m_odom.pose.pose.orientation));
            if( this->p_pubOdom )
            {
                this->m_odom.twist.twist.linear.x = twist.linear.x;
                this->m_odom.twist.twist.linear.y = twist.linear.y;
                this->m_odom.twist.twist.angular.z = twist.angular.z;
                this->m_odom.pose.pose.position.x += twist.linear.x * cos(th) - twist.linear.y * sin(th);
                this->m_odom.pose.pose.position.y += twist.linear.x * sin(th) + twist.linear.y * cos(th);
                //this->m_odom.pose.pose.orientation.z += twist.angular.z;
                //this->m_odom.pose.pose.orientation = geometry_msgs::msg::Quaternion(0.0,0.0,twist.angular.z);
                //this->m_odom.pose.pose.orientation.setRPY(0.0,0.0,yaw);
                this->m_odom.pose.pose.orientation = quat_from_yaw(th);
                this->m_odom.header.stamp = now();
                this->p_pubOdom->publish(m_odom);
            }
            if( this->p_tf_broadcaster )
            {
                this->m_tf.transform.translation.x += twist.linear.x * cos(th) - twist.linear.y * sin(th);
                this->m_tf.transform.translation.y += twist.linear.x * sin(th) - twist.linear.y * cos(th);
                this->m_tf.transform.rotation = quat_from_yaw(th);
                this->m_tf.header.stamp = now();
                this->p_tf_broadcaster->sendTransform(this->m_tf);
            }
        }

        // Publish Robot State
        if( this->p_pubRobotState )
        {
            // state of robot
            this->m_robot_state.state = this->m_state;

            // current of motors
            if( this-> p_servo )
            {
                this->p_servo->getCurrent(this->p_current_cur);
                for( unsigned int j = 0; j < this->p_servo->getNumberOfMotor(); j++ )
                    this->m_robot_state.current[j] = this->p_current_cur[j];    
            }

            // header
            this->m_robot_state.header.stamp = now();
            this->p_pubRobotState->publish(this->m_robot_state);
        }

        // Send command to motors
        this->p_servo->sendCmd(this->p_motor_vel);
        //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] wheel_vel_desired = [%.2f, %.2f].", this->p_motor_vel[0]*RAD2DEG, this->p_motor_vel[1]*RAD2DEG);
#if 0
        this->p_servo->getPosition(this->p_motor_pos);
        RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] p_motor_pos = [%.2f, %.2f].", this->p_motor_pos[0]*RAD2DEG, this->p_motor_pos[1]*RAD2DEG);

#endif
#if 0
        this->p_servo->getVelocity(this->p_current_vel);
        double position[2];
        this->p_servo->getPosition(position);
        RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] wheel_vel_desired = [%.2f, %.2f].", this->p_motor_vel[0]*RAD2DEG, this->p_motor_vel[1]*RAD2DEG);
        //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] wheel_vel_current = [%.2f, %.2f].", this->p_current_vel[0]*RAD2DEG, this->p_current_vel[1]*RAD2DEG);
        RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] wheel_vel_current = [%.2f, %.2f].", (position[0]-this->p_motor_pos[0])*RAD2DEG/0.05, (position[1]-this->p_motor_pos[1])*RAD2DEG/0.05);
        this->p_motor_pos[0] = position[0];
        this->p_motor_pos[1] = position[1];
        //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] wheel_vel_con_error = [%.2f, %.2f].", (this->p_motor_vel[0]-this->p_current_vel[0])*RAD2DEG, (this->p_motor_vel[1]-this->p_current_vel[1])*RAD2DEG);
#endif
    }
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] Send servo commands.");
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] twist_d = [%.2f, %.2f, %.1f].", this->m_cmd_d.linear.x, this->m_cmd_d.linear.y, this->m_cmd_d.angular.z*RAD2DEG);
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] twist_c = [%.2f, %.2f, %.1f].", this->m_cmd_c.linear.x, this->m_cmd_c.linear.y, this->m_cmd_c.angular.z*RAD2DEG);
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] wheel = [%.2f, %.2f, %.2f].", this->p_motor_vel[0]*RAD2DEG, this->p_motor_vel[1]*RAD2DEG, this->p_motor_vel[2]*RAD2DEG);
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] wheel = [%.2f, %.2f].", this->p_motor_vel[0]*RAD2DEG, this->p_motor_vel[1]*RAD2DEG);
    //RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::controller()] motor pulse = [%d, %d].", this->p_servo->getPulse(0), this->p_servo->getPulse(1));
}

bool skRobot::activateOdom(std::string name/* = "odom"*/)
{
    if( this->p_servo && this->p_solver)
    {
        this->p_servo->getPosition(this->p_motor_pos);
        this->p_solver->initOdom(this->p_motor_pos);
        this->p_pubOdom = create_publisher<nav_msgs::msg::Odometry>(name, 10);

        // constant coviarnace
        this->m_odom.pose.covariance[0] = 0.01; // x
        this->m_odom.pose.covariance[7] = 0.01; // y
        this->m_odom.pose.covariance[35] = 0.01; // yaw
        this->m_odom.twist.covariance[0] = 0.001; // x
        this->m_odom.twist.covariance[7] = 0.001; // y
        this->m_odom.twist.covariance[35] = 0.001; // yaw
        return (true);
    }

    return (false);
}

bool skRobot::activateCollisionAvoidance(const double& radius, const double& margin/* = 0.0*/)
{
    if( this->p_scan )
    {
        this->m_collision_radius = radius;
        this->m_collision_margin = margin;
        this->m_activate_collision_avoidance = true;

        return (true);
    }

    return (false);
}

void skRobot::updateDesiredTwistByScan()
{
    if( this->p_scan && this->m_activate_collision_avoidance && this->m_cmd_d.linear.x > 0.0 )
    {
        sPoint2D p;
        geometry_msgs::msg::Twist cmd;

        cmd = this->m_cmd_d;
        p = this->expetedPointFromTwist(cmd);
        const double dist0(this->minDistOfScan(p));

        cmd.angular.z = this->m_cmd_d.angular.z + this->m_angular_acc;
        p = this->expetedPointFromTwist(cmd);
        const double distP(this->minDistOfScan(p));

        cmd.angular.z = this->m_cmd_d.angular.z - this->m_angular_acc;
        p = this->expetedPointFromTwist(cmd);
        const double distN(this->minDistOfScan(p));

        //const double timeToStop((MAX(fabs(cmd.linear.x)/this->m_linear_acc,fabs(cmd.angular.z)/this->m_angular_acc)+2.0)*this->m_control_dt);
        const double timeToStop((fabs(cmd.linear.x)/this->m_linear_acc)*this->m_control_dt);
        const double distToStop(0.5*this->m_cmd_d.linear.x*timeToStop);

        double dist;
        if( dist0 >= distP && dist0 >= distN )
        {
            dist = MAX(0.0,dist0 - this->m_collision_radius - this->m_collision_margin);
        }
        else if( distP > distN )
        {
            dist = MAX(0.0,distP - this->m_collision_radius - this->m_collision_margin);
            this->m_cmd_d.angular.z += this->m_angular_acc;
        }
        else
        {
            dist = MAX(0.0,distN - this->m_collision_radius - this->m_collision_margin);
            this->m_cmd_d.angular.z -= this->m_angular_acc;
        }
        if( dist > distToStop )
            return;
        this->m_cmd_d.linear.x = MIN(sqrt(2.0*dist*this->m_linear_acc/this->m_control_dt), this->m_max_linear_vel);

/*        // 0 : follow current desired
        p = this->expetedPointFromTwist(this->m_cmd_d);
        RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::updateDesiredTwistByScan()] v = %.2f, p = (%.3f, %.3f).", this->m_cmd_d.linear.x, p.x, p.y);
        if( !this->ifCollide(p) )
            return;
        
        RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::updateDesiredTwistByScan()] Collide and modify the twist cmd.");
        geometry_msgs::msg::Twist cmd;
        cmd = this->m_cmd_d;
        if( fabs(this->m_cmd_d.angular.z) > 0.0 )
        {
            double dir(1.0);
            if( this->m_cmd_d.angular.z < 0.0 )
                dir = -1.0;
            // 1 : turn less
            cmd.angular.z = this->m_cmd_d.angular.z - dir*this->m_angular_acc;
            p = this->expetedPointFromTwist(cmd);
            if( !this->ifCollide(p) )
            {
                this->m_cmd_d = cmd;
                return;
            }
            // 2 : turn more
            cmd.angular.z = this->m_cmd_d.angular.z + dir*this->m_angular_acc;
            p = this->expetedPointFromTwist(cmd);
            if( !this->ifCollide(p) )
            {
                this->m_cmd_d = cmd;
                return;
            }
        }
        else
        {
            // 3 : reduce linear velocity
#if 0
            cmd = this->m_cmd_d;
            p = this->expetedPointFromTwist(cmd);
            const double dist0(this->minDistOfScan(p));
            cmd.angular.z = this->m_cmd_d.angular.z - this->m_angular_acc;
            p = this->expetedPointFromTwist(cmd);
            const double distN(this->minDistOfScan(p));
            cmd.angular.z = this->m_cmd_d.angular.z + this->m_angular_acc;
            p = this->expetedPointFromTwist(cmd);
            const double distP(this->minDistOfScan(p));
            //const double distP(this->minDistOfScan(0.5*this->m_angular_acc*this->m_control_dt,this->m_cmd_d.linear.x));
            //const double distN(this->minDistOfScan(0.5*this->m_angular_acc*this->m_control_dt,this->m_cmd_d.linear.x));
            double dist;
            if( dist0 >= distP && dist0 >= distN )
            {
                dist = MAX(0.0,dist0 - this->m_collision_radius - this->m_collision_margin);
                //this->m_cmd_d.angular.z = 0.0;
            }
            else if( distP > distN )
            {
                dist = MAX(0.0,distP - this->m_collision_radius - this->m_collision_margin);
                this->m_cmd_d.angular.z += this->m_angular_acc;
            }
            else
            {
                dist = MAX(0.0,distN - this->m_collision_radius - this->m_collision_margin);
                this->m_cmd_d.angular.z -= this->m_angular_acc;
            }
            this->m_cmd_d.linear.x = sqrt(2.0*dist*this->m_max_linear_vel*this->m_control_dt/this->m_linear_acc);
            RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::updateDesiredTwistByScan()] dist = %.3f, v = %.2f.", dist, this->m_cmd_d.linear.x);
#else
            p.x = 0.0;
            p.y = 0.0;
            const double dist(this->minDistOfScan(p));
            this->m_cmd_d.linear.x = sqrt(2.0*dist*this->m_max_linear_vel*this->m_control_dt/this->m_linear_acc);
            RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::updateDesiredTwistByScan()] dist = %.3f, v = %.2f.", dist, this->m_cmd_d.linear.x);
#endif
        }
*/
    }
}

sPoint2D skRobot::expetedPointFromTwist(const geometry_msgs::msg::Twist& cmd) const
{
    sPoint2D p;

    // time to stop from cmd speed
    const double timeToStop((MAX(fabs(cmd.linear.x)/this->m_linear_acc,fabs(cmd.angular.z)/this->m_angular_acc)+2.0)*this->m_control_dt);

    // expected position from cmd
    const double theta(0.5*(cmd.angular.z*timeToStop));
    p.x = 0.5*cmd.linear.x*timeToStop * cos(0.5*theta);
    p.y = 0.5*cmd.linear.x*timeToStop * sin(0.5*theta);

    return (p);
}

bool skRobot::ifCollide(sPoint2D p) const
{
    // move p to scanner coordinate
    p.x -= this->p_scan->getOffsetX();
    p.y -= this->p_scan->getOffsetY();

    double angle(atan2(p.y,p.x));
    int anchor(round((angle-this->m_scan.angle_min)/this->m_scan.angle_increment));
    projectToRange(anchor, this->m_scan.ranges.size());
    
    double delta(atan2(this->m_collision_radius,this->m_collision_radius+sqrt(p.x*p.x+p.y*p.y)));
    int range(ceil(delta/this->m_scan.angle_increment));

    const double collide_length(sqrt(p.x*p.x+p.y*p.y)+this->m_collision_radius+this->m_collision_margin);

    int k;
    for( int j = -range; j < range+1; j++ )
    {
        k = anchor + j;
        projectToRange(k, this->m_scan.ranges.size());
        if( this->m_scan.ranges[k] < collide_length )
            return (true);
    }

    return (false);
}

double skRobot::minDistOfScan(sPoint2D p) const
{
    // move p to scanner coordinate
    p.x -= this->p_scan->getOffsetX();
    p.y -= this->p_scan->getOffsetY();

    double angle(atan2(p.y,p.x));
    int anchor(round((angle-this->m_scan.angle_min)/this->m_scan.angle_increment));
    projectToRange(anchor, this->m_scan.ranges.size());
    
    double delta(atan2(this->m_collision_radius,this->m_collision_radius+sqrt(p.x*p.x+p.y*p.y)));
    int range(ceil(delta/this->m_scan.angle_increment));

    double minDist(this->m_scan.range_max);

    int k;

    for( int j = -range; j < range+1; j++ )
    {
        k = anchor + j;
        projectToRange(k, this->m_scan.ranges.size());
        minDist = MIN(minDist,this->m_scan.ranges[k]);
    }

    return (minDist-sqrt(p.x*p.x+p.y*p.y));
}

bool skRobot::updateState()
{
    return (false);
}

void skRobot::srvCall(const std::shared_ptr<sk_robot_msgs::srv::RobotCmd::Request> request, std::shared_ptr<sk_robot_msgs::srv::RobotCmd::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "[DEBUG][skRobot::srvCall()] I got a service call.");
    this->srvAct(request, response);
}

void skRobot::srvAct(const std::shared_ptr<sk_robot_msgs::srv::RobotCmd::Request> request, std::shared_ptr<sk_robot_msgs::srv::RobotCmd::Response> response)
{
}

void skRobot::updateCurrentFromDesiredTwist()
{
    this->m_cmd_c.linear.x = MAX(MIN(this->m_cmd_d.linear.x, this->m_cmd_c.linear.x+this->m_linear_acc), this->m_cmd_c.linear.x-this->m_linear_acc);
    this->m_cmd_c.linear.x = MAX(MIN(this->m_cmd_c.linear.x, this->m_max_linear_vel), -this->m_max_linear_vel);
    this->m_cmd_c.linear.y = MAX(MIN(this->m_cmd_d.linear.y, this->m_cmd_c.linear.y+this->m_linear_acc), this->m_cmd_c.linear.y-this->m_linear_acc);
    this->m_cmd_c.linear.y = MAX(MIN(this->m_cmd_c.linear.y, this->m_max_linear_vel), -this->m_max_linear_vel);
    this->m_cmd_c.angular.z = MAX(MIN(this->m_cmd_d.angular.z, this->m_cmd_c.angular.z+this->m_angular_acc), this->m_cmd_c.angular.z-this->m_angular_acc);
    this->m_cmd_c.angular.z = MAX(MIN(this->m_cmd_c.angular.z, this->m_max_angular_vel), -this->m_max_angular_vel);
}

double skRobot::getYaw(bool init/* = false*/)
{
    if( this->p_imu )
    {
        if( this->p_imu->haveMsg() )
        {
            this->m_imu = this->p_imu->getMsg();

            const double x(this->m_imu.orientation.x);
            const double y(this->m_imu.orientation.y);
            const double z(this->m_imu.orientation.z);
            const double w(this->m_imu.orientation.w);

            const double time(toSec(this->m_imu.header.stamp));

            double yaw(atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z)));
            if( init )
            {
                this->m_heading_time_start = time;
            }
            else
            {
                yaw -= this->m_heading_drift * (time - this->m_heading_time_start);
            }

            return (yaw);
        }
    }

    return (0.0);
}
