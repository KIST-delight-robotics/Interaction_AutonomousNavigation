#include "sk_robot_lib/skKinematicsMobile2WDD.h"

skKinematicsMobile2WDD::skKinematicsMobile2WDD()
{
    this->m_number_of_motor = 2;
    this->m_angular_vel_first = false;
}

skKinematicsMobile2WDD::skKinematicsMobile2WDD(const string name)
{
    this->m_number_of_motor = 2;
    this->m_angular_vel_first = false;
    if( name.compare("METABOT") == 0 || name.compare("MetaBOT") == 0 || name.compare("MetaBot") == 0 || name.compare("metabot") == 0 )
    {
        this->initialize(METABOT_WHEEL_RADIUS, METABOT_WHEEL_DIST, 1);
    }
    else if( name.compare("FISHBOT") == 0 || name.compare("FishBOT") == 0 || name.compare("FishBot") == 0 || name.compare("fishbot") == 0 || name.compare("Fishbot") == 0 )
    {
        this->initialize(OMNI_WHEEL_RADIUS, FISHBOT_WHEEL_DIST, 0);
    }
    else if( name.compare("VACUUMBOT") == 0 || name.compare("VacuumBOT") == 0 || name.compare("VacuumBot") == 0 || name.compare("vacuumbot") == 0 || name.compare("Vacuumhbot") == 0 )
    {
        this->initialize(ROBOTIS_WHEEL_RADIUS, FISHBOT_WHEEL_DIST, 0);
    }
    else if( name.compare("Kibot") == 0 || name.compare("KIBOT") == 0  || name.compare("kibot") == 0 )
    {
        this->initialize(ROBOTIS_WHEEL_RADIUS, KIBOT_WHEEL_DIST, 0);
    }
}

skKinematicsMobile2WDD::skKinematicsMobile2WDD(const double wheel_radius, const double wheel_dist, const int direction/* = 0*/)
{
    this->m_number_of_motor = 2;
    this->m_angular_vel_first = false;
    this->initialize(wheel_radius, wheel_dist, direction);
}

skKinematicsMobile2WDD::~skKinematicsMobile2WDD()
{
}

bool skKinematicsMobile2WDD::initialize(const double wheel_radius, const double wheel_dist, const int direction)
{
    this->m_wheel_radius = wheel_radius;
    this->m_wheel_dist = wheel_dist;
    if( direction == 1 )
    {
        this->m_wheel_dir[0] = -1.0;
        this->m_wheel_side[0] = -1.0;
        this->m_wheel_dir[1] = 1.0;
        this->m_wheel_side[1] = 1.0;
    }
    else if( direction == 0 )
    {
        // ID 1 : right, ID 2 : left
        this->m_wheel_dir[0] = -1.0;
        this->m_wheel_side[0] = 1.0;
        this->m_wheel_dir[1] = 1.0;
        this->m_wheel_side[1] = -1.0;
    }
    else if( direction == 2 )
    {
        // ID 1 : left, ID 2 : rightt
        this->m_wheel_dir[0] = -1.0;
        this->m_wheel_side[0] = -1.0;
        this->m_wheel_dir[1] = 1.0;
        this->m_wheel_side[1] = 1.0;
    }
    else
        return (false);

    // Jacobian Inverse, twist -> joint velocity
    this->m_iJ[0][0] = this->m_wheel_dir[0] / this->m_wheel_radius;
    //this->m_iJ[0][1] = this->m_wheel_dir[0] * this->m_wheel_side[0] * this->m_wheel_dist * 0.5 / this->m_wheel_radius;
    this->m_iJ[0][1] = this->m_wheel_dir[0] * this->m_wheel_side[0] * this->m_wheel_dist / this->m_wheel_radius;
    this->m_iJ[1][0] = this->m_wheel_dir[1] / this->m_wheel_radius;
    //this->m_iJ[1][1] = this->m_wheel_dir[1] * this->m_wheel_side[1] * this->m_wheel_dist * 0.5 / this->m_wheel_radius;
    this->m_iJ[1][1] = this->m_wheel_dir[1] * this->m_wheel_side[1] * this->m_wheel_dist / this->m_wheel_radius;

    // Jacobian, joint velocity -> Twist
    const double D(this->m_iJ[0][0]*this->m_iJ[1][1]-this->m_iJ[0][1]*this->m_iJ[1][0]);
    this->m_J[0][0] = this->m_iJ[1][1] / D;
    this->m_J[0][1] = -1.0*this->m_iJ[0][1] / D;
    this->m_J[1][0] = -1.0*this->m_iJ[1][0] / D;
    this->m_J[1][1] = this->m_iJ[0][0] / D;

#if 0
    printf("Det = %.6f\n", D);
    double I[2][2];
    I[0][0] = this->m_J[0][0]*this->m_iJ[0][0] + this->m_J[0][1]*this->m_iJ[1][0];
    I[0][1] = this->m_J[0][0]*this->m_iJ[0][1] + this->m_J[0][1]*this->m_iJ[1][1];
    I[1][0] = this->m_J[1][0]*this->m_iJ[0][0] + this->m_J[1][1]*this->m_iJ[1][0];
    I[1][1] = this->m_J[1][0]*this->m_iJ[0][1] + this->m_J[1][1]*this->m_iJ[1][1];
    printf("I = | %.2f, %.2f |\n", I[0][0], I[0][1]);
    printf("    | %.2f, %.2f |.\n", I[1][0], I[1][1]);
    I[0][0] = this->m_iJ[0][0]*this->m_J[0][0] + this->m_iJ[0][1]*this->m_J[1][0];
    I[0][1] = this->m_iJ[0][0]*this->m_J[0][1] + this->m_iJ[0][1]*this->m_J[1][1];
    I[1][0] = this->m_iJ[1][0]*this->m_J[0][0] + this->m_iJ[1][1]*this->m_J[1][0];
    I[1][1] = this->m_iJ[1][0]*this->m_J[0][1] + this->m_iJ[1][1]*this->m_J[1][1];
    printf("I = | %.2f, %.2f |\n", I[0][0], I[0][1]);
    printf("    | %.2f, %.2f |.\n", I[1][0], I[1][1]);
#endif

    this->m_motor_velocity.resize(2,0.0);
    this->m_motor_position.resize(2,0.0);

    return (true);
}

bool skKinematicsMobile2WDD::setCmd(const geometry_msgs::msg::Twist& cmd)
{
    //ROS_INFO("[DEBUG][2WDD] m_wheel_dist = %.3f, m_wheel_radius = %.3f.", this->m_wheel_dist, this->m_wheel_radius);
    //this->m_motor_velocity[0] = this->m_wheel_dir[0]*(cmd.linear.x + this->m_wheel_side[0] * this->m_wheel_dist*cmd.angular.z*0.5) / this->m_wheel_radius;
    //this->m_motor_velocity[1] = this->m_wheel_dir[1]*(cmd.linear.x + this->m_wheel_side[1] * this->m_wheel_dist*cmd.angular.z*0.5) / this->m_wheel_radius;
    this->m_motor_velocity[0] = this->m_iJ[0][0] * cmd.linear.x + this->m_iJ[0][1] * cmd.angular.z;
    this->m_motor_velocity[1] = this->m_iJ[1][0] * cmd.linear.x + this->m_iJ[1][1] * cmd.angular.z;

    if( this->m_angular_vel_first )
    {
        double linear;
        for( int j = 0; j < 2; j++ )
        {
            if( this->m_motor_velocity[j] > this->m_max_angular_vel )
            {
                linear = (this->m_max_angular_vel - this->m_iJ[j][1] * cmd.angular.z) / this->m_iJ[j][0];
                this->m_motor_velocity[j] = this->m_max_angular_vel;
                this->m_motor_velocity[(j+1)%2] = this->m_iJ[(j+1)%2][0] * linear + this->m_iJ[(j+1)%2][1] * cmd.angular.z;
            }
            else if( this->m_motor_velocity[j] < -this->m_max_angular_vel )
            {
                linear = (-this->m_max_angular_vel - this->m_iJ[j][1] * cmd.angular.z) / this->m_iJ[j][0];
                this->m_motor_velocity[j] = -this->m_max_angular_vel;
                this->m_motor_velocity[(j+1)%2] = this->m_iJ[(j+1)%2][0] * linear + this->m_iJ[(j+1)%2][1] * cmd.angular.z;
            }
        }
    }

    return (true);
}

geometry_msgs::msg::Twist skKinematicsMobile2WDD::getTwist(const double* position, const bool init/* = false*/)
{
    double dt[2];
    geometry_msgs::msg::Twist odom;

    for( int j = 0; j < 2; j++ )
    {
        dt[j] = position[j] - this->m_motor_position[j];
        trim(dt[j]);
        this->m_motor_position[j] = position[j];
    }

    odom.linear.x = this->m_J[0][0]*dt[0] + this->m_J[0][1]*dt[1];
    odom.linear.y = 0.0;
    odom.linear.z = 0.0;
    odom.angular.x = 0.0;
    odom.angular.y = 0.0;
    odom.angular.z = this->m_J[1][0]*dt[0] + this->m_J[1][1]*dt[1];

    return (odom);
}

void skKinematicsMobile2WDD::setAngularFirstMode(const double& maxAngularVelocity)
{
    this->m_max_angular_vel = maxAngularVelocity;
    this->m_angular_vel_first = true;
}

void skKinematicsMobile2WDD::releaseAngularFirstMode()
{
    this->m_angular_vel_first = false;
}
