#include "sk_robot_lib/skKinematicsMobile3WhOmni.h"

skKinematicsMobile3WhOmni::skKinematicsMobile3WhOmni()
{
    this->m_number_of_motor = 3;
    this->m_motor_velocity.resize(3,0.0);
    this->m_motor_position.resize(3,0.0);
}

skKinematicsMobile3WhOmni::skKinematicsMobile3WhOmni(const string name)
{
    this->m_number_of_motor = 3;
    this->m_motor_velocity.resize(3,0.0);
    this->m_motor_position.resize(3,0.0);
    this->initialize(name);
}

skKinematicsMobile3WhOmni::~skKinematicsMobile3WhOmni()
{
}

bool skKinematicsMobile3WhOmni::initialize(const string name)
{
    double arm, offset, step;

    if( name.compare("Kibot") == 0 || name.compare("KIBOT") == 0  || name.compare("kibot") == 0 )
    {
        this->m_wheel_radius = OMNI_WHEEL_RADIUS;

        arm = KIBOT_WHEEL_ARM;
        offset = KIBOT_WHEEL_OFFSET;
        step = KIBOT_WHEEL_STEP;
    }

    //const double r(arm);
    const double dth(step);
    const double th1(offset);
    const double th2(th1+dth);
    const double th3(th2+dth);

    // Inverse Jacobian : twist -> motor speed
    for( int j = 0; j < 3; j++ )
    {
        const double th(offset+((double)j)*step);
        this->m_iJ[j][0] = sin(th) / this->m_wheel_radius;
        this->m_iJ[j][1] = -1.0*cos(th) / this->m_wheel_radius;
        this->m_iJ[j][2] = -arm / this->m_wheel_radius;
    }

    // Jacobian : motor speed -> twist
    const double A(this->m_wheel_radius/(-3.0*arm*sin(step)));
    this->m_J[0][0] = A*arm*(cos(th2)-cos(th3));
    this->m_J[0][1] = A*arm*(cos(th3)-cos(th1));
    this->m_J[0][2] = A*arm*(cos(th1)-cos(th2));

    this->m_J[1][0] = A*arm*(sin(th2)-sin(th3));
    this->m_J[1][1] = A*arm*(sin(th3)-sin(th1));
    this->m_J[1][2] = A*arm*(sin(th1)-sin(th2));

    this->m_J[2][0] = A*sin(step);
    this->m_J[2][1] = A*sin(step);
    this->m_J[2][2] = A*sin(step);

#if 0
    double I[3][3];
    for( int i = 0; i < 3; i++ )
    {
        for( int j = 0; j < 3; j++ )
        {
            I[i][j] = 0.0;
            for( int k = 0; k < 3; k++ )
                I[i][j] += this->m_J[i][k] * this->m_iJ[k][j];
        }
        printf("| %.1f, %.1f, %.1f |\n", I[i][0], I[i][1], I[i][2]);
    }
#endif

    return (true);
}

bool skKinematicsMobile3WhOmni::setCmd(const geometry_msgs::msg::Twist& cmd)
{
    for( int j = 0; j < 3; j++ )
    {
        this->m_motor_velocity[j] = this->m_iJ[j][0]*cmd.linear.x + this->m_iJ[j][1]*cmd.linear.y + this->m_iJ[j][2]*cmd.angular.z;
    }

    return (true);
}

geometry_msgs::msg::Twist skKinematicsMobile3WhOmni::getTwist(const double* position, const bool init/* = false*/)
{
    geometry_msgs::msg::Twist odom;
    odom.linear.x = 0.0;
    odom.linear.y = 0.0;
    odom.linear.z = 0.0;
    odom.angular.x = 0.0;
    odom.angular.y = 0.0;
    odom.angular.z = 0.0;

    if( init )
    {
        for( unsigned int j = 0; j < 3; j++ )
            this->m_motor_position[j] = position[j];
        
        return (odom);
    }

    double diff[3];
    for( unsigned int j = 0; j < 3; j++ )
    {
        double dt(position[j] - this->m_motor_position[j]);
        trim(dt);
        diff[j] = dt;//*this->m_wheel_radius;

        odom.linear.x += this->m_J[0][j]*diff[j];
        odom.linear.y += this->m_J[1][j]*diff[j];
        odom.angular.z += this->m_J[2][j]*diff[j];

        this->m_motor_position[j] = position[j];
    }

    return (odom);
}
