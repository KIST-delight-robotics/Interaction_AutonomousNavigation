#include "sk_robot_lib/skServo.h"

skServo::skServo() : m_number_of_motor(0), p_node(NULL)
{}

skServo::~skServo()
{
    this->closeServo();
}

bool skServo::initialize(string port_name, int mode, int number_of_motor)
{
}

void skServo::closeServo()
{
}

bool skServo::sendCmd(double *cmd)
{
    for( unsigned int j = 0; j < this->m_number_of_motor; j++ )
        this->sendCmd(j,cmd[j]);

    return (true);
}
void skServo::setMode(int *mode)
{}
/*bool skServo::sendCmd(const int& j, const double& cmd){}
void skServo::setMode(const int& j, const int& mode){}
*/
bool skServo::getVelocity(double *vel) const
{
    for( unsigned int j = 0; j < this->m_number_of_motor; j++ )
        vel[j] = this->getVelocity(j);

    return (true);
}

bool skServo::getPosition(double *pos) const
{
    for( unsigned int j = 0; j < this->m_number_of_motor; j++ )
        pos[j] = this->getPosition(j);

    return (true);
}

bool skServo::getCurrent(double *cur) const
{
    for( unsigned int j = 0; j < this->m_number_of_motor; j++ )
        cur[j] = this->getCurrent(j);

    return (true);
}
/*
double skServo::getVelocity(const int& j) const
{
    return (0.0);
}

double skServo::getPosition(const int& j) const
{
    return (0.0);
}
*/
bool skServo::getState(int *state) const
{
    for( int j = 0; j < this->m_number_of_motor; j++ )
        state[j] = this->m_mode_current[j];

    return (true);
}
int skServo::getState(const int& j) const
{
    return (this->m_mode_current[j]);
}

int skServo::getNumberOfMotor() const
{
    return (this->m_number_of_motor);
}

void skServo::activateLog(rclcpp::Node* node)
{
    this->p_node = node;
    RCLCPP_INFO(this->p_node->get_logger(), "[Servo] Printing log is activated.");
}

int skServo::getPulse(unsigned int idx) const
{
    return (this->m_pulse[idx]);
}

