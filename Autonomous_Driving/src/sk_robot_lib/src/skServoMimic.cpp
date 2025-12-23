#include "sk_robot_lib/skServoMimic.h"

skServoMimic::skServoMimic()
{
}

skServoMimic::~skServoMimic()
{
}

bool skServoMimic::initialize(std::string port_name, int mode, int number_of_motor/* = 1*/)
{
    this->m_number_of_motor = number_of_motor;
    this->p_v = new double[this->m_number_of_motor];
    this->m_count = SK_SERVO_MIMIC_PERIOD;
    if( this->p_node )
    {
        RCLCPP_ERROR(this->p_node->get_logger(), "[ServoMimic] Initialized with %d motors to print log for every %d calls.", this->m_number_of_motor, SK_SERVO_MIMIC_PERIOD);
    }

    return (true);
}

void skServoMimic::closeServo()
{
}

bool skServoMimic::sendCmd(const int& j, const double& cmd)
{
    if( j < this->m_number_of_motor )
    {
        this->p_v[j] = cmd;
        if( j == this-> m_number_of_motor-1 )
        {
            this->m_count--;
            if( this->m_count < 1 )
            {
                if( this->p_node )
                {
                    RCLCPP_ERROR(this->p_node->get_logger(), "Got command [%.2f, %.2f].", this->p_v[0]*RAD2DEG, this->p_v[1]*RAD2DEG);
                }
                this->m_count = SK_SERVO_MIMIC_PERIOD;
            }
        }
    }
    return (true);
}

void skServoMimic::setMode(const int& j, const int& mode)
{
}

double skServoMimic::getVelocity(const int& j) const
{
    return (this->p_v[j]);
}

double skServoMimic::getPosition(const int& j) const
{
    return (0.0);
}

double skServoMimic::getCurrent(const int& j) const
{
    return (0.0);
}

void skServoMimic::setGain(const int& mode/*=SK_ROBOT_LIB_SERVO_GAIN_DEFAULT*/, const double& P/*=0.0*/, const double& D/*=0.0*/, const double& I/*=0.0*/)
{
}
