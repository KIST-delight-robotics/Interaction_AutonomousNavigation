#include "sk_robot_lib/skSubRobotState.h"

skSubRobotState::skSubRobotState()
{
}

skSubRobotState::~skSubRobotState()
{
}

double skSubRobotState::getTime() const
{
    double time = toMSec(this->m_msg.header.stamp);
    int trim(time/100.0);
    return (time - ((double)trim)*100,0);
}

double skSubRobotState::getTimeFrom(builtin_interfaces::msg::Time start) const
{
    return (toMSecBtw(this->m_msg.header.stamp, start));
}
