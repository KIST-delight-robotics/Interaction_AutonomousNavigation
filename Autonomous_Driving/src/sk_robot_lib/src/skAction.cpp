#include "sk_robot_lib/skAction.h"
#include "sk_robot_lib/skRobot.h"

skAction::skAction() : p_robot(NULL), m_state(SK_ACTION_STATE_IDLE)
{}

skAction::~skAction()
{}

bool skAction::isRunning() const
{
    if( this->p_robot )
        return (this->m_state != SK_ACTION_STATE_IDLE );

    return (false);
}

bool skAction::isDone() const
{
    if( this->p_robot )
        return (this->m_state == SK_ACTION_STATE_DONE );

    return (true);
}

void skAction::setDone()
{
    if( this->p_robot )
        this->m_state = SK_ACTION_STATE_DONE;
}

bool skAction::activate(skRobot* robot, const sActionData& data)
{
    return (false);
}

bool skAction::update()
{
    return (false);
}
