#include "sk_robot_lib/skSubJoy.h"

skSubJoy::skSubJoy()
{
    if( true )
    {
        // button map
        this->m_a = JOY_PAD_F710_BUTTON_A;
        this->m_b = JOY_PAD_F710_BUTTON_B;
        this->m_x = JOY_PAD_F710_BUTTON_X;
        this->m_y = JOY_PAD_F710_BUTTON_Y;
        this->m_lb = JOY_PAD_F710_BUTTON_LB;
        this->m_rb = JOY_PAD_F710_BUTTON_RB;
        this->m_back = JOY_PAD_F710_BUTTON_BACK;
        this->m_start = JOY_PAD_F710_BUTTON_START;
        this->m_logo = JOY_PAD_F710_BUTTON_LOGITECH;
        this->m_left_stick = JOY_PAD_F710_BUTTON_STICK_LEFT;
        this->m_right_stick = JOY_PAD_F710_BUTTON_STICK_RIGHT;
        // axis map
        this->m_left_stick_x = JOY_PAD_F710_AXIS_LEFT_STICK_LEFT_RIGHT;
        this->m_left_stick_y = JOY_PAD_F710_AXIS_LEFT_STICK_UP_DOWN;
        this->m_left_t = JOY_PAD_F710_AXIS_LT;
        this->m_right_stick_x =JOY_PAD_F710_AXIS_RIGHT_STICK_LEFT_RIGHT;
        this->m_right_stick_y = JOY_PAD_F710_AXIS_RIGHT_STICK_UP_DOWN;
        this->m_right_t = JOY_PAD_F710_AXIS_RT;
        this->m_key_x = JOY_PAD_F710_AXIS_KEY_LEFT_RIGHT;
        this->m_key_y = JOY_PAD_F710_AXIS_KEY_UP_DOWN;
        this->m_threashold = JOY_PAD_F710_AXIS_THREASHOLD;
    }

    // Initialize the msg to avoid runtime errors
    this->m_msg = sensor_msgs::msg::Joy();
    this->m_msg.buttons.resize(11,0);
    this->m_msg.axes.resize(8,0.0);
    this->m_msg.axes[this->m_left_t] = 1.0;
    this->m_msg.axes[this->m_right_t] = 1.0;
}

skSubJoy::~skSubJoy()
{
}

void skSubJoy::msgCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    this->m_msg = (*msg);

    this->updateJoy();

    this->m_have_msg = true;
}

void skSubJoy::updateJoy()
{
}

int skSubJoy::getCmd() const
{
    return (this->m_cmd);
}

int skSubJoy::getState() const
{
    return (this->m_state);
}

// Read Input
bool skSubJoy::A() const
{
    return (this->m_msg.buttons[this->m_a] > 0);
}

bool skSubJoy::B() const
{
    return (this->m_msg.buttons[this->m_b] > 0);
}

bool skSubJoy::X() const
{
    return (this->m_msg.buttons[this->m_x] > 0);
}

bool skSubJoy::Y() const
{
    return (this->m_msg.buttons[this->m_y] > 0);
}

bool skSubJoy::LB() const
{
    return (this->m_msg.buttons[this->m_lb] > 0);
}

bool skSubJoy::RB() const
{
    return (this->m_msg.buttons[this->m_rb] > 0);
}

bool skSubJoy::START() const
{
    return (this->m_msg.buttons[this->m_start] > 0);
}

bool skSubJoy::BACK() const
{
    return (this->m_msg.buttons[this->m_back] > 0);
}

bool skSubJoy::LOGO() const
{
    return (this->m_msg.buttons[this->m_logo] > 0);
}

bool skSubJoy::LEFTSTICK() const
{
    return (this->m_msg.buttons[this->m_left_stick] > 0);
}

bool skSubJoy::RIGHTSTICK() const
{
    return (this->m_msg.buttons[this->m_right_stick] > 0);
}

bool skSubJoy::LEFT() const
{
    return (this->m_msg.axes[this->m_key_x] < -this->m_threashold);
}

bool skSubJoy::RIGHT() const
{
    return (this->m_msg.axes[this->m_key_x] > this->m_threashold);
}

bool skSubJoy::UP() const
{
    return (this->m_msg.axes[this->m_key_y] > this->m_threashold);
}

bool skSubJoy::DOWN() const
{
    return (this->m_msg.axes[this->m_key_y] < -this->m_threashold);
}

// Check the value range of triggers
bool skSubJoy::LT() const
{
    return (this->m_msg.axes[this->m_left_t] < -this->m_threashold);
}

bool skSubJoy::RT() const
{
    return (this->m_msg.axes[this->m_right_t] < -this->m_threashold);
}

double skSubJoy::LEFT_X() const
{
    return (this->m_msg.axes[this->m_left_stick_x]);
}

double skSubJoy::LEFT_Y() const
{
    return (this->m_msg.axes[this->m_left_stick_y]);
}

double skSubJoy::RIGHT_X() const
{
    return (this->m_msg.axes[this->m_right_stick_x]);
}

double skSubJoy::RIGHT_Y() const
{
    return (this->m_msg.axes[this->m_right_stick_y]);
}
