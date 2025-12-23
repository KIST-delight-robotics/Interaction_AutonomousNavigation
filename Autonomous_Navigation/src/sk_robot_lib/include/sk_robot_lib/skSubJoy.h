#ifndef _SK_ROBOT_LIB_SUB_JOY_H_
#define _SK_ROBOT_LIB_SUB_JOY_H_

#include "rclcpp/rclcpp.hpp"
#include "sk_robot_lib/skRobotCommon.h"

// KEY Map
#define JOY_PAD_F710_BUTTON_A (0)
#define JOY_PAD_F710_BUTTON_B (1)
#define JOY_PAD_F710_BUTTON_X (2)
#define JOY_PAD_F710_BUTTON_Y (3)
#define JOY_PAD_F710_BUTTON_LB (4)
#define JOY_PAD_F710_BUTTON_RB (5)
#define JOY_PAD_F710_BUTTON_BACK (6)
#define JOY_PAD_F710_BUTTON_START (7)
#define JOY_PAD_F710_BUTTON_LOGITECH (8)
#define JOY_PAD_F710_BUTTON_STICK_LEFT (9)
#define JOY_PAD_F710_BUTTON_STICK_RIGHT (10)

#define JOY_PAD_F710_AXIS_LEFT_STICK_LEFT_RIGHT (0)
#define JOY_PAD_F710_AXIS_LEFT_STICK_UP_DOWN (1)
#define JOY_PAD_F710_AXIS_LT (2)
#define JOY_PAD_F710_AXIS_RIGHT_STICK_LEFT_RIGHT (3)
#define JOY_PAD_F710_AXIS_RIGHT_STICK_UP_DOWN (4)
#define JOY_PAD_F710_AXIS_RT (5)
#define JOY_PAD_F710_AXIS_KEY_LEFT_RIGHT (6)
#define JOY_PAD_F710_AXIS_KEY_UP_DOWN (7)

#define JOY_PAD_F710_AXIS_THREASHOLD (0.9)
#define JOY_PAD_F710_AXIS_RELEASE (JOY_PAD_F710_AXIS_THREASHOLD)
#define JOY_PAD_F710_AXIS_HOLD (-JOY_PAD_F710_AXIS_THREASHOLD)

#define SKRL_SUB_CLASS_NAME skSubJoy
#define SKRL_SUB_MSG_NAME sensor_msgs::msg::Joy
#define SKRL_SUB_JOY (1)

#include "sk_robot_lib/skSubMsg.h"

class skSubJoy : public skSubMsg<sensor_msgs::msg::Joy>
{
protected:
    int m_state;
    int m_cmd;

    // button map
    unsigned int m_a, m_b, m_x, m_y;
    unsigned int m_lb, m_rb, m_back, m_start, m_logo;
    unsigned int m_left_stick, m_right_stick;
    // axis map
    unsigned int m_left_stick_x, m_left_stick_y, m_left_t;
    unsigned int m_right_stick_x, m_right_stick_y, m_right_t;
    unsigned int m_key_x, m_key_y;

    double m_threashold;

    void msgCallback(const sensor_msgs::msg::Joy::SharedPtr msg);

public:
    skSubJoy();

    ~skSubJoy();

    virtual void updateJoy();

    int getState() const;
    int getCmd() const;

    // Read Input
    bool A() const;
    bool B() const;
    bool X() const;
    bool Y() const;
    bool LB() const;
    bool RB() const;
    bool START() const;
    bool BACK() const;
    bool LOGO() const;
    bool LEFTSTICK() const;
    bool RIGHTSTICK() const;
    bool LEFT() const;
    bool RIGHT() const;
    bool UP() const;
    bool DOWN() const;
    bool LT() const;
    bool RT() const;

    double LEFT_X() const;
    double LEFT_Y() const;
    double RIGHT_X() const;
    double RIGHT_Y() const;
};

#endif // _SK_ROBOT_LIB_SUB_JOY_H_
