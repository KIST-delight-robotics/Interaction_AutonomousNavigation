#ifndef _SK_ROBOT_LIB_SUB_MAG_H_
#define _SK_ROBOT_LIB_SUB_MAG_H_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubMag : public skSubMsg<sensor_msgs::msg::MagneticField>
{
protected:
    double m_offset;

    void saveMsgFilter();

    skAvgFilter* p_x;
    skAvgFilter* p_y;

public:
    skSubMag();

    ~skSubMag();

    void activateAvgFilter(const int length = 10);

    void setOffset(const double offset);
    //double getRoll();
    //double getPitch();
    double getYaw();
};

#endif // _SK_ROBOT_LIB_SUB_MAG_H_
