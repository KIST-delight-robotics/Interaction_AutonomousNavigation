#ifndef _SK_ROBOT_LIB_SUB_IMU_H_
#define _SK_ROBOT_LIB_SUB_IMU_H_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubImu : public skSubMsg<sensor_msgs::msg::Imu>
{
protected:
    double m_drift;
    double m_time_start;
    double m_yaw_offset;

public:
    skSubImu();

    ~skSubImu();

    void setDrift(const double& drift);
    double getRoll(bool init = false);
    double getPitch(bool init = false);
    double getYaw(bool init = false);
    bool setZeroYaw(const double offset = 0.0);
};

#endif // _SK_ROBOT_LIB_SUB_IMU_H_
