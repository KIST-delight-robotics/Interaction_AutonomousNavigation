#ifndef _SK_ROBOT_LIB_SUB_ARUCO_DETECTION_H_
#define _SK_ROBOT_LIB_SUB_ARUCO_DETECTION_H_

#include "rclcpp/rclcpp.hpp"
#include "aruco_opencv_msgs/msg/aruco_detection.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubArucoDetection : public skSubMsg<aruco_opencv_msgs::msg::ArucoDetection>
{
public:
    skSubArucoDetection();

    ~skSubArucoDetection();

    bool hasID(const int& id) const;
    geometry_msgs::msg::Pose getPose(const int& id) const;
    double getTime() const;
    double getTimeFrom(builtin_interfaces::msg::Time start) const;
    unsigned int sizeOfDetections() const;
};

#endif // _SK_ROBOT_LIB_SUB_ARUCO_DETECTION_H_
