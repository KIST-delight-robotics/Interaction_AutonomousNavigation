#ifndef _SK_MOBILE_LIB_SUB_NAV_SAT_FIX_H_
#define _SK_MOBILE_LIB_SUB_NAV_SAT_FIX_H_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"

#include "sk_robot_lib/skSubMsg.h"

class skSubNavSatFix : public skSubMsg<sensor_msgs::msg::NavSatFix>
{
public:
    skSubNavSatFix();

    ~skSubNavSatFix();

    int getState() const;
    bool valid() const;
    double getLatitude() const; // Degree
    double getLongitude() const; // Degree
    double getAltitude() const; // meter
};

#endif // _SK_MOBILE_LIB_SUB_NAV_SAT_FIX_H_
