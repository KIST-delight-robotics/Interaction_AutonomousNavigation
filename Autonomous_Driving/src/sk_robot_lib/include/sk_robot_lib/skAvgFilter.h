#ifndef _SK_ROBOT_LIB_AVG_FILTER_H_
#define _SK_ROBOT_LIB_AVG_FILTER_H_

#include "rclcpp/rclcpp.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono_literals;

class skAvgFilter
{
//private:
public:
    double m_sum;
    std::vector<double> m_data;
    double m_count;
    int m_idx;
    int m_max;
    bool m_full;

public:
    skAvgFilter(const int count = 0);
    ~skAvgFilter();

    void addData(const double& data);
    double getAvg() const;
    void print() const;
};

#endif
