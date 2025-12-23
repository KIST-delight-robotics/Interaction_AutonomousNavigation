#include "sub_test/mag.h"
#include "sub_test/imu.h"
#include "sub_test/scan.h"

// Text input and output
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    //rclcpp::spin(std::make_shared<CSubMag>());
    //rclcpp::spin(std::make_shared<CSubImu>());
    rclcpp::spin(std::make_shared<CSubScan>());

    rclcpp::shutdown();

    return 0;
}


