#include "joy2cmd/joy2cmd.h"

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

    rclcpp::spin(std::make_shared<CJoy2Cmd>());

    rclcpp::shutdown();

    return 0;
}


