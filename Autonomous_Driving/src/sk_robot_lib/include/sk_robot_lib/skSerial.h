#ifndef _SK_ROBOT_LIB_SERIAL_H_
#define _SK_ROBOT_LIB_SERIAL_H_

#include "rclcpp/rclcpp.hpp"

#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include <signal.h>

// Serial Communication
#include <errno.h>
#include <fcntl.h> 
#include <termios.h>
#include <unistd.h>

#include "sk_robot_lib/skRobotBasicFunctions.h"

#define SK_SERIAL_DEFAULT_DATA_LENGTH         (128)

typedef unsigned char 	      BYTE;
typedef unsigned int 	        WORD;

typedef struct {
	BYTE byLow;
	BYTE byHigh;
} IByte;

using namespace std;

class skSerial
{
private:
    int m_fd;
    struct termios m_tty;
  	BYTE* m_byD;
    BYTE* m_byInData;
    int m_max_length;
    std::string m_response;

    // for ROS Log
    rclcpp::Node* p_node;

public:
    skSerial();
    ~skSerial();

    bool initialize(std::string port_name, int baud_rate, int max_length = SK_SERIAL_DEFAULT_DATA_LENGTH);

    void setNode(rclcpp::Node* node);

    int sendData(const std::string data);
    bool readData(int count = 5);
    bool sendAndRead(const std::string data, int count = 5);

    std::string getResponse() const;
};

#endif
