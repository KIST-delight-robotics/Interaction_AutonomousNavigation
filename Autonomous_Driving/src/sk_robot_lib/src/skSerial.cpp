#include "sk_robot_lib/skSerial.h"

skSerial::skSerial() : p_node(NULL)
{
}

skSerial::~skSerial()
{
}

bool skSerial::initialize(std::string port_name, int baud_rate, int max_length/* = SK_SERIAL_DEFAULT_DATA_LENGTH*/)
{
    //this->m_fd = open (port_name.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    this->m_fd = open (port_name.c_str(), O_RDWR | O_NOCTTY);
    if( this->m_fd < 0 )
        if( this->p_node )
            RCLCPP_ERROR(this->p_node->get_logger(), "Error! in Opening serial port at %s", port_name.c_str());
    else
        if( this->p_node )
            RCLCPP_INFO(this->p_node->get_logger(), "Serial port at %s Opened Successfull. fd = %d", port_name.c_str(),this->m_fd);

    // Set up port
    if (tcgetattr(this->m_fd, &this->m_tty) < 0)
    {
        if( this->p_node )
            RCLCPP_INFO(this->p_node->get_logger(), "Error from tcgetattr: %s\n", strerror(errno));
    }

    // Add baud rate whenever consider new rate
    if( baud_rate == 115200 )
    {
        cfsetospeed(&m_tty, (speed_t)B115200);
        cfsetispeed(&m_tty, (speed_t)B115200);
    }
    else
    {
        return (false);
    }

    this->m_tty.c_cflag |= (CLOCAL | CREAD);    /* ignore modem controls */
    this->m_tty.c_cflag &= ~CSIZE;
    this->m_tty.c_cflag |= CS8;         /* 8-bit characters */
    this->m_tty.c_cflag &= ~PARENB;     /* no parity bit */
    this->m_tty.c_cflag &= ~CSTOPB;     /* only need 1 stop bit */
    this->m_tty.c_cflag &= ~CRTSCTS;    /* no hardware flowcontrol */

    /* setup for non-canonical mode */
    this->m_tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
    this->m_tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    this->m_tty.c_oflag &= ~OPOST;

    /* fetch bytes as they become available */
    this->m_tty.c_cc[VMIN] = 0;
    this->m_tty.c_cc[VTIME] = 1;

    if (tcsetattr(this->m_fd, TCSANOW, &this->m_tty) != 0)
    {
        if( this->p_node )
            RCLCPP_ERROR(this->p_node->get_logger(), "Error from tcsetattr: %s\n", strerror(errno));
        //return -1;
    }

    // read info
    this->m_max_length = max_length;
    this->m_byD = new BYTE[this->m_max_length];
    memset(this->m_byD, 0, sizeof(BYTE)*this->m_max_length);
    this->m_byInData = new BYTE[this->m_max_length];
    memset(this->m_byInData, 0, sizeof(BYTE)*this->m_max_length);
    this->m_response = std::string("");

    return (true);
}

void skSerial::setNode(rclcpp::Node* node)
{
    this->p_node = node;
}


int skSerial::sendData(const std::string data)
{
    int j(0);
    memset(this->m_byD, 0, sizeof(BYTE)*this->m_max_length);
    while( j < MIN(data.size(),this->m_max_length) )
    {
     	this->m_byD[j] = data.at(j);
        j++;
    }

 	//this->m_byD[j] = '\n';

    // Send cmd
    tcflush(this->m_fd, TCIOFLUSH);
  	return (write(this->m_fd, this->m_byD, j+1));
}

bool skSerial::readData(int count/* = 5*/)
{
    int m(1);

    // Read Response
    this->m_response = std::string("");

    while( (m > 0 || this->m_response.size() < 1) && count > 0 && this->m_response.back() != '\n' )
    {
        memset(this->m_byInData, 0, sizeof(BYTE)*this->m_max_length);
        tcdrain(this->m_fd);
        m = read(this->m_fd, this->m_byInData, this->m_max_length-1); /* Read the data */
        this->m_response.append(string((const char *)this->m_byInData));
        count--;
    }
    if( this->m_response.back() == '\n' )
        this->m_response.pop_back();

    return (this->m_response.length() > 0);
}

bool skSerial::sendAndRead(const std::string data, int count/* = 5*/)
{
    if( this->sendData(data) )
        return (this->readData(count));
    return (false);
}

std::string skSerial::getResponse() const
{
    return (this->m_response);
}

