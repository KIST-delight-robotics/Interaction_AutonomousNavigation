#include "sk_robot_lib/skServoWheelTec.h"

skServoWheelTec::skServoWheelTec() : m_fd(-1)
{
}

skServoWheelTec::~skServoWheelTec()
{
}

bool skServoWheelTec::initialize(string port_name, int mode = 1, int number_of_motor = 2)
{
    /*this->m_dxl_comm_result = COMM_TX_FAIL;
    this->m_dxl_error = 0;
    this->m_present_mode = DYXEL_MODE_IDLE;*/

    this->m_number_of_motor = 3;

    // Open port
    this->m_fd = open(port_name.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if( this->m_fd < 0 )
    {
        if( this->p_node )
            RCLCPP_ERROR(this->p_node->get_logger(), "Error! in Opening serial port at %s", port_name.c_str());
    }
    else
    {
        if( this->p_node )
            RCLCPP_INFO(this->p_node->get_logger(), "Serial port at %s Opened Successfull. fd = %d", port_name.c_str(), this->m_fd);
    }

    // Set up port
    if (tcgetattr(this->m_fd, &this->m_tty) < 0)
    {
        if( this->p_node )
            RCLCPP_ERROR(this->p_node->get_logger(), "Error from tcgetattr: %s\n", strerror(errno));
    }

    //cfsetospeed(&m_tty, (speed_t)B9600);
    //cfsetispeed(&m_tty, (speed_t)B9600);
    //cfsetospeed(&this->m_tty, (speed_t)B19200);
    //cfsetispeed(&this->m_tty, (speed_t)B19200);
    //cfsetospeed(&m_tty, (speed_t)B57600);
    //cfsetispeed(&m_tty, (speed_t)B57600);
    cfsetospeed(&m_tty, (speed_t)B115200);
    cfsetispeed(&m_tty, (speed_t)B115200);

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
            RCLCPP_ERROR(this->p_node->get_logger(), "Error from tcgetattr: %s\n", strerror(errno));
        //return -1;
    }

    if( this->p_node )
        RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][WheelTecServo] Initialized.");
    return (true);
}

void skServoWheelTec::closeServo()
{
    double cmd[3] = {0.0, 0.0, 0.0};

    this->sendCmd(cmd);
}

bool skServoWheelTec::sendCmd(double *cmd)
{
    short transition;
    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE_WHEELTEC);
	this->m_byD[0] = FRAME_HEADER;
	this->m_byD[1] = 0;
    this->m_byD[2] = 0;

    transition = (short)(cmd[0]*1000.0);
	this->m_byD[4] = transition;
    this->m_byD[3] = transition>>8;

    transition = (short)(cmd[1]*1000.0);
	this->m_byD[6] = transition;
    this->m_byD[5] = transition>>8;

    transition = (short)(cmd[2]*1000.0);
	this->m_byD[8] = transition;
    this->m_byD[7] = transition>>8;

  	this->m_byD[9] = this->check_sum();
	this->m_byD[10] = FRAME_TAIL;

    // Send cmd
    int m;
    tcflush(this->m_fd,TCIOFLUSH);
  	m = write(this->m_fd, this->m_byD, 11);

    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoWheelTec] send m = %d", m);
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoWheelTec] cmd = [%.2f, %.2f, %.2f]", cmd[0], cmd[1], cmd[2]*RAD2DEG);

    return (true);
}

#if 0
int skServoMD400T::sendData(const int& n, const int read_response/* = 0*/)
{
    int m;

    // Send cmd
    tcflush(this->m_fd,TCIOFLUSH);
  	m = write(this->m_fd, this->m_byD, n);
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] m = %d.", m);

    // wait for response
    if( read_response )
    {
        tcdrain(this->m_fd);    /* delay for output */
        m = 0;
        int r(0), c(2);
        BYTE data[MAX_PACKET_SIZE_MD400T];
        memset(this->m_byInData,0,sizeof(BYTE)*MAX_PACKET_SIZE_MD400T);
        while( m < read_response && c > 0 )
        {
            memset(data,0,sizeof(BYTE)*MAX_PACKET_SIZE_MD400T);
            //r = read(this->m_fd,data,read_response-m);
            r = read(this->m_fd,data,MAX_PACKET_SIZE_MD400T);
            if( r )
            {
                c = 2;
                for( int k = 0; k < r; k++ )
                    this->m_byInData[m++] = data[k];
            }
            else
                c--;
        }
    }
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] m = %d.", m);

    return (m);
}
#endif

bool skServoWheelTec::sendCmd(const int& j, const double& cmd)
{
    return (false);
}

void skServoWheelTec::setMode(const int& j, const int& mode){}

double skServoWheelTec::getVelocity(const int& j) const
{
    return (0.0);
}

double skServoWheelTec::getPosition(const int& j) const
{
    return (0.0);
}

double skServoWheelTec::getCurrent(const int& j) const
{
    return (0.0);
}

void skServoWheelTec::setGain(const int& mode/*=SK_ROBOT_LIB_SERVO_GAIN_VEL*/, const double& P/*=0.0*/, const double& I/*=0.0*/, const double& D/*=0.0*/){}

BYTE skServoWheelTec::check_sum()
{
    BYTE sum(0);

    // Send data
    for( int k = 0; k < 9; k++ )
    {
        sum = sum^this->m_byD[k];
    }

    return (sum);
}