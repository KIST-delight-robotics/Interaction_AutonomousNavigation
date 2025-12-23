#include "sk_robot_lib/skServoOdrive.h"

skServoOdrive::skServoOdrive()
{
}

skServoOdrive::~skServoOdrive()
{
}

bool skServoOdrive::initialize(const std::string port_name, int mode, int number_of_motor/* = 1*/)
{
    this->m_number_of_motor = number_of_motor;
    this->m_mode_current.resize(this->m_number_of_motor,SK_ROBOT_LIB_ODRIVE_MODE_IDLE);

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
            RCLCPP_ERROR(this->p_node->get_logger(), "Error from tcsetattr: %s\n", strerror(errno));
        //return -1;
    }

    // read info
    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE);
 	this->m_byD[0] = 'i';
 	this->m_byD[1] = '\n';
    sendData(1, true); // Created for user firmware

    //this->reboot();
    //ros::Duration(5.0).sleep();

    //this->setParam(0, "controller.config.control_mode CONTROL_MODE_VELOCITY_CONTROL"); // 2, CONTROL_MODE_VELOCITY_CONTROL
    this->setParam(0, "controller.config.control_mode 2"); // 2, CONTROL_MODE_VELOCITY_CONTROL
    sleep(0.05);
    this->readParam("axis0.controller.config.control_mode");
    //this->setParam(1, "controller.config.control_mode CONTROL_MODE_VELOCITY_CONTROL"); // CONTROL_MODE_VELOCITY_CONTROL
    this->setParam(1, "controller.config.control_mode 2"); // CONTROL_MODE_VELOCITY_CONTROL
    sleep(0.05);
    this->readParam("axis1.controller.config.control_mode");
    //this->save_conf();

    this->sendCmd(0,0.0);
    this->sendCmd(1,0.0);
    sleep(0.05);

    int count(10), ret0(0), ret1(0);
    while( (ret0 != 8 || ret1 != 8) && count > 0 )
    {
        //this->reset();
        if( ret0 != SK_ROBOT_LIB_ODRIVE_STATE_CLOSED_LOOP_CONTROL )
            //this->setParam(0, "requested_state AXIS_STATE_CLOSED_LOOP_CONTROL"); // 8, AXIS_STATE_CLOSED_LOOP_CONTROL
            this->setParam(0, "requested_state 8"); // 8, AXIS_STATE_CLOSED_LOOP_CONTROL
        if( ret1 != SK_ROBOT_LIB_ODRIVE_STATE_CLOSED_LOOP_CONTROL )
            //this->setParam(1, "requested_state AXIS_STATE_CLOSED_LOOP_CONTROL"); // AXIS_STATE_CLOSED_LOOP_CONTROL
            this->setParam(1, "requested_state 8"); // AXIS_STATE_CLOSED_LOOP_CONTROL
        sleep(0.05);
        ret0 = this->readParam("axis0.current_state");
        ret1 = this->readParam("axis1.current_state");
        count--;
    }

    return (ret0 == SK_ROBOT_LIB_ODRIVE_STATE_CLOSED_LOOP_CONTROL && ret1 == SK_ROBOT_LIB_ODRIVE_STATE_CLOSED_LOOP_CONTROL);
}

void skServoOdrive::closeServo()
{
}

/*bool skServoDynamixel::sendCmd(double *cmd)
{
    for( uint8_t j = 0; j < this->m_number_of_motor; j++ )
    {
        if( !this->sendCmd(j,cmd[j]) )
            return (false);
    }

    return (true);
}

void skServoDynamixel::setMode(int *mode)
{
    for( uint8_t j = 0; j < this->m_number_of_motor; j++ )
    {
        this->setMode(j,mode[j]);
    }
}
*/
bool skServoOdrive::sendCmd(const int& idx, const double& vel)
{
    unsigned int c(0);
    string s(to_string(vel/(2.0*M_PI)));

    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE);
 	this->m_byD[c++] = 'v';
 	this->m_byD[c++] = ' ';
 	this->m_byD[c++] = '0' + idx;
 	this->m_byD[c++] = ' ';
    for( int j = 0; j < MIN(s.size()-1,7); j++ )
     	this->m_byD[c++] = s.at(j);

 	this->m_byD[c] = '\n';

    this->sendData(c, false); // Created for user firmware

    return (true);
}

void skServoOdrive::setMode(const int& j, const int& mode)
{
}

double skServoOdrive::getVelocity(const int& j) const
{
}

double skServoOdrive::getPosition(const int& j) const
{
}

double skServoOdrive::getCurrent(const int& j) const
{
}

void skServoOdrive::setParam(const int& motor, const char* param, const char* value/* = NULL*/)
{
    string cmd("w ");
    if( motor == 0 )
        cmd.append("axis0.");
    else if( motor == 1 )
        cmd.append("axis1.");
    cmd.append(param);
    if( value )
    {
        cmd.append(" ");
        cmd.append(value);
    }

    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE);

    unsigned int c(0);
    while( c < cmd.size() )
    {
        this->m_byD[c] = cmd.at(c);
        c++;
    }
    this->m_byD[c] = '\n';

    this->sendData(c, false); // Created for user firmware
}

int skServoOdrive::readParam(const char* param)
{
    string cmd("r ");
    cmd.append(param);

    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE);

    unsigned int c(0);
    while( c < cmd.size() )
    {
        this->m_byD[c] = cmd.at(c);
        c++;
    }
    this->m_byD[c] = '\n';

    return (this->sendData(c, true)); // Created for user firmware
}

int skServoOdrive::sendData(const int& n, const bool& response/* = false*/)
{
    int m;

    int ret(0);

    // Send cmd
    tcflush(this->m_fd,TCIOFLUSH);
    //if( m_print_info )
    //    printf("[sendData] send Data = %s", m_byD);
  	m = write(this->m_fd, this->m_byD, n+1);
    //ROS_INFO("[DEBUG][skCMDServo] sendData() : write, m = %d.", m);

    // Read Response
    string data("");
    int count(10);
    if( response )
    {
        m = 1;
        while( (m > 0 || data.size() < 1) && count > 0 )
        {
            memset(this->m_byInData,0,sizeof(BYTE)*MAX_PACKET_SIZE);
            tcdrain(this->m_fd);    /* delay for output */
            m = read(this->m_fd,this->m_byInData,MAX_PACKET_SIZE-1); /* Read the data */
            data.append(string((const char *)this->m_byInData));
            count--;
            //ROS_INFO("[DEBUG][skCMDServo] sendData() : read, m = %d.", m);
        }
        /*if( m_print_info )
        {
            if( data.size() > 0 )
                printf("[sendData] read Data = %s", data.c_str());
            else if( count < 1 )
                printf("[sendData] Failed read response.");
        }
        if( data.size() > 0 )
        {
            ret = data.at(0)-'0';
            printf("ret = %d.\n", ret);
        }*/
    }

    return (ret);
}

void skServoOdrive::setGain(const int& mode/*=SK_ROBOT_LIB_ODRIVE_MODE_VEL*/, const double& P/*=SK_ROBOT_LIB_ODRIVE_VELOVITY_P_GAIN*/, const double& I/*=SK_ROBOT_LIB_ODRIVE_VELOVITY_I_GAIN*/, const double& D/*=SK_ROBOT_LIB_ODRIVE_VELOVITY_D_GAIN*/)
{
    if( mode == SK_ROBOT_LIB_ODRIVE_MODE_VEL )
    {
        string p(to_string(P));
        string i(to_string(I));
        this->setParam(0, "controller.config.vel_gain", p.c_str());
        this->setParam(0, "controller.config.vel_integrator_gain", i.c_str());
        this->setParam(1, "controller.config.vel_gain", p.c_str());
        this->setParam(1, "controller.config.vel_integrator_gain", i.c_str());
    }
}
