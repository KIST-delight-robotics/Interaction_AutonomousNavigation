#include "sk_robot_lib/skServoMD400T.h"

skServoMD400T::skServoMD400T()
: m_gear_ratio(MDH180_GEAR_RATIO), m_fd(-1)
{
}

skServoMD400T::~skServoMD400T()
{
}

bool skServoMD400T::initialize(string port_name, int mode, int number_of_motor = 2)
{
    /*this->m_dxl_comm_result = COMM_TX_FAIL;
    this->m_dxl_error = 0;
    this->m_present_mode = DYXEL_MODE_IDLE;*/

    this->m_number_of_motor = number_of_motor;

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
    cfsetospeed(&this->m_tty, (speed_t)B19200);
    cfsetispeed(&this->m_tty, (speed_t)B19200);
    //cfsetospeed(&m_tty, (speed_t)B57600);
    //cfsetispeed(&m_tty, (speed_t)B57600);
    //cfsetospeed(&m_tty, (speed_t)B115200);
    //cfsetispeed(&m_tty, (speed_t)B115200);

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

    BYTE byChkSum(0);
    int m;
#if 0
    // set baud rate of servo
    memset(m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE_MD400T);
 	this->m_byD[0] = MOTOR_CONTROLLER_MACHINE_ID;      //RMID, 183
    this->m_byD[1] = USER_MACHINE_ID;                  //TMID, 184
    this->m_byD[2] = 1;                               
   	this->m_byD[3] = 135;
    this->m_byD[4] = 2;
   	this->m_byD[5] = 170;
   	//m_byD[6] = 1; // Baud 9600
   	this->m_byD[6] = 2; // Baud 19200
   	//m_byD[6] = 4; // Baud 57600
   	//m_byD[6] = 5; // Baud 115200
 
    byChkSum = 0;
 	for(int i = 0 ; i < 7; i++)
        byChkSum += this->m_byD[i];
  	this->m_byD[7] = ~(byChkSum) + 1;
    this->sendData(8);
#endif

#if 0
    // Set encode
    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE_MD400T);
 	this->m_byD[0] = MOTOR_CONTROLLER_MACHINE_ID;      //RMID, 183
    this->m_byD[1] = 184;//USER_MACHINE_ID;                  //TMID, 184
    this->m_byD[2] = 1;                               
   	this->m_byD[3] = 46;
   	this->m_byD[4] = 1;
   	this->m_byD[5] = 1;
 
    byChkSum = 0;
 	for(int i = 0 ; i < 6; i++)
        byChkSum += this->m_byD[i];
  	this->m_byD[6] = ~(byChkSum) + 1;
    m = this->sendData(7,8);
    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][MD400TServo] Set encode m = %d", m);
    for( int j = 0; j < m; j++ )
        RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][MD400TServo] InData[%d] = %d.", j, this->m_byInData[j]);
#endif

#if 0
    // read encode enable
    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE_MD400T);
 	this->m_byD[0] = MOTOR_CONTROLLER_MACHINE_ID;      //RMID, 183
    this->m_byD[1] = 184;//USER_MACHINE_ID;                  //TMID, 184
    this->m_byD[2] = 1;                               
   	this->m_byD[3] = 4;
   	this->m_byD[4] = 1;
   	this->m_byD[5] = 46;
 
    byChkSum = 0;
 	for(int i = 0 ; i < 6; i++)
        byChkSum += this->m_byD[i];
  	this->m_byD[6] = ~(byChkSum) + 1;
    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][MD400TServo] Requset Encoder enable PID(46)");
    m = this->sendData(7,7);
    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][MD400TServo] Read encode m = %d", m);
    for( int j = 0; j < m; j++ )
        RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][MD400TServo] InData[%d] = %d.", j, this->m_byInData[j]);
#endif

    if( this->p_node )
        RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][MD400TServo] Initialized.");
    return (true);
}

void skServoMD400T::closeServo()
{
    for( unsigned int j = 0; j < this->m_number_of_motor; j++ )
        this->sendCmd(j,0.0);
}

bool skServoMD400T::sendCmd(double *cmd)
{
	BYTE byChkSum(0), byDataNum(7);
	IByte iData;
    int m;

    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE_MD400T);
	this->m_byD[0] = MOTOR_CONTROLLER_MACHINE_ID;      // RMID, 183
	this->m_byD[1] = 184;//128;//USER_MACHINE_ID;            // TMID, 184
    this->m_byD[2] = 1;                                // ID, 1
	this->m_byD[3] = 207;                          // PID, 207
    this->m_byD[4] = 7;                                // Data number
    iData = this->vel2Byte(cmd[0]);
  	this->m_byD[5] = 1;
  	this->m_byD[6] = iData.byLow;                      // Data
	this->m_byD[7] = iData.byHigh;                     // Data
    iData = this->vel2Byte(cmd[1]);
  	this->m_byD[8] = 1;
  	this->m_byD[9] = iData.byLow;                      // Data
	this->m_byD[10] = iData.byHigh;                     // Data
  	this->m_byD[11] = 1;                                // Ask return data

    byChkSum = 0;
  	for(int i = 0 ; i < 12; i++)
        byChkSum += this->m_byD[i];
    this->m_byD[12] = ~(byChkSum) + 1;

    m = this->sendData(13,20);
    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] send m = %d", m);

#if 1 // print return data
    if( m > 0 )
        for( int j = 0; j < m; j++ )
            RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][MD400TServo] InData[%d] = %d.", j, this->m_byInData[j]);
#endif

#if 1
    if( this->m_byD[11] && m > 0 )
    {
        this->m_motor_angle[0] = this->Byte2Pos(this->m_byInData+8);
        this->m_motor_angle[1] = this->Byte2Pos(this->m_byInData+15);
        RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][MD400TServo] Encoder = (%.1f,%.1f).", this->m_motor_angle[0]*RAD2DEG, this->m_motor_angle[1]*RAD2DEG);
    }

#endif

# if 0 // read position
    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE_MD400T);
	this->m_byD[0] = 183;//MOTOR_CONTROLLER_MACHINE_ID;       // RMID, 183
	this->m_byD[1] = 184;//USER_MACHINE_ID;             // TMID, 184
    this->m_byD[2] = 1;                                 // ID, 1
	this->m_byD[3] = 4;                               // PID, 188
  	this->m_byD[4] = 1;
	this->m_byD[5] = 188;
  	this->m_byD[6] = 207;                               // ChkSum

    /*byChkSum = 0;
  	for(int i = 0 ; i < 6; i++)
        byChkSum += this->m_byD[i];
    this->m_byD[6] = ~(byChkSum) + 1;*/
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] this->m_byD[7] = %d.", this->m_byD[7]);

    this->sendData(7);
    m = this->sendData(7,1);
    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] read m = %d.", m);
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] m_byD = [%d,%d,%d,%d'.", this->m_byD[5], this->m_byD[6], this->m_byD[7], this->m_byD[8]);
#endif

    return (true);

}

bool skServoMD400T::sendCmd(const int& j, const double& cmd)
{
    //if( this->p_node )
    //    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][MD400TServo] Joint = %d, vel = %.2f", j, cmd);

	BYTE byChkSum(0), byDataNum(7);
	IByte iData;
    int m;

    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE_MD400T);
	this->m_byD[0] = MOTOR_CONTROLLER_MACHINE_ID;      // RMID, 183
	this->m_byD[1] = 128;//USER_MACHINE_ID;            // TMID, 184
    this->m_byD[2] = 1;                                // ID, 1
	this->m_byD[3] = 130 + j;                          // PID, 207
    this->m_byD[4] = 2;                                // Data number
    iData = this->vel2Byte(cmd);
  	this->m_byD[5] = iData.byLow;                      // Data
	this->m_byD[6] = iData.byHigh;                     // Data

    byChkSum = 0;
  	for(int i = 0 ; i < 7; i++)
        byChkSum += this->m_byD[i];
    this->m_byD[7] = ~(byChkSum) + 1;

    m = this->sendData(8);
    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] send m = %d", m);

    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] m_byInData = %s", m_byInData);
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] SendCmd(%d, %.1f) = (%d,%d).", j, cmd*RAD2DEG, this->m_byD[5], this->m_byD[6]);
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] m_byInData = [%u,%u,%u,%u,%u,%u]", m_byInData[0], m_byInData[1], m_byInData[2], m_byInData[3], m_byInData[4], m_byInData[5]);
# if 1 // read position
    memset(this->m_byD,0,sizeof(BYTE)*MAX_PACKET_SIZE_MD400T);
	this->m_byD[0] = MOTOR_CONTROLLER_MACHINE_ID;       // RMID, 183
	this->m_byD[1] = 128;//USER_MACHINE_ID;             // TMID, 184
    this->m_byD[2] = 1;                                 // ID, 1
	this->m_byD[3] = 188;                               // PID, 188
    this->m_byD[4] = 8;                                 // 8
  	this->m_byD[5] = 0;
	this->m_byD[6] = 0;
  	this->m_byD[7] = 0;
	this->m_byD[8] = 0;
  	this->m_byD[9] = 0;
	this->m_byD[10] = 0;
  	this->m_byD[11] = 0;
	this->m_byD[12] = 0;

    byChkSum = 0;
  	for(int i = 0 ; i < 4; i++)
        byChkSum += this->m_byD[i];
    this->m_byD[4] = ~(byChkSum) + 1;

    this->sendData(5,13);
    m = read(this->m_fd,this->m_byInData,14); /* Read the data                   */
    RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] read m = %d.", m);
    for( int j = 0; j < m; j++ )
        RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][MD400TServo] InData[%d] = %d.", j, this->m_byInData[j]);
    //RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG][ServoMD400T] m_byD = [%d,%d,%d,%d'.", this->m_byD[5], this->m_byD[6], this->m_byD[7], this->m_byD[8]);
#endif

    return (true);
}

void skServoMD400T::setMode(const int& j, const int& mode)
{
}

double skServoMD400T::getVelocity(const int& j) const
{
    return (0.0);
}

double skServoMD400T::getPosition(const int& j) const
{
    return (m_motor_angle[j]);
}

double skServoMD400T::getCurrent(const int& j) const
{
    return (0.0);
}

void skServoMD400T::setGain(const int& mode/*=SK_ROBOT_LIB_SERVO_GAIN_VEL*/, const double& P/*=DYXEL_XM430_VELOCITY_P_GAIN*/, const double& I/*=DYXEL_XM430_VELOCITY_I_GAIN*/, const double& D/*=0.0*/)
{
}

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

IByte skServoMD400T::vel2Byte(const double &v)
{
    const int rpm((int)round(v*60.0/(2.0*M_PI)*this->m_gear_ratio));

	IByte Ret;

	Ret.byLow = rpm & 0xff;
	Ret.byHigh = rpm>>8 & 0xff;

	return (Ret);
}

double skServoMD400T::Byte2Pos(const BYTE* data)
{
    double pos(0.0);
    long encoder(0);

    encoder += (*(data+3));
    encoder *= 255;
    encoder += (*(data+2));
    encoder *= 255;
    encoder += (*(data+1));
    encoder *= 255;
    encoder += (*(data+0));

    //encoder = encoder % (1280 * 4);

    pos = ((double)encoder) / (5542.0*4.0) * (2.0*M_PI);

    trim(pos);
    if( pos < 0.0 )
        pos += 2.0*M_PI;

    return (pos);
}
