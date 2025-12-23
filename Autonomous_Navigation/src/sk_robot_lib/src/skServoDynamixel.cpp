#include "sk_robot_lib/skServoDynamixel.h"

skServoDynamixel::skServoDynamixel()
: m_dxl_comm_result(COMM_TX_FAIL), m_dxl_error(0)
{
}

skServoDynamixel::skServoDynamixel(const int& address, const int& param)
: m_dxl_comm_result(COMM_TX_FAIL), m_dxl_error(0)
{
    this->setting(address, param);
}

skServoDynamixel::~skServoDynamixel()
{
}

void skServoDynamixel::setting(const int& address, const int& param)
{
    if( address == 430 )
    {
        this->setAddress430();
        this->m_protocol = DYXEL_PROTOCOL_VERSION_XM;
        this->m_baudrate = DYXEL_BAUDRATE_XM;
        this->m_type = 'X';
        if( param == 350 )
            this->setParam350();
        else if( param == 210 )
            this->setParam210();
        else if( param == 150 )
            this->setParam150();
    }
    else if( address == 12 )
    {
        this->setAddressAX12();
        this->m_protocol = DYXEL_PROTOCOL_VERSION_AX;
        this->m_baudrate = DYXEL_BAUDRATE_AX;
        this->m_type = 'A';
        if( param == 'w' || param == 'W' )
            this->setParamW();
    }
}

void skServoDynamixel::setAddress430()
{
    this->m_add_drive_mode = DYXEL_XM430_ADDR_DRIVE_MODE;
    this->m_add_operating_mode = DYXEL_XM430_ADDR_OPERATING_MODE;
    this->m_add_torque_enable = DYXEL_XM430_ADDR_TORQUE_ENABLE;
    this->m_add_goal_current = DYXEL_XM430_ADDR_GOAL_CURRENT;
    this->m_add_goal_velocity = DYXEL_XM430_ADDR_GOAL_VELOCITY;
    this->m_add_goal_position = DYXEL_XM430_ADDR_GOAL_POSITION;
    this->m_add_present_current = DYXEL_XM430_ADDR_PRESENT_CURRENT;
    this->m_add_present_velocity = DYXEL_XM430_ADDR_PRESENT_VELOCITY;
    this->m_add_present_position = DYXEL_XM430_ADDR_PRESENT_POSITION;
    this->m_add_velocity_gain_P = DYXEL_XM430_ADDR_VELOCITY_P_GAIN;
    this->m_add_velocity_gain_I = DYXEL_XM430_ADDR_VELOCITY_I_GAIN;
}

void skServoDynamixel::setParam350()
{
    this->m_pulse2deg = DYXEL_XM430_W350_DEG_PER_PULSE;
    this->m_deg2pulse = 1.0 / this->m_pulse2deg;
    this->m_pulse2rad = this->m_pulse2deg*DEG2RAD;
    this->m_rad2pulse = 1.0 / this->m_pulse2rad;

    this->m_pulse2vel = DYXEL_XM430_W350_VELOCITY_PER_PULSE;
    this->m_vel2pulse = 1.0 / this->m_pulse2vel;

    this->m_pulse2tor = DYXEL_XM430_W350_TORQUE_PER_CURRENT * DYXEK_XM430_W350_CURRENT_PER_PULSE;
    this->m_tor2pulse = 1.0 / this->m_pulse2tor;
}

void skServoDynamixel::setParam210() // need to update
{
    this->m_pulse2deg = DYXEL_XM430_W350_DEG_PER_PULSE;
    this->m_deg2pulse = 1.0 / this->m_pulse2deg;
    this->m_pulse2rad = this->m_pulse2deg*DEG2RAD;
    this->m_rad2pulse = 1.0 / this->m_pulse2rad;

    this->m_pulse2vel = DYXEL_XM430_W350_VELOCITY_PER_PULSE;
    this->m_vel2pulse = 1.0 / this->m_pulse2vel;

    this->m_pulse2tor = DYXEL_XM430_W350_TORQUE_PER_CURRENT * DYXEK_XM430_W350_CURRENT_PER_PULSE;
    this->m_tor2pulse = 1.0 / this->m_pulse2tor;
}

void skServoDynamixel::setParam150() // need to update
{
    this->m_pulse2deg = DYXEL_XM430_W350_DEG_PER_PULSE;
    this->m_deg2pulse = 1.0 / this->m_pulse2deg;
    this->m_pulse2rad = this->m_pulse2deg*DEG2RAD;
    this->m_rad2pulse = 1.0 / this->m_pulse2rad;

    this->m_pulse2vel = DYXEL_XM430_W350_VELOCITY_PER_PULSE;
    this->m_vel2pulse = 1.0 / this->m_pulse2vel;
}

void skServoDynamixel::setAddressAX12()
{
    this->m_add_torque_enable = DYXEL_AX12_ADDR_TORQUE_ENABLE;
    this->m_add_goal_velocity = DYXEL_AX12_ADDR_GOAL_VELOCITY;
    this->m_add_goal_position = DYXEL_AX12_ADDR_GOAL_POSITION;
    this->m_add_present_velocity = DYXEL_AX12_ADDR_PRESENT_VELOCITY;
    this->m_add_present_position = DYXEL_AX12_ADDR_PRESENT_POSITION;
    this->m_add_present_current = DYXEL_AX12_ADDR_PRESENT_LOAD;
}

void skServoDynamixel::setParamW()
{
    this->m_pulse2deg = DYXEL_AX12_W_DEG_PER_PULSE;
    this->m_deg2pulse = 1.0/m_pulse2deg;
    this->m_pulse2rad = this->m_pulse2deg*DEG2RAD;
    this->m_rad2pulse = 1.0/m_pulse2rad;

    this->m_pulse2vel = DYXEL_AX12_W_RADSEC_PER_PULSE;
    this->m_vel2pulse = 1.0 / this->m_pulse2vel;

    this->m_pulse2tor = DYXEL_AX12_W_TORQUE_PER_PULSE;
    this->m_tor2pulse = 1.0 / this->m_pulse2tor;
}

bool skServoDynamixel::initialize(string port_name, int mode, int number_of_motor = 1)
{
    /*this->m_dxl_comm_result = COMM_TX_FAIL;
    this->m_dxl_error = 0;
    this->m_present_mode = DYXEL_MODE_IDLE;*/

    this->m_number_of_motor = number_of_motor;
    this->m_mode_current.resize(this->m_number_of_motor,DYXEL_MODE_IDLE);
    this->m_pulse.resize(this->m_number_of_motor,0);

    this->p_portHandler = dynamixel::PortHandler::getPortHandler(port_name.c_str());
    this->p_packetHandler = dynamixel::PacketHandler::getPacketHandler(this->m_protocol);

    // Open port
    if(this->p_portHandler->openPort())
    {
        if( this->p_node )
            RCLCPP_INFO(this->p_node->get_logger(), "Succeeded to open the dynamixel port.");
    }
    else
    {
        if( this->p_node )
            RCLCPP_ERROR(this->p_node->get_logger(), "Failed to open the dynamixel port!");
        return (false);
    }

    // Set port baudrate
    if (this->p_portHandler->setBaudRate(this->m_baudrate))
    {
        if( this->p_node )
            RCLCPP_INFO(this->p_node->get_logger(), "Succeeded to set the baudrate to %d.", p_portHandler->getBaudRate());
    }
    else
    {
        if( this->p_node )
            RCLCPP_ERROR(this->p_node->get_logger(), "Failed to change the baudrate!");
        return (false);
    }

    // Turn on motors
    int error(0);
    bool keep_waiting;
    uint8_t data;

    // Set mode of motors and turn on
    for( int j = 0; j < this->m_number_of_motor; j++ )
    {
        // turn off motor
        keep_waiting = true;
        while( keep_waiting )
        {
            this->m_dxl_comm_result = this->p_packetHandler->write1ByteTxRx(this->p_portHandler, (j+1), this->m_add_torque_enable, DYXEL_TORQUE_DISABLE);//, &m_dxl_error);
            if( this->m_dxl_comm_result != COMM_SUCCESS)
            {
                if( this->p_node )
                    RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with comm_result = %d.", j+1, this->m_dxl_comm_result);
                error++;
            }
            else if( this->m_dxl_error != 0 )
            {
                if( this->p_node )
                    RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with error = %d.", j+1,this->m_dxl_error);
                error++;
            }
            else
            {
                if( this->p_node )
                    RCLCPP_INFO(this->p_node->get_logger(), "Dynamixel of motor id %d has been successfully disabled.", j+1);
            }
            this->m_dxl_comm_result = this->p_packetHandler->read1ByteTxRx(this->p_portHandler, (j+1), this->m_add_torque_enable, (uint8_t*)&data);
            if( this->m_dxl_comm_result != COMM_SUCCESS )
            {
                if( this->p_node )
                    RCLCPP_ERROR(this->p_node->get_logger(), "Failed to read torque state with comm_result = %d.", this->m_dxl_comm_result);
            }
            if( data == DYXEL_TORQUE_DISABLE )
                keep_waiting = false;
        }

        // set mode
        if( this-> m_type == 'X' )
        {
            keep_waiting = true;
            while( keep_waiting )
            {
                this->m_dxl_comm_result = this->p_packetHandler->write1ByteTxRx(this->p_portHandler, (j+1), this->m_add_operating_mode, mode);//, &m_dxl_error);
                if (this->m_dxl_comm_result != COMM_SUCCESS)
                {
                    if( this->p_node )
                        RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with comm_result = %d.", j+1, this->m_dxl_comm_result);
                    error++;
                }
                else if (this->m_dxl_error != 0)
                {
                    if( this->p_node )
                        RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with error = %d.", j+1, this->m_dxl_error);
                    error++;
                }
                else
                {
                    if( this->p_node )
                        RCLCPP_INFO(this->p_node->get_logger(), "Dynamixel of motor id %d has been successfully setted to mode %d.", j+1, mode);
                }
                keep_waiting = false;
            }
        }

        // set goal zero for safety
        this->m_dxl_comm_result = COMM_SUCCESS;
        this->m_dxl_error = 0; 
        switch( mode )
        {
            case DYXEL_MODE_POSITION :
                this->m_dxl_comm_result = this->p_packetHandler->write2ByteTxRx(this->p_portHandler, j+1, this->m_add_goal_position, 0);
                this->m_add_goal_present = this->m_add_goal_position;
                break;
            case DYXEL_MODE_CURRENT :
                this->m_dxl_comm_result = this->p_packetHandler->write2ByteTxRx(this->p_portHandler, j+1, this->m_add_goal_current, 0);
                this->m_add_goal_present = this->m_add_goal_current;
                break;
            case DYXEL_MODE_VELOCITY :
                if( this->m_type == 'X' )
                {
                    this->m_dxl_comm_result = this->p_packetHandler->write4ByteTxRx(this->p_portHandler, j+1, this->m_add_goal_velocity, 0);
                }
                else if( this->m_type == 'A' )
                {
                    this->m_dxl_comm_result = this->p_packetHandler->write2ByteTxRx(this->p_portHandler, j+1, this->m_add_goal_velocity, 0);
                }
                this->m_add_goal_present = this->m_add_goal_velocity;
                break;
        }
        if( this->m_dxl_comm_result != COMM_SUCCESS )
        {
            if( this->p_node )
                RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with comm_result = %d.", j+1, this->m_dxl_comm_result);
            error++;
        }
        else if( this->m_dxl_error != 0 )
        {
            if( this->p_node )
                RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with error = %d.", j+1, this->m_dxl_error);
            error++;
        }
        else
        {
            if( this->p_node )
                RCLCPP_INFO(this->p_node->get_logger(), "Dynamixel of motor id %d has been successfully setted the initial goal to be zero.", j+1);
        }

        // turn on motor
        keep_waiting = true;
        while( keep_waiting )
        {
            this->m_dxl_comm_result = this->p_packetHandler->write1ByteTxRx(this->p_portHandler, j+1, this->m_add_torque_enable, DYXEL_TORQUE_ENABLE);//, &m_dxl_error);
            if (this->m_dxl_comm_result != COMM_SUCCESS)
            {
                if( this->p_node )
                    RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with comm_result = %d.", j+1, this->m_dxl_comm_result);
                error++;
            }
            else if (this->m_dxl_error != 0)
            {
                if( this->p_node )
                    RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with error = %d.", j+1, this->m_dxl_error);
                error++;
            }
            else
            {
                if( this->p_node )
                    RCLCPP_INFO(this->p_node->get_logger(), "Dynamixel of motor id %d has been successfully enabled.", j+1);
            }
            sleep(0.1);
            this->m_dxl_comm_result = this->p_packetHandler->read1ByteTxRx(this->p_portHandler, (j+1), this->m_add_torque_enable, (uint8_t*)&data);
            if (this->m_dxl_comm_result != COMM_SUCCESS)
            {
                if( this->p_node )
                    RCLCPP_ERROR(this->p_node->get_logger(), "Failed to read torque state with comm_result = %d.", this->m_dxl_comm_result);
            }
            if( data == DYXEL_TORQUE_ENABLE )
                keep_waiting = false;
        }
        this->m_mode_current[j] = mode;
        if( this->p_node )
            RCLCPP_INFO(this->p_node->get_logger(), "[DEBUG] this->m_mode_current[%d] = %d.", j, this->m_mode_current[j]);
    }
    if( error )
        return (false);        

    // read current angle
/*    m_dxl_comm_result = p_packetHandler->read4ByteTxRx(p_portHandler, DOOR_BOTTOM_ID, ADDR_PRO_PRESENT_POSITION, (uint32_t*)&current_pulse);
    if (m_dxl_comm_result != COMM_SUCCESS)
    {
        RCLCPP_ERROR("Failed to read current pulse with comm_result = %d.", m_dxl_comm_result);
    }
*/
    return (true);
}

void skServoDynamixel::closeServo()
{
    // Disable All motors
    for( int j = 0; j < this->m_number_of_motor; j++ )
    {
        this->m_dxl_comm_result = this->p_packetHandler->write1ByteTxRx(this->p_portHandler, j+1, this->m_add_torque_enable, DYXEL_TORQUE_DISABLE);//, &m_dxl_error);
        if (this->m_dxl_comm_result != COMM_SUCCESS || this->m_dxl_error != 0)
        {
            if( this->p_node )
                RCLCPP_ERROR(this->p_node->get_logger(), "Failed to servo off motor id %d with comm_result = %d, m_dxl_error = %d.", j+1, this->m_dxl_comm_result, this->m_dxl_error);
        }
    }

    // Close port Head
    this->p_portHandler->closePort();
}

bool skServoDynamixel::sendCmd(const int& j, const double& cmd)
{
    int32_t data4;
    int16_t data2;
    uint16_t add_goal;
    int data_length(4);
    switch( this->m_mode_current[j] )
    {
        case DYXEL_MODE_POSITION :
            add_goal = this->m_add_goal_position;
            data4 = this->rad2pulse(cmd); // need to change the sign
            break;
        case DYXEL_MODE_VELOCITY :
            add_goal = this->m_add_goal_velocity;
            if( this->m_type == 'X' )
            {
                data4 = this->vel2pulse(cmd); // need to change the sign
            }
            else if( this->m_type == 'A' )
            {
                data2 = this->vel2pulse(cmd); // need to change the sign
                data_length = 2;
                //RCLCPP_INFO("[DEBUG/DYNAMIXEL] %d : %.2f ==> %d.", j+1, cmd, data2);
            }
            //RCLCPP_INFO("[DEBUG/DYNAMIXEL] %d : %.2f ==> %d.", j+1, cmd[j], data);
            break;
        case DYXEL_MODE_CURRENT :
            add_goal = this->m_add_goal_current;
            data2 = this->tor2pulse(cmd); // need to change the sign
            data_length = 2;
            //RCLCPP_INFO("[DEBUG][DYNAMIXEL](sendCmd) (j,cmd,data) = (%d, %.3f, %d).", j, cmd, data2);
            break;
        default :
            if( this->p_node )
                RCLCPP_ERROR(this->p_node->get_logger(), "The current control mode is wrong with %d.", this->m_mode_current[j]);
            return (false);
    }
    // send cmd
    if( data_length == 4)
    {
        this->m_dxl_comm_result = this->p_packetHandler->write4ByteTxRx(this->p_portHandler, j+1, add_goal, data4);//, &m_dxl_error);
        this->m_pulse[j] = data4;
    }
    else if( data_length == 2 )
    {
        this->m_dxl_comm_result = this->p_packetHandler->write2ByteTxRx(this->p_portHandler, j+1, add_goal, data2);//, &m_dxl_error);
        this->m_pulse[j] = data2;
    }
    if (this->m_dxl_comm_result != COMM_SUCCESS || this->m_dxl_error != 0)
    {
        if( this->p_node )
            RCLCPP_ERROR(this->p_node->get_logger(), "Failed to send command to motor id %d with comm_result = %d, m_dxl_error = %d.", j+1, this->m_dxl_comm_result, this->m_dxl_error);
    }

    //RCLCPP_INFO("[DEBUG][ServoDynamixel] Mode = %d, SendCmd(%d,%.f).", this->m_mode_current[j], j, cmd*RAD2DEG);
    return (true);
}

void skServoDynamixel::setMode(const int& j, const int& mode)
{
    bool keep_waiting(true);

    if( this->m_type == 'A' )
        keep_waiting = false;
    // set mode
    while( keep_waiting )
    {
        this->m_dxl_comm_result = this->p_packetHandler->write1ByteTxRx(this->p_portHandler, (j+1), this->m_add_operating_mode, mode);//, &m_dxl_error);
        if (this->m_dxl_comm_result != COMM_SUCCESS)
        {
            if( this->p_node )
                RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with comm_result = %d.", j+1, this->m_dxl_comm_result);
        }
        else if (this->m_dxl_error != 0)
        {
            if( this->p_node )
                RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with error = %d.", j+1, this->m_dxl_error);
        }
        else
        {
            if( this->p_node )
                RCLCPP_INFO(this->p_node->get_logger(), "Dynamixel of motor id %d has been successfully setted to mode %d.", j+1, mode);
        }
        keep_waiting = false;
    }
    this->m_mode_current[j] = mode;
}

double skServoDynamixel::getVelocity(const int& j) const
{
    int32_t data4;
    int16_t data2;
    int dxl_comm_result;

    if( this->m_type == 'X' )
    {
        dxl_comm_result = this->p_packetHandler->read4ByteTxRx(this->p_portHandler, j+1, this->m_add_present_velocity, (uint32_t*)&data4);//, &m_dxl_error);
    }
    else if( this->m_type == 'A' )
    {
        dxl_comm_result = this->p_packetHandler->read2ByteTxRx(this->p_portHandler, j+1, this->m_add_present_velocity, (uint16_t*)&data2);//, &m_dxl_error);
    }
    if( dxl_comm_result != COMM_SUCCESS )
    {
        if( this->p_node )
            RCLCPP_ERROR(this->p_node->get_logger(), "Failed to read velocity of motor id %d with comm_result = %d.", j+1, dxl_comm_result);
        return (0.0);
    }

    if( this->m_type == 'X' )
    {
        return (this->pulse2vel(data4));
    }
    else if( this->m_type == 'A' )
    {
        return (this->pulse2vel(data2));
    }
    return (0.0);
}

double skServoDynamixel::getPosition(const int& j) const
{
    int32_t data4;
    int16_t data2;
    int dxl_comm_result;

    if( this->m_type == 'X' )
    {
        dxl_comm_result = this->p_packetHandler->read4ByteTxRx(this->p_portHandler, j+1, this->m_add_present_position, (uint32_t*)&data4);//, &m_dxl_error);
    }
    else if( this->m_type == 'A' )
    {
        dxl_comm_result = this->p_packetHandler->read2ByteTxRx(this->p_portHandler, j+1, this->m_add_present_position, (uint16_t*)&data2);//, &m_dxl_error);
    }
    if( dxl_comm_result != COMM_SUCCESS )
    {
        if( this->p_node )
            RCLCPP_ERROR(this->p_node->get_logger(), "Failed to read velocity of motor id %d with comm_result = %d.", j+1, dxl_comm_result);
        return (0.0);
    }

    if( this->m_type == 'X' )
    {
        return (this->pulse2rad(data4));
    }
    else if( this->m_type == 'A' )
    {
        return (this->pulse2rad(data2));
    }
    return (0.0);
}

double skServoDynamixel::getCurrent(const int& j) const
{
    int16_t data2;
    int dxl_comm_result;

    dxl_comm_result = this->p_packetHandler->read2ByteTxRx(this->p_portHandler, j+1, this->m_add_present_current, (uint16_t*)&data2);//, &m_dxl_error);
    if( dxl_comm_result != COMM_SUCCESS )
    {
        if( this->p_node )
            RCLCPP_ERROR(this->p_node->get_logger(), "Failed to read velocity of motor id %d with comm_result = %d.", j+1, dxl_comm_result);
        return (0.0);
    }

    return (this->pulse2rad(data2));
}

// private functions for unit convertion
int32_t skServoDynamixel::deg2pulse(const double& deg) const
{
    return ((int32_t)round(this->m_deg2pulse*deg));
}

int32_t skServoDynamixel::rad2pulse(const double& rad) const
{
    return ((int32_t)round(this->m_rad2pulse*rad));
}

double skServoDynamixel::pulse2deg(const int& pulse) const
{
    return (this->m_pulse2deg*(double)pulse);
}

double skServoDynamixel::pulse2rad(const int& pulse) const
{
    return (this->m_pulse2rad*(double)pulse);
}

int32_t skServoDynamixel::vel2pulse(const double& vel) const
{
    if( this->m_type == 'A' )
    {
        if( vel < 0.0 )
            return ((int32_t)round(this->m_vel2pulse*(-vel))+1024);
        else
            return ((int32_t)round(this->m_vel2pulse*vel));
    }
    return ((int32_t)round(this->m_vel2pulse*vel));
}

double skServoDynamixel::pulse2vel(const int& pulse) const
{
    if( this->m_type == 'A' )
    {
        if( pulse > 1024 )
            return (this->m_pulse2vel*(double)(1024-pulse));
        else
            return (this->m_pulse2vel*(double)pulse);
    }
    return (this->m_pulse2vel*(double)pulse);
}

int16_t skServoDynamixel::tor2pulse(const double& tor) const
{
    return ((int16_t)(this->m_tor2pulse*tor));
    //return ((int16_t)(tor*this->m_ration_torque_to_pulse));
    //return ((int16_t)((tor*DYXEL_XM430_W350_TORQUE_TO_CURRENT_A+(tor > 0.0 ? 1.0 : -1.0)*DYXEL_XM430_W350_TORQUE_TO_CURRENT_B)/DYXEK_XM430_W350_CURRENT_PER_PULSE));
    //return ((int32_t)(tor/(DYXEL_XM430_W350_TORQUE_PER_CURRENT*DYXEK_XM430_W350_CURRENT_PER_PULSE)));
}

double skServoDynamixel::pulse2tor(const int& pulse) const
{
    if( this->m_type == 'A' )
    {
        if( pulse > 1024 )
            return (this->m_pulse2tor*(double)(1024-pulse));
        else
            return (this->m_pulse2tor*(double)pulse);
    }
    return (this->m_pulse2tor*(double)pulse);

}


void skServoDynamixel::setGain(const int& mode/*=SK_ROBOT_LIB_SERVO_GAIN_VEL*/, const double& P/*=DYXEL_XM430_VELOCITY_P_GAIN*/, const double& I/*=DYXEL_XM430_VELOCITY_I_GAIN*/, const double& D/*=0.0*/)
{
    uint8_t gainP(P), gainI(I), gainD(D);
    gainP = (uint8_t)P;
    if( this->p_node )
        RCLCPP_INFO(this->p_node->get_logger(), "P = %.3f.", P);
    if( mode == SK_ROBOT_LIB_SERVO_GAIN_VEL)
    {
        for( uint8_t j = 0; j < this->m_number_of_motor; j++ )
        {
            this->m_dxl_comm_result = this->p_packetHandler->write2ByteTxRx(this->p_portHandler, (j+1), this->m_add_velocity_gain_P, gainP);// Velocity P Gain. default 100
            if( this->m_dxl_comm_result != COMM_SUCCESS || this->m_dxl_error != 0 )
            {
                if( this->p_node )
                    RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with comm_result = %d, dxl_error = %d.", j+1, this->m_dxl_comm_result, this->m_dxl_error);
            }
            else
            {
                if( this->p_node )
                    RCLCPP_INFO(this->p_node->get_logger(), "Dynamixel of motor id %d has been successfully set Velocity P Gain = %u.", j+1, gainP);
            }
            this->m_dxl_comm_result = this->p_packetHandler->write2ByteTxRx(this->p_portHandler, (j+1), this->m_add_velocity_gain_I, gainI);// Velocity I Gain, default 1920
            if( this->m_dxl_comm_result != COMM_SUCCESS || this->m_dxl_error != 0 )
            {
                if( this->p_node )
                    RCLCPP_ERROR(this->p_node->get_logger(), "Failed to connect motor id %d with comm_result = %d, dxl_error = %d.", j+1, this->m_dxl_comm_result, this->m_dxl_error);
            }
            else
            {
                if( this->p_node )
                    RCLCPP_INFO(this->p_node->get_logger(), "Dynamixel of motor id %d has been successfully set Velocity I Gain = %u.", j+1, gainI);
            }
/*
            m_dxl_comm_result = p_packetHandler->write4ByteTxRx(p_portHandler, (j+1), DYXEL_XM430_ADDR_VELOCITY_LIMIT, DYXEL_XM430_W350_MAX_VELOCITY_PULSE);// Velocity I Gain, default 1920
            if( m_dxl_comm_result != COMM_SUCCESS || m_dxl_error != 0 )
            {
                RCLCPP_ERROR("Failed to connect motor id %d with comm_result = %d, dxl_error = %d.", j+1, m_dxl_comm_result, m_dxl_error);
                error++;
            }
            else
            {
                RCLCPP_INFO("Dynamixel of motor id %d has been successfully set Velocity Limit.", j+1);
            }
*/
        }
    }
}
