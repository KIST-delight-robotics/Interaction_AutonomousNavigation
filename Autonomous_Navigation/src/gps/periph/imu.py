# ──────────────────────────────────────────────
# IMU Thread
# ──────────────────────────────────────────────
def checkSum(list_data, check_data):
    data = bytearray(list_data)
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for i in range(8):
            if (crc & 1) != 0:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    return hex(((crc & 0xff) << 8) + (crc >> 8)) == hex(check_data[0] << 8 | check_data[1])

def hex_to_ieee(raw_data):
    ieee_data = []
    raw_data.reverse()
    for i in range(0, len(raw_data), 4):
        data2str =hex(raw_data[i] | 0xff00)[4:6] + hex(raw_data[i + 1] | 0xff00)[4:6] + hex(raw_data[i + 2] | 0xff00)[4:6] + hex(raw_data[i + 3] | 0xff00)[4:6]
        ieee_data.append(struct.unpack('>f', bytes.fromhex(data2str))[0])
    ieee_data.reverse()
    return ieee_data

def handleSerialData(raw_data):
    global buff, key, magnetometer, angle_degree, acceleration, angularVelocity, pub_flag
    buff[key] = raw_data

    key += 1
    if buff[0] != 0xaa:
        key = 0
        return
    if key < 3:
        return
    if buff[1] != 0x55:
        key = 0
        return
    if key < buff[2] + 5:  # 根据数据长度位的判断, 来获取对应长度数据
        return

    else:
        data_buff = list(buff.values())  # 获取字典所以 value

        if buff[2] == 0x2c and pub_flag[0]:
            if checkSum(data_buff[2:47], data_buff[47:49]):
                data = hex_to_ieee(data_buff[7:47])
                angularVelocity = data[1:4]
                acceleration = data[4:7]
                magnetometer = data[7:10]
            else:
                print('校验失败')
            pub_flag[0] = False
        elif buff[2] == 0x14 and pub_flag[1]:
            if checkSum(data_buff[2:23], data_buff[23:25]):
                data = hex_to_ieee(data_buff[7:23])
                #angle_degree = data[1:4]
                angle_degree[0] = data[1]
                angle_degree[1] = data[2]
                angle_degree[2] = data[3]
            else:
                print('校验失败')
            pub_flag[1] = False
        else:
            print("该数据处理类没有提供该 " + str(buff[2]) + " 的解析")
            print("或数据错误")
            buff = {}
            key = 0

        buff = {}
        key = 0
        if pub_flag[0] == True or pub_flag[1] == True:
            return
        pub_flag[0] = pub_flag[1] = True
        acc_k = math.sqrt(acceleration[0] ** 2 + acceleration[1] ** 2 + acceleration[2] ** 2)
        """
        print('''
            acc (m/s²)：
                x: %.2f
                y: %.2f
                z: %.2f

            angular (rad/s)：
                x: %.2f
                y: %.2f
                z: %.2f

            欧拉角(°)：
                x: %.2f
                y: %.2f
                z: %.2f

            磁场：
                x: %.2f
                y: %.2f
                z: %.2f
''' % (acceleration[0] * -9.8 / acc_k, acceleration[1] * -9.8 / acc_k, acceleration[2] * -9.8 / acc_k,
       angularVelocity[0], angularVelocity[1], angularVelocity[2],
       angle_degree[0], angle_degree[1], angle_degree[2],
       magnetometer[0], magnetometer[1], magnetometer[2]
      ))
        """

def imu_thread(port='/dev/ttyUSB0', baud=921600):
    imu_ser = serial.Serial(port, baud, timeout=0.01)
    imu_print(f"Listening on {imu_ser.port}@{imu_ser.baudrate}")

    while(True):
        try:
            buff_count = imu_ser.inWaiting()
        except Exception as e:
            print("exception:" + str(e))
            print("imu 失去连接，接触不良，或断线")
            exit(0)
        else:
            if buff_count > 0:
                buff_data = imu_ser.read(buff_count)
                for i in range(0, buff_count):
                    handleSerialData(buff_data[i])
 
