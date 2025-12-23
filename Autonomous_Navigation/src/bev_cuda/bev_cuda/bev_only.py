#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import numpy as np
import struct
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from rclpy.qos import qos_profile_sensor_data

KERNEL_CODE = """
__global__ void bev_kernel(
    float *x, float *y, float *z, unsigned char *r, unsigned char *g, unsigned char *b,
    int num_points, int width, int height, float res, unsigned char *bev_img)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_points) return;

    // BEV 평면: X (좌우), Z (전방)
    int gx = (int)(x[idx] / res + width / 2);
    int gz = (int)(height - z[idx] / res);

    if (gx >= 0 && gx < width && gz >= 0 && gz < height) {
        // dilation: 반경 2픽셀(5x5 영역)에 걸쳐 색상 적용
        for (int dx=-5; dx<=5; dx++) {
            for (int dy=-5; dy<=5; dy++) {
                int nx = gx + dx;
                int ny = gz + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int offset = (ny * width + nx) * 3;
                    bev_img[offset + 0] = b[idx];
                    bev_img[offset + 1] = g[idx];
                    bev_img[offset + 2] = r[idx];
                }
            }
        }
    }
}
"""

class BEVNode(Node):
    def __init__(self):
        super().__init__('bev_node')

        self.create_subscription(
            PointCloud2,
            '/camera/depth_registered/points',
            self.pc_callback,
            qos_profile_sensor_data
        )

        self.pub = self.create_publisher(Image, '/bev/image', 10)
        self.bridge = CvBridge()

        self.res = 0.005  # 2cm per pixel
        self.width, self.height = 1000, 1000  # 0.8m x 0.8m

        self.mod = SourceModule(KERNEL_CODE)
        self.kernel = self.mod.get_function("bev_kernel")

    def pc_callback(self, msg):
        # PointCloud2 raw buffer → numpy
        cloud_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)

        # 좌표 추출 (offset은 보통 x:0, y:4, z:8, rgb:16) 
        x = np.frombuffer(cloud_arr[:,0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(cloud_arr[:,4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(cloud_arr[:,8:12].tobytes(), dtype=np.float32)
        rgb_float = np.frombuffer(cloud_arr[:,16:20].tobytes(), dtype=np.float32)

        rgb_int = rgb_float.view(np.uint32)
        r = ((rgb_int >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_int >> 8) & 0xFF).astype(np.uint8)
        b = (rgb_int & 0xFF).astype(np.uint8)

        num_points = len(x)
        if num_points == 0:
            return

        # GPU 전송
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(y)
        z_gpu = gpuarray.to_gpu(z)
        r_gpu = gpuarray.to_gpu(r)
        g_gpu = gpuarray.to_gpu(g)
        b_gpu = gpuarray.to_gpu(b)
        bev_gpu = gpuarray.zeros((self.height * self.width * 3), np.uint8)

        # CUDA 실행
        block = (256, 1, 1)
        grid = ((num_points + 255)//256, 1)
        self.kernel(
            x_gpu, y_gpu, z_gpu, r_gpu, g_gpu, b_gpu,
            np.int32(num_points), np.int32(self.width), np.int32(self.height),
            np.float32(self.res), bev_gpu,
            block=block, grid=grid
        )

        # CPU로 가져오기
        bev = bev_gpu.get().reshape((self.height, self.width, 3))

        # ROS2 퍼블리시
        img_msg = self.bridge.cv2_to_imgmsg(bev, encoding="bgr8")
        self.pub.publish(img_msg)

        self.get_logger().info(f"Published BEV image vvv | Points: {num_points}")

def main(args=None):
    rclpy.init(args=args)
    node = BEVNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
