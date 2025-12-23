#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from rclpy.qos import qos_profile_sensor_data

KERNEL_CODE = """
__global__ void bev_kernel(
    float *x, float *y, float *z,
    unsigned char *r, unsigned char *g, unsigned char *b,
    int num_points, int width, int height, float res,
    unsigned char *bev_img)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_points) return;

    int gx = (int)(x[idx] / res + width / 2);
    int gz = (int)(height - z[idx] / res);

    if (gx >= 0 && gx < width && gz >= 0 && gz < height) {
        int offset = (gz * width + gx) * 3;
        bev_img[offset + 0] = b[idx];
        bev_img[offset + 1] = g[idx];
        bev_img[offset + 2] = r[idx];
    }
}
"""

class BEVGridNode(Node):
    def __init__(self):
        super().__init__('bev_grid_node')

        # --------------------------
        # Declare parameters
        # --------------------------
        self.declare_parameter("res", 0.05)
        self.declare_parameter("width", 50)
        self.declare_parameter("height", 60)
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", -1.5) #왼쪽 아래 그리드 시작점 - (self.height * self.res) / 2.0
        self.declare_parameter("dx", -0.34)
        self.declare_parameter("dy", 0.0)
        self.declare_parameter("closing_kernel_size", 1)

        # --------------------------
        # Load parameters
        # --------------------------
        self.res = self.get_parameter("res").value
        self.width = self.get_parameter("width").value
        self.height = self.get_parameter("height").value
        self.origin_x = self.get_parameter("origin_x").value
        self.origin_y = self.get_parameter("origin_y").value
        self.dx = self.get_parameter("dx").value
        self.dy = self.get_parameter("dy").value
        closing_kernel_size = self.get_parameter("closing_kernel_size").value

        # Publishers
        self.pub_img = self.create_publisher(Image, '/bev/image', 10)
        self.pub_occ = self.create_publisher(OccupancyGrid, '/bev/occupancy_grid', 10)

        self.bridge = CvBridge()
        self.closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)

        # CUDA kernel 준비
        self.mod = SourceModule(KERNEL_CODE)
        self.kernel = self.mod.get_function("bev_kernel")

        # Subscribers
        self.create_subscription(
            PointCloud2,
            '/camera/depth_registered/points',
            self.pc_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"BEVGridNode launched with res={self.res}, size=({self.width},{self.height}), "
            f"origin=({self.origin_x},{self.origin_y}), dx={self.dx}, dy={self.dy}"
        )

    def pc_callback(self, msg):
        cloud_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        if cloud_arr.shape[0] == 0:
            return

        # >>> 여기 offset 수정 <<<
        x = np.frombuffer(cloud_arr[:, 0:4].tobytes(),  dtype=np.float32)
        y = np.frombuffer(cloud_arr[:, 4:8].tobytes(),  dtype=np.float32)
        z = np.frombuffer(cloud_arr[:, 8:12].tobytes(), dtype=np.float32)
        rgb_float = np.frombuffer(cloud_arr[:, 12:16].tobytes(), dtype=np.float32)

        rgb_int = rgb_float.view(np.uint32)
        r = ((rgb_int >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_int >> 8)  & 0xFF).astype(np.uint8)
        b = ( rgb_int        & 0xFF).astype(np.uint8)

        num_points = len(x)

        # Debbuging log
        if len(r) != num_points:
            self.get_logger().error(
                f"RGB length mismatch: num_points={num_points}, len(r)={len(r)}"
            )
            return


        # CUDA BEV image
        x_gpu, y_gpu, z_gpu = gpuarray.to_gpu(x), gpuarray.to_gpu(y), gpuarray.to_gpu(z)
        r_gpu, g_gpu, b_gpu = gpuarray.to_gpu(r), gpuarray.to_gpu(g), gpuarray.to_gpu(b)
        bev_gpu = gpuarray.zeros((self.height * self.width * 3), np.uint8)

        block = (256, 1, 1)
        grid = ((num_points + 255)//256, 1)
        self.kernel(x_gpu, y_gpu, z_gpu,
                    r_gpu, g_gpu, b_gpu,
                    np.int32(num_points),
                    np.int32(self.width), np.int32(self.height),
                    np.float32(self.res), bev_gpu,
                    block=block, grid=grid)
        bev = bev_gpu.get().reshape((self.height, self.width, 3))

        # Publish BEV image
        img_msg = self.bridge.cv2_to_imgmsg(bev, encoding="bgr8")
        self.pub_img.publish(img_msg)

        # OccupancyGrid 100으로 초기화
        grid_np = np.full((self.height, self.width), 100, dtype=np.int8) # -1(unknown) 안 가도록 하는게 나을듯?

        # 색 기반 마스크
        mask_obstacle = (r > 100) & (g < 80) & (b < 80) # red
        mask_person = (b > 100) & (r < 80) & (g < 80)  # blue
        mask_free     = (g > 100) & (r < 80) & (b < 80)    # green
        mask_avoid    = (r > 200) & (g > 200) & (b > 200)  # white

        # 좌표 변환
        x_fp, y_fp = z + self.dx, -x + self.dy
        j_grid = ((x_fp - self.origin_x) / self.res).astype(np.int32)
        i_grid = ((y_fp - self.origin_y) / self.res).astype(np.int32)

        valid_mask = (i_grid >= 0) & (i_grid < self.height) & \
                     (j_grid >= 0) & (j_grid < self.width)

        grid_np[i_grid[mask_obstacle & valid_mask],
                j_grid[mask_obstacle & valid_mask]] = 100   #장애물
        grid_np[i_grid[mask_person & valid_mask],
                j_grid[mask_person & valid_mask]] = 88      #사람 -> 88인식표로 씀
        grid_np[i_grid[mask_avoid & valid_mask],
                j_grid[mask_avoid & valid_mask]] = 70       #연석
        grid_np[i_grid[mask_free & valid_mask],
                j_grid[mask_free & valid_mask]] = 0         #feasible 

        occ_mask = (grid_np == 100).astype(np.uint8)
        occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_CLOSE,
                                    self.closing_kernel, iterations=3)
        grid_np[occ_mask > 0] = 100

        # 로봇 주변 free space
        # robot_radius = 0.5  # m
        # j_robot = int((0.0 - self.origin_x) / self.res)
        # i_robot = int((0.0 - self.origin_y) / self.res)

        # r_pix = int(robot_radius / self.res)
        # j_min, j_max = j_robot - r_pix, j_robot + r_pix
        # i_min, i_max = i_robot - r_pix, i_robot + r_pix

        # # 유효 범위 체크
        # j_min, j_max = max(0, j_min), min(self.width, j_max)
        # i_min, i_max = max(0, i_min), min(self.height, i_max)

        # grid_np[i_min:i_max, j_min:j_max] = 0
        # 메시지로 publish

        occ = OccupancyGrid()
        occ.header.stamp = self.get_clock().now().to_msg()
        occ.header.frame_id = "base_link"
        occ.info.resolution = self.res
        occ.info.width = self.width
        occ.info.height = self.height
        occ.info.origin.position.x = self.origin_x
        occ.info.origin.position.y = self.origin_y
        occ.info.origin.orientation.w = 1.0
        occ.data = grid_np.flatten().tolist()
        self.pub_occ.publish(occ)

        self.get_logger().info(f"Published BEV + OccupancyGrid | Points: {num_points}")

def main(args=None):
    rclpy.init(args=args)
    node = BEVGridNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
