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

        # Subscribers
        self.create_subscription(
            PointCloud2,
            '/camera/depth_registered/points',
            self.pc_callback,
            qos_profile_sensor_data
        )

        # Publishers
        self.pub_img = self.create_publisher(Image, '/bev/image', 10)
        self.pub_occ = self.create_publisher(OccupancyGrid, '/bev/occupancy_grid', 10)

        self.bridge = CvBridge()

        # Grid parameters
        self.res = 0.05  # 5 cm
        self.width, self.height = 80, 80  # 10m × 10m grid
        self.origin_x = -0.0 #- (self.width * self.res) / 2.0
        self.origin_y = - (self.height * self.res) / 2.0
        

        # Camera offset (base_footprint 기준)
        self.dx = -0.7
        self.dy = 0.0

        # CUDA kernel 준비
        self.mod = SourceModule(KERNEL_CODE)
        self.kernel = self.mod.get_function("bev_kernel")

        # morphology closing 커널
        self.closing_kernel = np.ones((1,1), np.uint8)

    def pc_callback(self, msg):
        # PointCloud2에서 x,y,z,rgb 추출
        cloud_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        if cloud_arr.shape[0] == 0:
            return

        x = np.frombuffer(cloud_arr[:,0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(cloud_arr[:,4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(cloud_arr[:,8:12].tobytes(), dtype=np.float32)
        rgb_float = np.frombuffer(cloud_arr[:,16:20].tobytes(), dtype=np.float32)
        rgb_int = rgb_float.view(np.uint32)
        r = ((rgb_int >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_int >> 8) & 0xFF).astype(np.uint8)
        b = (rgb_int & 0xFF).astype(np.uint8)

        num_points = len(x)

        # CUDA 실행 (BEV 이미지)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(y)
        z_gpu = gpuarray.to_gpu(z)
        r_gpu = gpuarray.to_gpu(r)
        g_gpu = gpuarray.to_gpu(g)
        b_gpu = gpuarray.to_gpu(b)
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

        # BEV 이미지 publish
        img_msg = self.bridge.cv2_to_imgmsg(bev, encoding="bgr8")
        self.pub_img.publish(img_msg)

        # OccupancyGrid 초기화
        grid_np = np.full((self.height, self.width), 0, dtype=np.int8)

        # 빠른 색 분류 (HSV 변환 제거, 직접 RGB 조건)
        mask_obstacle = (r > 100) & (g < 80) & (b < 80) | \
                        ((b > 100) & (r < 80) & (g < 80)) 
        mask_free     = (g > 100) & (r < 80) & (b < 80)
        mask_avoid    = (r > 200) & (g > 200) & (b > 200) 

        # --- 좌표 변환 (포인트 단위로) ---
        x_fp = z + self.dx
        y_fp = -x + self.dy
        j_grid = ((x_fp - self.origin_x) / self.res).astype(np.int32)
        i_grid = ((y_fp - self.origin_y) / self.res).astype(np.int32)

        valid_mask = (i_grid >= 0) & (i_grid < self.height) & \
                     (j_grid >= 0) & (j_grid < self.width)

        # OccupancyGrid 채우기
        grid_np[i_grid[mask_obstacle & valid_mask],
                j_grid[mask_obstacle & valid_mask]] = 100
        grid_np[i_grid[mask_avoid & valid_mask],
                j_grid[mask_avoid & valid_mask]] = 70
        grid_np[i_grid[mask_free & valid_mask],
                j_grid[mask_free & valid_mask]] = 0

        # morphology closing (빈칸 메우기)
        occ_mask = (grid_np == 100).astype(np.uint8)
        occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_CLOSE,
                                    self.closing_kernel, iterations=1)
        grid_np[occ_mask > 0] = 100

        # 로봇 주변 free space 확보
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
        occ.header.frame_id = "map"
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
