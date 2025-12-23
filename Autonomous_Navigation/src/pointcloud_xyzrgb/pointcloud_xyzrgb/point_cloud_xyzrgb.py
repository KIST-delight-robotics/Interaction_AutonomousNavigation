#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import time

from .pointcloud_gpu_xyzrgb import depth_rgb_to_xyzrgb_gpu


def stamp_to_float(stamp) -> float:
    """rclpy Time -> float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class PointCloudGPUNode(Node):
    def __init__(self):
        super().__init__("pointcloud_gpu_node")

        self.bridge = CvBridge()

        # ---- Parameters ----
        self.declare_parameter("range_max", 3.0)  # 전방 거리
        self.declare_parameter("stride", 2)       # 건너 뛸 셀
        self.declare_parameter("sync_slop", 0.05) # [s]

        qos = rclpy.qos.QoSProfile(depth=10)

        # ---- Latest messages for manual sync ----
        self.last_depth = None   # (t, msg)
        self.last_rgb = None    
        self.last_info = None    
        self.last_sync_time = None

        # ---- Subscribers ----
        self.sub_depth = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_cb,
            qos,
        )
        self.sub_rgb = self.create_subscription(
            Image,
            "/hugging/output/image",
            self.rgb_cb,
            qos,
        )
        self.sub_info = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.info_cb,
            qos,
        )

        # ---- Publisher ----
        self.pub_cloud = self.create_publisher(
            PointCloud2,
            "/camera/depth_registered/points",
            qos,
        )

        self.get_logger().info("PointCloudGPUNode (manual sync) started.")

    # =========================
    #   Individual Callbacks
    # =========================
    def depth_cb(self, msg: Image):
        t = stamp_to_float(msg.header.stamp)
        self.last_depth = (t, msg)
        self.try_sync()

    def rgb_cb(self, msg: Image):
        t = stamp_to_float(msg.header.stamp)
        self.last_rgb = (t, msg)
        self.try_sync()

    def info_cb(self, msg: CameraInfo):
        t = stamp_to_float(msg.header.stamp)
        self.last_info = (t, msg)
        self.try_sync()

    # =========================
    #     Manual Synchronizer
    # =========================
    def try_sync(self):
        if (
            self.last_depth is None
            or self.last_rgb is None
            or self.last_info is None
        ):
            return

        t_d, depth_msg = self.last_depth
        t_r, rgb_msg = self.last_rgb
        t_i, info_msg = self.last_info

        ts = [t_d, t_r, t_i]
        t_min = min(ts)
        t_max = max(ts)

        slop = float(
            self.get_parameter("sync_slop").get_parameter_value().double_value
        )

        # 1) 시간 되감김 감지
        if self.last_sync_time is not None and t_max < self.last_sync_time - 1e-3:
            self.get_logger().warn(
                f"Time jumped backwards (last_sync_time={self.last_sync_time:.3f}, "
                f"new={t_max:.3f}), reset sync buffers."
            )
            self.last_depth = None
            self.last_rgb = None
            self.last_info = None
            self.last_sync_time = None
            return

        # 2) 세 토픽의 시간 차이가 slop 안에 들어오면 한 번 처리
        if t_max - t_min <= slop:
            self.last_sync_time = t_max
            self.last_depth = None
            self.last_rgb = None
            self.last_info = None

            self.process_triplet(depth_msg, rgb_msg, info_msg)

    # =========================
    #    Core Processing
    # =========================
    def process_triplet(self, depth_msg, rgb_msg, info_msg):
        """depth/rgb/info가 잘 맞춰졌을 때 포인트클라우드 생성"""
        try:
            t0 = time.time()

            # ---- 1) depth → numpy ----
            depth_np = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )

            if depth_msg.encoding in ["16UC1", "mono16"]:
                depth_np = depth_np.astype(np.float32) * 0.001  # mm → m
            elif depth_msg.encoding == "32FC1":
                depth_np = depth_np.astype(np.float32)
            else:
                self.get_logger().warn(
                    f"Unexpected depth encoding: {depth_msg.encoding}, "
                    "treating as float32 meters."
                )
                depth_np = depth_np.astype(np.float32)

            H, W = depth_np.shape

            # ---- 2) rgb → numpy ----
            rgb_np = self.bridge.imgmsg_to_cv2(rgb_msg, rgb_msg.encoding)

            # ---- 3) intrinsics ----
            fx = info_msg.k[0]
            fy = info_msg.k[4]
            cx = info_msg.k[2]
            cy = info_msg.k[5]

            t1 = time.time()

            # ---- 4) GPU kernel ----
            range_max = float(
                self.get_parameter("range_max").get_parameter_value().double_value
            )

            cloud4_np = depth_rgb_to_xyzrgb_gpu(
                depth_np,
                rgb_np.astype(np.uint8, copy=False),
                fx,
                fy,
                cx,
                cy,
                encoding=rgb_msg.encoding,
                range_max=range_max,
            )

            t2 = time.time()

            # ---- 5) stride ----
            stride = int(
                self.get_parameter("stride")
                .get_parameter_value()
                .integer_value
            )
            if stride > 1:
                cloud4_np = cloud4_np[::stride, ::stride, :]

            # ---- 5.5) 거리 기반 필터 ----
            max_range = float(
                self.get_parameter("range_max").get_parameter_value().double_value
            )

            flat = cloud4_np.reshape(-1, 4)  # (H*W, 4)
            mask = (flat[:, 2] > 0.0) & (flat[:, 2] <= max_range)
            kept = flat[mask]               # (N_kept, 4)

            if kept.shape[0] == 0:
                self.get_logger().warn("No valid points within range_max; skip frame.")
                return

            # ---- 6) numpy → PointCloud2  ----
            cloud_msg = self.cloud4_to_pointcloud2_unorganized(
                depth_msg.header, kept
            )

            t3 = time.time()

            # ---- 7) Publish ----
            self.pub_cloud.publish(cloud_msg)

            t4 = time.time()

            self.get_logger().info(
                f"pc_node dt total={t4 - t0:.3f}s, "
                f"cv_bridge={t1 - t0:.3f}s, gpu={t2 - t1:.3f}s, "
                f"pack={t3 - t2:.3f}s, pub={t4 - t3:.3f}s"
            )

        except Exception as e:
            self.get_logger().error(f"Exception in process_triplet: {e}")

    # =========================
    #   Numpy → PointCloud2
    # =========================
    def cloud4_to_pointcloud2(self, header, cloud4_np: np.ndarray) -> PointCloud2:
        """organized 포맷 (H,W,4)를 그대로 PointCloud2로 변환."""
        H, W, C = cloud4_np.shape
        assert C == 4

        msg = PointCloud2()
        msg.header = header
        msg.height = H
        msg.width = W
        msg.is_bigendian = False
        msg.is_dense = False  # NaN 허용

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * W

        if not cloud4_np.flags["C_CONTIGUOUS"]:
            cloud4_np = np.ascontiguousarray(cloud4_np, dtype=np.float32)
        else:
            cloud4_np = cloud4_np.astype(np.float32, copy=False)

        msg.data = cloud4_np.tobytes()
        return msg

    def cloud4_to_pointcloud2_unorganized(self, header, arr: np.ndarray) -> PointCloud2:
        """비조직(PointCloud2 height=1) 형태로 변환: pack 속도 향상용"""
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = int(arr.shape[0])
        msg.is_bigendian = False
        msg.is_dense = False
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        msg.data = arr.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudGPUNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
