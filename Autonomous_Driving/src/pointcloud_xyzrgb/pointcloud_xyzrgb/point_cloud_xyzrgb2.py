#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import time

from .pointcloud_gpu_xyzrgb import depth_rgb_to_xyzrgb_gpu


class PointCloudGPUNode(Node):
    def __init__(self):
        super().__init__("pointcloud_gpu_node")

        self.bridge = CvBridge()

        # 파라미터 (원하면 launch에서 바꿀 수 있음)
        self.declare_parameter("range_max", 0.0)
        self.declare_parameter("stride", 3)  # 1=전체, 2=1/4, 4=1/16

        qos = rclpy.qos.QoSProfile(depth=10)

        # Subscribers
        self.sub_depth = message_filters.Subscriber(
            self,
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            qos_profile=qos,
        )

        self.sub_rgb = message_filters.Subscriber(
            self,
            Image,
            "/hugging/output/image",
            qos_profile=qos,
        )

        self.sub_info = message_filters.Subscriber(
            self,
            CameraInfo,
            "/camera/camera/color/camera_info",
            qos_profile=qos,
        )

        # ApproximateTimeSynchronizer
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_rgb, self.sub_info],
            queue_size=10,
            slop=0.05,
        )
        self.sync.registerCallback(self.callback)

        # Publisher
        self.pub_cloud = self.create_publisher(
            PointCloud2,
            "/camera/depth_registered/points",
            qos,
        )

    def callback(self, depth_msg, rgb_msg, info_msg):
        try:
            t0 = time.time()

            # 1) depth → numpy (단위 보정 포함)
            depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            if depth_msg.encoding in ["16UC1", "mono16"]:
                depth_np = depth_np.astype(np.float32) * 0.001  # mm → m
            elif depth_msg.encoding == "32FC1":
                depth_np = depth_np.astype(np.float32)
            else:
                self.get_logger().warn(
                    f"Unexpected depth encoding: {depth_msg.encoding}, "
                    f"treating as float32 meters."
                )
                depth_np = depth_np.astype(np.float32)

            H, W = depth_np.shape

            # 2) rgb → numpy
            rgb_np = self.bridge.imgmsg_to_cv2(rgb_msg, rgb_msg.encoding)

            # 3) intrinsics
            fx = info_msg.k[0]
            fy = info_msg.k[4]
            cx = info_msg.k[2]
            cy = info_msg.k[5]

            t1 = time.time()

            # 4) GPU 커널
            range_max = float(
                self.get_parameter("range_max").get_parameter_value().double_value
            )
            cloud4_np = depth_rgb_to_xyzrgb_gpu(
                depth_np,
                rgb_np.astype(np.uint8, copy=False),
                fx, fy, cx, cy,
                encoding=rgb_msg.encoding,
                range_max=range_max,
            )

            t2 = time.time()

            # 5) 다운샘플 (stride>1이면 해상도 줄이기)
            stride = int(self.get_parameter("stride").get_parameter_value().integer_value)
            if stride > 1:
                cloud4_np = cloud4_np[::stride, ::stride, :]

            # 6) pack: numpy → PointCloud2
            cloud_msg = self.cloud4_to_pointcloud2(depth_msg.header, cloud4_np)

            t3 = time.time()

            # 7) Publish
            self.pub_cloud.publish(cloud_msg)

            t4 = time.time()

            self.get_logger().info(
                f"pc_node dt total={t4 - t0:.3f}s, "
                f"cv_bridge={t1 - t0:.3f}s, gpu={t2 - t1:.3f}s, "
                f"pack={t3 - t2:.3f}s, pub={t4 - t3:.3f}s"
            )

        except Exception as e:
            # throttle_duration_sec 인자는 rclpy에는 없음
            self.get_logger().error(f"Exception in callback: {e}")

    def cloud4_to_pointcloud2(self, header, cloud4_np: np.ndarray) -> PointCloud2:
        H, W, C = cloud4_np.shape
        assert C == 4

        msg = PointCloud2()
        msg.header = header
        msg.height = H
        msg.width = W
        msg.is_bigendian = False
        msg.is_dense = False  # NaN 허용

        msg.fields = [
            PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * W

        # C-contiguous float32 [H,W,4] → 그대로 bytes만 사용
        cloud4_np = np.ascontiguousarray(cloud4_np, dtype=np.float32)
        msg.data = cloud4_np.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudGPUNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
