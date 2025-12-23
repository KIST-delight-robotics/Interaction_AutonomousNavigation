#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import message_filters
from sensor_msgs_py import point_cloud2
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
        self.declare_parameter("stride", 3)  # 다운샘플링: 1이면 전체, 2면 1/4, 4면 1/16 포인트

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

        # Publisher (하나만!)
        self.pub_cloud = self.create_publisher(
            PointCloud2,
            "/camera/depth_registered/points",
            qos,
        )


    def callback(self, depth_msg, rgb_msg, info_msg):
        try:
            t0 = time.time()

            # 1) depth → numpy (단위 보정 포함)
            # 원본 encoding 그대로 받아온 뒤, encoding 보고 mm → m 변환
            depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            if depth_msg.encoding in ["16UC1", "mono16"]:
                # mm → m
                depth_np = depth_np.astype(np.float32) * 0.001
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
            xyz_np, rgb_packed_np = depth_rgb_to_xyzrgb_gpu(
                depth_np,
                rgb_np.astype(np.uint8, copy=False),
                fx, fy, cx, cy,
                encoding=rgb_msg.encoding,
                range_max=range_max,
            )

            t2 = time.time()

            # 5) 다운샘플 옵션 (속도 올리고 싶으면 stride=2,4 사용)
            stride = int(
                self.get_parameter("stride").get_parameter_value().integer_value
            )
            if stride > 1:
                xyz_np = xyz_np[::stride, ::stride, :]
                rgb_packed_np = rgb_packed_np[::stride, ::stride]

            # 6) PointCloud2로 패킹
            cloud_msg = self.numpy_to_cloud_struct(depth_msg.header, xyz_np, rgb_packed_np)

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
            self.get_logger().error(
                f"Exception in callback: {e}",
                throttle_duration_sec=1.0
            )

    def numpy_to_cloud_struct(self, header, xyz_np, rgb_packed_np):
        """
        xyz_np: (H, W, 3) float32
        rgb_packed_np: (H, W) float32
        """
        H, W, _ = xyz_np.shape

        # PointField 정의
        fields = [
            PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg = PointCloud2()
        msg.header = header
        msg.height = H
        msg.width = W
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * W
        msg.is_dense = False  # NaN 허용

        # NumPy 구조체 배열 생성
        dtype = np.dtype([
            ("x",   np.float32),
            ("y",   np.float32),
            ("z",   np.float32),
            ("rgb", np.float32),
        ])

        cloud_arr = np.empty((H, W), dtype=dtype)

        cloud_arr["x"]   = xyz_np[:, :, 0].astype(np.float32, copy=False)
        cloud_arr["y"]   = xyz_np[:, :, 1].astype(np.float32, copy=False)
        cloud_arr["z"]   = xyz_np[:, :, 2].astype(np.float32, copy=False)
        cloud_arr["rgb"] = rgb_packed_np.astype(np.float32, copy=False)

        msg.data = cloud_arr.tobytes()

        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudGPUNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
