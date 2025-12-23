
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge

import os, sys
import numpy as np
import torch
import types
import cv2
from PIL import Image as PILImage
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

sys.path.append('/usr/lib/python3.10/dist-packages')
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.trt_utils.trt_engine import TRTEngine

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
ENGINE_PATH = "/home/nvidia/jetson_quantization-master/engines/trt/ddrnet23_fp16_kist-v1-80k_1x480x640.engine" 

class HuggingMask2FormerTRTNode(Node):
    def __init__(self):
        super().__init__('hugging_mask2former_trt_node')
        self.bridge = CvBridge()

        self.get_logger().info("Loading TensorRT engine and image processor...")
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic", use_fast=True, do_resize=True,
        size={"height": 480, "width": 640},  # 엔진과 동일하게
        do_rescale=True)
        self.model_config = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-mapillary-vistas-semantic"
        ).config

        self.trt_model = TRTEngine(ENGINE_PATH)
        self.subscription = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            qos
        )

        self.image_pub = self.create_publisher(Image, "/hugging/output/image", 10)
        self.class_pub = self.create_publisher(Int32MultiArray, "/hugging/output/classes", 10)

    def image_callback(self, msg):
        start = self.get_clock().now().nanoseconds / 1e9

        # ROS Image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # OpenCV - RGB 변환 - numpy 전처리
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=img_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].numpy().copy() 

        # 추론 
        outputs = self.trt_model.infer(pixel_values)
        logits = torch.from_numpy(outputs[0])   # [1, 65, 60, 80]

        # 가장 높은 클래스 선택 → segmentation map
        predicted_map = logits.argmax(dim=1).cpu().numpy()[0]  # [60, 80]

        # 원본 해상도로 업샘플
        predicted_map = cv2.resize(
            predicted_map.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # semantic_map 생성
        semantic_map = np.zeros_like(predicted_map, dtype=np.uint8)
        for cid in np.unique(predicted_map):
            if cid in [15, 23, 7, 11, 41, 8]:           # 주행 가능 (1) 
                #sidewalk,lane,bikelane,pedestrian area,manhole,crosswalk
                semantic_map[predicted_map == cid] = 1
            elif cid in [19]:                           # 사람 (2)
                semantic_map[predicted_map == cid] = 2
            elif cid in [13, 6, 29, 30, 55, 59, 57, 54, 52]:   # 피해야 할 영역 (3)
                #road,wall,terrain,vegetation,car,other vehicle,motorcycle,bus,bicycle
                semantic_map[predicted_map == cid] = 3
            elif cid in [2, 24]:           
                #lane mk-general,curb                    # 연석
                semantic_map[predicted_map == cid] = 4
            # elif cid in [19]:
            #     semantic_map[predicted_map == cid] = 5
            else:                                    # 나머지 (0)
                semantic_map[predicted_map == cid] = 0

        # 색상 맵 오버레이
        color_map = {
            0: (128,128,128),  # 회색
            1: (0,255,0),      # 초록
            2: (255,0,0),      # 파랑
            3: (0,0,255),      # 빨강
            4: (255,255,255)
            # 5: (0,255,255)
        }
        h, w = semantic_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for k, color in color_map.items():
            colored[semantic_map == k] = color

        alpha = 0.9
        overlay = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)

        # Publish
        ros_img = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        ros_img.header = msg.header
        self.image_pub.publish(ros_img)
        self.class_pub.publish(Int32MultiArray(data=np.unique(predicted_map).tolist()))

        elapsed = self.get_clock().now().nanoseconds / 1e9 - start
        self.get_logger().info(f"Inference time: {elapsed:.3f}s ({1/elapsed:.2f} FPS)")



def main(args=None):
    rclpy.init(args=args)
    node = HuggingMask2FormerTRTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
