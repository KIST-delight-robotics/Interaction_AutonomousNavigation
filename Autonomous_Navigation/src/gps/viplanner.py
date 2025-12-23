## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
## Code from https://aisj.tistory.com/153    ##
###############################################
import copy
import time

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F


from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from viplanner.vip_inference import ViPlannerInference
from utils import palette

def realsense_thread(use_segment=False):
    global current_frame, color_image, depth_image
    current_frame = np.zeros((480,640*2,3), np.uint8)
    color_image = np.zeros((480,640*3,3), np.uint8)
    depth_image = np.zeros((480,640*3,1), np.uint8)
 
    """
    checkpoint = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
    segmentator = Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
    ).cuda().half() # NHWC; TensorRT 변환 시 유리
    segmentator.eval()        
    print('Init mask2former as segmentation model')
    """

    viplanner = ViPlannerInference()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    print('Intel Realsense Camera Streaming Start.')

    previous_time = time.time()
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap = cv2.resize(depth_colormap, dsize=(color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_AREA)
            current_frame = np.hstack((color_image, depth_colormap)).copy()

            """
            # 1) numpy array -> torch cuda tensor -> resize
            img = T.ToTensor()(color_image).cuda().half()          # (C,H,W) in [0,1]
            img = F.interpolate(img.unsqueeze(0), (150,200), mode="bilinear", align_corners=False)

            # 2) 모델이 기대하는 정규화 (processor 메타데이터 활용)
            mean = torch.tensor(processor.image_mean, device=img.device, dtype=img.dtype).view(1,3,1,1)
            std  = torch.tensor(processor.image_std , device=img.device, dtype=img.dtype).view(1,3,1,1)
            pixel_values = (img - mean) / std

            #with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = segmentator(pixel_values=pixel_values)

            # pass through image_processor for postprocessing
            seg_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[[color_image.shape[0], color_image.shape[1]]])[0]
            pred = seg_map.clone().cpu().numpy().astype(np.uint8)
            seg_pil = Image.fromarray(pred, mode="P")   # 팔레트 모드
            seg_pil.putpalette(palette)
            seg_pil = seg_pil.convert('RGB')
            seg_image = np.asanyarray(seg_pil)
            #seg_image = cv2.resize(seg_image, dsize=(160, 120), interpolation=cv2.INTER_AREA)

            goal_cam_frame = torch.Tensor([10.0, 0.0, 0.0]).unsqueeze(0)
            traj, waypoints, fear = viplanner.plan(depth_image, seg_image, goal_cam_frame)
            """
            goal_cam_frame = torch.Tensor([2.0, 0.0, 0.0]).unsqueeze(0)
            traj, waypoints, fear = viplanner.plan_depth(depth_image, goal_cam_frame)
            print(waypoints, fear)


    finally:
        # Stop streaming
        pipeline.stop()

realsense_thread()
