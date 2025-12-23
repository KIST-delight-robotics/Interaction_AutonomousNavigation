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
from PIL import Image

from termcolor import colored
def print_info(headline, text, color='green'):
    print(colored(headline + "\t", color), text)

def realsense_thread(use_segment=False):
    global current_frame, color_image, depth_image
    current_frame = np.zeros((480,640*2,3), np.uint8)
    color_image = np.zeros((480,640*3,3), np.uint8)
    depth_image = np.zeros((480,640*3,1), np.uint8)
 
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
        print_info("d435i", "The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    print_info("d435i", 'Intel Realsense Camera Streaming Start.')

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

    finally:
        # Stop streaming
        pipeline.stop()
