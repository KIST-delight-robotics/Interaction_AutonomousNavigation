import os, sys, glob, types, time

# tensorrt 
import ctypes
import pycuda.driver as cuda
#import pycuda.autoinit
import tensorrt as trt

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F

from termcolor import colored


# vi-planner
from viplanner.vip_inference import ViPlannerInference

# segmentation models
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import AutoImageProcessor, Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

sys.path.append(os.path.abspath("../jetson_quantization"))
from utils.trt_utils.trt_engine import TRTEngine

# ddrnet
#this_dir = os.path.dirname(__file__)
#ddrnet_path = os.path.join(this_dir, '..', 'ddrnet', 'lib')
#sys.path.insert(0, ddrnet_path)

#import models

#torch._dynamo.config.suppress_errors = True
#torch._dynamo.config.verbose = True


def print_info(headline, text, color='green'):
    print(colored(headline + "\t", color), text)

def plan_viplanner(color_image, depth_image, goal_xy, processor, segmentator, palette, viplanner,trt=False):
    start_time = time.time()
    #with torch.inference_mode():
    if trt:
        inputs = processor(images=color_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].numpy()
        model_outputs = segmentator.infer(pixel_values)

        # 출력을 torch tensor로 변환 (후처리를 위해)
        class_queries_logits = torch.from_numpy(model_outputs[0])
        masks_queries_logits = torch.from_numpy(model_outputs[1])

        print(class_queries_logits.shape)         # [1, num_queries, num_classes]

        # 후처리를 위한 출력 객체 생성
        outputs = types.SimpleNamespace(
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits
        )
     
    else:
        # Conduct rgb image segmentation
        # global processor, segmentator, seg_image, palette
        # 1) 입력 이미지를 바로 CUDA Tensor 로
        img = T.ToTensor()(color_image).cuda().half().unsqueeze(0) # (1,C,H,W) in [0,1]
            
        # 2) 모델이 기대하는 정규화 (processor 메타데이터 활용)
        if processor is None:
            pixel_values = (img - 0.5) / 0.5
        else:
            img = F.interpolate(img, (240,320), mode="bilinear", align_corners=False)
            mean = torch.tensor(processor.image_mean, device=img.device, dtype=img.dtype).view(1,3,1,1)
            std  = torch.tensor(processor.image_std , device=img.device, dtype=img.dtype).view(1,3,1,1)
            pixel_values = (img - mean) / std

       
        with torch.cuda.amp.autocast(dtype=torch.float16):
            if processor is not None:
                outputs = segmentator(pixel_values=pixel_values)
            else:
                outputs = segmentator(pixel_values)[1]
                outputs = F.interpolate(outputs, (pixel_values.shape[2:]), mode="bilinear")

    # pass through image_processor for postprocessing
    if processor is not None:
        seg_image = processor.post_process_semantic_segmentation(outputs, target_sizes=[[color_image.shape[0], color_image.shape[1]]])[0]
    else:
        seg_image = torch.argmax(outputs, dim=1)[0]
    seg_image = superclass_palette[seg_image.cpu()].to_dense()
    
    traj_time = time.time()
    goal_cam_frame = torch.Tensor([goal_xy[0], goal_xy[1], 0]).unsqueeze(0)
    traj, points, fear = viplanner.plan(depth_image, seg_image.cpu().numpy(), goal_cam_frame)
    
    plan_time = time.time() - traj_time
    seg_time = traj_time - start_time

    print_info('SEGMENT', f'time took:{seg_time}, {plan_time}')

    return traj, points, fear, seg_image.clone().cpu().numpy().astype(np.uint8)

def plan_iplanner(depth_image, goal_xy):
    goal_cam_frame = torch.Tensor([goal_xy[0], goal_xy[1], 0]).unsqueeze(0)
    return viplanner.plan_depth(depth_image, goal_cam_frame)

def segment_thread(queue_in, queue_out, USE_SEGMENTATION, rate=1.0):
    viplanner = ViPlannerInference()
   
    # Loading a single segmentation model
    if USE_SEGMENTATION:
        #seg_model='mask2former'
        seg_model='mask2former_trt'
        #seg_model = 'ddrnet23_slim'
        if seg_model == 'oneformer':
            checkpoint = "shi-labs/oneformer_cityscapes_swin_large"
            processor = OneFormerProcessor.from_pretrained(checkpoint)
            segmentator = OneFormerForUniversalSegmentation.from_pretrained(checkpoint).cuda()
        elif seg_model == 'mask2former':
            checkpoint = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
            processor = Mask2FormerImageProcessor.from_pretrained(checkpoint, use_fast=True, do_resize=True, size=(320,240))
            print_info("Segment", f"Initialization at {checkpoint}")
            segmentator = Mask2FormerForUniversalSegmentation.from_pretrained(
                checkpoint,
                torch_dtype=torch.float16,
            )
            #segmentator = torch.compile(segmentator, mode="reduce-overhead")
            segmentator.cuda().half().eval()
            print_info("Segment", f'Init mask2former as segmentation model. Is queue_out empty? {queue_out.empty()}')
        elif seg_model == 'ddrnet23_slim':
            processor = None
            libusb-1.0 #segmentator = models.ddrnet_23_slim.DualResNet(models.ddrnet_23_slim.BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64, augment=True)
            #pretrained_state = torch.load('../ddrnet/pretrained_models/DDRNet23s_cityscape.pth', map_location='cpu') 
            segmentator = models.ddrnet_23_slim.DualResNet(models.ddrnet_23_slim.BasicBlock, [2, 2, 2, 2], num_classes=65, planes=32, spp_planes=128, head_planes=64, augment=True)
            pretrained_state = torch.load('../ddrnet/pretrained_models/DDRNet23s_mapillary_temp.pth', map_location='cpu') 
           
            model_dict = segmentator.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
            model_dict.update(pretrained_state)
            segmentator.load_state_dict(model_dict, strict=True)
            segmentator.cuda().half()
            segmentator.eval()
            print_info("Segment", f'Init DDRNet23 slim as segmentation model. Is queue_out empty? {queue_out.empty()}')
        elif seg_model == 'mask2former_trt':
            checkpoint = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
            ENGINE_PATH = "../jetson_quantization/engines/trt/fp16.engine"  # 또는 fp32

            processor = AutoImageProcessor.from_pretrained(checkpoint)

            #TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            ## 1) libnvinfer_plugin.so 로드 (경로는 x86/Jetson에 따라 다릅니다)
            #for base in ("/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib"):
            #    cands = glob.glob(os.path.join(base, "libnvinfer_plugin.so*"))
            #    if cands:
            #        ctypes.CDLL(cands[0], mode=ctypes.RTLD_GLOBAL)
            #        break

            ## 2) 플러그인 레지스트리 초기화
            #trt.init_libnvinfer_plugins(TRT_LOGGER, "")    

            # TensorRT 엔진 초기화
            print('engine initializaion')
            segmentator = TRTEngine(ENGINE_PATH)


        queue_out.get()
        queue_out.task_done()

        print_info("Segment", f"initialization done. Is queue_out empty? {queue_out.empty()}")       


        while True:
            print_info("Segment", 'wait for rgb+depth')
            color, depth, goal = queue_in.get()
            queue_in.task_done()
            traj, points, fear, seg_image = plan_viplanner(color, depth, goal, processor, segmentator, palette, viplanner, 'trt' in seg_model)
            print("Segment", "put seg+waypoints")
            queue_out.put((traj, points, fear, seg_image))

# ─────────────────────────────────────────────────────────────
# 1. Mapillary Vistas 팔레트 (65 클래스 + void = 총 66)
#    [R, G, B] 순; 주석은 원본 라벨명을 간략히 표시
# ─────────────────────────────────────────────────────────────
mapillary_vistas_palette = [
    (165,  42,  42),   # 0  Bird
    (  0, 192,   0),   # 1  Ground Animal
    (196, 196, 196),   # 2  Curb
    (190, 153, 153),   # 3  Fence
    (180, 165, 180),   # 4  Guard Rail
    (102, 102, 156),   # 5  Barrier
    (102, 102, 156),   # 6  Wall
    (128,  64, 255),   # 7  Bike Lane
    (140, 140, 200),   # 8  Crosswalk-Plain
    (170, 170, 170),   # 9  Curb Cut
    (250, 170, 160),   # 10 Parking
    ( 96,  96,  96),   # 11 Pedestrian Area
    (230, 150, 140),   # 12 Rail Track
    (128,  64, 128),   # 13 Road
    (110, 110, 110),   # 14 Service Lane
    (244,  35, 232),   # 15 Sidewalk
    (150, 100, 100),   # 16 Bridge
    ( 70,  70,  70),   # 17 Building
    (150, 120,  90),   # 18 Tunnel
    (220,  20,  60),   # 19 Person
    (255,   0,   0),   # 20 Bicyclist
    (255,   0,   0),   # 21 Motorcyclist
    (255,   0,   0),   # 22 Other Rider
    (200, 128, 128),   # 23 Lane Mk-Cross
    (255, 255, 255),   # 24 Lane Mk-General
    ( 64, 170,  64),   # 25 Mountain
    (128,  64,  64),   # 26 Sand
    ( 70, 130, 180),   # 27 Sky
    (255, 255, 255),   # 28 Snow
    (152, 251, 152),   # 29 Terrain
    (107, 142,  35),   # 30 Vegetation
    (  0, 170,  30),   # 31 Water
    (255, 255, 128),   # 32 Banner
    (250,   0,  30),   # 33 Bench
    (  0,   0,   0),   # 34 Bike Rack
    (220, 220, 220),   # 35 Billboard
    (170, 170, 170),   # 36 Catch Basin
    (222,  40,  40),   # 37 CCTV Camera
    (100, 170,  30),   # 38 Fire Hydrant
    ( 40,  40,  40),   # 39 Junction Box
    ( 33,  33,  33),   # 40 Mailbox
    (170, 170, 170),   # 41 Manhole
    (  0,   0, 142),   # 42 Phone Booth
    (170, 170, 170),   # 43 Pothole
    (210, 170, 100),   # 44 Street Light
    (153, 153, 153),   # 45 Pole
    (128, 128, 128),   # 46 Traffic Sign Frame
    (  0,   0, 142),   # 47 Utility Pole
    (250, 170,  30),   # 48 Traffic Light
    (192, 192, 192),   # 49 Traffic Sign Back
    (220, 220,   0),   # 50 Traffic Sign Front
    (180, 165, 180),   # 51 Trash Can
    (119,  11,  32),   # 52 Bicycle
    (  0,   0, 142),   # 53 Boat
    (  0,  60, 100),   # 54 Bus
    (  0,   0, 142),   # 55 Car
    (  0,   0,  90),   # 56 Caravan
    (  0,   0, 230),   # 57 Motorcycle
    (  0,  80, 100),   # 58 On Rails
    (128,  64,  64),   # 59 Other Vehicle
    (  0,   0, 110),   # 60 Trailer
    (  0,   0,  70),   # 61 Truck
    (  0,   0, 192),   # 62 Wheeled Slow
    ( 32,  32,  32),   # 63 Car Mount
    (  0,   0,   0),   # 64 Ego-Vehicle
    (  0,   0,   0),   # 65 Void / Unlabeled
]

palette = sum(mapillary_vistas_palette, ())                      # 평탄화
palette += (0, 0, 0) * (256 - len(mapillary_vistas_palette))     # 256*3 채우기

# ─────────────────────────────────────────────────────────────
# 1. Mapillary Vistas (65 클래스 + void = 총 66) -> Mapping
# sidewalk, crosswalk, floor, stairs
# ─────────────────────────────────────────────────────────────
free        = [0,   255, 0] # green: sidewalk, crosswalk, floor, stairs
drivable    = [127, 255, 0] # yellow~green: gravel, sand, snow
terrain     = [255, 255, 0] # yellow: terrain (grass, dirt)
road        = [255, 165, 0] # orange: road 
obstacle    = [255, 0,   0]   # red: person, animal, vehicle, trains, motorcycle, bicycle
building    = [160, 32,  240] # purple: building, wall, fence, bridge, tunnel, furniture, tree, water surface
traffic     = [0,   0,   255] # blue: pole, traffic sign, traffic light, bench
others      = [46,  17,  35] #sky, ceiling, unknown 

superclass_palette = torch.Tensor([
    obstacle,   # 0  Bird
    obstacle,   # 1  Ground Animal
    drivable,   # 2  Curb
    building,   # 3  Fence
    building,   # 4  Guard Rail
    building,   # 5  Barrier
    building,   # 6  Wall
    road,       # 7  Bike Lane
    free,       # 8  Crosswalk-Plain
    drivable,   # 9  Curb Cut
    drivable,   # 10 Parking
    free,       # 11 Pedestrian Area
    road,       # 12 Rail Track
    road,       # 13 Road
    road,       # 14 Service Lane
    free,       # 15 Sidewalk
    building,   # 16 Bridge
    building,   # 17 Building
    building,   # 18 Tunnel
    obstacle,   # 19 Person
    obstacle,   # 20 Bicyclist
    obstacle,   # 21 Motorcyclist
    obstacle,   # 22 Other Rider
    road,       # 23 Lane Mk-Cross
    road,       # 24 Lane Mk-General
    terrain,    # 25 Mountain
    drivable,   # 26 Sand
    others,     # 27 Sky
    drivable,   # 28 Snow
    terrain,    # 29 Terrain
    building,   # 30 Vegetation
    building,   # 31 Water
    traffic,    # 32 Banner
    traffic,    # 33 Bench
    traffic,    # 34 Bike Rack
    traffic,    # 35 Billboard
    traffic,    # 36 Catch Basin
    traffic,    # 37 CCTV Camera
    traffic,    # 38 Fire Hydrant
    traffic,    # 39 Junction Box
    traffic,    # 40 Mailbox
    traffic,    # 41 Manhole
    traffic,    # 42 Phone Booth
    traffic,    # 43 Pothole
    traffic,    # 44 Street Light
    traffic,    # 45 Pole
    traffic,    # 46 Traffic Sign Frame
    traffic,    # 47 Utility Pole
    traffic,    # 48 Traffic Light
    traffic,    # 49 Traffic Sign Back
    traffic,    # 50 Traffic Sign Front
    traffic,    # 51 Trash Can
    obstacle,   # 52 Bicycle
    obstacle,   # 53 Boat
    obstacle,   # 54 Bus
    obstacle,   # 55 Car
    obstacle,   # 56 Caravan
    obstacle,   # 57 Motorcycle
    obstacle,   # 58 On Rails
    obstacle,   # 59 Other Vehicle
    obstacle,   # 60 Trailer
    obstacle,   # 61 Truck
    obstacle,   # 62 Wheeled Slow
    others,     # 63 Car Mount
    others,     # 64 Ego-Vehicle
    others,     # 65 Void / Unlabeled
])

