from PIL import Image
from transformers import AutoImageProcessor
import os, sys 
sys.path.append(os.path.abspath(".."))
from utils.trt_utils.trt_engine import TRTEngine



def trt_inference(engine_path, image_path):
    # 이미지 프로세서 로드
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")

    # TensorRT 엔진 초기화
    trt_model = TRTEngine(engine_path)
    
    # 이미지 로드 및 전처리
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    # TensorRT 추론
    pixel_values = inputs["pixel_values"].numpy()
    outputs = trt_model.infer(pixel_values)

    return outputs

