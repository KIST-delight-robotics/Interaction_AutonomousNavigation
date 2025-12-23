from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from types import SimpleNamespace
from utils.visualization.register_mapillary import MAPILLARY_VISTAS_SEM_SEG_CATEGORIES

# metadata 준비 (MAPILLARY_VISTAS_SEM_SEG_CATEGORIES는 이미 정의되어 있다고 가정)
metadata = SimpleNamespace()
metadata.stuff_classes = [k["readable"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES if k["evaluate"]]
metadata.stuff_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES if k["evaluate"]]


def visualize_output(image, pred_map, metadata=metadata, alpha=0.5, min_area=100):


    from utils.visualization.visualizer import Visualizer, ColorMode
    
    predictions = pred_map
    # print(predictions)
    # print(predictions.shape)
    visualizer = Visualizer(image, metadata=metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_sem_seg(
                predictions.to(torch.device("cpu")), alpha=0.7
            )
    vis_output = vis_output.get_image()

    return vis_output
    plt.axis("off")
    plt.imshow(vis_output)
    plt.show()
