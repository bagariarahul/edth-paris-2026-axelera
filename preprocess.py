"""Voyager axcompile calibration transform for RF-DETR Nano (384x384, ImageNet-normalised NCHW RGB)."""
import numpy as np
import torch
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_preprocess_transform(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    x = np.asarray(image.convert("RGB").resize((384, 384), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())  # CHW float32
