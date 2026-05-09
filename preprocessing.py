# src/preprocessing.py

from PIL import Image
import torch
from torchvision import transforms


# SigLIP image normalization values (from official HF config)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


# Build the preprocessing pipeline once
siglip_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
])


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img   # return PIL image, NOT tensor

