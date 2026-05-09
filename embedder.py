from transformers import SiglipVisionModel, SiglipImageProcessor
import torch
import numpy as np

processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-384")
model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-384")

def embed_image(pil_image):
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(pixel_values=inputs.pixel_values)
    emb = outputs.pooler_output
    return emb / emb.norm(dim=1, keepdim=True)

