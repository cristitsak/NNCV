from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)

from model import Model

IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"
CONFIG_PATH = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
# "/app/segformer_b3_config"  # Add this


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=(512, 1024), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # pred shape: [1, 19, 512, 1024]
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)  # [1, 1, 512, 1024]
    
    # Resize back to original image size
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)
    # prediction shape: [1, 1, original_h, original_w]
    
    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # [original_h, original_w]
    
    return prediction_numpy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model with correct parameters
    model = Model(n_classes=19, config_path=CONFIG_PATH)
    
    # Load weights
    state_dict = torch.load(
        MODEL_PATH, 
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass
            pred = model(img_tensor)

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred, original_shape)

            # Save predicted mask
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()