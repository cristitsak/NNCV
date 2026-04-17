from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode

from helpers import convert_train_id_to_color
from model import Model

IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"
CONFIG_PATH = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"

# def preprocess(img: Image.Image) -> torch.Tensor:
#     transform = Compose([
#         ToImage(),
#         Resize(size=(512, 1024), interpolation=InterpolationMode.BILINEAR),
#         ToDtype(dtype=torch.float32, scale=True),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     return transform(img).unsqueeze(0)

def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        # Increase this to 1024x2048 if your GPU handles it!
        Resize(size=(1024, 2048), interpolation=InterpolationMode.BILINEAR), 
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    pred_max = torch.argmax(pred, dim=1, keepdim=True)
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)
    return prediction.cpu().detach().numpy().squeeze()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running inference on {device} ---")

    model = Model(n_classes=19, config_path=CONFIG_PATH)
    
    # LOAD CHECKPOINT
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)
            pred = model(img_tensor)
            seg_pred = postprocess(pred, original_shape)

            # # COLOR VISUALIZATION
            # seg_tensor = torch.from_numpy(seg_pred).unsqueeze(0).unsqueeze(0) 
            # color_mask = convert_train_id_to_color(seg_tensor)
            # color_mask_np = color_mask.squeeze().permute(1, 2, 0).cpu().numpy()
            
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Image.fromarray(color_mask_np.astype(np.uint8)).save(out_path)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)

            print(f"Processed: {img_path.name}")

if __name__ == "__main__":
    main()