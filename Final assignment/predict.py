from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode

from model import Model

IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"
CONFIG_PATH = "/app/segformer_b3_config"

def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=(512, 1024), interpolation=InterpolationMode.BILINEAR), 
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def postprocess(pred: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    # Get the class with highest probability
    pred_max = torch.argmax(pred, dim=1, keepdim=True)
    # Resize back to original image dimensions
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)
    return prediction

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running inference on {device} ---")

    model = Model(n_classes=19, config_path=CONFIG_PATH)
    
    # LOAD LOGIC: Handles the KeyError you saw earlier
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Use the key if it exists, otherwise assume the whole file is the state_dict
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    
    # Load with strict=False to bypass those "Missing Keys" errors
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))

    with torch.no_grad():
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                original_shape = (img.height, img.width)

                img_tensor = preprocess(img).to(device)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    pred = model(img_tensor)
                
                # --- YOUR ORIGINAL POSTPROCESSING ---
                seg_tensor = postprocess(pred, original_shape)
                raw_ids = seg_tensor.squeeze().cpu().numpy().astype(np.uint8)

                out_path = Path(OUTPUT_DIR) / img_path.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(raw_ids, mode='L').save(out_path)
                # ------------------------------------

                print(f"Processed and Saved: {img_path.name}")

                del img_tensor, pred, seg_tensor, raw_ids
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

if __name__ == "__main__":
    main()