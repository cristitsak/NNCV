import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class Model(nn.Module):
    def __init__(self, n_classes=19, config_path="/app/segformer_b3_config"):
        super().__init__()
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config path not found: {config_path}")
        
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            config_path, 
            num_labels=n_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        """
        Forward pass for SegFormer.
        Input: (Batch, 3, H, W)
        Output: (Batch, n_classes, H, W)
        """
        # SegFormer expects the argument 'pixel_values'
        outputs = self.segformer(pixel_values=x)
        
        # SegFormer returns logits that are 1/4 of the input resolution
        logits = outputs.logits 
        
        # We MUST upscale the logits back to the original input size 
        # so the loss function in train.py can compare them to the masks.
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[-2:], # Takes (H, W) from input tensor
            mode="bilinear", 
            align_corners=False
        )
        
        return upsampled_logits