import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class Model(nn.Module):
    """
    Peak Performance Model using NVIDIA's SegFormer.
    This replaces the vanilla U-Net with a Transformer-based architecture.
    """
    def __init__(
        self, 
        in_channels=3, 
        n_classes=19,
        model_name="nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    ):
        super().__init__()
        
        # Load the pre-trained SegFormer from Hugging Face
        # We use 'ignore_mismatched_sizes' to ensure it adapts to our n_classes
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
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
        

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)