"""
Helper functions and classes for robust Cityscapes segmentation training.
Contains loss functions, data transforms, schedulers, and utility functions.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    Compose, Normalize, Resize, ToImage, ToDtype, InterpolationMode,
    RandomHorizontalFlip, ColorJitter, RandomErasing, RandomCrop,
    GaussianBlur, RandomGrayscale, RandomApply
)
from torchvision.transforms import functional as TF
import segmentation_models_pytorch as smp


# ==================== CUSTOM TRANSFORMS ====================

class AddGaussianNoise:
    """Add Gaussian noise to a tensor image (applied after ToFloat/Normalize)."""
    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"



# copy paste from original train.py file to ensure consistent class ID handling across training and prediction.
# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    """Convert Cityscapes class IDs to train IDs."""
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to colors for visualization
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Black for ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    """Convert train ID predictions to RGB color images."""
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


# ==================== LOSS FUNCTIONS ====================

class RobustCombinedLoss(nn.Module):
    """
    Combined loss with OHEM (Online Hard Example Mining) focusing on difficult pixels.
    Improves robustness by making the model pay more attention to challenging areas.
    """
    def __init__(
        self, 
        ce_weight: float = 0.4,
        dice_weight: float = 0.6,
        label_smoothing: float = 0.05,
        ohem_ratio: float = 0.3,
        use_ohem: bool = True,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            ignore_index=255, 
            label_smoothing=label_smoothing,
            reduction='none',
            weight=class_weights,
        )
        self.dice = smp.losses.DiceLoss(
            mode='multiclass', 
            ignore_index=255,
            smooth=1e-5
        )
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ohem_ratio = ohem_ratio
        self.use_ohem = use_ohem
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE loss per pixel
        ce_loss_per_pixel = self.ce(preds, targets)
        
        if self.use_ohem and self.ohem_ratio < 1.0:
            valid_mask = targets != 255
            if valid_mask.sum() > 0:
                ce_loss_valid = ce_loss_per_pixel[valid_mask]
                k = max(1, int(self.ohem_ratio * len(ce_loss_valid)))
                top_k_losses, _ = torch.topk(ce_loss_valid, k)
                ce_loss = top_k_losses.mean()
            else:
                ce_loss = ce_loss_per_pixel.mean()
        else:
            ce_loss = ce_loss_per_pixel.mean()
        
        dice_loss = self.dice(preds, targets)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class ClassWeightedLoss(nn.Module):
    """Loss function with class weights to handle imbalance."""
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        ce_weight: float = 0.6,
        dice_weight: float = 0.4
    ):
        super().__init__()
        
        if class_weights is not None:
            self.ce = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=255
            )
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=255)
            
        self.dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
    def forward(self, preds, targets):
        return (self.ce_weight * self.ce(preds, targets) + 
                self.dice_weight * self.dice(preds, targets))


def compute_class_weights(dataset=None, num_classes=19, device='cpu') -> torch.Tensor:
    """
    Return inverse-frequency class weights for Cityscapes (19 train classes).

    Weights are derived from published Cityscapes pixel-frequency statistics.
    Rare classes (train, rider, motorcycle, bicycle, traffic light, etc.) get
    significantly higher weight so the model pays more attention to them.

    If `dataset` is provided the function still uses the pre-computed values —
    computing live requires a full pass over ~3000 images which is very slow.

    Train-ID order (0-18):
      road, sidewalk, building, wall, fence, pole, traffic light,
      traffic sign, vegetation, terrain, sky, person, rider, car,
      truck, bus, train, motorcycle, bicycle
    """
    print("--- Validating Class Mapping ---")
    for cls in Cityscapes.classes:
        if 'sky' in cls.name.lower():
            # This confirms name, raw ID, and the 0-18 Train ID used for weighting
            print(f"Class: {cls.name}, ID: {cls.id}, Train ID: {cls.train_id}, Color: {cls.color}")
    print("--------------------------------")
    
    # Approximate pixel frequencies from Cityscapes training set
    frequencies = torch.tensor([
        0.3674,  # 0  road          — very common
        0.0615,  # 1  sidewalk
        0.2070,  # 2  building
        0.0048,  # 3  wall
        0.0076,  # 4  fence
        0.0124,  # 5  pole
        0.0034,  # 6  traffic light — rare
        0.0053,  # 7  traffic sign
        0.1914,  # 8  vegetation
        0.0186,  # 9  terrain
        0.0521,  # 10 sky
        0.0196,  # 11 person
        0.0025,  # 12 rider         — very rare
        0.1213,  # 13 car
        0.0076,  # 14 truck
        0.0049,  # 15 bus
        0.0010,  # 16 train         — very rare
        0.0013,  # 17 motorcycle    — very rare
        0.0103,  # 18 bicycle
    ], dtype=torch.float32)

    # Inverse-frequency weighting, then median normalisation (more stable than sum)
    raw_weights = 1.0 / (frequencies + 1e-6)
    weights = raw_weights / raw_weights.median()

    # Cap extreme weights to avoid numerical instability for the rarest classes
    weights = torch.clamp(weights, max=20.0)

    return weights.to(device)


# ==================== DATA TRANSFORMS ====================

def get_train_transforms():
    """Get training data transforms with augmentation."""
    return Compose([
        ToImage(),
        Resize((512, 1024), interpolation=InterpolationMode.BILINEAR),
        RandomHorizontalFlip(p=0.5),
        # Stronger colour jitter: hue/saturation shifts help robustness to lighting
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # Occasional grayscale simulates cameras / edge cases
        RandomGrayscale(p=0.05),
        # Blur simulates motion / focus variation
        RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Gaussian noise — simulates sensor noise, improves robustness
        RandomApply([AddGaussianNoise(mean=0.0, std=0.02)], p=0.4),
        # Random erasing (pixel block masking) — applied after normalisation
        # Forces model to reason from context rather than local texture
        RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    ])


def get_val_transforms():
    """Get validation transforms (no augmentation)."""
    return Compose([
        ToImage(),
        Resize((512, 1024), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_target_transforms():
    """Get target transforms for both train and validation.
    
    NOTE: Geometric augmentations like flipping must NOT be applied here
    independently — they must be synchronised with the image transform.
    Joint geometric augmentation is handled in MultiScaleTransform / the
    train loop instead.
    """
    return Compose([
        ToImage(),
        Resize((512, 1024), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])


class MultiScaleTransform:
    """
    Joint spatial augmentation applied to both image and label together,
    ensuring they always stay in sync.

    Includes:
    - Random scale (resize)
    - Random horizontal flip
    - Random crop (when scaled-up, avoids zero-padding artefacts)
    """
    def __init__(
        self,
        base_size: tuple = (512, 1024),
        scales: list = [0.75, 1.0, 1.25, 1.5],
        crop_size: tuple = (512, 1024),
        flip_prob: float = 0.5,
    ):
        self.base_size = base_size
        self.scales = scales
        self.crop_size = crop_size
        self.flip_prob = flip_prob

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
        scale = np.random.choice(self.scales)
        new_h = int(self.base_size[0] * scale)
        new_w = int(self.base_size[1] * scale)

        # Keep dimensions as multiples of 32 (SegFormer patch requirement)
        new_h = max(32, (new_h // 32) * 32)
        new_w = max(32, (new_w // 32) * 32)

        image = TF.resize(image, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        label = TF.resize(label, (new_h, new_w), interpolation=InterpolationMode.NEAREST)

        # Random crop back to crop_size when bigger than crop_size
        crop_h, crop_w = self.crop_size
        if new_h > crop_h or new_w > crop_w:
            # Compute valid top-left corner range
            top  = np.random.randint(0, max(1, new_h - crop_h + 1))
            left = np.random.randint(0, max(1, new_w - crop_w + 1))
            image = TF.crop(image, top, left, min(crop_h, new_h), min(crop_w, new_w))
            label = TF.crop(label, top, left, min(crop_h, new_h), min(crop_w, new_w))

            # Pad back if crop ended up smaller (rare edge case with small scales)
            if image.shape[-2] < crop_h or image.shape[-1] < crop_w:
                pad_h = max(0, crop_h - image.shape[-2])
                pad_w = max(0, crop_w - image.shape[-1])
                image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
                label = TF.pad(label, [0, 0, pad_w, pad_h], fill=255)  # 255 = ignore

        # Joint horizontal flip
        if np.random.random() < self.flip_prob:
            image = TF.hflip(image)
            label = TF.hflip(label)

        return image, label


# ==================== SCHEDULERS ====================

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 10,
    min_lr: float = 1e-7
) -> SequentialLR:
    """Create learning rate scheduler with warmup and cosine annealing."""
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=min_lr
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    return scheduler


class WeightDecayScheduler:
    """Gradually reduce weight decay during training."""
    def __init__(self, optimizer, initial_wd=0.01, final_wd=1e-5, decay_epochs=50):
        self.optimizer = optimizer
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.decay_epochs = decay_epochs
        
    def step(self, epoch):
        if epoch < self.decay_epochs:
            wd = self.initial_wd - (self.initial_wd - self.final_wd) * (epoch / self.decay_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = wd


# ==================== METRICS ====================

def compute_iou(preds, labels, num_classes=19, ignore_index=255):
    """Compute IoU per class."""
    ious = []
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        ignore_mask = (labels == ignore_index)
        
        pred_mask[ignore_mask] = False
        label_mask[ignore_mask] = False
        
        intersection = (pred_mask & label_mask).sum()
        union = (pred_mask | label_mask).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious


# ==================== TEST TIME AUGMENTATION ====================

@torch.no_grad()
def test_time_augmentation(model, image, device):
    """Apply TTA by averaging predictions from original and flipped images."""
    model.eval()
    predictions = []
    
    # Original
    pred = model(image.to(device))
    predictions.append(pred.softmax(1))
    
    # Horizontal flip
    flipped = torch.flip(image, dims=[3])
    pred_flipped = model(flipped.to(device))
    pred_flipped = torch.flip(pred_flipped, dims=[3])
    predictions.append(pred_flipped.softmax(1))
    
    # Average predictions
    avg_pred = torch.stack(predictions).mean(0)
    return avg_pred


# ==================== MODEL SAVING/LOADING ====================

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_miou, filepath):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'val_miou': val_miou,
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cuda'):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint