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
    RandomHorizontalFlip, ColorJitter
)
from torchvision.transforms import functional as TF
import segmentation_models_pytorch as smp


# ==================== CLASS MAPPING UTILITIES ====================
# copy pest from original train.py file to ensure consistent class ID handling across training and prediction.
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
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.05,
        ohem_ratio: float = 0.3,
        use_ohem: bool = True
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            ignore_index=255, 
            label_smoothing=label_smoothing,
            reduction='none'
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


def compute_class_weights(dataset, num_classes=19):
    """Compute inverse frequency class weights from dataset."""
    class_counts = torch.zeros(num_classes)
    total_pixels = 0
    
    for _, label in dataset:
        label = label.squeeze().numpy()
        for cls in range(num_classes):
            class_counts[cls] += (label == cls).sum()
        total_pixels += (label != 255).sum()
    
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)
    return class_weights / class_weights.sum()


# ==================== DATA TRANSFORMS ====================

def get_train_transforms():
    """Get training data transforms with augmentation."""
    return Compose([
        ToImage(),
        Resize((512, 1024), interpolation=InterpolationMode.BILINEAR),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    """Get target transforms for both train and validation."""
    return Compose([
        ToImage(),
        Resize((512, 1024), interpolation=InterpolationMode.NEAREST),
        RandomHorizontalFlip(p=0.5),  # Only applied during training
        ToDtype(torch.int64),
    ])


class MultiScaleTransform:
    """Apply random scaling during training for improved robustness."""
    def __init__(self, base_size=(512, 1024), scales=[0.75, 1.0, 1.25, 1.5]):
        self.base_size = base_size
        self.scales = scales
        
    def __call__(self, image, label):
        scale = np.random.choice(self.scales)
        new_h = int(self.base_size[0] * scale)
        new_w = int(self.base_size[1] * scale)
        
        # Ensure dimensions are multiples of 32 (important for transformer)
        new_h = (new_h // 32) * 32
        new_w = (new_w // 32) * 32
        
        image = TF.resize(image, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        label = TF.resize(label, (new_h, new_w), interpolation=InterpolationMode.NEAREST)
        
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