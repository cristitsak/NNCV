"""
Main training script for robust Cityscapes segmentation using SegFormer.
Implements advanced techniques for robustness including:
- OHEM loss focusing on hard pixels
- Multi-scale training
- Gradient clipping
- Learning rate warmup with cosine annealing
- Weight decay scheduling
"""

import os
import sys
import numpy as np
from argparse import ArgumentParser

import wandb
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torchvision.datasets import Cityscapes

from model import Model
from helpers import (
    convert_to_train_id,
    convert_train_id_to_color,
    RobustCombinedLoss,
    compute_class_weights,
    get_train_transforms,
    get_val_transforms,
    get_target_transforms,
    MultiScaleTransform,
    create_scheduler,
    WeightDecayScheduler,
    compute_iou,
    save_checkpoint,
    load_checkpoint
)


def get_args_parser():
    """Parse command line arguments."""
    parser = ArgumentParser("Robustness-focused training for Cityscapes segmentation")
    
    # Basic arguments
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to Cityscapes dataset")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=4e-5, help="Base learning rate")
    parser.add_argument("--num-workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-id", type=str, default="segformer-robustness", help="WandB experiment name")
    
    # Robustness-specific arguments
    parser.add_argument("--use-ohem", action='store_true', default=True, help="Use OHEM loss")
    parser.add_argument("--ohem-ratio", type=float, default=0.3, help="Ratio of hardest pixels to keep")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing factor")
    parser.add_argument("--use-multi-scale", action='store_true', default=True, help="Use multi-scale training")
    parser.add_argument("--use-gradient-clipping", action='store_true', default=True, help="Use gradient clipping")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--use-weight-decay-decay", action='store_true', default=True, help="Decay weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Number of warmup epochs")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from checkpoint")
    
    return parser


def main(args):
    """Main training function."""
    
    # Initialize wandb
    wandb.init(
        project="cityscapes-segmentation-robustness",
        name=args.experiment_id,
        config=vars(args),
    )
    
    # Create output directory
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========== DATA LOADING ==========
    
    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=get_train_transforms(),
        target_transform=get_target_transforms(),
    )
    
    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=get_val_transforms(),
        target_transform=get_target_transforms(),
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
        
    # ========== MODEL ==========
    model = Model(
        n_classes=19,
        config_path="nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    ).to(device)
    
    # ========== LOSS FUNCTION ==========
    # Compute inverse-frequency class weights to focus on rare classes
    # (rider, train, motorcycle, traffic light, etc.)
    class_weights = compute_class_weights(device=device)
    
    criterion = RobustCombinedLoss(
        ce_weight=0.4,       # reduced: CE alone is noisy on edges
        dice_weight=0.6,     # increased: Dice captures shape/boundary better
        label_smoothing=args.label_smoothing,
        ohem_ratio=args.ohem_ratio,
        use_ohem=args.use_ohem,
        class_weights=class_weights,
    )
    
    # ========== OPTIMIZER ==========
    # Layer-wise learning rates: backbone lower, head higher
    optimizer = AdamW([
        {"params": model.segformer.segformer.parameters(), "lr": args.lr},
        {"params": model.segformer.decode_head.parameters(), "lr": args.lr * 5},
    ], weight_decay=0.01)
    
    # ========== SCHEDULERS ==========
    scheduler = create_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=1e-7
    )
    
    wd_scheduler = None
    if args.use_weight_decay_decay:
        wd_scheduler = WeightDecayScheduler(optimizer, initial_wd=0.01, final_wd=1e-5)
    
    # ========== MIXED PRECISION ==========
    scaler = torch.amp.GradScaler('cuda')    

    # ========== MULTI-SCALE TRANSFORM ==========
    multi_scale = MultiScaleTransform() if args.use_multi_scale else None
    
    # ========== RESUME TRAINING ==========
    start_epoch = 0
    best_valid_loss = float('inf')
    best_valid_miou = 0.0
    
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = load_checkpoint(args.resume_from, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch'] + 1
        best_valid_loss = checkpoint.get('val_loss', float('inf'))
        best_valid_miou = checkpoint.get('val_miou', 0.0)
    
    # ========== TRAINING LOOP ==========
    global_step = start_epoch * len(train_dataloader)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04} \n")
        
        # ---------- Training Phase ----------
        model.train()
        train_losses = []
        
        for i, (images, labels) in enumerate(train_dataloader):
            # Apply multi-scale if enabled
            if multi_scale is not None:
                images, labels = multi_scale(images, labels)
            
            # Convert labels to train IDs
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if args.use_gradient_clipping:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=args.grad_clip_norm
                )
            
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            
            # Log every 50 BATCHES (not global steps) with the batch-level step
            # Using global_step as the x-axis keeps train and val on the same timeline
            if i % 50 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/lr_backbone": optimizer.param_groups[0]['lr'],
                    "train/lr_head": optimizer.param_groups[1]['lr'],
                }, step=global_step)
        
            global_step += 1
        
        # Update weight decay scheduler
        if wd_scheduler is not None:
            wd_scheduler.step(epoch)
        
        avg_train_loss = np.mean(train_losses)
        
        # Log epoch-level training summary — this is the reliable number to watch
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "train/lr_backbone": optimizer.param_groups[0]['lr'],
            "train/lr_head": optimizer.param_groups[1]['lr'],
            "epoch": epoch,
        }, step=global_step)
        
        # ---------- Validation Phase ----------
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        
        print("\nValidating...")
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                
                predictions = outputs.softmax(1).argmax(1)
                all_preds.append(predictions.cpu())
                all_labels.append(labels.cpu())
                
                if i == 0 and (epoch + 1) % 5 == 0:
                    pred_vis = convert_train_id_to_color(predictions.unsqueeze(1).cpu())
                    label_vis = convert_train_id_to_color(labels.unsqueeze(1).cpu())
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    orig_imgs = (images.cpu() * std + mean).clamp(0, 1)

                    wandb.log({
                        "val/sample_input": wandb.Image(orig_imgs[0].permute(1, 2, 0).numpy()),
                        "val/sample_prediction": wandb.Image(pred_vis[0].permute(1, 2, 0).numpy()),
                        "val/sample_ground_truth": wandb.Image(label_vis[0].permute(1, 2, 0).numpy()),
                    }, step=global_step)
        
        # Calculate metrics
        avg_val_loss = np.mean(val_losses)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        ious = compute_iou(all_preds, all_labels)
        avg_miou = np.nanmean(ious)
        
        # Log class-wise IoU
        class_names = [cls.name for cls in Cityscapes.classes if cls.train_id not in [-1, 255]]
        iou_dict = {}
        for i, name in enumerate(class_names):
            if not np.isnan(ious[i]):
                iou_dict[f"val/iou_{name}"] = ious[i]
        
        wandb.log({
            "val/loss": avg_val_loss,
            "val/miou": avg_miou,
            **iou_dict,
        }, step=global_step)
        
        print(f"\nTrain Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mIoU: {avg_miou:.4f}")
        
        # Save best model
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            best_valid_miou = avg_miou
            
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, avg_val_loss, avg_miou, checkpoint_path)
            
            wandb.log({
                "best/val_loss": best_valid_loss,
                "best/val_miou": best_valid_miou,
                "best/epoch": epoch + 1,
            }, step=global_step)
            
            print(f"New best model saved! mIoU: {avg_miou:.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                output_dir,
                f"checkpoint-epoch={epoch+1:04}.pt"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, avg_val_loss, avg_miou, checkpoint_path)
    
    print("Training complete!")
    print(f"Best validation mIoU: {best_valid_miou:.4f}")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)