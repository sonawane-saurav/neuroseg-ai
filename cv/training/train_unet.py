"""
Baseline U-Net Model Training Pipeline.
Implements reproducible data loading, AdamW optimization, CosineAnnealingLR scheduling,
early stopping, checkpoint saving, metric tracking, and loss visualization.
"""

from typing import Dict, Any, Optional
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cv.data.dataset import BRISCDataset
from cv.models.unet import UNet
from cv.training.losses import BCEDiceLoss
from cv.training.metrics import calculate_segmentation_metrics

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_unet(
    data_dir: str,
    output_dir: str,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    features: int = 32,
    target_size: tuple = (256, 256),
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    patience: int = 5,
    device: Optional[str] = None
) -> Dict[str, Any]:
    set_seed(42)
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Starting U-Net Training on Device: {device_obj} ===")

    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    metrics_dir = os.path.join(output_dir, 'metrics')
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Dataset & Loaders
    train_dataset = BRISCDataset(
        data_dir=data_dir,
        split="train",
        target_size=target_size,
        in_channels=1,
        augment=True,
        max_samples=max_train_samples
    )
    val_dataset = BRISCDataset(
        data_dir=data_dir,
        split="val",
        target_size=target_size,
        in_channels=1,
        augment=False,
        max_samples=max_val_samples
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Dataset Loaded: Train Samples = {len(train_dataset)}, Val Samples = {len(val_dataset)}")

    # 2. Model & Optimization
    model = UNet(in_channels=1, out_classes=1, features=features, dropout_p=0.2).to(device_obj)
    param_info = model.get_parameter_count()
    print(f"Model Parameters: Total = {param_info['total_parameters']:,}, Trainable = {param_info['trainable_parameters']:,}")

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 3. Training Loop
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": [],
        "epoch_times": []
    }

    best_val_dice = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
        train_dices, train_ious = [], []

        for batch in train_loader:
            images = batch["image"].to(device_obj)
            masks = batch["mask"].to(device_obj)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * images.size(0)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                for p, m in zip(probs, masks):
                    met = calculate_segmentation_metrics(p[0], m[0])
                    train_dices.append(met["dice"])
                    train_ious.append(met["iou"])

        scheduler.step()
        epoch_train_loss = train_loss_accum / len(train_dataset)
        epoch_train_dice = float(np.mean(train_dices))
        epoch_train_iou = float(np.mean(train_ious))

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_dices, val_ious = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device_obj)
                masks = batch["mask"].to(device_obj)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss_accum += loss.item() * images.size(0)

                probs = torch.sigmoid(logits)
                for p, m in zip(probs, masks):
                    met = calculate_segmentation_metrics(p[0], m[0])
                    val_dices.append(met["dice"])
                    val_ious.append(met["iou"])

        epoch_val_loss = val_loss_accum / len(val_dataset)
        epoch_val_dice = float(np.mean(val_dices))
        epoch_val_iou = float(np.mean(val_ious))
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_dice"].append(epoch_train_dice)
        history["val_dice"].append(epoch_val_dice)
        history["train_iou"].append(epoch_train_iou)
        history["val_iou"].append(epoch_val_iou)
        history["epoch_times"].append(epoch_duration)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f}, Dice: {epoch_train_dice:.4f}, IoU: {epoch_train_iou:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Dice: {epoch_val_dice:.4f}, IoU: {epoch_val_iou:.4f}")

        # Checkpointing
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            best_epoch = epoch
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(checkpoints_dir, 'best_unet.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'val_iou': epoch_val_iou,
                'features': features,
                'in_channels': 1,
                'out_classes': 1,
                'target_size': target_size
            }, best_ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Dice: {best_val_dice:.4f} (Epoch {best_epoch})")
                break

    # Save final checkpoint
    final_ckpt_path = os.path.join(checkpoints_dir, 'final_unet.pth')
    torch.save({
        'epoch': len(history['train_loss']),
        'model_state_dict': model.state_dict(),
        'val_dice': history['val_dice'][-1],
        'features': features,
        'target_size': target_size
    }, final_ckpt_path)

    # Save training history JSON
    history_path = os.path.join(metrics_dir, 'unet_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    _plot_curves(history, os.path.join(viz_dir, 'unet_training_curves.png'))

    total_time = time.time() - start_time
    summary = {
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_iou": history["val_iou"][best_epoch - 1] if best_epoch > 0 else 0.0,
        "total_training_time_seconds": total_time,
        "parameter_count": param_info
    }
    with open(os.path.join(metrics_dir, 'unet_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"=== U-Net Training Complete in {total_time:.1f}s! Best Val Dice: {best_val_dice:.4f} ===")
    return summary

def _plot_curves(history: Dict[str, Any], save_path: str):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#38bdf8", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#f43f5e", linewidth=2)
    axes[0].set_title("Training & Validation Loss (BCE + Dice)", color='#f8fafc', fontsize=12)
    axes[0].set_xlabel("Epoch", color='#cbd5e1')
    axes[0].set_ylabel("Loss", color='#cbd5e1')
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(fontsize=10)

    # Dice & IoU curve
    axes[1].plot(epochs, history["train_dice"], label="Train Dice", color="#10b981", linewidth=2)
    axes[1].plot(epochs, history["val_dice"], label="Val Dice", color="#34d399", linestyle='--', linewidth=2)
    axes[1].plot(epochs, history["train_iou"], label="Train IoU", color="#fbbf24", linewidth=2)
    axes[1].plot(epochs, history["val_iou"], label="Val IoU", color="#f59e0b", linestyle='--', linewidth=2)
    axes[1].set_title("Dice & IoU Metrics", color='#f8fafc', fontsize=12)
    axes[1].set_xlabel("Epoch", color='#cbd5e1')
    axes[1].set_ylabel("Score", color='#cbd5e1')
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor='#090d16')
    plt.close()

if __name__ == "__main__":
    train_unet(
        data_dir='cv/data/brisc2025',
        output_dir='artifacts',
        epochs=10,
        batch_size=8,
        features=32,
        max_train_samples=200,
        max_val_samples=50
    )
