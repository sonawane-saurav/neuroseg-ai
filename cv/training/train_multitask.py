"""
Feature Three: Multi-Task Learning Training Pipeline.
Jointly trains shared CNN encoder on pixel-level tumor segmentation and
pathological tumor classification (Glioma, Meningioma, Pituitary, No Tumor).
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

from cv.data.dataset import BRISCDataset, TUMOR_LABEL_TO_IDX, IDX_TO_TUMOR_LABEL
from cv.models.multitask_unet import MultiTaskUNet
from cv.training.losses import MultiTaskLoss
from cv.training.metrics import calculate_segmentation_metrics, calculate_classification_metrics

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_multitask(
    data_dir: str,
    output_dir: str,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_cls: float = 0.5,
    features: int = 32,
    target_size: tuple = (256, 256),
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    patience: int = 5,
    device: Optional[str] = None
) -> Dict[str, Any]:
    set_seed(42)
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Starting Multi-Task Training on Device: {device_obj} (lambda_cls={lambda_cls}) ===")

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
        is_multitask=True,
        augment=True,
        max_samples=max_train_samples
    )
    val_dataset = BRISCDataset(
        data_dir=data_dir,
        split="val",
        target_size=target_size,
        in_channels=1,
        is_multitask=True,
        augment=False,
        max_samples=max_val_samples
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Multi-Task Dataset: Train = {len(train_dataset)}, Val = {len(val_dataset)}")

    # 2. Model & Loss
    model = MultiTaskUNet(
        in_channels=1,
        out_classes=1,
        num_classes=4,
        features=features,
        dropout_p=0.2
    ).to(device_obj)
    param_info = model.get_parameter_count()
    print(f"Multi-Task Model Parameters: Total = {param_info['total_parameters']:,}")

    criterion = MultiTaskLoss(bce_weight=0.5, dice_weight=0.5, lambda_cls=lambda_cls)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [], "val_loss": [],
        "train_seg_loss": [], "val_seg_loss": [],
        "train_cls_loss": [], "val_cls_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": [],
        "train_cls_acc": [], "val_cls_acc": [],
        "epoch_times": []
    }

    best_val_score = -1.0 # Combined score: val_dice + val_cls_acc
    best_epoch = -1
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_tot_loss, train_s_loss, train_c_loss = 0.0, 0.0, 0.0
        train_dices, train_ious = [], []
        train_preds, train_targets = [], []

        for batch in train_loader:
            images = batch["image"].to(device_obj)
            masks = batch["mask"].to(device_obj)
            labels = batch["label_idx"].to(device_obj)

            optimizer.zero_grad()
            seg_logits, cls_logits = model(images)
            total_loss, s_loss, c_loss = criterion(seg_logits, masks, cls_logits, labels)
            total_loss.backward()
            optimizer.step()

            train_tot_loss += total_loss.item() * images.size(0)
            train_s_loss += s_loss.item() * images.size(0)
            train_c_loss += c_loss.item() * images.size(0)

            with torch.no_grad():
                probs = torch.sigmoid(seg_logits)
                for p, m in zip(probs, masks):
                    met = calculate_segmentation_metrics(p[0], m[0])
                    train_dices.append(met["dice"])
                    train_ious.append(met["iou"])
                cls_preds = torch.argmax(cls_logits, dim=-1).cpu().numpy()
                train_preds.extend(cls_preds)
                train_targets.extend(labels.cpu().numpy())

        scheduler.step()
        n_train = len(train_dataset)
        ep_t_loss = train_tot_loss / n_train
        ep_s_loss = train_s_loss / n_train
        ep_c_loss = train_c_loss / n_train
        ep_t_dice = float(np.mean(train_dices))
        ep_t_iou = float(np.mean(train_ious))
        ep_t_acc = float(np.mean(np.array(train_preds) == np.array(train_targets)))

        # Validation
        model.eval()
        val_tot_loss, val_s_loss, val_c_loss = 0.0, 0.0, 0.0
        val_dices, val_ious = [], []
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device_obj)
                masks = batch["mask"].to(device_obj)
                labels = batch["label_idx"].to(device_obj)

                seg_logits, cls_logits = model(images)
                total_loss, s_loss, c_loss = criterion(seg_logits, masks, cls_logits, labels)
                val_tot_loss += total_loss.item() * images.size(0)
                val_s_loss += s_loss.item() * images.size(0)
                val_c_loss += c_loss.item() * images.size(0)

                probs = torch.sigmoid(seg_logits)
                for p, m in zip(probs, masks):
                    met = calculate_segmentation_metrics(p[0], m[0])
                    val_dices.append(met["dice"])
                    val_ious.append(met["iou"])
                cls_preds = torch.argmax(cls_logits, dim=-1).cpu().numpy()
                val_preds.extend(cls_preds)
                val_targets.extend(labels.cpu().numpy())

        n_val = len(val_dataset)
        ep_v_loss = val_tot_loss / n_val
        ep_v_s_loss = val_s_loss / n_val
        ep_v_c_loss = val_c_loss / n_val
        ep_v_dice = float(np.mean(val_dices))
        ep_v_iou = float(np.mean(val_ious))
        ep_v_acc = float(np.mean(np.array(val_preds) == np.array(val_targets)))
        ep_dur = time.time() - epoch_start

        history["train_loss"].append(ep_t_loss)
        history["val_loss"].append(ep_v_loss)
        history["train_seg_loss"].append(ep_s_loss)
        history["val_seg_loss"].append(ep_v_s_loss)
        history["train_cls_loss"].append(ep_c_loss)
        history["val_cls_loss"].append(ep_v_c_loss)
        history["train_dice"].append(ep_t_dice)
        history["val_dice"].append(ep_v_dice)
        history["train_iou"].append(ep_t_iou)
        history["val_iou"].append(ep_v_iou)
        history["train_cls_acc"].append(ep_t_acc)
        history["val_cls_acc"].append(ep_v_acc)
        history["epoch_times"].append(ep_dur)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_dur:.1f}s) | "
              f"Train Dice: {ep_t_dice:.4f}, Acc: {ep_t_acc:.4f} | "
              f"Val Dice: {ep_v_dice:.4f}, Acc: {ep_v_acc:.4f}")

        # Combined validation score
        val_score = ep_v_dice + ep_v_acc
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': ep_v_dice,
                'val_iou': ep_v_iou,
                'val_cls_acc': ep_v_acc,
                'features': features,
                'lambda_cls': lambda_cls,
                'target_size': target_size
            }, os.path.join(checkpoints_dir, 'best_multitask.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Score: {best_val_score:.4f}")
                break

    # Save final checkpoint
    torch.save({
        'epoch': len(history['train_loss']),
        'model_state_dict': model.state_dict(),
        'val_dice': history['val_dice'][-1],
        'val_cls_acc': history['val_cls_acc'][-1],
        'features': features
    }, os.path.join(checkpoints_dir, 'final_multitask.pth'))

    # Save history
    with open(os.path.join(metrics_dir, 'multitask_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    summary = {
        "best_epoch": best_epoch,
        "best_val_dice": history["val_dice"][best_epoch - 1] if best_epoch > 0 else 0.0,
        "best_val_iou": history["val_iou"][best_epoch - 1] if best_epoch > 0 else 0.0,
        "best_val_cls_acc": history["val_cls_acc"][best_epoch - 1] if best_epoch > 0 else 0.0,
        "total_training_time_seconds": total_time,
        "parameter_count": param_info
    }
    with open(os.path.join(metrics_dir, 'multitask_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"=== Multi-Task Training Complete in {total_time:.1f}s! Best Val Dice: {summary['best_val_dice']:.4f}, Acc: {summary['best_val_cls_acc']:.4f} ===")
    return summary

if __name__ == "__main__":
    train_multitask(
        data_dir='cv/data/brisc2025',
        output_dir='artifacts',
        epochs=10,
        batch_size=8,
        features=32,
        max_train_samples=200,
        max_val_samples=50
    )
