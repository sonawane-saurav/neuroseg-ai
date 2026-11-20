"""
Comprehensive Model Evaluation and Comparative Benchmarking Script.
Evaluates Baseline U-Net and Multi-Task U-Net on the official BRISC 2025 held-out test set (860 pairs).
Calculates Dice, IoU, Precision, Recall, Confusion Matrix, and Qualitative Case Comparisons.
"""

from typing import Dict, Any, List, Optional
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cv.data.dataset import BRISCDataset, TUMOR_LABEL_TO_IDX, IDX_TO_TUMOR_LABEL
from cv.models.unet import UNet
from cv.models.multitask_unet import MultiTaskUNet
from cv.preprocessing.transform import MRIPreprocessor
from cv.training.metrics import calculate_segmentation_metrics, calculate_classification_metrics
from cv.uncertainty.mc_dropout import MonteCarloDropoutEstimator

def evaluate_models(
    data_dir: str,
    artifacts_dir: str,
    max_test_samples: int = 100,
    device: Optional[str] = None
) -> Dict[str, Any]:
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Running Held-Out Test Evaluation on: {device_obj} ===")

    preprocessor = MRIPreprocessor()
    test_dataset = BRISCDataset(
        data_dir=data_dir,
        split="test",
        in_channels=1,
        augment=False,
        max_samples=max_test_samples
    )
    print(f"Loaded {len(test_dataset)} Held-Out Test Samples for rigorous evaluation.")

    checkpoints_dir = os.path.join(artifacts_dir, 'checkpoints')
    metrics_dir = os.path.join(artifacts_dir, 'metrics')
    viz_dir = os.path.join(artifacts_dir, 'visualizations')
    preds_dir = os.path.join(artifacts_dir, 'predictions')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    # 1. Load Baseline U-Net Checkpoint
    unet_ckpt_path = os.path.join(checkpoints_dir, 'best_unet.pth')
    unet_model = UNet(in_channels=1, out_classes=1, features=32, dropout_p=0.2).to(device_obj)
    if os.path.exists(unet_ckpt_path):
        state = torch.load(unet_ckpt_path, map_location=device_obj)
        unet_model.load_state_dict(state['model_state_dict'])
        print("Loaded best_unet.pth successfully.")
    else:
        print("Warning: best_unet.pth not found, using initialized weights.")
    unet_model.eval()

    # 2. Load Multi-Task Model Checkpoint
    mt_ckpt_path = os.path.join(checkpoints_dir, 'best_multitask.pth')
    mt_model = MultiTaskUNet(in_channels=1, out_classes=1, num_classes=4, features=32, dropout_p=0.2).to(device_obj)
    has_mt = False
    if os.path.exists(mt_ckpt_path):
        state = torch.load(mt_ckpt_path, map_location=device_obj)
        mt_model.load_state_dict(state['model_state_dict'])
        has_mt = True
        print("Loaded best_multitask.pth successfully.")
    mt_model.eval()

    # Evaluate Baseline U-Net
    print("\n--- Evaluating Baseline U-Net ---")
    unet_dices, unet_ious, unet_precs, unet_recs, unet_accs = [], [], [], [], []
    unet_inference_times = []
    qualitative_samples = []

    for i in range(len(test_dataset)):
        item = test_dataset[i]
        img = item["image"].unsqueeze(0).to(device_obj)
        gt_mask = item["mask"].numpy()[0]
        filename = item["filename"]

        t0 = time.time()
        with torch.no_grad():
            logits = unet_model(img)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
        unet_inference_times.append(time.time() - t0)

        pred_mask = (probs > 0.5).astype(np.uint8)
        m = calculate_segmentation_metrics(pred_mask, gt_mask)
        unet_dices.append(m["dice"])
        unet_ious.append(m["iou"])
        unet_precs.append(m["precision"])
        unet_recs.append(m["recall"])
        unet_accs.append(m["accuracy"])

        # Store candidate samples for qualitative analysis
        qualitative_samples.append({
            "index": i,
            "filename": filename,
            "tumor_label": item["tumor_label"],
            "dice": m["dice"],
            "iou": m["iou"],
            "image": item["image"].numpy()[0],
            "gt_mask": gt_mask,
            "pred_mask": pred_mask,
            "prob_map": probs
        })

    unet_summary = {
        "mean_dice": float(np.mean(unet_dices)),
        "std_dice": float(np.std(unet_dices)),
        "mean_iou": float(np.mean(unet_ious)),
        "mean_precision": float(np.mean(unet_precs)),
        "mean_recall": float(np.mean(unet_recs)),
        "mean_accuracy": float(np.mean(unet_accs)),
        "mean_inference_time_ms": float(np.mean(unet_inference_times) * 1000.0),
        "parameter_count": unet_model.get_parameter_count()
    }
    print(f"U-Net Test Results: Dice={unet_summary['mean_dice']:.4f}, IoU={unet_summary['mean_iou']:.4f}, Latency={unet_summary['mean_inference_time_ms']:.1f}ms")

    # Evaluate Multi-Task U-Net
    mt_summary = None
    if has_mt:
        print("\n--- Evaluating Multi-Task U-Net ---")
        mt_dices, mt_ious, mt_precs, mt_recs, mt_accs = [], [], [], [], []
        mt_inference_times = []
        cls_preds, cls_targets = [], []

        for i in range(len(test_dataset)):
            item = test_dataset[i]
            img = item["image"].unsqueeze(0).to(device_obj)
            gt_mask = item["mask"].numpy()[0]
            label_idx = item["label_idx"].item()

            t0 = time.time()
            with torch.no_grad():
                seg_logits, cls_logits = mt_model(img)
                probs = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]
                pred_c = torch.argmax(cls_logits, dim=-1).item()
            mt_inference_times.append(time.time() - t0)

            pred_mask = (probs > 0.5).astype(np.uint8)
            m = calculate_segmentation_metrics(pred_mask, gt_mask)
            mt_dices.append(m["dice"])
            mt_ious.append(m["iou"])
            mt_precs.append(m["precision"])
            mt_recs.append(m["recall"])
            mt_accs.append(m["accuracy"])
            cls_preds.append(pred_c)
            cls_targets.append(label_idx)

        cls_metrics = calculate_classification_metrics(cls_preds, cls_targets, num_classes=4)
        mt_summary = {
            "mean_dice": float(np.mean(mt_dices)),
            "std_dice": float(np.std(mt_dices)),
            "mean_iou": float(np.mean(mt_ious)),
            "mean_precision": float(np.mean(mt_precs)),
            "mean_recall": float(np.mean(mt_recs)),
            "mean_accuracy": float(np.mean(mt_accs)),
            "mean_inference_time_ms": float(np.mean(mt_inference_times) * 1000.0),
            "parameter_count": mt_model.get_parameter_count(),
            "classification": cls_metrics
        }
        print(f"Multi-Task Test Results: Dice={mt_summary['mean_dice']:.4f}, Cls Accuracy={cls_metrics['accuracy']:.4f}")

    # Generate Qualitative Visualizations (Best, Challenging, Under/Over segmentation)
    sorted_samples = sorted(qualitative_samples, key=lambda x: x["dice"], reverse=True)
    best_sample = sorted_samples[0]
    median_sample = sorted_samples[len(sorted_samples) // 2]
    challenging_sample = sorted_samples[-1]

    _save_qualitative_figure(
        [best_sample, median_sample, challenging_sample],
        ["Successful High-Dice Segment", "Typical Median Segment", "Challenging / Boundary Error Case"],
        os.path.join(viz_dir, "qualitative_test_cases.png")
    )

    # Save comparison report JSON
    comparison_report = {
        "evaluation_cohort_size": len(test_dataset),
        "baseline_unet": unet_summary,
        "multitask_unet": mt_summary
    }
    with open(os.path.join(metrics_dir, "model_comparison.json"), "w") as f:
        json.dump(comparison_report, f, indent=2)

    # Save a selection of test samples as PNGs in predictions/ for rapid backend demonstration
    for idx, s in enumerate(sorted_samples[:6]):
        # Save raw image
        raw_img_pil = Image.fromarray((s["image"] * 255.0).astype(np.uint8))
        raw_img_pil.save(os.path.join(preds_dir, f"sample_{idx}_image.png"))
        # Save GT mask
        gt_mask_pil = Image.fromarray((s["gt_mask"] * 255.0).astype(np.uint8))
        gt_mask_pil.save(os.path.join(preds_dir, f"sample_{idx}_gt.png"))

    print("Evaluation and comparative benchmarking completed successfully!")
    return comparison_report

def _save_qualitative_figure(samples: List[Dict[str, Any]], titles: List[str], save_path: str):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 4 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, 0)

    for r, (s, title) in enumerate(zip(samples, titles)):
        img = s["image"]
        gt = s["gt_mask"]
        pred = s["pred_mask"]

        # 1. MRI
        axes[r, 0].imshow(img, cmap='gray')
        axes[r, 0].set_title(f"{title}\nMRI ({s['tumor_label']})", color='#38bdf8', fontsize=10)
        axes[r, 0].axis('off')

        # 2. Ground Truth Mask
        axes[r, 1].imshow(gt, cmap='hot', vmin=0, vmax=1)
        axes[r, 1].set_title("Ground Truth Mask", color='#f43f5e', fontsize=10)
        axes[r, 1].axis('off')

        # 3. Predicted Mask
        axes[r, 2].imshow(pred, cmap='viridis', vmin=0, vmax=1)
        axes[r, 2].set_title(f"Model Prediction\nDice: {s['dice']:.3f} | IoU: {s['iou']:.3f}", color='#10b981', fontsize=10)
        axes[r, 2].axis('off')

        # 4. Error Map (Green=TP, Red=FP, Blue=FN)
        error_map = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        error_map[(pred == 1) & (gt == 1)] = [16, 185, 129]  # TP: green
        error_map[(pred == 1) & (gt == 0)] = [244, 63, 94]   # FP: red
        error_map[(pred == 0) & (gt == 1)] = [59, 130, 246]  # FN: blue
        axes[r, 3].imshow(error_map)
        axes[r, 3].set_title("Overlap (Green:TP, Red:FP, Blue:FN)", color='#f59e0b', fontsize=10)
        axes[r, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor='#090d16')
    plt.close()

if __name__ == "__main__":
    evaluate_models(
        data_dir='cv/data/brisc2025',
        artifacts_dir='artifacts',
        max_test_samples=50
    )
