"""
Evaluation Metrics for Medical Image Segmentation and Classification.
Provides numerically stable implementations of Dice, IoU, Precision, Recall,
Specificity, F1, Accuracy, and Confusion Matrix.
"""

from typing import Dict, Any, Union
import numpy as np
import torch

def calculate_segmentation_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Computes pixel-level medical segmentation metrics.
    Supports either probabilities or binary masks as input.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Binarize prediction
    pred_bin = (pred > threshold).astype(np.uint8).flatten()
    target_bin = (target > 0.5).astype(np.uint8).flatten()

    tp = np.sum((pred_bin == 1) & (target_bin == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (target_bin == 1))
    tn = np.sum((pred_bin == 0) & (target_bin == 0))

    # Dice (F1)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    # IoU (Jaccard)
    iou = (tp + eps) / (tp + fp + fn + eps)
    # Precision (Positive Predictive Value)
    precision = (tp + eps) / (tp + fp + eps)
    # Recall (Sensitivity / True Positive Rate)
    recall = (tp + eps) / (tp + fn + eps)
    # Specificity (True Negative Rate)
    specificity = (tn + eps) / (tn + fp + eps)
    # Pixel Accuracy
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(pixel_acc)
    }

def calculate_classification_metrics(
    pred_classes: np.ndarray,
    target_classes: np.ndarray,
    num_classes: int = 4
) -> Dict[str, Any]:
    """Computes multiclass accuracy, macro precision/recall/F1, and confusion matrix."""
    pred_classes = np.array(pred_classes)
    target_classes = np.array(target_classes)

    total = len(target_classes)
    accuracy = float(np.sum(pred_classes == target_classes) / (total + 1e-7))

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(pred_classes, target_classes):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_matrix[t, p] += 1

    # Per-class metrics
    precisions = []
    recalls = []
    f1s = []
    for c in range(num_classes):
        tp = conf_matrix[c, c]
        fp = np.sum(conf_matrix[:, c]) - tp
        fn = np.sum(conf_matrix[c, :]) - tp
        p = (tp + 1e-7) / (tp + fp + 1e-7)
        r = (tp + 1e-7) / (tp + fn + 1e-7)
        f1 = (2 * p * r) / (p + r + 1e-7)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": conf_matrix.tolist()
    }
