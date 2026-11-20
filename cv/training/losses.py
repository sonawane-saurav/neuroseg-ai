"""
Loss Functions for Brain Tumor Segmentation and Multi-Task Learning.
Includes DiceLoss, combined BCEDiceLoss, and configurable MultiTaskLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw unnormalized logits.
            targets: (B, 1, H, W) binary ground truth tensor {0.0, 1.0}.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        lambda_cls: float = 0.5
    ):
        """
        Combined Multi-Task Loss:
        Total Loss = Segmentation Loss (BCE + Dice) + lambda_cls * Classification Loss (CrossEntropy).
        """
        super().__init__()
        self.seg_loss_fn = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.lambda_cls = lambda_cls

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor
    ) -> torch.Tensor:
        seg_loss = self.seg_loss_fn(seg_logits, seg_targets)
        cls_loss = self.cls_loss_fn(cls_logits, cls_targets)
        total_loss = seg_loss + self.lambda_cls * cls_loss
        return total_loss, seg_loss, cls_loss
