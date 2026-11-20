"""
Multi-Task U-Net Architecture.
Integrates a shared CNN encoder with dual task-specific output heads:
1. Segmentation Decoder: Predicts pixel-level tumor segmentation mask.
2. Classification Head: Predicts tumor pathological classification (Glioma, Meningioma, Pituitary, No Tumor).
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from cv.models.unet import DoubleConv, Down, Up, OutConv

class MultiTaskUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 1,
        num_classes: int = 4,
        features: int = 32,
        bilinear: bool = False,
        dropout_p: float = 0.2
    ):
        """
        Args:
            in_channels: 1 for grayscale MRI, 3 for RGB.
            out_classes: 1 for binary segmentation mask.
            num_classes: Number of classification categories (4: glioma, meningioma, pituitary, no_tumor).
            features: Base feature channel width.
            bilinear: Whether to use bilinear upsampling.
            dropout_p: Dropout probability in bottleneck and decoder.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.num_classes = num_classes
        self.features = features
        self.bilinear = bilinear
        self.dropout_p = dropout_p

        f = features
        # 1. Shared Encoder
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2, dropout_p=0.0)
        self.down2 = Down(f * 2, f * 4, dropout_p=0.0)
        self.down3 = Down(f * 4, f * 8, dropout_p=dropout_p)

        factor = 2 if bilinear else 1
        self.bottleneck = Down(f * 8, (f * 16) // factor, dropout_p=dropout_p)
        bottleneck_channels = (f * 16) // factor

        # 2. Branch A: Segmentation Decoder
        self.up1 = Up(f * 16, (f * 8) // factor, bilinear, dropout_p=dropout_p)
        self.up2 = Up(f * 8, (f * 4) // factor, bilinear, dropout_p=dropout_p)
        self.up3 = Up(f * 4, (f * 2) // factor, bilinear, dropout_p=0.0)
        self.up4 = Up(f * 2, f, bilinear, dropout_p=0.0)
        self.seg_outc = OutConv(f, out_classes)

        # 3. Branch B: Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            seg_logits: (B, out_classes, H, W)
            cls_logits: (B, num_classes)
        """
        # Shared Encoder Feature Extraction
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        feat_bottleneck = self.bottleneck(x4)

        # Branch A: Segmentation Pass
        u1 = self.up1(feat_bottleneck, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        seg_logits = self.seg_outc(u4)

        # Branch B: Classification Pass
        pooled = self.global_pool(feat_bottleneck)
        cls_logits = self.cls_head(pooled)

        return seg_logits, cls_logits

    def enable_mc_dropout(self):
        """Enable MC Dropout for uncertainty evaluation in segmentation decoder."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def get_parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}
