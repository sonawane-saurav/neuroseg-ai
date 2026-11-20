"""
Attention U-Net Architecture.
Incorporates Attention Gates (Oktay et al., 2018) on skip connections
to suppress irrelevant background regions and focus feature activation on the tumor lesion.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from cv.models.unet import DoubleConv, Down, OutConv

class AttentionGate(nn.Module):
    """
    Attention Gate for medical image segmentation.
    g: Gating signal from coarser resolution.
    x: Feature map from skip connection.
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        # Match dimensions if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha

class AttentionUp(nn.Module):
    """Upscaling with Attention Gate and DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.ag = AttentionGate(f_g=in_channels // 2, f_l=in_channels // 2, f_int=in_channels // 4)
        self.conv = DoubleConv(in_channels, out_channels)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        g_att = self.ag(g=x, x=g)
        out = torch.cat([g_att, x], dim=1)
        out = self.conv(out)
        return self.dropout(out)

class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 1,
        features: int = 32,
        dropout_p: float = 0.2
    ):
        super().__init__()
        f = features
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2, dropout_p=0.0)
        self.down2 = Down(f * 2, f * 4, dropout_p=0.0)
        self.down3 = Down(f * 4, f * 8, dropout_p=dropout_p)
        self.down4 = Down(f * 8, f * 16, dropout_p=dropout_p)

        self.up1 = AttentionUp(f * 16, f * 8, dropout_p=dropout_p)
        self.up2 = AttentionUp(f * 8, f * 4, dropout_p=dropout_p)
        self.up3 = AttentionUp(f * 4, f * 2, dropout_p=0.0)
        self.up4 = AttentionUp(f * 2, f, dropout_p=0.0)
        self.outc = OutConv(f, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        return self.outc(u4)

    def get_parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}
