"""
Baseline U-Net Architecture with Monte Carlo Dropout Support.
Implements modular DoubleConv blocks, skip connections, configurable channels,
and selective Monte Carlo Dropout activation for uncertainty estimation.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv, with optional Dropout."""
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.maxpool_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then DoubleConv with Skip Connection and optional Dropout."""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False, dropout_p: float = 0.0):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if shapes differ slightly due to odd resolutions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return self.dropout(x)

class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 1,
        features: int = 32,
        bilinear: bool = False,
        dropout_p: float = 0.2
    ):
        """
        Baseline U-Net Architecture.
        Args:
            in_channels: 1 for grayscale MRI, 3 for RGB.
            out_classes: 1 for binary tumor segmentation.
            features: Base channel width (default: 32).
            bilinear: If True, uses bilinear interpolation instead of transposed conv.
            dropout_p: Monte Carlo dropout probability in bottleneck and decoder.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.features = features
        self.bilinear = bilinear
        self.dropout_p = dropout_p

        f = features
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2, dropout_p=0.0)
        self.down2 = Down(f * 2, f * 4, dropout_p=0.0)
        self.down3 = Down(f * 4, f * 8, dropout_p=dropout_p)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, (f * 16) // factor, dropout_p=dropout_p)

        self.up1 = Up(f * 16, (f * 8) // factor, bilinear, dropout_p=dropout_p)
        self.up2 = Up(f * 8, (f * 4) // factor, bilinear, dropout_p=dropout_p)
        self.up3 = Up(f * 4, (f * 2) // factor, bilinear, dropout_p=0.0)
        self.up4 = Up(f * 2, f, bilinear, dropout_p=0.0)
        self.outc = OutConv(f, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (B, out_classes, H, W)."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def enable_mc_dropout(self):
        """
        Keeps dropout layers in training mode while preserving eval mode
        for batch normalization and all other layers.
        Enables rigorous Monte Carlo Dropout inference.
        """
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def get_parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}
