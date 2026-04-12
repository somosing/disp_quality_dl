"""Lightweight U-Net for disparity reliability map prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two Conv-BN-ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling block: maxpool + double conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upsampling block with skip connection and double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial sizes by center-cropping skip to x.
        if x.shape[-2:] != skip.shape[-2:]:
            dy = skip.size(-2) - x.size(-2)
            dx = skip.size(-1) - x.size(-1)
            skip = skip[
                :,
                :,
                dy // 2 : skip.size(-2) - (dy - dy // 2),
                dx // 2 : skip.size(-1) - (dx - dx // 2),
            ]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ReliabilityUNet(nn.Module):
    """Small U-Net with sigmoid output in [0, 1]."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        self.up1 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch, base_ch)

        self.head = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.head(x)
        return self.activation(x)
