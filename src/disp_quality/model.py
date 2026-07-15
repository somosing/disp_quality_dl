from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, dropout: float):
        super().__init__()
        norm_groups = min(groups, out_channels)
        while out_channels % norm_groups != 0 and norm_groups > 1:
            norm_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class RoiReliabilityUNet(nn.Module):
    """Small fully convolutional U-Net with three output heads.

    The ROI mask is an explicit input channel and is also used for supervision
    and object-level summarisation. Deployment therefore requires disparity and ROI.
    """

    def __init__(self, in_channels: int, base_channels: int = 16, groups: int = 8, dropout: float = 0.05):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b, groups, dropout)
        self.enc2 = ConvBlock(b, b * 2, groups, dropout)
        self.enc3 = ConvBlock(b * 2, b * 4, groups, dropout)
        self.enc4 = ConvBlock(b * 4, b * 8, groups, dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(b * 8, b * 16, groups, dropout)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = ConvBlock(b * 16, b * 8, groups, dropout)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4, groups, dropout)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2, groups, dropout)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b, groups, dropout)
        self.head = nn.Conv2d(b, 3, kernel_size=1)

    @staticmethod
    def _match(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self._match(self.up4(b), e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self._match(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self._match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def build_model(cfg: dict, in_channels: int) -> RoiReliabilityUNet:
    model_cfg = cfg["model"]
    return RoiReliabilityUNet(
        in_channels=in_channels,
        base_channels=int(model_cfg.get("base_channels", 16)),
        groups=int(model_cfg.get("group_norm_groups", 8)),
        dropout=float(model_cfg.get("dropout", 0.05)),
    )


def decode_logits(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "reliability": torch.sigmoid(logits[:, 0:1]),
        "error_normalized": torch.sigmoid(logits[:, 1:2]),
        "bad_score": torch.sigmoid(logits[:, 2:3]),
    }
