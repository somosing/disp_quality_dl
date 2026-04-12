"""Loss utilities for disparity reliability prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ROIWeightConfig:
    """Weights for ROI-aware map regression."""

    interior: float = 4.0
    boundary: float = 8.0
    outside: float = 0.25
    boundary_kernel_size: int = 3
    boundary_iterations: int = 1


def _morphological_dilate(mask: torch.Tensor, kernel_size: int, iterations: int) -> torch.Tensor:
    out = mask
    for _ in range(iterations):
        out = F.max_pool2d(out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (out > 0.5).float()


def _morphological_erode(mask: torch.Tensor, kernel_size: int, iterations: int) -> torch.Tensor:
    # Erode using duality: erode(m) = 1 - dilate(1-m)
    out = mask
    for _ in range(iterations):
        out = 1.0 - F.max_pool2d(
            1.0 - out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
    return (out > 0.5).float()


def compute_roi_boundary_band(
    roi_mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1
) -> torch.Tensor:
    """Build a boundary band from ROI mask via dilate-erode difference.

    Args:
        roi_mask: [B, 1, H, W] binary-ish tensor.
    Returns:
        boundary_band: [B, 1, H, W] in {0,1}
    """
    roi = (roi_mask > 0.5).float()
    dilated = _morphological_dilate(roi, kernel_size=kernel_size, iterations=iterations)
    eroded = _morphological_erode(roi, kernel_size=kernel_size, iterations=iterations)
    boundary = ((dilated - eroded) > 0.0).float()
    return boundary


def build_roi_weight_map(
    roi_mask: torch.Tensor, cfg: ROIWeightConfig | None = None
) -> torch.Tensor:
    """Build per-pixel weight map using interior/boundary/outside ROI regions."""
    if cfg is None:
        cfg = ROIWeightConfig()
    roi = (roi_mask > 0.5).float()
    boundary = compute_roi_boundary_band(
        roi, kernel_size=cfg.boundary_kernel_size, iterations=cfg.boundary_iterations
    )
    interior = (roi > 0.5).float() * (1.0 - boundary)
    outside = 1.0 - roi

    weights = (
        interior * float(cfg.interior)
        + boundary * float(cfg.boundary)
        + outside * float(cfg.outside)
    )
    return weights


def roi_weighted_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    roi_mask: torch.Tensor,
    cfg: ROIWeightConfig | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute ROI-weighted SmoothL1 loss for map regression."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction={prediction.shape} target={target.shape}"
        )
    if roi_mask.shape != prediction.shape:
        raise ValueError(f"roi_mask must match prediction shape, got {roi_mask.shape}")

    if cfg is None:
        cfg = ROIWeightConfig()
    per_pixel = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    weights = build_roi_weight_map(roi_mask, cfg=cfg)
    weighted = per_pixel * weights
    return weighted.sum() / (weights.sum() + 1e-6)


def compute_predicted_roi_score(
    predicted_map: torch.Tensor, roi_mask: torch.Tensor
) -> torch.Tensor:
    """Mean predicted reliability inside ROI, safe for empty ROI.

    Args:
        predicted_map: [B, 1, H, W] in [0,1]
        roi_mask: [B, 1, H, W] binary-ish
    Returns:
        scores: [B] with 0 for empty ROI masks
    """
    if predicted_map.shape != roi_mask.shape:
        raise ValueError(
            f"Shape mismatch: predicted_map={predicted_map.shape} roi_mask={roi_mask.shape}"
        )
    roi = (roi_mask > 0.5).float()
    area = roi.sum(dim=(1, 2, 3))
    summed = (predicted_map * roi).sum(dim=(1, 2, 3))
    scores = torch.where(area > 0.0, summed / area.clamp_min(1e-6), torch.zeros_like(area))
    return scores


def roi_score_smooth_l1_loss(
    predicted_map: torch.Tensor, roi_mask: torch.Tensor, target_roi_score: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """SmoothL1 loss on scalar ROI score derived from predicted map."""
    pred_roi_score = compute_predicted_roi_score(predicted_map, roi_mask)
    target = target_roi_score.view(-1).to(pred_roi_score.dtype)
    return F.smooth_l1_loss(pred_roi_score, target, reduction="mean", beta=beta)


def compute_losses(
    predicted_map: torch.Tensor,
    target_map: torch.Tensor,
    roi_mask: torch.Tensor,
    target_roi_score: torch.Tensor,
    map_weight_cfg: ROIWeightConfig | None = None,
    map_loss_beta: float = 1.0,
    roi_loss_beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Convenience helper returning both map and scalar ROI-score losses."""
    map_loss = roi_weighted_smooth_l1_loss(
        prediction=predicted_map,
        target=target_map,
        roi_mask=roi_mask,
        cfg=map_weight_cfg,
        beta=map_loss_beta,
    )
    score_loss = roi_score_smooth_l1_loss(
        predicted_map=predicted_map,
        roi_mask=roi_mask,
        target_roi_score=target_roi_score,
        beta=roi_loss_beta,
    )
    return {"map_loss": map_loss, "roi_score_loss": score_loss}
