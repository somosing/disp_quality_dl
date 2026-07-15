from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import decode_logits


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() + 1e-6)


def compute_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    pred = decode_logits(logits)
    reliability_target = target[:, 0:1]
    error_target = target[:, 1:2]
    bad_target = target[:, 2:3]

    reliability_loss = masked_mean(
        F.smooth_l1_loss(pred["reliability"], reliability_target, reduction="none"), mask
    )
    error_loss = masked_mean(
        F.smooth_l1_loss(pred["error_normalized"], error_target, reduction="none"), mask
    )

    positive_weight = float(cfg["loss"].get("bad_pixel_positive_weight", 1.0))
    pos_weight = torch.tensor(positive_weight, device=logits.device, dtype=logits.dtype)
    bad_loss_map = F.binary_cross_entropy_with_logits(
        logits[:, 2:3], bad_target, reduction="none", pos_weight=pos_weight
    )
    bad_loss = masked_mean(bad_loss_map, mask)

    max_error = float(cfg["targets"]["max_error_px"])
    tau = float(cfg["targets"]["reliability_tau_px"])
    predicted_error_px = torch.expm1(pred["error_normalized"] * math.log1p(max_error))
    reliability_from_error = torch.exp(-predicted_error_px / max(tau, 1e-6))
    consistency_loss = masked_mean(
        F.smooth_l1_loss(pred["reliability"], reliability_from_error, reduction="none"), mask
    )

    weights = cfg["loss"]
    total = (
        float(weights.get("reliability_weight", 4.0)) * reliability_loss
        + float(weights.get("error_weight", 1.0)) * error_loss
        + float(weights.get("bad_pixel_weight", 1.0)) * bad_loss
        + float(weights.get("consistency_weight", 0.25)) * consistency_loss
    )
    logs = {
        "total": float(total.detach().cpu()),
        "reliability": float(reliability_loss.detach().cpu()),
        "error": float(error_loss.detach().cpu()),
        "bad_pixel": float(bad_loss.detach().cpu()),
        "consistency": float(consistency_loss.detach().cpu()),
    }
    return total, logs
