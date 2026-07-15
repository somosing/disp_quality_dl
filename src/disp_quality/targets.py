from __future__ import annotations

import numpy as np


TARGET_NAMES = ["reliability", "error_normalized", "bad_pixel"]


def error_px_to_normalized(error_px: np.ndarray, max_error_px: float) -> np.ndarray:
    clipped = np.clip(error_px, 0.0, max_error_px)
    return (np.log1p(clipped) / np.log1p(max_error_px)).astype(np.float32)


def error_normalized_to_px(error_normalized, max_error_px: float):
    return np.expm1(error_normalized * np.log1p(max_error_px))


def build_targets(disparity: np.ndarray, ground_truth: np.ndarray, roi: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    target_cfg = cfg["targets"]
    threshold = float(target_cfg["bad_pixel_threshold_px"])
    tau = float(target_cfg["reliability_tau_px"])
    max_error = float(target_cfg["max_error_px"])

    sensor_valid = np.isfinite(disparity) & (disparity > 0)
    gt_valid = np.isfinite(ground_truth) & (ground_truth > 0)
    supervision = roi & gt_valid
    jointly_valid = supervision & sensor_valid

    error_px = np.full_like(disparity, max_error, dtype=np.float32)
    error_px[jointly_valid] = np.abs(disparity[jointly_valid] - ground_truth[jointly_valid])
    error_px[~supervision] = 0.0

    reliability = np.zeros_like(disparity, dtype=np.float32)
    reliability[jointly_valid] = np.exp(-error_px[jointly_valid] / max(tau, 1e-6))

    bad_pixel = np.zeros_like(disparity, dtype=np.float32)
    bad_pixel[supervision] = ((~sensor_valid[supervision]) | (error_px[supervision] >= threshold)).astype(np.float32)

    error_normalized = error_px_to_normalized(error_px, max_error)
    error_normalized[~supervision] = 0.0

    target = np.stack([reliability, error_normalized, bad_pixel], axis=0).astype(np.float32)
    mask = supervision[None].astype(np.float32)
    debug = {
        "error_px": error_px,
        "sensor_valid": sensor_valid,
        "gt_valid": gt_valid,
        "supervision": supervision,
    }
    return target, mask, debug
