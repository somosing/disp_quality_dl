from __future__ import annotations

import cv2
import numpy as np


def safe_percentile(values: np.ndarray, q: float, default: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.percentile(finite, q))


def robust_normalize(values: np.ndarray, valid: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    selected = values[valid & np.isfinite(values)]
    if selected.size == 0:
        return output
    low = safe_percentile(selected, low_q, 0.0)
    high = safe_percentile(selected, high_q, 1.0)
    if high <= low + 1e-6:
        return output
    output = np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
    output[~valid] = 0.0
    return output


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def build_features(
    disparity: np.ndarray,
    roi: np.ndarray,
    intensity: np.ndarray | None,
    cfg: dict,
) -> tuple[np.ndarray, list[str]]:
    valid = np.isfinite(disparity) & (disparity > 0)
    max_disp = float(cfg["features"].get("max_disparity_px", 256.0))
    residual_clip = float(cfg["features"].get("local_residual_clip_px", 6.0))

    disparity_clean = np.where(valid, disparity, 0.0).astype(np.float32)
    disparity_scaled = np.clip(disparity_clean / max(max_disp, 1e-6), 0.0, 1.0)
    disparity_robust = robust_normalize(disparity_clean, valid)

    grad = gradient_magnitude(disparity_scaled)
    grad = robust_normalize(grad, valid, low_q=0.0, high_q=95.0)

    local_median = cv2.medianBlur(disparity_clean, 5)
    local_residual = np.abs(disparity_clean - local_median)
    local_residual = np.clip(local_residual / max(residual_clip, 1e-6), 0.0, 1.0)
    local_residual[~valid] = 0.0

    if roi.shape != disparity.shape:
        raise ValueError(f"ROI shape {roi.shape} does not match disparity shape {disparity.shape}.")
    roi_channel = roi.astype(np.float32)

    channels = [
        disparity_scaled.astype(np.float32),
        disparity_robust.astype(np.float32),
        valid.astype(np.float32),
        grad.astype(np.float32),
        local_residual.astype(np.float32),
        roi_channel,
    ]
    names = [
        "disparity_scaled",
        "disparity_robust",
        "sensor_valid_mask",
        "disparity_gradient",
        "local_disparity_residual",
        "roi_mask",
    ]

    if cfg["data"].get("use_intensity", False):
        if intensity is None:
            raise ValueError("This checkpoint/config requires an intensity image.")
        intensity_01 = np.clip(intensity.astype(np.float32) / 255.0, 0.0, 1.0)
        intensity_grad = gradient_magnitude(intensity_01)
        intensity_grad = robust_normalize(intensity_grad, np.ones_like(valid, dtype=bool), low_q=0.0, high_q=95.0)
        channels.extend([intensity_01, intensity_grad])
        names.extend(["intensity", "intensity_gradient"])

    return np.stack(channels, axis=0).astype(np.float32), names
