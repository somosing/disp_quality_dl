"""Target generation utilities for disparity reliability prediction.

Assumptions documented in this module:
- Sensor and RAFT disparities are loaded as float32 arrays in pixel disparity units.
- Invalid disparity values are <= 0 or non-finite.
- RAFT disparity is aligned to sensor resolution via resize and disparity scaling.
- Reliability is computed only on overlapping valid pixels, then set to 0 for
  invalid sensor pixels (including inside ROI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from detection.runtime_roi import (
    RuntimeROIConfig,
    RuntimeROIDetector,
    choose_final_roi_mask,
    resize_mask_to_shape,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class TargetBuilderConfig:
    """Configuration for target generation."""

    yolo_model_path: str
    yolo_img_size: int = 1024
    conf_threshold: float = 0.25
    reliability_threshold: float = 5.0
    yolo_device: Optional[str] = None


def load_grayscale_intensity(path: str | Path) -> np.ndarray:
    """Load grayscale/intensity image as uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read intensity image: {path}")
    return img


def load_disparity(path: str | Path) -> np.ndarray:
    """Load disparity map (.npy or image format) as float32."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        disp = np.load(p).astype(np.float32)
    else:
        disp = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if disp is None:
            raise FileNotFoundError(f"Could not read disparity file: {path}")
        disp = disp.astype(np.float32)
    if disp.ndim == 3:
        # Use first channel if disparity was saved in multi-channel container.
        disp = disp[..., 0]
    return disp


def align_raft_to_sensor(raft_disp: np.ndarray, sensor_hw: Tuple[int, int]) -> np.ndarray:
    """Resize RAFT disparity to sensor resolution and scale disparity values.

    Important: horizontal resize changes disparity magnitude, so disparity values
    are scaled by horizontal factor (sensor_width / raft_width).
    """
    sensor_h, sensor_w = sensor_hw
    raft_h, raft_w = raft_disp.shape[:2]
    scale_x = float(sensor_w) / float(raft_w)

    raft_resized = cv2.resize(
        raft_disp, (sensor_w, sensor_h), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    raft_resized *= scale_x
    return raft_resized


def create_valid_mask(disp: np.ndarray) -> np.ndarray:
    """Define valid disparity mask: finite and > 0."""
    return np.isfinite(disp) & (disp > 0.0)


def compute_error_and_reliability(
    sensor_disp: np.ndarray,
    raft_disp_aligned: np.ndarray,
    reliability_threshold: float,
    sensor_valid: Optional[np.ndarray] = None,
    raft_valid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute absolute error and soft reliability target map in [0, 1].

    Reliability definition:
        R = clip(1 - |D_sensor - D_raft| / T, 0, 1)
    on overlapping valid pixels.

    Invalid handling:
    - If sensor pixel is invalid, reliability is forced to 0.
    - If both sensor and RAFT are invalid, pixel is ignored in overlap-valid mask.
      Reliability remains 0 by default.
    """
    if sensor_valid is None:
        sensor_valid = create_valid_mask(sensor_disp)
    if raft_valid is None:
        raft_valid = create_valid_mask(raft_disp_aligned)

    overlap_valid = sensor_valid & raft_valid
    abs_error = np.zeros_like(sensor_disp, dtype=np.float32)
    abs_error[overlap_valid] = np.abs(
        sensor_disp[overlap_valid] - raft_disp_aligned[overlap_valid]
    )

    reliability = np.zeros_like(sensor_disp, dtype=np.float32)
    reliability[overlap_valid] = np.clip(
        1.0 - (abs_error[overlap_valid] / float(reliability_threshold)), 0.0, 1.0
    )

    # Requirement: invalid sensor pixels inside ROI must be 0.
    # We enforce this globally so behavior is explicit and stable.
    reliability[~sensor_valid] = 0.0

    return {
        "abs_error": abs_error,
        "reliability": reliability,
        "sensor_valid": sensor_valid,
        "raft_valid": raft_valid,
        "overlap_valid": overlap_valid,
    }


def compute_roi_score_target(
    reliability: np.ndarray, roi_mask: Optional[np.ndarray]
) -> float:
    """Compute ROI score target as mean reliability inside ROI."""
    if roi_mask is None:
        return float("nan")
    roi = roi_mask.astype(bool)
    area = int(np.count_nonzero(roi))
    if area == 0:
        return float("nan")
    return float(np.mean(reliability[roi]))


class TargetBuilder:
    """End-to-end per-sample target builder for reliability supervision."""

    def __init__(self, cfg: TargetBuilderConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or LOGGER
        self.roi_detector = RuntimeROIDetector(
            RuntimeROIConfig(
                model_path=cfg.yolo_model_path,
                img_size=cfg.yolo_img_size,
                conf_threshold=cfg.conf_threshold,
                device=cfg.yolo_device,
            )
        )

    def build_sample(
        self,
        sample_id: str,
        intensity_path: str | Path,
        sensor_disp_path: str | Path,
        raft_disp_path: str | Path,
    ) -> Dict[str, np.ndarray | float | Tuple[int, int] | None | str]:
        """Build all intermediate maps and scalar targets for one sample."""
        intensity = load_grayscale_intensity(intensity_path)
        sensor_disp = load_disparity(sensor_disp_path)
        raft_disp = load_disparity(raft_disp_path)

        sensor_h, sensor_w = sensor_disp.shape[:2]
        raft_aligned = align_raft_to_sensor(raft_disp, (sensor_h, sensor_w))

        candidate_masks, confidences = self.roi_detector.detect_instance_masks(intensity)
        roi_mask = choose_final_roi_mask(
            candidate_masks, confidences, self.cfg.conf_threshold, logger=self.logger
        )
        if roi_mask is not None:
            roi_mask = resize_mask_to_shape(roi_mask, (sensor_h, sensor_w))
            roi_area = int(np.count_nonzero(roi_mask))
        else:
            self.logger.warning("Sample %s: ROI is None after runtime detection.", sample_id)
            roi_area = 0

        maps = compute_error_and_reliability(
            sensor_disp=sensor_disp,
            raft_disp_aligned=raft_aligned,
            reliability_threshold=self.cfg.reliability_threshold,
        )
        roi_score = compute_roi_score_target(maps["reliability"], roi_mask)

        result: Dict[str, np.ndarray | float | Tuple[int, int] | None | str] = {
            "sample_id": sample_id,
            "intensity": intensity,
            "sensor_disp": sensor_disp,
            "raft_disp_raw": raft_disp,
            "raft_disp_aligned": raft_aligned,
            "roi_mask": roi_mask,
            "roi_area": float(roi_area),
            "roi_score": roi_score,
            "abs_error": maps["abs_error"],
            "reliability": maps["reliability"],
            "sensor_valid": maps["sensor_valid"],
            "raft_valid": maps["raft_valid"],
            "overlap_valid": maps["overlap_valid"],
            "sensor_shape": sensor_disp.shape[:2],
            "raft_shape_raw": raft_disp.shape[:2],
            "raft_shape_aligned": raft_aligned.shape[:2],
            "valid_ratio": float(np.mean(maps["overlap_valid"])),
        }
        return result
