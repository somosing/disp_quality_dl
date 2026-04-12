"""Runtime ROI detection using Ultralytics YOLO segmentation masks.

This module mirrors the classical runtime behavior:
1) load intensity image
2) run YOLO detector
3) get candidate instance masks
4) pick one final ROI mask by confidence + largest area
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class RuntimeROIConfig:
    """Configuration for runtime ROI detection."""

    model_path: str
    img_size: int = 1024
    conf_threshold: float = 0.25
    device: Optional[str] = None


class RuntimeROIDetector:
    """YOLO-based runtime ROI detector that returns instance masks."""

    def __init__(self, cfg: RuntimeROIConfig) -> None:
        self.cfg = cfg
        self._model = None

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO  # Imported lazily to keep module lightweight.

            self._model = YOLO(self.cfg.model_path)
        return self._model

    def detect_instance_masks(
        self, intensity_image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Run YOLO segmentation and return candidate masks and confidences.

        Args:
            intensity_image: Grayscale or BGR image as numpy array.

        Returns:
            A tuple of (masks, confidences), where masks are boolean arrays
            at the detector output image size.
        """
        if intensity_image.ndim == 3 and intensity_image.shape[2] == 3:
            model_input = cv2.cvtColor(intensity_image, cv2.COLOR_BGR2GRAY)
        else:
            model_input = intensity_image

        model = self._get_model()
        results = model.predict(
            source=model_input,
            imgsz=self.cfg.img_size,
            conf=self.cfg.conf_threshold,
            device=self.cfg.device,
            verbose=False,
        )
        if not results:
            return [], []

        result = results[0]
        if result.masks is None or result.boxes is None:
            return [], []

        mask_data = result.masks.data
        conf_data = result.boxes.conf
        if mask_data is None or conf_data is None:
            return [], []

        mask_np = mask_data.detach().cpu().numpy()
        conf_np = conf_data.detach().cpu().numpy()

        masks: List[np.ndarray] = []
        confs: List[float] = []
        for idx in range(mask_np.shape[0]):
            conf = float(conf_np[idx])
            if conf < self.cfg.conf_threshold:
                continue
            masks.append(mask_np[idx] > 0.5)
            confs.append(conf)
        return masks, confs


def choose_final_roi_mask(
    candidate_masks: Sequence[np.ndarray],
    confidences: Sequence[float],
    conf_threshold: float,
    logger: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """Pick final ROI by confidence threshold and largest area."""
    log = logger or LOGGER
    if not candidate_masks:
        log.warning("ROI detection: no candidate masks returned by detector.")
        return None

    kept: List[np.ndarray] = []
    for mask, conf in zip(candidate_masks, confidences):
        if conf >= conf_threshold:
            kept.append(mask.astype(bool))

    if not kept:
        log.warning(
            "ROI detection: no detections above confidence threshold=%.3f.",
            conf_threshold,
        )
        return None

    areas = [int(np.count_nonzero(mask)) for mask in kept]
    best_idx = int(np.argmax(areas))
    return kept[best_idx]


def resize_mask_to_shape(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize boolean mask to (H, W) using nearest-neighbor interpolation."""
    target_h, target_w = target_hw
    if mask.shape[:2] == (target_h, target_w):
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)
