"""PyTorch dataset for disparity reliability training data preparation.

This module reuses TargetBuilder for all target-generation logic to keep behavior
consistent with the visualization/debug pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.target_builder import TargetBuilder, TargetBuilderConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class SceneRecord:
    """Index entry for one scene sample."""

    dataset_name: str
    scene_id: str
    intensity_path: Path
    sensor_disp_path: Path
    raft_disp_path: Path

    @property
    def unique_id(self) -> str:
        return f"{self.dataset_name}/{self.scene_id}"


@dataclass
class CropConfig:
    """ROI-centered cropping configuration."""

    enabled: bool = False
    margin_ratio: float = 0.25
    resize_to: Optional[Tuple[int, int]] = None  # (H, W)


@dataclass
class ReliabilityDatasetConfig:
    """Configuration for scene indexing and dataset preparation."""

    root_dir: str
    yolo_model_path: str
    yolo_img_size: int = 1024
    conf_threshold: float = 0.25
    reliability_threshold: float = 5.0
    yolo_device: Optional[str] = None
    intensity_filename: str = "kl_depth_intensity_0.png"
    sensor_disp_filename: str = "kl_depth_disparity_0.png"
    raft_disp_filename: str = "raft_disp.npy"
    crop: CropConfig = field(default_factory=CropConfig)
    # Per-sample disparity normalization applied to sensor disparity input channel only.
    # Options:
    # - "none": keep original disparity values
    # - "minmax_valid": scale valid pixels to [0, 1] using valid min/max
    # - "zscore_valid": normalize valid pixels with valid mean/std
    disparity_norm_mode: Literal["none", "minmax_valid", "zscore_valid"] = "none"
    # If true, filter invalid ROI scenes at dataset construction time.
    prefilter_invalid: bool = True


def _find_scene_records(cfg: ReliabilityDatasetConfig) -> List[SceneRecord]:
    """Build scene index supporting both nested and flat layouts.

    Supported layouts:
    1) Nested: root/<dataset_name>/<scene_id>/
    2) Flat:   root/<scene_id>/
    """
    root = Path(cfg.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    def _scene_paths(scene_dir: Path) -> Tuple[Path, Path, Path]:
        return (
            scene_dir / cfg.intensity_filename,
            scene_dir / cfg.sensor_disp_filename,
            scene_dir / cfg.raft_disp_filename,
        )

    def _is_scene_dir(scene_dir: Path) -> bool:
        intensity_path, sensor_disp_path, raft_disp_path = _scene_paths(scene_dir)
        return intensity_path.exists() and sensor_disp_path.exists() and raft_disp_path.exists()

    records: List[SceneRecord] = []

    # Iterate root children once and handle:
    # - flat scene directories directly under root
    # - nested dataset directories containing scene subfolders
    for child_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        # Flat layout: root/<scene_id>/
        if _is_scene_dir(child_dir):
            intensity_path, sensor_disp_path, raft_disp_path = _scene_paths(child_dir)
            records.append(
                SceneRecord(
                    dataset_name=root.name,
                    scene_id=child_dir.name,
                    intensity_path=intensity_path,
                    sensor_disp_path=sensor_disp_path,
                    raft_disp_path=raft_disp_path,
                )
            )
            continue

        # Nested layout: root/<dataset_name>/<scene_id>/
        dataset_name = child_dir.name
        for scene_dir in sorted([p for p in child_dir.iterdir() if p.is_dir()]):
            if not _is_scene_dir(scene_dir):
                continue
            intensity_path, sensor_disp_path, raft_disp_path = _scene_paths(scene_dir)
            records.append(
                SceneRecord(
                    dataset_name=dataset_name,
                    scene_id=scene_dir.name,
                    intensity_path=intensity_path,
                    sensor_disp_path=sensor_disp_path,
                    raft_disp_path=raft_disp_path,
                )
            )
    return records


def compute_roi_bbox(mask: np.ndarray, margin_ratio: float = 0.25) -> Tuple[int, int, int, int]:
    """Return expanded ROI bbox as (y0, y1, x0, x1), with y1/x1 exclusive."""
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Cannot compute ROI bbox for empty mask.")

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    h = y_max - y_min + 1
    w = x_max - x_min + 1
    pad_h = int(round(h * margin_ratio))
    pad_w = int(round(w * margin_ratio))

    y0 = max(0, y_min - pad_h)
    y1 = min(mask.shape[0], y_max + pad_h + 1)
    x0 = max(0, x_min - pad_w)
    x1 = min(mask.shape[1], x_max + pad_w + 1)
    return y0, y1, x0, x1


def crop_array(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop array using bbox=(y0, y1, x0, x1)."""
    y0, y1, x0, x1 = bbox
    return arr[y0:y1, x0:x1]


def resize_array(
    arr: np.ndarray, out_hw: Tuple[int, int], is_mask: bool = False
) -> np.ndarray:
    """Resize 2D map to out_hw=(H, W)."""
    out_h, out_w = out_hw
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(arr.astype(np.float32), (out_w, out_h), interpolation=interp)
    if is_mask:
        return (resized > 0.5).astype(np.float32)
    return resized.astype(np.float32)


def apply_roi_crop_and_resize(
    sensor_disp: np.ndarray,
    sensor_valid: np.ndarray,
    intensity: np.ndarray,
    target_map: np.ndarray,
    roi_mask: np.ndarray,
    crop_cfg: CropConfig,
) -> Dict[str, np.ndarray]:
    """Apply optional ROI-centered crop and resize to all arrays consistently."""
    arrays = {
        "sensor_disp": sensor_disp.astype(np.float32),
        "sensor_valid": sensor_valid.astype(np.float32),
        "intensity": intensity.astype(np.float32),
        "target_map": target_map.astype(np.float32),
        "roi_mask": roi_mask.astype(np.float32),
    }

    if crop_cfg.enabled:
        bbox = compute_roi_bbox(roi_mask, margin_ratio=crop_cfg.margin_ratio)
        for key in list(arrays.keys()):
            arrays[key] = crop_array(arrays[key], bbox)

    if crop_cfg.resize_to is not None:
        for key in list(arrays.keys()):
            arrays[key] = resize_array(
                arrays[key],
                out_hw=crop_cfg.resize_to,
                is_mask=key in {"sensor_valid", "roi_mask"},
            )
    return arrays


def normalize_sensor_disparity(
    sensor_disp: np.ndarray,
    sensor_valid: np.ndarray,
    mode: Literal["none", "minmax_valid", "zscore_valid"] = "none",
) -> np.ndarray:
    """Normalize sensor disparity per sample using valid pixels only.

    Invalid pixels remain 0 after normalization for all modes.
    """
    disp = sensor_disp.astype(np.float32).copy()
    valid = sensor_valid.astype(bool)

    if mode == "none":
        # Keep original values while enforcing invalid pixels at 0 for consistency.
        disp[~valid] = 0.0
        return disp

    if not np.any(valid):
        return np.zeros_like(disp, dtype=np.float32)

    valid_vals = disp[valid]
    if mode == "minmax_valid":
        vmin = float(np.min(valid_vals))
        vmax = float(np.max(valid_vals))
        if vmax <= vmin:
            out = np.zeros_like(disp, dtype=np.float32)
        else:
            out = np.zeros_like(disp, dtype=np.float32)
            out[valid] = (disp[valid] - vmin) / (vmax - vmin)
    elif mode == "zscore_valid":
        mean = float(np.mean(valid_vals))
        std = float(np.std(valid_vals))
        if std <= 1e-8:
            out = np.zeros_like(disp, dtype=np.float32)
        else:
            out = np.zeros_like(disp, dtype=np.float32)
            out[valid] = (disp[valid] - mean) / std
    else:
        raise ValueError(
            f"Unsupported disparity_norm_mode={mode}. "
            "Expected one of: none, minmax_valid, zscore_valid."
        )

    out[~valid] = 0.0
    return out


class ReliabilityDataset(Dataset):
    """Dataset that returns model inputs + reliability targets per scene."""

    def __init__(
        self,
        cfg: ReliabilityDatasetConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or LOGGER
        if cfg.disparity_norm_mode not in {"none", "minmax_valid", "zscore_valid"}:
            raise ValueError(
                f"Invalid disparity_norm_mode={cfg.disparity_norm_mode}. "
                "Expected: none | minmax_valid | zscore_valid"
            )
        self.target_builder = TargetBuilder(
            TargetBuilderConfig(
                yolo_model_path=cfg.yolo_model_path,
                yolo_img_size=cfg.yolo_img_size,
                conf_threshold=cfg.conf_threshold,
                reliability_threshold=cfg.reliability_threshold,
                yolo_device=cfg.yolo_device,
            ),
            logger=self.logger,
        )

        raw_records = _find_scene_records(cfg)
        if not raw_records:
            raise RuntimeError(f"No scenes found under root_dir={cfg.root_dir}")

        self.records: List[SceneRecord] = []
        # Cache avoids recomputing target generation in __getitem__ if prefiltering is used.
        self._result_cache: Dict[str, Dict[str, Any]] = {}

        if cfg.prefilter_invalid:
            self._prefilter_records(raw_records)
        else:
            self.records = raw_records

        if not self.records:
            raise RuntimeError(
                "No valid samples left after filtering (ROI None/empty or NaN ROI score)."
            )

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        roi_mask = result["roi_mask"]
        roi_area = float(result["roi_area"])
        roi_score = float(result["roi_score"])
        if roi_mask is None:
            return False
        if roi_area <= 0:
            return False
        if np.isnan(roi_score):
            return False
        return True

    def _prefilter_records(self, raw_records: Sequence[SceneRecord]) -> None:
        """Filter out invalid scenes by running runtime target generation once."""
        kept = 0
        for rec in raw_records:
            result = self.target_builder.build_sample(
                sample_id=rec.scene_id,
                intensity_path=rec.intensity_path,
                sensor_disp_path=rec.sensor_disp_path,
                raft_disp_path=rec.raft_disp_path,
            )
            if not self._is_valid_result(result):
                self.logger.info(
                    "Skipping invalid sample %s (ROI missing/empty or NaN score).",
                    rec.unique_id,
                )
                continue
            self.records.append(rec)
            self._result_cache[rec.unique_id] = result
            kept += 1
        self.logger.info(
            "Dataset index built: %d raw scenes, %d valid scenes kept.",
            len(raw_records),
            kept,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _get_target_result(self, rec: SceneRecord) -> Dict[str, Any]:
        cache_key = rec.unique_id
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]
        return self.target_builder.build_sample(
            sample_id=rec.scene_id,
            intensity_path=rec.intensity_path,
            sensor_disp_path=rec.sensor_disp_path,
            raft_disp_path=rec.raft_disp_path,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        result = self._get_target_result(rec)

        if not self._is_valid_result(result):
            # Guard for prefilter_invalid=False path.
            raise RuntimeError(
                f"Sample became invalid at runtime: {rec.unique_id}. "
                "Enable prefilter_invalid=True to remove such samples up-front."
            )

        sensor_disp = result["sensor_disp"].astype(np.float32)
        sensor_valid = result["sensor_valid"].astype(np.float32)
        intensity = result["intensity"].astype(np.float32) / 255.0
        target_map = result["reliability"].astype(np.float32)
        roi_mask = result["roi_mask"].astype(np.float32)
        roi_score = float(result["roi_score"])

        arrays = apply_roi_crop_and_resize(
            sensor_disp=sensor_disp,
            sensor_valid=sensor_valid,
            intensity=intensity,
            target_map=target_map,
            roi_mask=roi_mask,
            crop_cfg=self.cfg.crop,
        )
        # Normalize only the sensor disparity channel, using valid pixels only.
        # This does not affect target generation.
        arrays["sensor_disp"] = normalize_sensor_disparity(
            sensor_disp=arrays["sensor_disp"],
            sensor_valid=arrays["sensor_valid"],
            mode=self.cfg.disparity_norm_mode,
        )

        model_input = np.stack(
            [arrays["sensor_disp"], arrays["sensor_valid"], arrays["intensity"]], axis=0
        ).astype(np.float32)

        target_tensor = torch.from_numpy(arrays["target_map"][None, ...].astype(np.float32))
        roi_mask_tensor = torch.from_numpy(arrays["roi_mask"][None, ...].astype(np.float32))

        return {
            "input": torch.from_numpy(model_input),
            "target_map": target_tensor,
            "roi_mask": roi_mask_tensor,
            "roi_score": torch.tensor(roi_score, dtype=torch.float32),
            "sample_id": rec.scene_id,
            "dataset_name": rec.dataset_name,
        }
