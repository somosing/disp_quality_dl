from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import build_features
from .io import load_scene
from .targets import build_targets
from .transforms import augment_scene, validation_transform


class DisparityQualityDataset(Dataset):
    def __init__(self, scene_dirs: list[str | Path], cfg: dict, training: bool):
        self.scene_dirs = [Path(p) for p in scene_dirs]
        self.cfg = cfg
        self.training = training
        self.base_seed = int(cfg.get("seed", 42))

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def __getitem__(self, index: int):
        scene = load_scene(self.scene_dirs[index], self.cfg["data"], require_gt=True)
        if scene.ground_truth is None:
            raise RuntimeError(f"Ground truth unexpectedly missing for {scene.directory}")

        # Training draws from the worker-seeded NumPy stream, so augmentation changes
        # across epochs while remaining reproducible for a fixed run seed. Validation
        # uses a stable scene-specific seed.
        if self.training:
            seed = int(np.random.randint(0, 2**31 - 1))
        else:
            seed = self.base_seed + index * 1009
        rng = np.random.default_rng(seed)

        if self.training:
            disparity, gt, roi, intensity = augment_scene(
                scene.disparity, scene.ground_truth, scene.roi, scene.intensity, self.cfg, rng
            )
        else:
            disparity, gt, roi, intensity = validation_transform(
                scene.disparity, scene.ground_truth, scene.roi, scene.intensity, self.cfg, rng
            )

        x, feature_names = build_features(disparity, roi, intensity, self.cfg)
        y, supervision_mask, debug = build_targets(disparity, gt, roi, self.cfg)

        metadata = {
            "scene_name": scene.name,
            "scene_dir": str(scene.directory),
            "feature_names": feature_names,
            "roi_pixels": int(roi.sum()),
            "supervised_pixels": int(debug["supervision"].sum()),
        }
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(supervision_mask),
            metadata,
        )
