from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ScenePaths:
    directory: Path
    disparity: Path
    ground_truth: Path | None
    roi: Path | None
    intensity: Path | None


@dataclass
class SceneData:
    name: str
    directory: Path
    disparity: np.ndarray
    ground_truth: np.ndarray | None
    roi: np.ndarray
    intensity: np.ndarray | None


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read grayscale image: {path}")
    return image


def read_disparity(path: Path, scale: float = 1.0) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Disparity scale must be positive.")
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise FileNotFoundError(f"Could not read disparity: {path}")
    if array.ndim != 2:
        raise ValueError(f"Expected single-channel disparity, got {array.shape}: {path}")
    return array.astype(np.float32) / float(scale)


def resize_image_to_shape(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def resize_mask_to_shape(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def make_scene_paths(directory: Path, data_cfg: dict, require_gt: bool = True) -> ScenePaths:
    names = data_cfg["file_names"]
    disparity = directory / names["disparity"]
    gt = directory / names["ground_truth"]
    roi = directory / names["roi"]
    intensity = directory / names["intensity"]
    return ScenePaths(
        directory=directory,
        disparity=disparity,
        ground_truth=gt if require_gt else (gt if gt.exists() else None),
        roi=roi if roi.exists() else None,
        intensity=intensity if intensity.exists() else None,
    )


def required_files(data_cfg: dict, require_gt: bool = True) -> list[str]:
    names = data_cfg["file_names"]
    required = [names["disparity"], names["roi"]]
    if require_gt:
        required.append(names["ground_truth"])
    if data_cfg.get("use_intensity", False):
        required.append(names["intensity"])
    return required


def scan_scene_directories(roots: Iterable[str | Path], data_cfg: dict, require_gt: bool = True) -> tuple[list[Path], list[dict]]:
    valid: list[Path] = []
    rejected: list[dict] = []
    required = required_files(data_cfg, require_gt=require_gt)

    for root_value in roots:
        root = Path(root_value).expanduser()
        if not root.exists():
            rejected.append({"scene": str(root), "reason": "root_missing"})
            continue
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            missing = [name for name in required if not (directory / name).exists()]
            if missing:
                rejected.append({"scene": str(directory), "reason": "missing_files", "missing": missing})
            else:
                valid.append(directory.resolve())
    return valid, rejected


def load_scene(directory: str | Path, data_cfg: dict, require_gt: bool = True) -> SceneData:
    directory = Path(directory)
    paths = make_scene_paths(directory, data_cfg, require_gt=require_gt)

    disparity = read_disparity(paths.disparity, scale=float(data_cfg.get("sensor_scale", 1.0)))
    h, w = disparity.shape

    ground_truth = None
    if paths.ground_truth is not None:
        ground_truth = read_disparity(paths.ground_truth, scale=float(data_cfg.get("gt_scale", 1.0)))
        if ground_truth.shape != disparity.shape:
            if data_cfg.get("strict_gt_shape", True):
                raise ValueError(
                    f"Ground-truth shape {ground_truth.shape} does not match sensor disparity "
                    f"shape {disparity.shape} in {directory}. Do not silently resize disparity GT."
                )
            old_h, old_w = ground_truth.shape
            ground_truth = cv2.resize(ground_truth, (w, h), interpolation=cv2.INTER_LINEAR)
            ground_truth *= w / max(old_w, 1)

    if paths.roi is None:
        roi = np.ones((h, w), dtype=bool)
    else:
        roi = read_gray(paths.roi) > 0
        if roi.shape != disparity.shape:
            roi = resize_mask_to_shape(roi, (h, w))

    intensity = None
    if data_cfg.get("use_intensity", False):
        if paths.intensity is None:
            raise FileNotFoundError(f"Intensity image is required but missing in {directory}")
        intensity = read_gray(paths.intensity).astype(np.float32)
        if intensity.shape != disparity.shape:
            intensity = resize_image_to_shape(intensity, (h, w))

    return SceneData(
        name=directory.name,
        directory=directory,
        disparity=disparity.astype(np.float32),
        ground_truth=None if ground_truth is None else ground_truth.astype(np.float32),
        roi=roi.astype(bool),
        intensity=None if intensity is None else intensity.astype(np.float32),
    )


def write_float_png(path: Path, array_01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.nan_to_num(array_01), 0.0, 1.0)
    cv2.imwrite(str(path), np.round(image * 65535.0).astype(np.uint16))


def write_color_map(path: Path, array_01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.nan_to_num(array_01), 0.0, 1.0)
    image_u8 = np.round(image * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), color)
