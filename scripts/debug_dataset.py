#!/usr/bin/env python3
"""Debug script for ReliabilityDataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from data.reliability_dataset import CropConfig, ReliabilityDataset, ReliabilityDatasetConfig


CONFIG = {
    "dataset_root": "raft_data",
    "yolo_model_path": "weights/yolo_roi.pt",
    "yolo_img_size": 1024,
    "confidence_threshold": 0.25,
    "reliability_threshold": 5.0,
    "prefilter_invalid": True,
    "crop_enabled": False,
    "crop_margin_ratio": 0.25,
    "resize_h": 512,
    "resize_w": 512,
    "output_root": "outputs/dataset_debug",
    "num_samples": 5,
    "disparity_norm_mode": "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=str, default=CONFIG["dataset_root"])
    parser.add_argument("--yolo-model", type=str, default=CONFIG["yolo_model_path"])
    parser.add_argument("--yolo-imgsz", type=int, default=CONFIG["yolo_img_size"])
    parser.add_argument("--conf-thr", type=float, default=CONFIG["confidence_threshold"])
    parser.add_argument(
        "--reliability-thr", type=float, default=CONFIG["reliability_threshold"]
    )
    parser.add_argument(
        "--prefilter-invalid",
        action="store_true",
        default=CONFIG["prefilter_invalid"],
        help="Precompute and drop invalid ROI samples during index build.",
    )
    parser.add_argument("--crop-enabled", action="store_true", default=CONFIG["crop_enabled"])
    parser.add_argument(
        "--crop-margin-ratio", type=float, default=CONFIG["crop_margin_ratio"]
    )
    parser.add_argument("--resize-h", type=int, default=CONFIG["resize_h"])
    parser.add_argument("--resize-w", type=int, default=CONFIG["resize_w"])
    parser.add_argument("--output-root", type=str, default=CONFIG["output_root"])
    parser.add_argument("--num-samples", type=int, default=CONFIG["num_samples"])
    parser.add_argument(
        "--disparity-norm-mode",
        type=str,
        default=CONFIG["disparity_norm_mode"],
        choices=["none", "minmax_valid", "zscore_valid"],
        help="Per-sample normalization for sensor disparity channel using valid pixels only.",
    )
    return parser.parse_args()


def _norm_u8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    cv2.imwrite(str(path), _norm_u8(arr))


def _save_mask(path: Path, arr: np.ndarray) -> None:
    cv2.imwrite(str(path), (arr.astype(np.float32) > 0.5).astype(np.uint8) * 255)


def _save_heat(path: Path, arr: np.ndarray, cmap: int = cv2.COLORMAP_TURBO) -> None:
    cv2.imwrite(str(path), cv2.applyColorMap(_norm_u8(arr), cmap))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    resize_to = None
    if args.crop_enabled:
        resize_to = (args.resize_h, args.resize_w)

    ds = ReliabilityDataset(
        ReliabilityDatasetConfig(
            root_dir=args.dataset_root,
            yolo_model_path=args.yolo_model,
            yolo_img_size=args.yolo_imgsz,
            conf_threshold=args.conf_thr,
            reliability_threshold=args.reliability_thr,
            crop=CropConfig(
                enabled=args.crop_enabled,
                margin_ratio=args.crop_margin_ratio,
                resize_to=resize_to,
            ),
            disparity_norm_mode=args.disparity_norm_mode,
            prefilter_invalid=args.prefilter_invalid,
        )
    )

    print(f"Dataset size: {len(ds)}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n = min(args.num_samples, len(ds))
    for i in range(n):
        sample = ds[i]
        x = sample["input"].numpy()  # [3, H, W]
        target = sample["target_map"].numpy()[0]
        roi_mask = sample["roi_mask"].numpy()[0]
        roi_score = float(sample["roi_score"].item())
        dataset_name = sample["dataset_name"]
        sample_id = sample["sample_id"]

        print(
            f"[{i}] {dataset_name}/{sample_id} | "
            f"input={tuple(sample['input'].shape)} | "
            f"target={tuple(sample['target_map'].shape)} | "
            f"roi_mask={tuple(sample['roi_mask'].shape)} | "
            f"roi_score={roi_score:.4f} | "
            f"disp_norm_mode={args.disparity_norm_mode}"
        )

        sample_dir = out_root / f"{dataset_name}__{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        sensor_disp = x[0]
        valid_mask = x[1]
        intensity = x[2]

        # This channel reflects the configured normalized disparity input.
        _save_heat(
            sample_dir / "input_sensor_disparity_normalized.png",
            sensor_disp,
            cv2.COLORMAP_TURBO,
        )
        _save_gray(sample_dir / "input_sensor_disparity_normalized_gray.png", sensor_disp)
        _save_mask(sample_dir / "input_sensor_valid_mask.png", valid_mask)
        _save_gray(sample_dir / "input_intensity.png", intensity)
        _save_heat(sample_dir / "target_map.png", target, cv2.COLORMAP_VIRIDIS)
        _save_mask(sample_dir / "roi_mask.png", roi_mask)


if __name__ == "__main__":
    main()
