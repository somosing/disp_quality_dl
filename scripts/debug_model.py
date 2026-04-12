#!/usr/bin/env python3
"""Debug forward/loss pipeline for reliability model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from data.reliability_dataset import CropConfig, ReliabilityDataset, ReliabilityDatasetConfig
from models.reliability_unet import ReliabilityUNet
from training.losses import ROIWeightConfig, compute_losses, compute_predicted_roi_score


CONFIG = {
    "dataset_root": "raft_data",
    "yolo_model_path": "weights/yolo_roi.pt",
    "yolo_img_size": 1024,
    "confidence_threshold": 0.25,
    "reliability_threshold": 5.0,
    "crop_enabled": True,
    "crop_margin_ratio": 0.25,
    "resize_h": 512,
    "resize_w": 512,
    "prefilter_invalid": True,
    "num_samples": 5,
    "output_root": "outputs/model_debug",
    "device": "cpu",
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
    parser.add_argument("--crop-enabled", action="store_true", default=CONFIG["crop_enabled"])
    parser.add_argument(
        "--crop-margin-ratio", type=float, default=CONFIG["crop_margin_ratio"]
    )
    parser.add_argument("--resize-h", type=int, default=CONFIG["resize_h"])
    parser.add_argument("--resize-w", type=int, default=CONFIG["resize_w"])
    parser.add_argument(
        "--prefilter-invalid", action="store_true", default=CONFIG["prefilter_invalid"]
    )
    parser.add_argument("--num-samples", type=int, default=CONFIG["num_samples"])
    parser.add_argument("--output-root", type=str, default=CONFIG["output_root"])
    parser.add_argument("--device", type=str, default=CONFIG["device"])
    return parser.parse_args()


def _norm_u8(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def _save_mask(path: Path, arr: np.ndarray) -> None:
    cv2.imwrite(str(path), (arr > 0.5).astype(np.uint8) * 255)


def _save_heat(path: Path, arr: np.ndarray, cmap: int = cv2.COLORMAP_VIRIDIS) -> None:
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
            prefilter_invalid=args.prefilter_invalid,
        )
    )

    device = torch.device(args.device)
    model = ReliabilityUNet(in_channels=3, out_channels=1, base_ch=32).to(device)
    model.eval()

    print(f"Dataset size: {len(ds)}")
    print(f"Model: {model.__class__.__name__}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n = min(max(2, args.num_samples), len(ds))
    weight_cfg = ROIWeightConfig(interior=4.0, boundary=8.0, outside=0.25)

    for i in range(n):
        sample = ds[i]
        x = sample["input"].unsqueeze(0).to(device)  # [1, 3, H, W]
        target_map = sample["target_map"].unsqueeze(0).to(device)  # [1, 1, H, W]
        roi_mask = sample["roi_mask"].unsqueeze(0).to(device)  # [1, 1, H, W]
        roi_score = sample["roi_score"].view(1).to(device)  # [1]

        with torch.no_grad():
            pred_map = model(x)
            losses = compute_losses(
                predicted_map=pred_map,
                target_map=target_map,
                roi_mask=roi_mask,
                target_roi_score=roi_score,
                map_weight_cfg=weight_cfg,
            )
            pred_roi_score = compute_predicted_roi_score(pred_map, roi_mask)

        dataset_name = sample["dataset_name"]
        sample_id = sample["sample_id"]
        print(
            f"[{i}] {dataset_name}/{sample_id} | "
            f"input={tuple(x.shape)} pred={tuple(pred_map.shape)} "
            f"target={tuple(target_map.shape)} roi={tuple(roi_mask.shape)} | "
            f"map_loss={float(losses['map_loss']):.6f} "
            f"roi_score_loss={float(losses['roi_score_loss']):.6f} | "
            f"pred_roi_score={float(pred_roi_score.item()):.4f} "
            f"target_roi_score={float(roi_score.item()):.4f}"
        )

        pred_np = pred_map[0, 0].detach().cpu().numpy()
        target_np = target_map[0, 0].detach().cpu().numpy()
        roi_np = roi_mask[0, 0].detach().cpu().numpy()

        sample_dir = out_root / f"{dataset_name}__{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _save_heat(sample_dir / "predicted_reliability_map.png", pred_np, cv2.COLORMAP_VIRIDIS)
        _save_heat(sample_dir / "target_reliability_map.png", target_np, cv2.COLORMAP_VIRIDIS)
        _save_mask(sample_dir / "roi_mask.png", roi_np)


if __name__ == "__main__":
    main()
