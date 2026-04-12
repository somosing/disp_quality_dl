#!/usr/bin/env python3
"""Evaluate a saved checkpoint on the validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.reliability_dataset import CropConfig, ReliabilityDataset, ReliabilityDatasetConfig
from models.reliability_unet import ReliabilityUNet
from train_minimal import make_train_val_indices
from training.losses import ROIWeightConfig, compute_losses, compute_predicted_roi_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="raft_data")
    parser.add_argument("--yolo-model", type=str, default="weights/yolo_roi.pt")
    parser.add_argument(
        "--crop-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--crop-margin-ratio", type=float, default=0.25)
    parser.add_argument("--resize-h", type=int, default=512)
    parser.add_argument("--resize-w", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--disparity-norm-mode",
        type=str,
        default="none",
        choices=["none", "minmax_valid", "zscore_valid"],
    )
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _norm_u8(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def _save_heat(path: Path, arr: np.ndarray, cmap: int = cv2.COLORMAP_VIRIDIS) -> None:
    cv2.imwrite(str(path), cv2.applyColorMap(_norm_u8(arr), cmap))


def _save_mask(path: Path, arr: np.ndarray) -> None:
    cv2.imwrite(str(path), (arr > 0.5).astype(np.uint8) * 255)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    """Numpy rankdata equivalent with average tie handling."""
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_a = a[order]

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        # 1-based average rank for tie group [i, j)
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    return _pearson(rx, ry)


def _select_mid_indices(sorted_indices: List[int], count: int) -> List[int]:
    n = len(sorted_indices)
    if n == 0:
        return []
    k = min(count, n)
    start = max(0, (n - k) // 2)
    return sorted_indices[start : start + k]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resize_to = (args.resize_h, args.resize_w) if args.crop_enabled else None
    dataset = ReliabilityDataset(
        ReliabilityDatasetConfig(
            root_dir=args.dataset_root,
            yolo_model_path=args.yolo_model,
            crop=CropConfig(
                enabled=args.crop_enabled,
                margin_ratio=args.crop_margin_ratio,
                resize_to=resize_to,
            ),
            disparity_norm_mode=args.disparity_norm_mode,
            prefilter_invalid=True,
        )
    )

    train_idx, val_idx = make_train_val_indices(
        total_size=len(dataset),
        val_ratio=args.val_ratio,
        seed=args.seed,
        overfit_small_subset=False,
        overfit_count=8,
    )
    _ = train_idx
    val_set = Subset(dataset, val_idx)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = ReliabilityUNet(in_channels=3, out_channels=1, base_ch=32).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Validation samples: {len(val_set)}")
    print(f"Disparity normalization mode: {args.disparity_norm_mode}")

    weight_cfg = ROIWeightConfig(interior=4.0, boundary=8.0, outside=0.25)

    total_map_loss = 0.0
    total_roi_loss = 0.0
    n_batches = 0

    per_sample_rows: List[Dict[str, object]] = []
    per_sample_vis: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["input"].to(device)
            target_map = batch["target_map"].to(device)
            roi_mask = batch["roi_mask"].to(device)
            target_roi_score = batch["roi_score"].to(device).view(-1)

            pred_map = model(x)
            losses = compute_losses(
                predicted_map=pred_map,
                target_map=target_map,
                roi_mask=roi_mask,
                target_roi_score=target_roi_score,
                map_weight_cfg=weight_cfg,
            )

            total_map_loss += float(losses["map_loss"].item())
            total_roi_loss += float(losses["roi_score_loss"].item())
            n_batches += 1

            pred_roi_score = compute_predicted_roi_score(pred_map, roi_mask)

            for i in range(x.shape[0]):
                dataset_name = batch["dataset_name"][i]
                sample_id = batch["sample_id"][i]
                tgt = float(target_roi_score[i].item())
                pred = float(pred_roi_score[i].item())
                abs_err = abs(pred - tgt)

                per_sample_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "sample_id": sample_id,
                        "target_roi_score": tgt,
                        "predicted_roi_score": pred,
                        "absolute_error": abs_err,
                    }
                )
                per_sample_vis.append(
                    {
                        "dataset_name": dataset_name,
                        "sample_id": sample_id,
                        "target_roi_score": tgt,
                        "predicted_roi_score": pred,
                        "pred_map": pred_map[i, 0].detach().cpu().numpy(),
                        "target_map": target_map[i, 0].detach().cpu().numpy(),
                        "roi_mask": roi_mask[i, 0].detach().cpu().numpy(),
                    }
                )

    val_map_loss = total_map_loss / max(1, n_batches)
    val_roi_score_loss = total_roi_loss / max(1, n_batches)

    target_scores = np.array([float(r["target_roi_score"]) for r in per_sample_rows], dtype=np.float64)
    pred_scores = np.array([float(r["predicted_roi_score"]) for r in per_sample_rows], dtype=np.float64)
    abs_errs = np.abs(pred_scores - target_scores)

    mae = float(np.mean(abs_errs)) if abs_errs.size else float("nan")
    rmse = float(np.sqrt(np.mean((pred_scores - target_scores) ** 2))) if abs_errs.size else float("nan")
    pearson = _pearson(pred_scores, target_scores)
    spearman = _spearman(pred_scores, target_scores)

    print(f"Validation map loss: {val_map_loss:.6f}")
    print(f"Validation ROI score loss: {val_roi_score_loss:.6f}")
    print(f"ROI score MAE: {mae:.6f}")
    print(f"ROI score RMSE: {rmse:.6f}")
    print(f"ROI score Pearson correlation: {pearson:.6f}")
    print(f"ROI score Spearman correlation: {spearman:.6f}")

    csv_path = output_dir / "val_sample_scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_name",
                "sample_id",
                "target_roi_score",
                "predicted_roi_score",
                "absolute_error",
            ],
        )
        writer.writeheader()
        for row in per_sample_rows:
            writer.writerow(row)
    print(f"Saved per-sample CSV: {csv_path}")

    # Qualitative selection by target ROI score.
    sorted_idx = sorted(range(len(per_sample_vis)), key=lambda i: per_sample_vis[i]["target_roi_score"])
    low_idx = sorted_idx[: min(5, len(sorted_idx))]
    mid_idx = _select_mid_indices(sorted_idx, count=5)
    high_idx = sorted_idx[max(0, len(sorted_idx) - 5) :]

    selection_groups = {
        "low_target_scores": low_idx,
        "mid_target_scores": mid_idx,
        "high_target_scores": high_idx,
    }

    qual_root = output_dir / "qualitative"
    for group_name, idxs in selection_groups.items():
        group_dir = qual_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for idx in idxs:
            item = per_sample_vis[idx]
            sample_dir = group_dir / f"{item['dataset_name']}__{item['sample_id']}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            _save_heat(sample_dir / "predicted_map.png", item["pred_map"], cv2.COLORMAP_VIRIDIS)
            _save_heat(sample_dir / "target_map.png", item["target_map"], cv2.COLORMAP_VIRIDIS)
            _save_mask(sample_dir / "roi_mask.png", item["roi_mask"])

            with (sample_dir / "scores.json").open("w") as f:
                json.dump(
                    {
                        "dataset_name": item["dataset_name"],
                        "sample_id": item["sample_id"],
                        "target_roi_score": float(item["target_roi_score"]),
                        "predicted_roi_score": float(item["predicted_roi_score"]),
                        "absolute_error": abs(
                            float(item["predicted_roi_score"]) - float(item["target_roi_score"])
                        ),
                    },
                    f,
                    indent=2,
                )
    print(f"Saved qualitative outputs under: {qual_root}")


if __name__ == "__main__":
    main()
