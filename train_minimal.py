#!/usr/bin/env python3
"""Minimal trainer for disparity reliability prediction.

This script is intentionally small and focused on sanity-checking learning.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.reliability_dataset import CropConfig, ReliabilityDataset, ReliabilityDatasetConfig
from models.reliability_unet import ReliabilityUNet
from training.losses import ROIWeightConfig, compute_losses, compute_predicted_roi_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=str, default="raft_data")
    parser.add_argument("--yolo-model", type=str, default="weights/yolo_roi.pt")
    parser.add_argument(
        "--disparity-norm-mode",
        type=str,
        default="none",
        choices=["none", "minmax_valid", "zscore_valid"],
    )
    parser.add_argument(
        "--crop-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resize-h", type=int, default=512)
    parser.add_argument("--resize-w", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit-small-subset", action="store_true")
    parser.add_argument("--overfit-count", type=int, default=8)
    parser.add_argument("--lambda-roi", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vis-every", type=int, default=2, help="Save debug predictions every N epochs.")
    parser.add_argument("--print-every-iters", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument(
        "--use-plateau-scheduler",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_train_val_indices(
    total_size: int,
    val_ratio: float,
    seed: int,
    overfit_small_subset: bool = False,
    overfit_count: int = 8,
) -> Tuple[List[int], List[int]]:
    """Split indices into train/val with optional overfit subset mode."""
    if total_size <= 0:
        raise ValueError("Dataset is empty.")

    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if overfit_small_subset:
        count = max(1, min(overfit_count, total_size))
        subset = indices[:count]
        # Overfit sanity mode: train and validate on the same tiny subset.
        return subset, subset

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    val_count = max(1, int(round(total_size * val_ratio)))
    if val_count >= total_size:
        val_count = total_size - 1
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    return train_indices, val_indices


def _norm_u8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def _save_heat(path: Path, arr: np.ndarray, cmap: int = cv2.COLORMAP_VIRIDIS) -> None:
    cv2.imwrite(str(path), cv2.applyColorMap(_norm_u8(arr), cmap))


def _save_mask(path: Path, arr: np.ndarray) -> None:
    cv2.imwrite(str(path), (arr > 0.5).astype(np.uint8) * 255)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    weight_cfg: ROIWeightConfig,
    lambda_roi: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_map = 0.0
    total_roi = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            target_map = batch["target_map"].to(device)
            roi_mask = batch["roi_mask"].to(device)
            roi_score = batch["roi_score"].to(device).view(-1)

            pred_map = model(x)
            losses = compute_losses(
                predicted_map=pred_map,
                target_map=target_map,
                roi_mask=roi_mask,
                target_roi_score=roi_score,
                map_weight_cfg=weight_cfg,
            )
            map_loss = losses["map_loss"]
            roi_loss = losses["roi_score_loss"]
            loss = map_loss + (lambda_roi * roi_loss)

            total_loss += float(loss.item())
            total_map += float(map_loss.item())
            total_roi += float(roi_loss.item())
            n_batches += 1

    if n_batches == 0:
        return {"loss": float("nan"), "map_loss": float("nan"), "roi_score_loss": float("nan")}
    return {
        "loss": total_loss / n_batches,
        "map_loss": total_map / n_batches,
        "roi_score_loss": total_roi / n_batches,
    }


def save_debug_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_samples: int = 3,
) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            target_map = batch["target_map"].to(device)
            roi_mask = batch["roi_mask"].to(device)
            target_roi = batch["roi_score"].to(device).view(-1)

            pred_map = model(x)
            pred_roi = compute_predicted_roi_score(pred_map, roi_mask)

            bsz = x.shape[0]
            for bi in range(bsz):
                if saved >= max_samples:
                    return
                dataset_name = batch["dataset_name"][bi]
                sample_id = batch["sample_id"][bi]
                sample_dir = out_dir / f"{dataset_name}__{sample_id}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                pred_np = pred_map[bi, 0].detach().cpu().numpy()
                tgt_np = target_map[bi, 0].detach().cpu().numpy()
                roi_np = roi_mask[bi, 0].detach().cpu().numpy()

                _save_heat(sample_dir / "predicted_map.png", pred_np, cv2.COLORMAP_VIRIDIS)
                _save_heat(sample_dir / "target_map.png", tgt_np, cv2.COLORMAP_VIRIDIS)
                _save_mask(sample_dir / "roi_mask.png", roi_np)

                print(
                    f"  val-sample {dataset_name}/{sample_id}: "
                    f"pred_roi_score={float(pred_roi[bi].item()):.4f}, "
                    f"target_roi_score={float(target_roi[bi].item()):.4f}"
                )
                saved += 1


def append_history_row(history_csv_path: Path, row: Dict[str, float | int]) -> None:
    """Append one epoch row to CSV; write header once."""
    header = [
        "epoch",
        "train_loss",
        "train_map_loss",
        "train_roi_loss",
        "val_loss",
        "val_map_loss",
        "val_roi_loss",
        "learning_rate",
    ]
    is_new_file = not history_csv_path.exists()
    with history_csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    train_debug_dir = output_dir / "train_debug"
    train_log_dir = output_dir / "train_logs"
    history_csv_path = train_log_dir / "history.csv"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_debug_dir.mkdir(parents=True, exist_ok=True)
    train_log_dir.mkdir(parents=True, exist_ok=True)

    resize_to = (args.resize_h, args.resize_w) if args.crop_enabled else None
    dataset = ReliabilityDataset(
        ReliabilityDatasetConfig(
            root_dir=args.dataset_root,
            yolo_model_path=args.yolo_model,
            crop=CropConfig(
                enabled=args.crop_enabled,
                margin_ratio=0.25,
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
        overfit_small_subset=args.overfit_small_subset,
        overfit_count=args.overfit_count,
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = ReliabilityUNet(in_channels=3, out_channels=1, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.use_plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )
    weight_cfg = ROIWeightConfig(interior=4.0, boundary=8.0, outside=0.25)

    print(f"Device: {device}")
    print(f"Disparity normalization mode: {args.disparity_norm_mode}")
    print(f"Dataset size: {len(dataset)} | Train: {len(train_set)} | Val: {len(val_set)}")
    if args.overfit_small_subset:
        print(f"Overfit mode enabled on {len(train_set)} samples")

    best_val = float("inf")
    early_stop_counter = 0
    for epoch in range(1, args.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        model.train()
        running_loss = 0.0
        running_map = 0.0
        running_roi = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            x = batch["input"].to(device)
            target_map = batch["target_map"].to(device)
            roi_mask = batch["roi_mask"].to(device)
            roi_score = batch["roi_score"].to(device).view(-1)

            optimizer.zero_grad(set_to_none=True)
            pred_map = model(x)
            losses = compute_losses(
                predicted_map=pred_map,
                target_map=target_map,
                roi_mask=roi_mask,
                target_roi_score=roi_score,
                map_weight_cfg=weight_cfg,
            )
            map_loss = losses["map_loss"]
            roi_loss = losses["roi_score_loss"]
            total_loss = map_loss + (args.lambda_roi * roi_loss)
            total_loss.backward()
            optimizer.step()

            running_loss += float(total_loss.item())
            running_map += float(map_loss.item())
            running_roi += float(roi_loss.item())
            n_batches += 1

            if args.overfit_small_subset and (step % args.print_every_iters == 0):
                print(
                    f"epoch {epoch:03d} iter {step:04d} | "
                    f"total={total_loss.item():.6f} map={map_loss.item():.6f} roi={roi_loss.item():.6f}"
                )

        train_stats = {
            "loss": running_loss / max(1, n_batches),
            "map_loss": running_map / max(1, n_batches),
            "roi_score_loss": running_roi / max(1, n_batches),
        }
        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            weight_cfg=weight_cfg,
            lambda_roi=args.lambda_roi,
        )
        if scheduler is not None:
            scheduler.step(val_stats["loss"])

        print(
            f"epoch {epoch:03d} | "
            f"lr={current_lr:.6e} | "
            f"train_loss={train_stats['loss']:.6f} "
            f"(map={train_stats['map_loss']:.6f}, roi={train_stats['roi_score_loss']:.6f}) | "
            f"val_loss={val_stats['loss']:.6f} "
            f"(map={val_stats['map_loss']:.6f}, roi={val_stats['roi_score_loss']:.6f})"
        )
        append_history_row(
            history_csv_path=history_csv_path,
            row={
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_map_loss": train_stats["map_loss"],
                "train_roi_loss": train_stats["roi_score_loss"],
                "val_loss": val_stats["loss"],
                "val_map_loss": val_stats["map_loss"],
                "val_roi_loss": val_stats["roi_score_loss"],
                "learning_rate": current_lr,
            },
        )

        last_ckpt = ckpt_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_stats": train_stats,
                "val_stats": val_stats,
                "args": vars(args),
            },
            last_ckpt,
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            early_stop_counter = 0
            best_ckpt = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"  saved new best checkpoint: {best_ckpt}")
        else:
            early_stop_counter += 1

        if (epoch % args.vis_every == 0) or (epoch == 1) or (epoch == args.epochs):
            epoch_dir = train_debug_dir / f"epoch_{epoch:03d}"
            print(f"  saving validation debug predictions to: {epoch_dir}")
            save_debug_predictions(
                model=model,
                loader=val_loader,
                device=device,
                out_dir=epoch_dir,
                max_samples=3,
            )

        if early_stop_counter >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:03d} "
                f"(no val improvement for {args.early_stopping_patience} epochs)."
            )
            break


if __name__ == "__main__":
    main()
