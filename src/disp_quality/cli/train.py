from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from disp_quality.checkpoint import save_checkpoint
from disp_quality.cli.make_split import create_split
from disp_quality.config import load_config, resolve_device, save_config
from disp_quality.dataset import DisparityQualityDataset
from disp_quality.losses import compute_loss
from disp_quality.model import build_model, decode_logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_or_create_split(cfg: dict) -> dict:
    split_path = Path(cfg["paths"]["split_file"])
    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8"))
    print(f"Split file does not exist; creating {split_path}")
    return create_split(cfg, split_path)


def average_logs(weighted_logs: list[tuple[dict[str, float], int]]) -> dict[str, float]:
    if not weighted_logs:
        return {}
    total_weight = sum(weight for _, weight in weighted_logs)
    return {
        key: float(sum(row[key] * weight for row, weight in weighted_logs) / max(total_weight, 1))
        for key in weighted_logs[0][0]
    }


def run_epoch(model, loader, cfg, device, optimizer=None, scaler=None) -> dict[str, float]:
    """Run one epoch with optional micro-batching.

    `training.batch_size` is the logical batch size. `micro_batch_size` controls
    how many full-resolution samples are placed on the GPU at once. Gradients
    from all micro-batches are accumulated before one optimizer step, so a
    logical batch size of four can be used even when four full-resolution images
    do not fit in GPU memory simultaneously.
    """
    training = optimizer is not None
    model.train(training)
    collected: list[tuple[dict[str, float], int]] = []
    device_type = "cuda" if device == "cuda" else "cpu"
    amp_enabled = bool(cfg["training"].get("amp", True)) and device == "cuda"
    configured_micro = int(cfg["training"].get("micro_batch_size", cfg["training"]["batch_size"]))
    if configured_micro < 1:
        raise ValueError("training.micro_batch_size must be at least 1.")
    progress = tqdm(loader, desc="train" if training else "validation", leave=False)

    for x_cpu, y_cpu, mask_cpu, _metadata in progress:
        logical_batch = int(x_cpu.shape[0])
        micro_size = min(configured_micro, logical_batch)
        starts = list(range(0, logical_batch, micro_size))
        if training:
            optimizer.zero_grad(set_to_none=True)

        batch_weighted: list[tuple[dict[str, float], int]] = []
        for start_index in starts:
            end_index = min(start_index + micro_size, logical_batch)
            x = x_cpu[start_index:end_index].to(device, non_blocking=True)
            y = y_cpu[start_index:end_index].to(device, non_blocking=True)
            mask = mask_cpu[start_index:end_index].to(device, non_blocking=True)

            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                    logits = model(x)
                    loss, logs = compute_loss(logits, y, mask, cfg)
                if training:
                    assert scaler is not None
                    # Average gradients over all samples in the logical batch.
                    chunk_weight = (end_index - start_index) / logical_batch
                    scaler.scale(loss * chunk_weight).backward()

            weight = end_index - start_index
            batch_weighted.append((logs, weight))
            collected.append((logs, weight))

        if training:
            assert scaler is not None
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["training"].get("gradient_clip_norm", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()

        batch_logs = average_logs(batch_weighted)
        progress.set_postfix(loss=f"{batch_logs['total']:.4f}", rel=f"{batch_logs['reliability']:.4f}")
    return average_logs(collected)


@torch.no_grad()
def save_preview(model, dataset, output_path: Path, device: str, cfg: dict) -> None:
    x, y, mask, metadata = dataset[0]
    device_type = "cuda" if device == "cuda" else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=(device == "cuda" and cfg["training"].get("amp", True))):
        pred = decode_logits(model(x[None].to(device)))
    pred_rel = pred["reliability"][0, 0].float().cpu().numpy()
    pred_err = pred["error_normalized"][0, 0].float().cpu().numpy()
    pred_bad = pred["bad_score"][0, 0].float().cpu().numpy()
    target_rel, target_err, target_bad = y.numpy()
    supervised = mask[0].numpy().astype(bool)
    disp = x[0].numpy()

    def panel(arr, label):
        image = np.clip(arr, 0.0, 1.0)
        image = cv2.applyColorMap(np.round(image * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    mask_panel = panel(supervised.astype(np.float32), "supervision mask")
    panels = [
        panel(disp, "disparity scaled"), mask_panel,
        panel(target_rel, "target reliability"), panel(pred_rel, "pred reliability"),
        panel(target_err, "target error norm"), panel(pred_err, "pred error norm"),
        panel(target_bad, "target bad pixel"), panel(pred_bad, "pred bad score"),
    ]
    grid = np.concatenate([np.concatenate(panels[:4], axis=1), np.concatenate(panels[4:], axis=1)], axis=0)
    cv2.putText(grid, str(metadata["scene_name"]), (8, grid.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ROI-supervised disparity reliability model.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(str(cfg.get("device", "auto")))
    if device == "cpu":
        torch.set_num_threads(int(cfg["training"].get("cpu_threads", 4)))
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.yaml")

    split = load_or_create_split(cfg)
    train_dataset = DisparityQualityDataset(split["train"], cfg, training=True)
    val_dataset = DisparityQualityDataset(split["val"], cfg, training=False)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Training and validation datasets must be non-empty.")

    sample_x, _, _, sample_meta = train_dataset[0]
    feature_names = list(sample_meta["feature_names"])
    model = build_model(cfg, in_channels=sample_x.shape[0]).to(device)

    training_cfg = cfg["training"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_workers = int(training_cfg.get("num_workers", 1))
    loader_kwargs = dict(
        batch_size=int(training_cfg["batch_size"]),
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 1))
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
    if scheduler_name in {"none", "fixed", "off"}:
        scheduler = None
    elif scheduler_name in {"reduce_on_plateau", "plateau"}:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_cfg.get("scheduler_factor", 0.5)),
            patience=int(training_cfg.get("scheduler_patience", 6)),
            min_lr=float(training_cfg.get("minimum_learning_rate", 5e-6)),
        )
    else:
        raise ValueError(f"Unsupported training.scheduler: {scheduler_name}")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and training_cfg.get("amp", True)))

    start_epoch = 1
    best_val = math.inf
    resume = training_cfg.get("resume_checkpoint")
    if resume:
        checkpoint = torch.load(Path(resume), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val_loss", math.inf))
        print(f"Resumed from epoch {start_epoch - 1}: {resume}")

    print(f"Device: {device}")
    print(f"Train scenes: {len(train_dataset)} | Validation scenes: {len(val_dataset)}")
    print(f"Feature channels ({len(feature_names)}): {', '.join(feature_names)}")
    print(
        f"Logical batch size: {int(training_cfg['batch_size'])} | "
        f"GPU micro-batch size: {int(training_cfg.get('micro_batch_size', training_cfg['batch_size']))}"
    )
    print("Outputs: reliability, normalized absolute error, bad-pixel score")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    history_path = output_dir / "training_history.csv"
    history_fields = [
        "epoch", "learning_rate", "train_total", "train_reliability", "train_error", "train_bad_pixel",
        "train_consistency", "val_total", "val_reliability", "val_error", "val_bad_pixel", "val_consistency",
        "best_val_total",
    ]
    if not history_path.exists() or start_epoch == 1:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=history_fields).writeheader()

    patience = int(training_cfg.get("early_stopping_patience", 25))
    min_delta = float(training_cfg.get("early_stopping_min_delta", 1e-5))
    epochs_without_improvement = 0
    total_epochs = int(training_cfg["epochs"])

    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")
        train_logs = run_epoch(model, train_loader, cfg, device, optimizer=optimizer, scaler=scaler)
        val_logs = run_epoch(model, val_loader, cfg, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(val_logs["total"])

        improved = val_logs["total"] < (best_val - min_delta)
        if improved:
            best_val = val_logs["total"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, epoch, best_val, cfg, feature_names)
        if improved:
            save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, epoch, best_val, cfg, feature_names)
        if epoch % int(training_cfg.get("save_every_epochs", 10)) == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scheduler, epoch, best_val, cfg, feature_names)
        if epoch == 1 or epoch % int(training_cfg.get("preview_every_epochs", 5)) == 0:
            save_preview(model, val_dataset, output_dir / "previews" / f"epoch_{epoch:03d}.png", device, cfg)

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            **{f"train_{k}": v for k, v in train_logs.items()},
            **{f"val_{k}": v for k, v in val_logs.items()},
            "best_val_total": best_val,
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=history_fields).writerow(row)
        print(f"train={train_logs['total']:.6f} val={val_logs['total']:.6f} best={best_val:.6f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without validation improvement.")
            break

    print(f"Best checkpoint: {output_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
