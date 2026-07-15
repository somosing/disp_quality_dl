from __future__ import annotations

from pathlib import Path

import torch

from .model import build_model


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_val_loss: float, cfg: dict, feature_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "config": cfg,
            "feature_names": feature_names,
            "output_names": ["reliability", "error_normalized", "bad_score"],
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str, load_optimizer: bool = False):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if "config" not in checkpoint or "feature_names" not in checkpoint:
        raise ValueError("Checkpoint is not from the rebuilt ROI disparity-quality repository.")
    cfg = checkpoint["config"]
    feature_names = list(checkpoint["feature_names"])
    model = build_model(cfg, in_channels=len(feature_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
