from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, env=env)


def make_scene(directory: Path, index: int, h: int = 72, w: int = 104) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:h, :w]
    gt = 40.0 + 0.08 * xx + 0.03 * yy
    sensor = gt.copy()
    sensor += np.sin(xx / 8.0 + index) * 0.7
    roi = ((xx - w / 2) ** 2 / (0.36 * w) ** 2 + (yy - h / 2) ** 2 / (0.34 * h) ** 2) < 1.0
    sensor[(xx + index * 3) % 31 == 0] = 0.0
    sensor[(xx > 55) & (yy > 35) & roi] += 4.5

    cv2.imwrite(str(directory / "disparity.png"), np.round(sensor).astype(np.uint16))
    np.save(directory / "gt_disparity.npy", (gt * 16.0).astype(np.float32))
    cv2.imwrite(str(directory / "roi_mask.png"), (roi.astype(np.uint8) * 255))
    intensity = np.clip(80 + xx + yy + index * 3, 0, 255).astype(np.uint8)
    cv2.imwrite(str(directory / "intensity_left.png"), intensity)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-epoch end-to-end synthetic smoke test.")
    parser.add_argument("--work-dir", type=Path, default=Path("smoke_test_output"))
    args = parser.parse_args()
    work = args.work_dir.resolve()
    data_root = work / "data"
    train_root = data_root / "train"
    val_root = data_root / "validation"
    for index in range(6):
        make_scene(train_root / f"scene_{index:03d}", index)
    for index in range(2):
        make_scene(val_root / f"scene_{index:03d}", index + 100)

    cfg = {
        "seed": 7,
        "device": "cpu",
        "paths": {
            "output_dir": str(work / "training"),
            "split_file": str(work / "split.json"),
        },
        "data": {
            "train_roots": [str(train_root)],
            "val_roots": [str(val_root)],
            "validation_fraction": None,
            "validation_count": None,
            "file_names": {
                "disparity": "disparity.png",
                "ground_truth": "gt_disparity.npy",
                "roi": "roi_mask.png",
                "intensity": "intensity_left.png",
            },
            "sensor_scale": 1.0,
            "gt_scale": 16.0,
            "strict_gt_shape": True,
            "min_roi_pixels": 32,
            "use_intensity": False,
            "group_by_capture": False,
        },
        "features": {"max_disparity_px": 128.0, "local_residual_clip_px": 6.0},
        "patch": {
            "height": 64,
            "width": 96,
            "roi_aware_crop": True,
            "horizontal_flip_probability": 0.5,
            "brightness_contrast_probability": 0.0,
            "gaussian_noise_probability": 0.0,
        },
        "model": {"base_channels": 4, "group_norm_groups": 4, "dropout": 0.0},
        "targets": {
            "bad_pixel_threshold_px": 3.0,
            "reliability_tau_px": 3.0,
            "max_error_px": 16.0,
        },
        "loss": {
            "reliability_weight": 4.0,
            "error_weight": 1.0,
            "bad_pixel_weight": 1.0,
            "consistency_weight": 0.25,
            "bad_pixel_positive_weight": 2.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "micro_batch_size": 1,
            "num_workers": 0,
            "cpu_threads": 1,
            "learning_rate": 0.0002,
            "minimum_learning_rate": 0.00001,
            "scheduler_factor": 0.5,
            "scheduler_patience": 1,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "amp": False,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
            "save_every_epochs": 1,
            "preview_every_epochs": 1,
            "resume_checkpoint": None,
        },
        "evaluation": {"tile_overlap": 16, "max_pixel_samples": 50000},
    }
    config_path = work / "smoke_config.yaml"
    work.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    py = sys.executable
    run([py, "-m", "disp_quality.cli.audit_dataset", "--config", str(config_path), "--output-dir", str(work / "audit")], env)
    run([py, "-m", "disp_quality.cli.make_split", "--config", str(config_path)], env)
    run([py, "-m", "disp_quality.cli.train", "--config", str(config_path)], env)
    sample = train_root / "scene_000"
    run([
        py, "-m", "disp_quality.cli.infer",
        "--checkpoint", str(work / "training" / "best.pth"),
        "--disparity", str(sample / "disparity.png"),
        "--roi-mask", str(sample / "roi_mask.png"),
        "--output-dir", str(work / "inference"),
        "--device", "cpu",
    ], env)
    run([
        py, "-m", "disp_quality.cli.evaluate",
        "--checkpoint", str(work / "training" / "best.pth"),
        "--split-file", str(work / "split.json"),
        "--split-name", "val",
        "--output-dir", str(work / "evaluation"),
        "--device", "cpu",
    ], env)
    print(f"Smoke test completed successfully: {work}")


if __name__ == "__main__":
    main()
