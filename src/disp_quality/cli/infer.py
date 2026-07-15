from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from disp_quality.checkpoint import load_checkpoint
from disp_quality.config import resolve_device
from disp_quality.features import build_features
from disp_quality.io import read_disparity, read_gray, resize_image_to_shape, resize_mask_to_shape, write_color_map, write_float_png
from disp_quality.predict import apply_deployment_rules, predict_tiled


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict thesis-aligned disparity reliability outputs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--disparity", type=Path, required=True)
    parser.add_argument("--roi-mask", type=Path, required=True)
    parser.add_argument("--intensity", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sensor-scale", type=float, default=None)
    parser.add_argument("--tile-overlap", type=int, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = checkpoint["config"]
    data_cfg = cfg["data"]

    sensor_scale = float(data_cfg.get("sensor_scale", 1.0) if args.sensor_scale is None else args.sensor_scale)
    disparity = read_disparity(args.disparity, scale=sensor_scale)
    h, w = disparity.shape
    valid = np.isfinite(disparity) & (disparity > 0)

    intensity = None
    if data_cfg.get("use_intensity", False):
        if args.intensity is None:
            raise ValueError("This checkpoint requires --intensity.")
        intensity = read_gray(args.intensity).astype(np.float32)
        if intensity.shape != disparity.shape:
            intensity = resize_image_to_shape(intensity, (h, w))

    roi = read_gray(args.roi_mask) > 0
    if roi.shape != disparity.shape:
        roi = resize_mask_to_shape(roi, (h, w))
    if not roi.any():
        raise ValueError("The ROI mask is empty.")

    features, feature_names = build_features(disparity, roi, intensity, cfg)
    expected_names = list(checkpoint["feature_names"])
    if feature_names != expected_names:
        raise RuntimeError(f"Feature mismatch. Checkpoint expects {expected_names}, built {feature_names}")

    overlap = int(args.tile_overlap if args.tile_overlap is not None else cfg.get("evaluation", {}).get("tile_overlap", 0))
    pred = predict_tiled(model, features, cfg, device, bool(cfg["training"].get("amp", True)), overlap)
    pred = apply_deployment_rules(pred, disparity)

    reliability = pred["reliability"]
    reliability_raw = pred["reliability_raw"]
    error_normalized = pred["error_normalized"]
    error_px = pred["error_px"]
    bad_score = pred["bad_score"]

    roi_reliability = np.where(roi, reliability, 0.0).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "reliability_raw.npy", reliability_raw)
    np.save(args.output_dir / "reliability_map.npy", reliability)
    np.save(args.output_dir / "roi_reliability_map.npy", roi_reliability)
    np.save(args.output_dir / "predicted_normalized_error.npy", error_normalized)
    np.save(args.output_dir / "predicted_error_px.npy", error_px)
    np.save(args.output_dir / "bad_pixel_score.npy", bad_score)

    write_float_png(args.output_dir / "reliability_map.png", reliability)
    write_color_map(args.output_dir / "reliability_map_color.png", reliability)
    write_float_png(args.output_dir / "roi_reliability_map.png", roi_reliability)
    write_color_map(args.output_dir / "roi_reliability_map_color.png", roi_reliability)
    write_float_png(args.output_dir / "predicted_normalized_error.png", error_normalized)
    write_color_map(args.output_dir / "predicted_normalized_error_color.png", error_normalized)
    write_float_png(args.output_dir / "bad_pixel_score.png", bad_score)
    write_color_map(args.output_dir / "bad_pixel_score_color.png", bad_score)

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "disparity": str(args.disparity.resolve()),
        "roi_mask": str(args.roi_mask.resolve()),
        "device": device,
        "shape": [h, w],
        "feature_names": feature_names,
        "sensor_scale": sensor_scale,
        "tile_overlap": overlap,
        "valid_pixel_ratio": float(valid.mean()),
        "roi_pixel_count": int(roi.sum()),
        "roi_valid_pixel_count": int((roi & valid).sum()),
        "roi_mean_reliability_complete_roi": float(reliability[roi].mean()),
        "roi_mean_predicted_normalized_error": float(error_normalized[roi].mean()),
        "roi_mean_bad_pixel_score": float(bad_score[roi].mean()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
