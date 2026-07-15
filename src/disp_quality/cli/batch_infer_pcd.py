from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from disp_quality.checkpoint import load_checkpoint
from disp_quality.config import resolve_device
from disp_quality.features import build_features
from disp_quality.io import (
    load_scene,
    read_gray,
    resize_image_to_shape,
    scan_scene_directories,
    write_color_map,
    write_float_png,
)
from disp_quality.metrics import quality_coverage_curve, safe_binary_metrics, safe_correlation
from disp_quality.pointcloud import (
    CameraCalibration,
    disparity_to_xyz_map,
    extract_points_colors,
    grayscale_rgb,
    load_camera_json,
    scalar_colormap_rgb,
    stride_mask,
    write_cloud,
)
from disp_quality.predict import apply_deployment_rules, predict_tiled
from disp_quality.targets import build_targets


def _find_first(directory: Path, names: list[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def _read_raw_disparity(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"Could not read disparity: {path}")
    if arr.ndim != 2:
        raise ValueError(f"Expected single-channel disparity, got {arr.shape}: {path}")
    return arr.astype(np.float32)


def _scene_directories(args: argparse.Namespace, cfg: dict) -> list[Path]:
    if args.split_file is not None:
        split = json.loads(args.split_file.read_text(encoding="utf-8"))
        if args.split_name not in split:
            raise KeyError(f"Split file has no '{args.split_name}' list")
        scenes = [Path(value).expanduser().resolve() for value in split[args.split_name]]
    else:
        roots = args.scene_root or [Path(value) for value in cfg["data"].get("train_roots", [])]
        scenes, rejected = scan_scene_directories(roots, cfg["data"], require_gt=False)
        if rejected:
            print(f"Scene scan skipped {len(rejected)} directories with missing required files.")
    if args.scene_regex:
        pattern = re.compile(args.scene_regex)
        scenes = [path for path in scenes if pattern.search(path.name)]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise RuntimeError("No inference scenes found.")
    return scenes


def _select_checkpoint_features(all_features: np.ndarray, all_names: list[str], expected_names: list[str]) -> np.ndarray:
    by_name = {name: all_features[index] for index, name in enumerate(all_names)}
    missing = [name for name in expected_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Could not build checkpoint features {missing}. Built features: {all_names}")
    return np.stack([by_name[name] for name in expected_names], axis=0).astype(np.float32)


def _optional_intensity(scene_dir: Path, shape_hw: tuple[int, int], configured_name: str) -> tuple[np.ndarray | None, Path | None]:
    path = _find_first(scene_dir, [configured_name, "intensity_left.png", "kl_depth_intensity_left_0.png", "kl_depth_intensity_0.png"])
    if path is None:
        return None, None
    image = read_gray(path).astype(np.float32)
    if image.shape != shape_hw:
        image = resize_image_to_shape(image, shape_hw)
    return image, path


def _optional_confidence(scene_dir: Path, shape_hw: tuple[int, int]) -> tuple[np.ndarray | None, Path | None]:
    path = _find_first(scene_dir, ["confidence.png", "kl_depth_confidence_0.png", "sensor_confidence.png"])
    if path is None:
        return None, None
    image = read_gray(path).astype(np.float32)
    if image.shape != shape_hw:
        image = resize_image_to_shape(image, shape_hw)
    # Normalize common uint8/uint16 confidence representations to [0,1].
    maximum = float(np.nanmax(image)) if image.size else 0.0
    divisor = 255.0 if maximum <= 255.0 else 65535.0
    return np.clip(image / divisor, 0.0, 1.0).astype(np.float32), path


def _save_prediction_outputs(output_dir: Path, pred: dict[str, np.ndarray], roi: np.ndarray, cfg: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reliability = pred["reliability"].astype(np.float32)
    error_px = pred["error_px"].astype(np.float32)
    bad_score = pred["bad_score"].astype(np.float32)
    max_error = float(cfg["targets"]["max_error_px"])

    arrays = {
        "reliability_map": reliability,
        "roi_reliability_map": np.where(roi, reliability, 0.0).astype(np.float32),
        "predicted_error": error_px,
        "bad_pixel_score": bad_score,
    }
    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array)

    write_float_png(output_dir / "reliability_map.png", reliability)
    write_color_map(output_dir / "reliability_map_color.png", reliability)
    write_float_png(output_dir / "roi_reliability_map.png", arrays["roi_reliability_map"])
    write_color_map(output_dir / "roi_reliability_map_color.png", arrays["roi_reliability_map"])
    error_n = np.clip(error_px / max(max_error, 1e-6), 0.0, 1.0)
    write_float_png(output_dir / "predicted_error_normalized.png", error_n)
    write_color_map(output_dir / "predicted_error_color.png", error_n)
    write_float_png(output_dir / "bad_pixel_score.png", bad_score)
    write_color_map(output_dir / "bad_pixel_score_color.png", bad_score)


def _panel(title: str, image: np.ndarray, shape_wh: tuple[int, int]) -> np.ndarray:
    w, h = shape_wh
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h + 28, w, 3), dtype=np.uint8)
    canvas[28:] = image
    cv2.putText(canvas, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _save_overview(
    path: Path,
    disparity: np.ndarray,
    roi: np.ndarray,
    pred: dict[str, np.ndarray],
    max_error: float,
    intensity: np.ndarray | None,
    true_error: np.ndarray | None,
) -> None:
    valid = np.isfinite(disparity) & (disparity > 0)
    disp_vis = np.zeros_like(disparity, dtype=np.float32)
    vals = disparity[valid]
    if vals.size:
        lo, hi = np.percentile(vals, [2, 98])
        if hi > lo:
            disp_vis = np.clip((disparity - lo) / (hi - lo), 0.0, 1.0)
    disp_color = cv2.applyColorMap((disp_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    roi_vis = (roi.astype(np.uint8) * 255)
    rel_color = cv2.applyColorMap((np.clip(pred["reliability"], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    err_color = cv2.applyColorMap((np.clip(pred["error_px"] / max(max_error, 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    bad_color = cv2.applyColorMap((np.clip(pred["bad_score"], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if true_error is not None:
        sixth = cv2.applyColorMap((np.clip(true_error / max(max_error, 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        sixth_title = "Reference error"
    elif intensity is not None:
        sixth = np.clip(intensity, 0, 255).astype(np.uint8)
        sixth_title = "Intensity"
    else:
        sixth = np.zeros_like(roi_vis)
        sixth_title = ""
    panels = [
        _panel("Sensor disparity", disp_color, (420, 300)),
        _panel("ROI", roi_vis, (420, 300)),
        _panel("Predicted reliability", rel_color, (420, 300)),
        _panel("Predicted error", err_color, (420, 300)),
        _panel("Bad-pixel score", bad_color, (420, 300)),
        _panel(sixth_title, sixth, (420, 300)),
    ]
    top = np.concatenate(panels[:3], axis=1)
    bottom = np.concatenate(panels[3:], axis=1)
    cv2.imwrite(str(path), np.concatenate([top, bottom], axis=0))


def _camera_side(scene_name: str, default_side: str) -> str:
    lower = scene_name.lower()
    if lower.endswith("_l"):
        return "left"
    if lower.endswith("_r"):
        return "right"
    return default_side


def _choose_camera(
    scene_name: str,
    single: CameraCalibration | None,
    left: CameraCalibration | None,
    right: CameraCalibration | None,
    default_side: str,
) -> tuple[CameraCalibration, str]:
    if single is not None:
        return single, "single"
    side = _camera_side(scene_name, default_side)
    camera = left if side == "left" else right
    if camera is None:
        raise RuntimeError(f"No {side} camera calibration available for scene {scene_name}")
    return camera, side


def _write_pointclouds(
    output_dir: Path,
    raw_disparity: np.ndarray,
    roi: np.ndarray,
    pred: dict[str, np.ndarray],
    intensity: np.ndarray | None,
    confidence: np.ndarray | None,
    true_bad: np.ndarray | None,
    calibration: CameraCalibration,
    args: argparse.Namespace,
) -> dict[str, Any]:
    calibration = calibration.scaled_to_shape(raw_disparity.shape)
    xyz, corrected, disparity_valid = disparity_to_xyz_map(raw_disparity, calibration)
    depth = xyz[..., 2]
    depth_valid = np.isfinite(depth) & (depth >= args.min_depth_mm) & (depth <= args.max_depth_mm)
    cloud_mask = roi & disparity_valid & depth_valid & stride_mask(raw_disparity.shape, args.point_stride)
    if not cloud_mask.any():
        raise RuntimeError("No valid ROI points remain after disparity/depth filtering")

    base_rgb = grayscale_rgb(intensity, raw_disparity.shape)
    rel_rgb = scalar_colormap_rgb(pred["reliability"], cv2.COLORMAP_TURBO)
    bad_rgb = scalar_colormap_rgb(pred["bad_score"], cv2.COLORMAP_INFERNO)
    low_rel = roi & (pred["reliability"] <= args.low_reliability_threshold)
    high_bad = roi & (pred["bad_score"] >= args.bad_score_threshold)
    debug_rgb = base_rgb.copy()
    debug_rgb[low_rel] = np.array([255, 0, 0], dtype=np.uint8)
    debug_rgb[high_bad] = np.array([255, 220, 0], dtype=np.uint8)
    debug_rgb[low_rel & high_bad] = np.array([255, 0, 255], dtype=np.uint8)

    outputs: dict[str, list[str]] = {}
    for name, colors in [("cloud_reliability", rel_rgb), ("cloud_debug", debug_rgb)]:
        points, rgb = extract_points_colors(xyz, colors, cloud_mask)
        outputs[name] = write_cloud(output_dir / name, points, rgb, args.pointcloud_format)
    if args.cloud_set in {"diagnostic", "all"}:
        points, rgb = extract_points_colors(xyz, bad_rgb, cloud_mask)
        outputs["cloud_bad_score"] = write_cloud(output_dir / "cloud_bad_score", points, rgb, args.pointcloud_format)
    if true_bad is not None and args.cloud_set in {"diagnostic", "all"}:
        colors = base_rgb.copy()
        colors[true_bad] = np.array([255, 0, 255], dtype=np.uint8)
        points, rgb = extract_points_colors(xyz, colors, cloud_mask)
        outputs["cloud_reference_bad"] = write_cloud(output_dir / "cloud_reference_bad", points, rgb, args.pointcloud_format)
    conflict_count = None
    if confidence is not None and args.cloud_set == "all":
        roi_values = confidence[cloud_mask]
        conf_thr = float(np.percentile(roi_values, args.high_confidence_percentile)) if roi_values.size else 1.0
        conflict = cloud_mask & (confidence >= conf_thr) & low_rel
        colors = base_rgb.copy()
        colors[conflict] = np.array([255, 0, 0], dtype=np.uint8)
        points, rgb = extract_points_colors(xyz, colors, cloud_mask)
        outputs["cloud_high_conf_low_rel"] = write_cloud(output_dir / "cloud_high_conf_low_rel", points, rgb, args.pointcloud_format)
        conflict_count = int(conflict.sum())

    return {
        "point_count": int(cloud_mask.sum()),
        "point_stride": int(args.point_stride),
        "depth_min_mm": float(np.nanmin(depth[cloud_mask])),
        "depth_max_mm": float(np.nanmax(depth[cloud_mask])),
        "corrected_disparity_min": float(np.nanmin(corrected[cloud_mask])),
        "corrected_disparity_max": float(np.nanmax(corrected[cloud_mask])),
        "low_reliability_point_count": int((cloud_mask & low_rel).sum()),
        "high_bad_score_point_count": int((cloud_mask & high_bad).sum()),
        "high_confidence_low_reliability_point_count": conflict_count,
        "calibration": calibration.to_dict(),
        "files": outputs,
        "color_legend": {
            "cloud_reliability": "TURBO colormap of predicted reliability",
            "cloud_bad_score": "INFERNO colormap of predicted bad-pixel score",
            "cloud_debug": "gray=intensity, red=low reliability, yellow=high bad score, magenta=both",
            "cloud_reference_bad": "gray=intensity, magenta=reference-defined bad pixel",
            "cloud_high_conf_low_rel": "gray=intensity, red=high sensor confidence and low predicted reliability",
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference for disparity reliability with optional GT metrics and calibrated point clouds."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, action="append", default=[])
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split-name", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--scene-regex", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--ignore-gt",
        action="store_true",
        help="Ignore ground-truth files completely and perform prediction-only inference.",
    )

    parser.add_argument(
        "--align-gt-like-old-repo",
        action="store_true",
        help=(
            "Align a differently sized GT disparity map to the sensor grid "
            "for post-inference evaluation."
        ),
    )
    parser.add_argument("--no-pointcloud", action="store_true")
    parser.add_argument("--camera-json", type=Path, default=None, help="Use one calibration for every scene.")
    parser.add_argument("--camera-left-json", type=Path, default=Path("configs/cameras/camera_left_kl.json"))
    parser.add_argument("--camera-right-json", type=Path, default=Path("configs/cameras/camera_right_kr.json"))
    parser.add_argument("--default-camera-side", choices=["left", "right"], default="left")
    parser.add_argument("--pointcloud-format", choices=["ply", "pcd", "both"], default="ply")
    parser.add_argument("--cloud-set", choices=["minimal", "diagnostic", "all"], default="diagnostic")
    parser.add_argument("--point-stride", type=int, default=2, help="2 keeps one point per 2x2 image block; use 1 for full density.")
    parser.add_argument("--min-depth-mm", type=float, default=500.0)
    parser.add_argument("--max-depth-mm", type=float, default=6000.0)
    parser.add_argument("--low-reliability-threshold", type=float, default=0.5)
    parser.add_argument("--bad-score-threshold", type=float, default=0.5)
    parser.add_argument("--high-confidence-percentile", type=float, default=75.0)
    parser.add_argument("--max-pixel-samples", type=int, default=None)
    args = parser.parse_args()

    if args.ignore_gt and args.align_gt_like_old_repo:
        raise ValueError(
            "--ignore-gt and --align-gt-like-old-repo cannot be used together."
        )


    if args.split_file is not None and args.scene_root:
        raise ValueError("Use either --split-file or --scene-root, not both.")
    if args.min_depth_mm >= args.max_depth_mm:
        raise ValueError("min depth must be smaller than max depth")

    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = checkpoint["config"]
    if args.align_gt_like_old_repo:
        cfg = {
            **cfg,
            "data": {
                **cfg["data"],
                "strict_gt_shape": False,
            },
        }

    expected_features = list(checkpoint["feature_names"])
    if device == "cpu":
        torch.set_num_threads(int(cfg.get("training", {}).get("cpu_threads", 4)))
    overlap = int(args.tile_overlap if args.tile_overlap is not None else cfg.get("evaluation", {}).get("tile_overlap", 64))
    scenes = _scene_directories(args, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    single_camera = load_camera_json(args.camera_json) if args.camera_json else None
    left_camera = None
    right_camera = None
    if not args.no_pointcloud and single_camera is None:
        if args.camera_left_json and args.camera_left_json.exists():
            left_camera = load_camera_json(args.camera_left_json)
        if args.camera_right_json and args.camera_right_json.exists():
            right_camera = load_camera_json(args.camera_right_json)
        if left_camera is None and right_camera is None:
            raise FileNotFoundError(
                "Point-cloud generation requested, but camera JSON files were not found. "
                "Use --camera-json or --camera-left-json/--camera-right-json."
            )

    max_samples = int(
        args.max_pixel_samples
        if args.max_pixel_samples is not None
        else cfg.get("evaluation", {}).get("max_pixel_samples", 2_000_000)
    )
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    sample_per_scene = max(1, max_samples // max(len(scenes), 1))

    rows: list[dict] = []
    failures: list[dict] = []
    sampled_rel: list[np.ndarray] = []
    sampled_bad_prob: list[np.ndarray] = []
    sampled_bad_true: list[np.ndarray] = []
    sampled_pred_error: list[np.ndarray] = []
    sampled_true_error: list[np.ndarray] = []
    started = time.perf_counter()

    for scene_dir in tqdm(scenes, desc="Batch inference"):
        scene_output = args.output_dir / scene_dir.name
        done_file = scene_output / "summary.json"
        if args.skip_existing and done_file.exists():
            try:
                rows.append(json.loads(done_file.read_text(encoding="utf-8"))["flat_metrics"])
                continue
            except Exception:
                pass
        try:
            data_cfg = cfg["data"]
            if args.ignore_gt:
                data_cfg = {
                    **cfg["data"],
                    "file_names": {
                        **cfg["data"]["file_names"],
                        "ground_truth": "__ignore_ground_truth__.npy",
                    },
                }
            scene = load_scene(scene_dir, data_cfg, require_gt=False)
            disparity = scene.disparity
            roi = scene.roi
            valid = np.isfinite(disparity) & (disparity > 0)
            intensity, intensity_path = _optional_intensity(
                scene_dir, disparity.shape, cfg["data"]["file_names"].get("intensity", "intensity_left.png")
            )
            # Use the checkpoint's configured intensity for the model, while still loading optional intensity for PCD coloring.
            model_intensity = scene.intensity if cfg["data"].get("use_intensity", False) else None
            # Support both repository feature APIs:
            # old: build_features(disparity, intensity, cfg)
            # ROI-input: build_features(disparity, roi, intensity, cfg)
            import inspect

            feature_parameter_count = len(inspect.signature(build_features).parameters)

            if feature_parameter_count == 4:
                all_features, all_names = build_features(
                    disparity,
                    roi,
                    model_intensity,
                    cfg,
                )
            elif feature_parameter_count == 3:
                all_features, all_names = build_features(
                    disparity,
                    model_intensity,
                    cfg,
                )

                # Permit inference with an ROI-input checkpoint even when the
                # local feature builder is the older three-argument version.
                if "roi_mask" in expected_features and "roi_mask" not in all_names:

                    all_features = np.concatenate(
                        [
                            all_features,
                            roi.astype(np.float32)[None, ...],
                        ],
                        axis=0,
                    )
                    all_names = [*all_names, "roi_mask"]
            else:
                raise RuntimeError(
                    "Unsupported build_features signature with "
                    f"{feature_parameter_count} parameters."
                )

            features = _select_checkpoint_features(
                all_features,
                all_names,
                expected_features,
            )
            pred = predict_tiled(
                model=model,
                features=features,
                cfg=cfg,
                device=device,
                use_amp=bool(cfg.get("training", {}).get("amp", True)),
                overlap=overlap,
            )
            pred = apply_deployment_rules(pred, disparity)
            _save_prediction_outputs(scene_output, pred, roi, cfg)

            confidence, confidence_path = _optional_confidence(scene_dir, disparity.shape)
            true_error = None
            true_bad = None
            metrics: dict[str, Any] = {}
            if scene.ground_truth is not None:
                target, mask_1, debug = build_targets(disparity, scene.ground_truth, roi, cfg)
                mask = mask_1[0].astype(bool)
                true_error = debug["error_px"]
                true_bad = target[2].astype(bool)
                if mask.any():
                    rel_values = pred["reliability"][mask]
                    bad_score_values = pred["bad_score"][mask]
                    bad_true_values = target[2][mask]
                    pred_error_values = pred["error_px"][mask]
                    true_error_values = true_error[mask]
                    metrics = {
                        "supervised_pixels": int(mask.sum()),
                        "mean_true_error": float(true_error_values.mean()),
                        "true_bad_pixel_ratio": float(bad_true_values.mean()),
                        "reliability_target_mae": float(np.abs(rel_values - target[0][mask]).mean()),
                        "predicted_error_mae": float(np.abs(pred_error_values - true_error_values).mean()),
                        "bad_head_auroc": safe_binary_metrics(bad_true_values, bad_score_values)["auroc"],
                        "bad_head_auprc": safe_binary_metrics(bad_true_values, bad_score_values)["auprc"],
                        "inverted_reliability_auroc": safe_binary_metrics(bad_true_values, 1.0 - rel_values)["auroc"],
                        "inverted_reliability_auprc": safe_binary_metrics(bad_true_values, 1.0 - rel_values)["auprc"],
                    }
                    count = len(bad_true_values)
                    idx = rng.choice(count, size=sample_per_scene, replace=False) if count > sample_per_scene else np.arange(count)
                    sampled_rel.append(rel_values[idx])
                    sampled_bad_prob.append(bad_score_values[idx])
                    sampled_bad_true.append(bad_true_values[idx])
                    sampled_pred_error.append(pred_error_values[idx])
                    sampled_true_error.append(true_error_values[idx])

            _save_overview(
                scene_output / "overview.png",
                disparity,
                roi,
                pred,
                float(cfg["targets"]["max_error_px"]),
                intensity,
                true_error,
            )

            pcd_summary = None
            camera_side = None
            if not args.no_pointcloud:
                camera, camera_side = _choose_camera(
                    scene_dir.name, single_camera, left_camera, right_camera, args.default_camera_side
                )
                disparity_path = scene_dir / cfg["data"]["file_names"]["disparity"]
                raw_disparity = _read_raw_disparity(disparity_path)
                pcd_summary = _write_pointclouds(
                    scene_output,
                    raw_disparity,
                    roi,
                    pred,
                    intensity,
                    confidence,
                    true_bad,
                    camera,
                    args,
                )

            roi_count = int(roi.sum())
            flat = {
                "scene": scene.name,
                "path": str(scene.directory),
                "camera_side": camera_side,
                "height": int(disparity.shape[0]),
                "width": int(disparity.shape[1]),
                "roi_pixels": roi_count,
                "sensor_valid_ratio_roi": float(valid[roi].mean()) if roi_count else None,
                "mean_predicted_reliability_roi": float(pred["reliability"][roi].mean()) if roi_count else None,
                "mean_predicted_error_roi": float(pred["error_px"][roi].mean()) if roi_count else None,
                "mean_bad_score_roi": float(pred["bad_score"][roi].mean()) if roi_count else None,
                "point_count": None if pcd_summary is None else pcd_summary["point_count"],
                **metrics,
            }
            summary = {
                "checkpoint": str(args.checkpoint.resolve()),
                "scene": scene.name,
                "scene_dir": str(scene.directory),
                "output_dir": str(scene_output.resolve()),
                "device": device,
                "feature_names": expected_features,
                "tile_overlap": overlap,
                "files": {
                    "intensity": None if intensity_path is None else str(intensity_path),
                    "confidence": None if confidence_path is None else str(confidence_path),
                    "ground_truth": None if scene.ground_truth is None else str(scene_dir / cfg["data"]["file_names"]["ground_truth"]),
                },
                "flat_metrics": flat,
                "pointcloud": pcd_summary,
            }
            scene_output.mkdir(parents=True, exist_ok=True)
            done_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            rows.append(flat)
        except Exception as exc:
            failure = {"scene": scene_dir.name, "path": str(scene_dir), "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            if args.fail_fast:
                raise

    _write_csv(args.output_dir / "batch_scene_summary.csv", rows)
    _write_csv(args.output_dir / "batch_failures.csv", failures)

    global_metrics: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "requested_scene_count": len(scenes),
        "successful_scene_count": len(rows),
        "failed_scene_count": len(failures),
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_dir": str(args.output_dir.resolve()),
        "pointcloud_enabled": not args.no_pointcloud,
        "pointcloud_format": None if args.no_pointcloud else args.pointcloud_format,
        "point_stride": None if args.no_pointcloud else args.point_stride,
    }
    if sampled_bad_true:
        rel = np.concatenate(sampled_rel)
        bad_prob = np.concatenate(sampled_bad_prob)
        bad_true = np.concatenate(sampled_bad_true)
        pred_error = np.concatenate(sampled_pred_error)
        true_error = np.concatenate(sampled_true_error)
        global_metrics.update({
            "sampled_pixel_count": int(len(bad_true)),
            "bad_detection_from_bad_head": safe_binary_metrics(bad_true, bad_prob),
            "bad_detection_from_inverted_reliability": safe_binary_metrics(bad_true, 1.0 - rel),
            "predicted_error_mae": float(np.abs(pred_error - true_error).mean()),
            "predicted_error_correlation": safe_correlation(pred_error, true_error),
        })
        if len(rows) >= 3:
            with_gt = [r for r in rows if r.get("mean_true_error") is not None]
            if len(with_gt) >= 3:
                global_metrics["roi_reliability_vs_mean_error"] = safe_correlation(
                    np.asarray([r["mean_predicted_reliability_roi"] for r in with_gt], dtype=float),
                    np.asarray([r["mean_true_error"] for r in with_gt], dtype=float),
                )
                global_metrics["roi_reliability_vs_bad_ratio"] = safe_correlation(
                    np.asarray([r["mean_predicted_reliability_roi"] for r in with_gt], dtype=float),
                    np.asarray([r["true_bad_pixel_ratio"] for r in with_gt], dtype=float),
                )
        coverage = quality_coverage_curve(rel, bad_true)
        _write_csv(args.output_dir / "quality_coverage.csv", coverage)

    (args.output_dir / "batch_metrics.json").write_text(json.dumps(global_metrics, indent=2), encoding="utf-8")
    print(json.dumps(global_metrics, indent=2))
    print(f"Scene summary: {args.output_dir / 'batch_scene_summary.csv'}")
    if failures:
        print(f"Failures:      {args.output_dir / 'batch_failures.csv'}")


if __name__ == "__main__":
    main()
