from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from disp_quality.checkpoint import load_checkpoint
from disp_quality.config import resolve_device
from disp_quality.features import build_features
from disp_quality.io import load_scene, scan_scene_directories
from disp_quality.metrics import safe_binary_metrics, safe_correlation
from disp_quality.predict import apply_deployment_rules, predict_tiled
from disp_quality.targets import build_targets


def read_scene_list(args, cfg: dict) -> list[str]:
    if args.split_file is not None:
        split = json.loads(args.split_file.read_text(encoding="utf-8"))
        if args.split_name not in split:
            raise KeyError(f"Split does not contain '{args.split_name}'")
        return list(split[args.split_name])
    if args.scene_root:
        scenes, _ = scan_scene_directories(args.scene_root, cfg["data"], require_gt=True)
        return [str(path) for path in scenes]
    default_split = Path(cfg["paths"]["split_file"])
    split = json.loads(default_split.read_text(encoding="utf-8"))
    return list(split[args.split_name])


def scene_quality_coverage(rows: list[dict], score_key: str, coverages=None) -> list[dict]:
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 10)
    ordered = sorted(rows, key=lambda row: row[score_key], reverse=True)
    output = []
    for coverage in coverages:
        count = max(1, int(math.ceil(float(coverage) * len(ordered))))
        kept = ordered[:count]
        output.append({
            "coverage": float(coverage),
            "scenes_kept": count,
            "mean_reference_error_px": float(np.mean([row["mean_reference_error_px"] for row in kept])),
            "mean_bad_pixel_ratio": float(np.mean([row["bad_pixel_ratio"] for row in kept])),
        })
    return output


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the thesis-aligned three-output model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split-name", choices=["train", "val", "test"], default="val")
    parser.add_argument("--scene-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = checkpoint["config"]
    scene_dirs = read_scene_list(args, cfg)
    if args.max_scenes is not None:
        scene_dirs = scene_dirs[:args.max_scenes]
    if not scene_dirs:
        raise RuntimeError("No evaluation scenes found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlap = int(args.tile_overlap if args.tile_overlap is not None else cfg.get("evaluation", {}).get("tile_overlap", 0))
    per_scene_limit = int(cfg.get("evaluation", {}).get("max_pixels_per_scene", 20_000))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    combined_rel, combined_err, combined_bad_score, combined_target = [], [], [], []
    joint_rel, joint_err, joint_bad_score, joint_target = [], [], [], []
    scene_rows: list[dict] = []

    for directory in tqdm(scene_dirs, desc="Evaluating scenes"):
        scene = load_scene(directory, cfg["data"], require_gt=True)
        assert scene.ground_truth is not None
        features, names = build_features(scene.disparity, scene.roi, scene.intensity, cfg)
        if names != list(checkpoint["feature_names"]):
            raise RuntimeError(f"Feature mismatch in {directory}")
        raw_pred = predict_tiled(model, features, cfg, device, bool(cfg["training"].get("amp", True)), overlap)
        pred = apply_deployment_rules(raw_pred, scene.disparity)
        target, _supervision_mask, debug = build_targets(scene.disparity, scene.ground_truth, scene.roi, cfg)

        roi = scene.roi.astype(bool)
        sensor_valid = debug["sensor_valid"].astype(bool)
        reference_valid = debug["gt_valid"].astype(bool)
        eval_mask = roi & reference_valid
        joint_mask = eval_mask & sensor_valid
        if not eval_mask.any() or not roi.any():
            continue

        error_px = debug["error_px"]
        bad_target = target[2]
        threshold = float(cfg["targets"]["bad_pixel_threshold_px"])

        mean_reference_error = float(error_px[joint_mask].mean()) if joint_mask.any() else float("nan")
        bad_ratio = float(bad_target[eval_mask].mean())
        joint_bad_ratio = float((error_px[joint_mask] >= threshold).mean()) if joint_mask.any() else float("nan")

        scene_rows.append({
            "scene": scene.name,
            "path": str(scene.directory),
            "roi_pixels": int(roi.sum()),
            "reference_valid_roi_pixels": int(eval_mask.sum()),
            "jointly_valid_roi_pixels": int(joint_mask.sum()),
            "mean_roi_reliability_complete_roi": float(pred["reliability"][roi].mean()),
            "mean_reference_error_px": mean_reference_error,
            "bad_pixel_ratio": bad_ratio,
            "mean_reliability_joint_valid": float(pred["reliability"][joint_mask].mean()) if joint_mask.any() else float("nan"),
            "mean_predicted_normalized_error_eval": float(pred["error_normalized"][eval_mask].mean()),
            "mean_bad_pixel_score_eval": float(pred["bad_score"][eval_mask].mean()),
            "joint_valid_bad_pixel_ratio": joint_bad_ratio,
        })

        def sampled(mask: np.ndarray) -> np.ndarray:
            flat = np.flatnonzero(mask)
            if flat.size > per_scene_limit:
                return rng.choice(flat, size=per_scene_limit, replace=False)
            return flat

        idx = sampled(eval_mask)
        combined_rel.append(pred["reliability"].ravel()[idx])
        combined_err.append(pred["error_normalized"].ravel()[idx])
        combined_bad_score.append(pred["bad_score"].ravel()[idx])
        combined_target.append(bad_target.ravel()[idx])

        idx_joint = sampled(joint_mask)
        if idx_joint.size:
            joint_rel.append(pred["reliability"].ravel()[idx_joint])
            joint_err.append(pred["error_normalized"].ravel()[idx_joint])
            joint_bad_score.append(pred["bad_score"].ravel()[idx_joint])
            joint_target.append((error_px.ravel()[idx_joint] >= threshold).astype(np.float32))

    if not scene_rows:
        raise RuntimeError("No scenes had evaluable ROI pixels.")

    def cat(items):
        return np.concatenate(items) if items else np.empty(0, dtype=np.float32)

    c_rel, c_err, c_bad_score, c_target = map(cat, [combined_rel, combined_err, combined_bad_score, combined_target])
    j_rel, j_err, j_bad_score, j_target = map(cat, [joint_rel, joint_err, joint_bad_score, joint_target])

    roi_rel = np.asarray([r["mean_roi_reliability_complete_roi"] for r in scene_rows])
    roi_err = np.asarray([r["mean_reference_error_px"] for r in scene_rows])
    roi_bad = np.asarray([r["bad_pixel_ratio"] for r in scene_rows])
    roi_pred_err = np.asarray([r["mean_predicted_normalized_error_eval"] for r in scene_rows])
    roi_bad_score = np.asarray([r["mean_bad_pixel_score_eval"] for r in scene_rows])

    metrics = {
        "scene_count": len(scene_rows),
        "sampling": {"maximum_reference_valid_roi_pixels_per_scene": per_scene_limit, "seed": int(cfg.get("seed", 42))},
        "combined_pixel_count": int(c_target.size),
        "combined_bad_prevalence": float(c_target.mean()),
        "combined_bad_detection": {
            "inverted_reliability": safe_binary_metrics(c_target, 1.0 - c_rel),
            "predicted_normalized_error": safe_binary_metrics(c_target, c_err),
            "bad_pixel_score": safe_binary_metrics(c_target, c_bad_score),
        },
        "jointly_valid_pixel_count": int(j_target.size),
        "jointly_valid_bad_prevalence": float(j_target.mean()) if j_target.size else None,
        "jointly_valid_bad_detection": {
            "inverted_reliability": safe_binary_metrics(j_target, 1.0 - j_rel),
            "predicted_normalized_error": safe_binary_metrics(j_target, j_err),
            "bad_pixel_score": safe_binary_metrics(j_target, j_bad_score),
        },
        "roi_correlations": {
            "mean_reliability_vs_mean_reference_error": safe_correlation(roi_rel, roi_err),
            "mean_reliability_vs_bad_pixel_ratio": safe_correlation(roi_rel, roi_bad),
            "mean_predicted_normalized_error_vs_mean_reference_error": safe_correlation(roi_pred_err, roi_err),
            "mean_predicted_normalized_error_vs_bad_pixel_ratio": safe_correlation(roi_pred_err, roi_bad),
            "mean_bad_pixel_score_vs_bad_pixel_ratio": safe_correlation(roi_bad_score, roi_bad),
        },
    }

    write_rows(args.output_dir / "scene_metrics.csv", scene_rows)
    write_rows(args.output_dir / "quality_coverage_by_reliability.csv", scene_quality_coverage(scene_rows, "mean_roi_reliability_complete_roi"))
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
