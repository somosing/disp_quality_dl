from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from disp_quality.config import load_config
from disp_quality.io import load_scene, scan_scene_directories


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit disparity-quality dataset scenes.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("audit_output"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    roots = list(data_cfg.get("train_roots", [])) + list(data_cfg.get("val_roots", []) or [])
    candidates, scan_rejected = scan_scene_directories(roots, data_cfg, require_gt=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rejected = list(scan_rejected)
    min_roi = int(data_cfg.get("min_roi_pixels", 1))

    for directory in tqdm(candidates, desc="Auditing scenes"):
        try:
            scene = load_scene(directory, data_cfg, require_gt=True)
            assert scene.ground_truth is not None
            sensor_valid = np.isfinite(scene.disparity) & (scene.disparity > 0)
            gt_valid = np.isfinite(scene.ground_truth) & (scene.ground_truth > 0)
            roi_pixels = int(scene.roi.sum())
            joint = scene.roi & sensor_valid & gt_valid
            error = np.abs(scene.disparity[joint] - scene.ground_truth[joint]) if joint.any() else np.array([])
            row = {
                "scene": scene.name,
                "path": str(scene.directory),
                "height": scene.disparity.shape[0],
                "width": scene.disparity.shape[1],
                "roi_pixels": roi_pixels,
                "sensor_valid_ratio_roi": float((scene.roi & sensor_valid).sum() / max(roi_pixels, 1)),
                "gt_valid_ratio_roi": float((scene.roi & gt_valid).sum() / max(roi_pixels, 1)),
                "joint_valid_pixels": int(joint.sum()),
                "mean_abs_error_px": float(error.mean()) if error.size else None,
                "p95_abs_error_px": float(np.percentile(error, 95)) if error.size else None,
                "status": "ok" if roi_pixels >= min_roi and joint.any() else "weak_supervision",
            }
            rows.append(row)
            if row["status"] != "ok":
                rejected.append({"scene": str(directory), "reason": row["status"]})
        except Exception as exc:  # audit must continue across corrupt scenes
            rejected.append({"scene": str(directory), "reason": "read_error", "error": str(exc)})

    csv_path = args.output_dir / "dataset_audit.csv"
    fieldnames = [
        "scene", "path", "height", "width", "roi_pixels", "sensor_valid_ratio_roi",
        "gt_valid_ratio_roi", "joint_valid_pixels", "mean_abs_error_px", "p95_abs_error_px", "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "roots": roots,
        "file_scan_candidates": len(candidates),
        "readable_scenes": len(rows),
        "ok_scenes": sum(row["status"] == "ok" for row in rows),
        "weak_supervision_scenes": sum(row["status"] != "ok" for row in rows),
        "rejected_or_failed": len(rejected),
        "rejected": rejected,
    }
    summary_path = args.output_dir / "dataset_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rejected"}, indent=2))
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
