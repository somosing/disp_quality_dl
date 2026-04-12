#!/usr/bin/env python3
"""Visualize reliability targets for a subset of samples.

Outputs are written to:
    outputs/target_debug/<sample_id>/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from data.target_builder import TargetBuilder, TargetBuilderConfig


# Config section. Update these paths/patterns for your dataset.
CONFIG = {
    "dataset_root": "dataset",
    "intensity_glob": "intensity/*",
    "sensor_disp_glob": "sensor_disparity/*",
    "raft_disp_glob": "raft_disparity/*",
    "yolo_model_path": "weights/yolo_roi.pt",
    "yolo_img_size": 1024,
    "confidence_threshold": 0.25,
    "reliability_threshold_T": 5.0,
    "num_samples": 10,
    "output_root": "outputs/target_debug",
}


LOGGER = logging.getLogger("visualize_targets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=str, default=CONFIG["dataset_root"])
    parser.add_argument("--intensity-glob", type=str, default=CONFIG["intensity_glob"])
    parser.add_argument("--sensor-glob", type=str, default=CONFIG["sensor_disp_glob"])
    parser.add_argument("--raft-glob", type=str, default=CONFIG["raft_disp_glob"])
    parser.add_argument("--yolo-model", type=str, default=CONFIG["yolo_model_path"])
    parser.add_argument("--yolo-imgsz", type=int, default=CONFIG["yolo_img_size"])
    parser.add_argument("--conf-thr", type=float, default=CONFIG["confidence_threshold"])
    parser.add_argument(
        "--reliability-thr", type=float, default=CONFIG["reliability_threshold_T"]
    )
    parser.add_argument("--num-samples", type=int, default=CONFIG["num_samples"])
    parser.add_argument("--output-root", type=str, default=CONFIG["output_root"])
    return parser.parse_args()


def _index_by_stem(paths: List[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in sorted(paths):
        index[p.stem] = p
    return index


def discover_samples(
    dataset_root: Path, intensity_glob: str, sensor_glob: str, raft_glob: str
) -> List[Tuple[str, Path, Path, Path]]:
    intensity_paths = list(dataset_root.glob(intensity_glob))
    sensor_paths = list(dataset_root.glob(sensor_glob))
    raft_paths = list(dataset_root.glob(raft_glob))

    intensity_idx = {p.parent.name: p for p in intensity_paths}
    sensor_idx = {p.parent.name: p for p in sensor_paths}
    raft_idx = {p.parent.name: p for p in raft_paths}

    common_ids = sorted(set(intensity_idx) & set(sensor_idx) & set(raft_idx))

    return [
        (sid, intensity_idx[sid], sensor_idx[sid], raft_idx[sid])
        for sid in common_ids
    ]


def _normalize_to_uint8(arr: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    x = arr.astype(np.float32).copy()
    if valid_mask is not None and np.any(valid_mask):
        vals = x[valid_mask]
        lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
    else:
        lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _save_colormap(path: Path, arr: np.ndarray, cmap: int, valid_mask: np.ndarray | None = None):
    arr_u8 = _normalize_to_uint8(arr, valid_mask=valid_mask)
    color = cv2.applyColorMap(arr_u8, cmap)
    cv2.imwrite(str(path), color)


def _save_roi_overlay(path: Path, intensity: np.ndarray, roi_mask: np.ndarray | None):
    base = cv2.cvtColor(intensity, cv2.COLOR_GRAY2BGR)
    if roi_mask is None:
        cv2.imwrite(str(path), base)
        return
    overlay = base.copy()
    overlay[roi_mask.astype(bool)] = (0, 255, 0)
    blended = cv2.addWeighted(base, 0.7, overlay, 0.3, 0.0)
    contours, _ = cv2.findContours(
        roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(str(path), blended)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(
        dataset_root=dataset_root,
        intensity_glob=args.intensity_glob,
        sensor_glob=args.sensor_glob,
        raft_glob=args.raft_glob,
    )
    if not samples:
        raise RuntimeError(
            "No matching samples found. Check dataset root and glob patterns."
        )

    builder = TargetBuilder(
        TargetBuilderConfig(
            yolo_model_path=args.yolo_model,
            yolo_img_size=args.yolo_imgsz,
            conf_threshold=args.conf_thr,
            reliability_threshold=args.reliability_thr,
        ),
        logger=LOGGER,
    )

    for sample_id, intensity_path, sensor_path, raft_path in samples[: args.num_samples]:
        result = builder.build_sample(
            sample_id=sample_id,
            intensity_path=intensity_path,
            sensor_disp_path=sensor_path,
            raft_disp_path=raft_path,
        )

        out_dir = output_root / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)

        intensity = result["intensity"]
        sensor_disp = result["sensor_disp"]
        raft_aligned = result["raft_disp_aligned"]
        abs_error = result["abs_error"]
        reliability = result["reliability"]
        roi_mask = result["roi_mask"]
        sensor_valid = result["sensor_valid"]

        cv2.imwrite(str(out_dir / "intensity.png"), intensity)
        _save_roi_overlay(out_dir / "intensity_roi_overlay.png", intensity, roi_mask)
        _save_colormap(
            out_dir / "sensor_disparity.png",
            sensor_disp,
            cmap=cv2.COLORMAP_TURBO,
            valid_mask=sensor_valid,
        )
        _save_colormap(
            out_dir / "raft_disparity_aligned.png",
            raft_aligned,
            cmap=cv2.COLORMAP_TURBO,
            valid_mask=result["raft_valid"],
        )
        _save_colormap(
            out_dir / "absolute_error.png",
            abs_error,
            cmap=cv2.COLORMAP_INFERNO,
            valid_mask=result["overlap_valid"],
        )
        _save_colormap(
            out_dir / "target_reliability.png",
            reliability,
            cmap=cv2.COLORMAP_VIRIDIS,
            valid_mask=None,
        )
        if roi_mask is None:
            cv2.imwrite(
                str(out_dir / "final_roi_mask.png"),
                np.zeros(result["sensor_shape"], dtype=np.uint8),
            )
        else:
            cv2.imwrite(
                str(out_dir / "final_roi_mask.png"),
                (roi_mask.astype(np.uint8) * 255),
            )

        print(
            "sample_id={sid} | sensor_shape={s_shape} | raft_before={r_before} "
            "| raft_after={r_after} | roi_area={roi_area} | valid_ratio={v_ratio:.4f} "
            "| roi_reliability_mean={roi_mean:.4f}".format(
                sid=sample_id,
                s_shape=result["sensor_shape"],
                r_before=result["raft_shape_raw"],
                r_after=result["raft_shape_aligned"],
                roi_area=int(result["roi_area"]),
                v_ratio=float(result["valid_ratio"]),
                roi_mean=float(result["roi_score"])
                if not np.isnan(result["roi_score"])
                else float("nan"),
            )
        )


if __name__ == "__main__":
    main()
