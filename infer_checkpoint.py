#!/usr/bin/env python3
"""Inference-only checkpoint runner (no RAFT required)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch

from data.reliability_dataset import CropConfig, apply_roi_crop_and_resize, normalize_sensor_disparity
from data.target_builder import create_valid_mask, load_disparity, load_grayscale_intensity
from detection.runtime_roi import (
    RuntimeROIConfig,
    RuntimeROIDetector,
    choose_final_roi_mask,
    resize_mask_to_shape,
)
from models.reliability_unet import ReliabilityUNet
from training.losses import compute_predicted_roi_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--yolo-model", type=str, required=True)
    parser.add_argument("--yolo-imgsz", type=int, default=1024)
    parser.add_argument("--conf-thr", type=float, default=0.25)
    parser.add_argument(
        "--crop-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--crop-margin-ratio", type=float, default=0.25)
    parser.add_argument("--resize-h", type=int, default=512)
    parser.add_argument("--resize-w", type=int, default=512)
    parser.add_argument(
        "--disparity-norm-mode",
        type=str,
        default="none",
        choices=["none", "minmax_valid", "zscore_valid"],
    )
    parser.add_argument(
        "--save-input-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=str, default="outputs/infer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_scene_paths(scene_dir: Path) -> Optional[Tuple[Path, Path]]:
    """Resolve intensity/disparity files for supported per-scene layouts.

    Supported layouts:
    1) <scene_id>/kl_depth_intensity_0.png
       <scene_id>/kl_depth_disparity_0.png
    2) <scene_id>/depth_data/...
    3) <scene_id>/data/...
    """
    candidates = [scene_dir, scene_dir / "depth_data", scene_dir / "data"]
    for base in candidates:
        intensity_path = base / "kl_depth_intensity_0.png"
        sensor_disp_path = base / "kl_depth_disparity_0.png"
        if intensity_path.exists() and sensor_disp_path.exists():
            return intensity_path, sensor_disp_path
    return None


def discover_inference_scenes(root: Path) -> List[Path]:
    """Discover scene directories under root that contain supported kl_* files."""
    scenes: List[Path] = []
    for scene_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if resolve_scene_paths(scene_dir) is not None:
            scenes.append(scene_dir)
    return scenes


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


# def _save_roi_overlay(path: Path, intensity: np.ndarray, roi_mask: np.ndarray | None) -> None:
#     base = cv2.cvtColor(intensity, cv2.COLOR_GRAY2BGR)
#     if roi_mask is None:
#         cv2.imwrite(str(path), base)
#         return
#     overlay = base.copy()
#     overlay[roi_mask.astype(bool)] = (0, 255, 0)
#     blended = cv2.addWeighted(base, 0.7, overlay, 0.3, 0.0)
#     cv2.imwrite(str(path), blended)
def _save_roi_overlay(path: Path, intensity: np.ndarray, roi_mask: np.ndarray | None) -> None:
    base = cv2.cvtColor(intensity, cv2.COLOR_GRAY2BGR)
    if roi_mask is None:
        cv2.imwrite(str(path), base)
        return

    if roi_mask.shape[:2] != intensity.shape[:2]:
        roi_mask = cv2.resize(
            roi_mask.astype(np.uint8),
            (intensity.shape[1], intensity.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        roi_mask = roi_mask.astype(bool)

    overlay = base.copy()
    overlay[roi_mask] = (0, 255, 0)
    blended = cv2.addWeighted(base, 0.7, overlay, 0.3, 0.0)

    contours, _ = cv2.findContours(
        roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(str(path), blended)

def prepare_model_input(
    intensity: np.ndarray,
    sensor_disp: np.ndarray,
    roi_mask: np.ndarray,
    crop_cfg: CropConfig,
    disparity_norm_mode: Literal["none", "minmax_valid", "zscore_valid"],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build input channels [sensor_disp, sensor_valid, intensity] like training."""
    sensor_valid = create_valid_mask(sensor_disp).astype(np.float32)
    intensity_f = intensity.astype(np.float32) / 255.0
    dummy_target = np.zeros_like(sensor_disp, dtype=np.float32)

    arrays = apply_roi_crop_and_resize(
        sensor_disp=sensor_disp.astype(np.float32),
        sensor_valid=sensor_valid,
        intensity=intensity_f,
        target_map=dummy_target,
        roi_mask=roi_mask.astype(np.float32),
        crop_cfg=crop_cfg,
    )
    arrays["sensor_disp"] = normalize_sensor_disparity(
        sensor_disp=arrays["sensor_disp"],
        sensor_valid=arrays["sensor_valid"],
        mode=disparity_norm_mode,
    )

    model_input = np.stack(
        [arrays["sensor_disp"], arrays["sensor_valid"], arrays["intensity"]], axis=0
    ).astype(np.float32)
    return model_input, arrays["roi_mask"], arrays["sensor_disp"]


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = discover_inference_scenes(input_root)
    if not scenes:
        raise RuntimeError(f"No scene folders found under {input_root}")

    device = torch.device(args.device)
    model = ReliabilityUNet(in_channels=3, out_channels=1, base_ch=32).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    roi_detector = RuntimeROIDetector(
        RuntimeROIConfig(
            model_path=args.yolo_model,
            img_size=args.yolo_imgsz,
            conf_threshold=args.conf_thr,
            device=args.device,
        )
    )
    crop_cfg = CropConfig(
        enabled=args.crop_enabled,
        margin_ratio=args.crop_margin_ratio,
        resize_to=(args.resize_h, args.resize_w) if args.crop_enabled else None,
    )

    print(f"Scenes: {len(scenes)} | device={device} | disparity_norm_mode={args.disparity_norm_mode}")

    for scene_dir in scenes:
        scene_id = scene_dir.name
        resolved = resolve_scene_paths(scene_dir)
        if resolved is None:
            # Should not happen because discovery already filters this.
            continue
        intensity_path, sensor_disp_path = resolved
        intensity = load_grayscale_intensity(intensity_path)
        sensor_disp = load_disparity(sensor_disp_path).astype(np.float32)

        cand_masks, confidences = roi_detector.detect_instance_masks(intensity)
        roi_mask = choose_final_roi_mask(cand_masks, confidences, args.conf_thr)
        if roi_mask is not None:
            roi_mask = resize_mask_to_shape(roi_mask, sensor_disp.shape[:2])
        else:
            # Safe fallback for inference if no ROI is detected.
            roi_mask = np.ones(sensor_disp.shape[:2], dtype=bool)

        model_input_np, roi_mask_proc, disp_proc = prepare_model_input(
            intensity=intensity,
            sensor_disp=sensor_disp,
            roi_mask=roi_mask,
            crop_cfg=crop_cfg,
            disparity_norm_mode=args.disparity_norm_mode,
        )

        x = torch.from_numpy(model_input_np).unsqueeze(0).to(device)
        roi_t = torch.from_numpy(roi_mask_proc.astype(np.float32))[None, None, ...].to(device)

        with torch.no_grad():
            pred_map = model(x)  # [1,1,H,W]
            pred_roi_score = compute_predicted_roi_score(pred_map, roi_t)

        pred_np = pred_map[0, 0].detach().cpu().numpy()
        roi_np = roi_mask_proc.astype(np.float32)
        roi_area = int(np.count_nonzero(roi_np > 0.5))
        predicted_roi_score = float(pred_roi_score.item())

        scene_out = output_dir / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)
        _save_heat(scene_out / "predicted_map.png", pred_np, cv2.COLORMAP_VIRIDIS)
        _save_mask(scene_out / "roi_mask.png", roi_np)

        if args.save_input_debug:
            _save_heat(scene_out / "input_disparity.png", disp_proc, cv2.COLORMAP_TURBO)
            _save_roi_overlay(scene_out / "intensity_roi_overlay.png", intensity, roi_mask)

        with (scene_out / "scores.json").open("w") as f:
            json.dump(
                {
                    "scene_id": scene_id,
                    "predicted_roi_score": predicted_roi_score,
                    "roi_area": roi_area,
                },
                f,
                indent=2,
            )

        print(
            f"scene_id={scene_id} | predicted_roi_score={predicted_roi_score:.4f} | roi_area={roi_area}"
        )


if __name__ == "__main__":
    main()
