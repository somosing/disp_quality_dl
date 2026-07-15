from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from disp_quality.config import load_config
from disp_quality.io import load_scene, scan_scene_directories


def filter_trainable(candidates: list[Path], data_cfg: dict, rejected: list[dict]) -> list[Path]:
    """Reject corrupt scenes and scenes without enough ROI/reference supervision."""
    accepted: list[Path] = []
    min_pixels = int(data_cfg.get("min_roi_pixels", 1))
    for directory in tqdm(candidates, desc="Validating split scenes", leave=False):
        try:
            scene = load_scene(directory, data_cfg, require_gt=True)
            assert scene.ground_truth is not None
            gt_valid = np.isfinite(scene.ground_truth) & (scene.ground_truth > 0)
            supervised = scene.roi & gt_valid
            if int(scene.roi.sum()) < min_pixels:
                rejected.append({"scene": str(directory), "reason": "roi_too_small"})
                continue
            if int(supervised.sum()) < min_pixels:
                rejected.append({"scene": str(directory), "reason": "insufficient_reference_supervision"})
                continue
            accepted.append(directory)
        except Exception as exc:
            rejected.append({"scene": str(directory), "reason": "read_or_alignment_error", "error": str(exc)})
    return accepted


def capture_group_key(path: Path, suffix_regex: str) -> str:
    """Map left/right folders from one capture to the same split group."""
    return re.sub(suffix_regex, "", path.name, flags=re.IGNORECASE)


def grouped_partition(
    candidates: list[Path],
    target_validation_scenes: int,
    rng: random.Random,
    suffix_regex: str,
) -> tuple[list[Path], list[Path], dict]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in candidates:
        groups[capture_group_key(path, suffix_regex)].append(path)

    keys = sorted(groups)
    rng.shuffle(keys)
    val_keys: list[str] = []
    val_scene_count = 0

    # Add complete capture groups until the requested scene count is reached.
    # This may differ by one group from the exact requested count, but prevents
    # left/right observations from the same capture leaking across the split.
    for key in keys:
        if val_scene_count >= target_validation_scenes:
            break
        if len(val_keys) >= len(keys) - 1:
            break
        val_keys.append(key)
        val_scene_count += len(groups[key])

    val_key_set = set(val_keys)
    val_paths = sorted(path for key in val_key_set for path in groups[key])
    train_paths = sorted(path for key in keys if key not in val_key_set for path in groups[key])
    metadata = {
        "grouping_enabled": True,
        "group_suffix_regex": suffix_regex,
        "total_capture_groups": len(groups),
        "train_capture_groups": len(groups) - len(val_key_set),
        "val_capture_groups": len(val_key_set),
        "target_validation_scenes": target_validation_scenes,
        "actual_validation_scenes": len(val_paths),
    }
    return train_paths, val_paths, metadata


def create_split(
    cfg: dict,
    output_path: Path | None = None,
    validation_count_override: int | None = None,
    validation_fraction_override: float | None = None,
) -> dict:
    data_cfg = cfg["data"]
    train_candidates, train_rejected = scan_scene_directories(
        data_cfg.get("train_roots", []), data_cfg, require_gt=True
    )
    rejected = list(train_rejected)
    train_candidates = filter_trainable(train_candidates, data_cfg, rejected)
    val_roots = data_cfg.get("val_roots", []) or []

    configured_count = data_cfg.get("validation_count")
    configured_fraction = data_cfg.get("validation_fraction")
    if validation_count_override is not None:
        validation_count = validation_count_override
        validation_fraction = None
    elif validation_fraction_override is not None:
        validation_count = None
        validation_fraction = validation_fraction_override
    else:
        validation_count = configured_count
        validation_fraction = configured_fraction

    if val_roots and (validation_count is not None or validation_fraction is not None):
        raise ValueError(
            "Choose one validation method only: either val_roots, validation_count, "
            "or validation_fraction."
        )
    if validation_count is not None and validation_fraction is not None:
        raise ValueError("Set only one of data.validation_count and data.validation_fraction.")

    rng = random.Random(int(cfg.get("seed", 42)))
    split_metadata: dict = {}
    if val_roots:
        val_candidates, val_rejected = scan_scene_directories(
            val_roots, data_cfg, require_gt=True
        )
        rejected.extend(val_rejected)
        val_candidates = filter_trainable(val_candidates, data_cfg, rejected)
        train_paths = sorted(set(train_candidates))
        val_paths = sorted(set(val_candidates) - set(train_paths))
        split_mode = "separate_validation_roots"
    else:
        candidates = sorted(set(train_candidates))
        if len(candidates) < 2:
            raise RuntimeError(
                "At least two trainable scenes are required when splitting one root into "
                "training and validation data."
            )

        if validation_count is not None:
            requested = int(validation_count)
            if requested <= 0 or requested >= len(candidates):
                raise ValueError(
                    "data.validation_count must be greater than zero and smaller than "
                    "the number of accepted scenes."
                )
            chosen_count = requested
            split_mode = "deterministic_count_split"
        elif validation_fraction is not None:
            fraction = float(validation_fraction)
            if not 0.0 < fraction < 1.0:
                raise ValueError("data.validation_fraction must be between zero and one.")
            chosen_count = int(round(fraction * len(candidates)))
            chosen_count = min(max(chosen_count, 1), len(candidates) - 1)
            split_mode = "deterministic_fraction_split"
        else:
            raise ValueError(
                "Validation selection is intentionally unspecified. Set val_roots, "
                "data.validation_count, or data.validation_fraction before creating the split."
            )

        if bool(data_cfg.get("group_by_capture", False)):
            suffix_regex = str(data_cfg.get("capture_suffix_regex", r"_(l|r)$"))
            train_paths, val_paths, split_metadata = grouped_partition(
                candidates, chosen_count, rng, suffix_regex
            )
            split_mode = "grouped_" + split_mode
        else:
            rng.shuffle(candidates)
            val_paths = sorted(candidates[:chosen_count])
            train_paths = sorted(candidates[chosen_count:])
            split_metadata = {"grouping_enabled": False}

    if not train_paths or not val_paths:
        raise RuntimeError(
            f"Need non-empty train and validation sets; got train={len(train_paths)}, val={len(val_paths)}"
        )

    split = {
        "format_version": 2,
        "seed": int(cfg.get("seed", 42)),
        "mode": split_mode,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "train": [str(path) for path in train_paths],
        "val": [str(path) for path in val_paths],
        "rejected": rejected,
        **split_metadata,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic train/validation split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--validation-count",
        type=int,
        default=None,
        help="Optional one-run override for data.validation_count.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="Optional one-run override for data.validation_fraction.",
    )
    args = parser.parse_args()

    if args.validation_count is not None and args.validation_fraction is not None:
        parser.error("Use only one of --validation-count and --validation-fraction.")

    cfg = load_config(args.config)
    output = args.output or Path(cfg["paths"]["split_file"])
    split = create_split(
        cfg,
        output,
        validation_count_override=args.validation_count,
        validation_fraction_override=args.validation_fraction,
    )
    print(f"Split mode: {split['mode']}")
    print(f"Train scenes: {split['train_count']}")
    print(f"Validation scenes: {split['val_count']}")
    if split.get("grouping_enabled"):
        print(
            "Capture groups: "
            f"train={split['train_capture_groups']} val={split['val_capture_groups']} "
            f"total={split['total_capture_groups']}"
        )
    print(f"Rejected during validation: {len(split['rejected'])}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
