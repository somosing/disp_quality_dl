from __future__ import annotations

import numpy as np


def _choose_crop_origin(
    h: int,
    w: int,
    target_h: int,
    target_w: int,
    roi: np.ndarray,
    random_crop: bool,
    rng: np.random.Generator,
    roi_aware: bool,
) -> tuple[int, int]:
    max_y = max(h - target_h, 0)
    max_x = max(w - target_w, 0)
    if max_y == 0 and max_x == 0:
        return 0, 0

    ys, xs = np.where(roi)
    if roi_aware and ys.size > 0:
        if random_crop:
            index = int(rng.integers(0, ys.size))
            center_y, center_x = int(ys[index]), int(xs[index])
        else:
            center_y = int(round(float(np.mean(ys))))
            center_x = int(round(float(np.mean(xs))))
        y0 = int(np.clip(center_y - target_h // 2, 0, max_y))
        x0 = int(np.clip(center_x - target_w // 2, 0, max_x))
        return y0, x0

    if random_crop:
        y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        return y0, x0
    return max_y // 2, max_x // 2


def crop_or_pad_scene(
    disparity: np.ndarray,
    ground_truth: np.ndarray,
    roi: np.ndarray,
    intensity: np.ndarray | None,
    target_h: int,
    target_w: int,
    random_crop: bool,
    roi_aware: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Crop and pad aligned arrays without resizing or changing disparity units."""
    h, w = disparity.shape
    y0, x0 = _choose_crop_origin(h, w, target_h, target_w, roi, random_crop, rng, roi_aware)
    y1, x1 = min(y0 + target_h, h), min(x0 + target_w, w)

    disparity_crop = disparity[y0:y1, x0:x1]
    gt_crop = ground_truth[y0:y1, x0:x1]
    roi_crop = roi[y0:y1, x0:x1]
    intensity_crop = None if intensity is None else intensity[y0:y1, x0:x1]

    out_disp = np.zeros((target_h, target_w), dtype=np.float32)
    out_gt = np.zeros((target_h, target_w), dtype=np.float32)
    out_roi = np.zeros((target_h, target_w), dtype=bool)
    out_intensity = None if intensity is None else np.zeros((target_h, target_w), dtype=np.float32)

    crop_h, crop_w = disparity_crop.shape
    pad_y = max((target_h - crop_h) // 2, 0)
    pad_x = max((target_w - crop_w) // 2, 0)
    out_disp[pad_y:pad_y + crop_h, pad_x:pad_x + crop_w] = disparity_crop
    out_gt[pad_y:pad_y + crop_h, pad_x:pad_x + crop_w] = gt_crop
    out_roi[pad_y:pad_y + crop_h, pad_x:pad_x + crop_w] = roi_crop
    if out_intensity is not None and intensity_crop is not None:
        out_intensity[pad_y:pad_y + crop_h, pad_x:pad_x + crop_w] = intensity_crop

    return out_disp, out_gt, out_roi, out_intensity


def augment_scene(
    disparity: np.ndarray,
    ground_truth: np.ndarray,
    roi: np.ndarray,
    intensity: np.ndarray | None,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    patch_cfg = cfg["patch"]
    disparity, ground_truth, roi, intensity = crop_or_pad_scene(
        disparity=disparity,
        ground_truth=ground_truth,
        roi=roi,
        intensity=intensity,
        target_h=int(patch_cfg["height"]),
        target_w=int(patch_cfg["width"]),
        random_crop=True,
        roi_aware=bool(patch_cfg.get("roi_aware_crop", True)),
        rng=rng,
    )

    if rng.random() < float(patch_cfg.get("horizontal_flip_probability", 0.0)):
        disparity = np.fliplr(disparity).copy()
        ground_truth = np.fliplr(ground_truth).copy()
        roi = np.fliplr(roi).copy()
        if intensity is not None:
            intensity = np.fliplr(intensity).copy()

    if intensity is not None:
        if rng.random() < float(patch_cfg.get("brightness_contrast_probability", 0.0)):
            alpha = float(rng.uniform(
                patch_cfg.get("brightness_alpha_min", 0.85),
                patch_cfg.get("brightness_alpha_max", 1.15),
            ))
            beta = float(rng.uniform(
                patch_cfg.get("brightness_beta_min", -12.0),
                patch_cfg.get("brightness_beta_max", 12.0),
            ))
            intensity = np.clip(intensity * alpha + beta, 0.0, 255.0)

        if rng.random() < float(patch_cfg.get("gaussian_noise_probability", 0.0)):
            std = float(rng.uniform(
                patch_cfg.get("gaussian_noise_std_min", 1.0),
                patch_cfg.get("gaussian_noise_std_max", 5.0),
            ))
            intensity = np.clip(
                intensity + rng.normal(0.0, std, size=intensity.shape).astype(np.float32),
                0.0,
                255.0,
            )

    return disparity, ground_truth, roi, intensity


def validation_transform(
    disparity: np.ndarray,
    ground_truth: np.ndarray,
    roi: np.ndarray,
    intensity: np.ndarray | None,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    patch_cfg = cfg["patch"]
    return crop_or_pad_scene(
        disparity=disparity,
        ground_truth=ground_truth,
        roi=roi,
        intensity=intensity,
        target_h=int(patch_cfg["height"]),
        target_w=int(patch_cfg["width"]),
        random_crop=False,
        roi_aware=bool(patch_cfg.get("roi_aware_crop", True)),
        rng=rng,
    )
