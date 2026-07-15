from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .model import decode_logits
from .targets import error_normalized_to_px


def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(tile - overlap, 1)
    positions = list(range(0, max(length - tile, 0) + 1, stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.no_grad()
def predict_tiled(model, features: np.ndarray, cfg: dict, device: str, use_amp: bool = True, overlap: int = 64) -> dict[str, np.ndarray]:
    _, h, w = features.shape
    tile_h = int(cfg["patch"]["height"])
    tile_w = int(cfg["patch"]["width"])
    y_positions = _positions(h, tile_h, overlap)
    x_positions = _positions(w, tile_w, overlap)

    sums = np.zeros((3, h, w), dtype=np.float32)
    counts = np.zeros((1, h, w), dtype=np.float32)
    device_type = "cuda" if device == "cuda" else "cpu"

    for y0 in y_positions:
        for x0 in x_positions:
            y1, x1 = min(y0 + tile_h, h), min(x0 + tile_w, w)
            tile = features[:, y0:y1, x0:x1]
            actual_h, actual_w = tile.shape[-2:]
            pad_h, pad_w = tile_h - actual_h, tile_w - actual_w
            if pad_h > 0 or pad_w > 0:
                tile_tensor = torch.from_numpy(tile[None]).to(device)
                tile_tensor = F.pad(tile_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            else:
                tile_tensor = torch.from_numpy(tile[None]).to(device)

            with torch.amp.autocast(device_type=device_type, enabled=(use_amp and device == "cuda")):
                decoded = decode_logits(model(tile_tensor))
            prediction = torch.cat(
                [decoded["reliability"], decoded["error_normalized"], decoded["bad_score"]], dim=1
            )[0, :, :actual_h, :actual_w].float().cpu().numpy()
            sums[:, y0:y1, x0:x1] += prediction
            counts[:, y0:y1, x0:x1] += 1.0

    averaged = sums / np.maximum(counts, 1e-6)
    error_px = error_normalized_to_px(averaged[1], float(cfg["targets"]["max_error_px"])).astype(np.float32)
    return {
        "reliability": averaged[0].astype(np.float32),
        "error_normalized": averaged[1].astype(np.float32),
        "error_px": error_px,
        "bad_score": averaged[2].astype(np.float32),
    }


def apply_deployment_rules(prediction: dict[str, np.ndarray], disparity: np.ndarray) -> dict[str, np.ndarray]:
    """Apply only the deterministic deployment rule stated in the thesis.

    The raw reliability sigmoid is multiplied by the sensor-validity mask.
    The normalized-error and bad-pixel outputs remain learned scores; they are
    not overwritten at invalid sensor pixels.
    """
    valid = np.isfinite(disparity) & (disparity > 0)
    output = {key: np.asarray(value).copy() for key, value in prediction.items()}
    output["reliability_raw"] = output["reliability"].copy()
    output["reliability"] = (output["reliability"] * valid.astype(np.float32)).astype(np.float32)
    return output
