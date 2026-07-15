from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    K: np.ndarray
    baseline_mm: float
    disp_scale: float = 1.0
    disp_offset: float = 0.0
    image_width: int | None = None
    image_height: int | None = None
    source: str | None = None

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def scaled_to_shape(self, shape_hw: tuple[int, int]) -> "CameraCalibration":
        """Scale intrinsics if the disparity resolution differs from calibration resolution."""
        h, w = shape_hw
        ref_w = self.image_width
        ref_h = self.image_height
        if ref_w is None:
            ref_w = int(round(2.0 * self.cx)) if self.cx > 0 else w
        if ref_h is None:
            ref_h = int(round(2.0 * self.cy)) if self.cy > 0 else h
        if ref_w <= 0 or ref_h <= 0 or (ref_w == w and ref_h == h):
            return self
        sx = w / float(ref_w)
        sy = h / float(ref_h)
        K = self.K.astype(np.float64).copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return CameraCalibration(
            K=K,
            baseline_mm=self.baseline_mm,
            disp_scale=self.disp_scale,
            disp_offset=self.disp_offset,
            image_width=w,
            image_height=h,
            source=self.source,
        )

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "K": self.K.tolist(),
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "baseline_mm": self.baseline_mm,
            "disp_scale": self.disp_scale,
            "disp_offset": self.disp_offset,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


def load_camera_json(path: str | Path) -> CameraCalibration:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "K" not in data:
        raise ValueError(f"Camera JSON has no K matrix: {path}")
    K = np.asarray(data["K"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Camera K must be 3x3, got {K.shape}: {path}")
    baseline = data.get("baseline_mm", data.get("baseline"))
    if baseline is None:
        raise ValueError(f"Camera JSON has no baseline_mm: {path}")
    return CameraCalibration(
        K=K,
        baseline_mm=float(baseline),
        disp_scale=float(data.get("disp_scale", data.get("stereoScale", 1.0))),
        disp_offset=float(data.get("disp_offset", data.get("stereoOffset", 0.0))),
        image_width=int(data["image_width"]) if data.get("image_width") is not None else None,
        image_height=int(data["image_height"]) if data.get("image_height") is not None else None,
        source=str(path),
    )


def disparity_to_xyz_map(
    disparity_raw: np.ndarray,
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if disparity_raw.ndim != 2:
        raise ValueError(f"Expected 2D disparity, got {disparity_raw.shape}")
    calibration = calibration.scaled_to_shape(disparity_raw.shape)
    corrected = disparity_raw.astype(np.float32) * float(calibration.disp_scale) + float(calibration.disp_offset)
    # Preserve the usual invalid-zero convention even when an offset is present.
    corrected = np.where(disparity_raw > 0, corrected, 0.0).astype(np.float32)
    valid = np.isfinite(corrected) & (corrected > 0)

    h, w = corrected.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = np.full((h, w), np.nan, dtype=np.float32)
    x = np.full((h, w), np.nan, dtype=np.float32)
    y = np.full((h, w), np.nan, dtype=np.float32)
    z[valid] = calibration.fx * calibration.baseline_mm / corrected[valid]
    x[valid] = (u[valid] - calibration.cx) * z[valid] / calibration.fx
    y[valid] = (v[valid] - calibration.cy) * z[valid] / calibration.fy
    return np.stack([x, y, z], axis=-1), corrected, valid


def grayscale_rgb(image: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if image is None:
        gray = np.full((h, w), 150, dtype=np.uint8)
    else:
        if image.shape != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        gray = np.clip(image, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def scalar_colormap_rgb(values: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    u8 = (np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def stride_mask(shape_hw: tuple[int, int], stride: int) -> np.ndarray:
    h, w = shape_hw
    stride = max(int(stride), 1)
    if stride == 1:
        return np.ones((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=bool)
    mask[::stride, ::stride] = True
    return mask


def extract_points_colors(
    xyz: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(xyz).all(axis=-1)
    points = xyz[valid].astype(np.float32, copy=False)
    rgb = colors[valid].astype(np.uint8, copy=False)
    return points.reshape(-1, 3), rgb.reshape(-1, 3)


def write_binary_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("points and colors must contain the same number of rows")
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def write_binary_pcd(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary PCD readable by PCL/Open3D."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("points and colors must contain the same number of rows")
    packed_u32 = (
        (colors[:, 0].astype(np.uint32) << 16)
        | (colors[:, 1].astype(np.uint32) << 8)
        | colors[:, 2].astype(np.uint32)
    )
    packed_rgb = packed_u32.view(np.float32)
    records = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    records["x"], records["y"], records["z"] = points.T
    records["rgb"] = packed_rgb
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)


def write_cloud(path_without_suffix: str | Path, points: np.ndarray, colors: np.ndarray, fmt: str) -> list[str]:
    base = Path(path_without_suffix)
    written: list[str] = []
    if fmt in {"ply", "both"}:
        path = base.with_suffix(".ply")
        write_binary_ply(path, points, colors)
        written.append(str(path))
    if fmt in {"pcd", "both"}:
        path = base.with_suffix(".pcd")
        write_binary_pcd(path, points, colors)
        written.append(str(path))
    return written
