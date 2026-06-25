"""
Depth utilities: loading, scaling to metres, validity masking and
back-projection of pixels into a 3D camera-frame point cloud.

All 3D coordinates are expressed in the camera frame (metres):
    x = (u - cx) * z / fx      (right)
    y = (v - cy) * z / fy      (down)
    z = metric depth           (forward, away from camera)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics (pixels)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_dict(cls, d: dict) -> "CameraIntrinsics":
        return cls(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
        )

    def scaled(self, sx: float, sy: float) -> "CameraIntrinsics":
        """Return intrinsics for an image resized by (sx, sy)."""
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=int(round(self.width * sx)),
            height=int(round(self.height * sy)),
        )


def load_depth(path: str) -> np.ndarray:
    """Load a depth image as a float32 array of RAW (un-scaled) values.

    16-bit PNGs (the common OCID/RealSense format) are read losslessly.
    """
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def to_metres(
    depth_raw: np.ndarray,
    depth_scale: float,
    min_depth_m: float = 0.0,
    max_depth_m: float = np.inf,
) -> np.ndarray:
    """Convert raw depth to metres and zero out invalid / out-of-range pixels.

    Invalid pixels (0, NaN, inf, or outside [min, max]) become 0.0, which is
    the convention used everywhere downstream to mark "no depth".
    """
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    invalid = ~np.isfinite(depth_m)
    depth_m[invalid] = 0.0
    if min_depth_m > 0:
        depth_m[depth_m < min_depth_m] = 0.0
    if np.isfinite(max_depth_m):
        depth_m[depth_m > max_depth_m] = 0.0
    return depth_m


def valid_depth_mask(depth_m: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels with usable (> 0, finite) metric depth."""
    return np.isfinite(depth_m) & (depth_m > 0.0)


def backproject(
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    pixel_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project depth pixels into 3D camera-frame points.

    Parameters
    ----------
    depth_m : (H, W) metric depth (0 == invalid).
    intr    : camera intrinsics.
    pixel_mask : optional (H, W) boolean mask restricting which pixels to use.

    Returns
    -------
    points : (N, 3) float32 array of (x, y, z) in metres.
    pix    : (N, 2) int array of the source (v, u) pixel coordinates.
    """
    h, w = depth_m.shape[:2]
    valid = valid_depth_mask(depth_m)
    if pixel_mask is not None:
        valid &= pixel_mask.astype(bool)

    vs, us = np.nonzero(valid)
    z = depth_m[vs, us]
    x = (us - intr.cx) * z / intr.fx
    y = (vs - intr.cy) * z / intr.fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    pix = np.stack([vs, us], axis=1)
    return points, pix


def valid_depth_ratio(depth_m: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of pixels inside `mask` that carry valid depth."""
    mask = mask.astype(bool)
    n = int(mask.sum())
    if n == 0:
        return 0.0
    return float((valid_depth_mask(depth_m) & mask).sum()) / float(n)
