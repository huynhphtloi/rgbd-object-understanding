"""Colorize a metric depth map for visualization."""
from __future__ import annotations

import cv2
import numpy as np


def colorize_depth(depth_m: np.ndarray, max_depth_m: float | None = None) -> np.ndarray:
    """Map metric depth to a BGR color image (invalid pixels -> black)."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not valid.any():
        return np.zeros((*depth_m.shape[:2], 3), dtype=np.uint8)

    d = depth_m.copy()
    hi = max_depth_m if max_depth_m else float(np.percentile(d[valid], 95))
    lo = float(d[valid].min())
    hi = max(hi, lo + 1e-3)

    norm = np.clip((d - lo) / (hi - lo), 0, 1)
    norm8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm8, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color
