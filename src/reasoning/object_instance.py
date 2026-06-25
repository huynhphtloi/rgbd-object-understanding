"""
ObjectInstance: the unit passed to the spatial-relation reasoner.

It bundles the 2D mask/box with depth-derived signals. We use the MEDIAN
depth inside the mask as the primary depth signal because it is far more
robust to noisy/invalid depth pixels than the mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..depth.depth_utils import CameraIntrinsics, backproject, valid_depth_ratio
from ..depth.dimension_estimation import MeasurementResult, measure_object


@dataclass
class ObjectInstance:
    id: int
    mask: np.ndarray                 # (H, W) bool
    bbox: tuple                       # (x1, y1, x2, y2) pixels
    score: float = 1.0
    area_px: int = 0
    centroid_2d: tuple = (0.0, 0.0)   # (u, v)
    centroid_3d: tuple = (0.0, 0.0, 0.0)
    median_depth: float = 0.0         # metres
    mean_depth: float = 0.0           # metres
    valid_depth_ratio: float = 0.0
    n_points: int = 0
    measurement: MeasurementResult | None = field(default=None, repr=False)

    @property
    def has_depth(self) -> bool:
        return self.n_points > 0 and self.median_depth > 0.0


def _mask_bbox(mask: np.ndarray) -> tuple:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def build_instance(
    object_id: int,
    mask: np.ndarray,
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    score: float = 1.0,
    measure: bool = True,
    **measure_kwargs,
) -> ObjectInstance:
    """Construct an ObjectInstance from a mask + depth + intrinsics."""
    mask = mask.astype(bool)
    ys, xs = np.nonzero(mask)
    area = int(mask.sum())

    if area > 0:
        cu, cv = float(xs.mean()), float(ys.mean())
    else:
        cu, cv = 0.0, 0.0

    points, _ = backproject(depth_m, intr, pixel_mask=mask)
    if len(points):
        z = points[:, 2]
        median_d = float(np.median(z))
        mean_d = float(np.mean(z))
        c3d = tuple(float(v) for v in points.mean(axis=0))
    else:
        median_d = mean_d = 0.0
        c3d = (0.0, 0.0, 0.0)

    measurement = None
    if measure:
        measurement = measure_object(object_id, mask, depth_m, intr, **measure_kwargs)

    return ObjectInstance(
        id=int(object_id),
        mask=mask,
        bbox=_mask_bbox(mask),
        score=float(score),
        area_px=area,
        centroid_2d=(cu, cv),
        centroid_3d=c3d,
        median_depth=median_d,
        mean_depth=mean_d,
        valid_depth_ratio=valid_depth_ratio(depth_m, mask),
        n_points=int(len(points)),
        measurement=measurement,
    )
