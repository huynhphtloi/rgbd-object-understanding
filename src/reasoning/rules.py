"""
Low-level geometric predicates used by the relation reasoner.

Everything here is deterministic and side-effect free so the rules can be unit
tested and tuned independently of the relation logic.
"""
from __future__ import annotations

import numpy as np

from .object_instance import ObjectInstance


def bbox_iou(a: tuple, b: tuple) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / float(union)


def mask_overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of mask `a` that is also covered by mask `b` (intersection/area_a)."""
    a = a.astype(bool)
    b = b.astype(bool)
    area_a = int(a.sum())
    if area_a == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return inter / float(area_a)


def depth_order(a: ObjectInstance, b: ObjectInstance) -> float:
    """Signed median-depth gap b - a (metres).

    Positive  -> a is closer to the camera (in front of b).
    """
    return float(b.median_depth - a.median_depth)


def vertical_position_frac(a: ObjectInstance, b: ObjectInstance) -> float:
    """Where a's centroid sits within the vertical span of (a union b).

    0 = top of the union, 1 = bottom. Smaller means a is visually above b.
    (Image v increases downward.)
    """
    va = a.centroid_2d[1]
    y_top = min(a.bbox[1], b.bbox[1])
    y_bot = max(a.bbox[3], b.bbox[3])
    span = max(1.0, y_bot - y_top)
    return float((va - y_top) / span)


def centroid_distance_3d(a: ObjectInstance, b: ObjectInstance) -> float:
    """Euclidean distance between 3D centroids (metres). inf if depth missing."""
    if not (a.has_depth and b.has_depth):
        return float("inf")
    ca = np.asarray(a.centroid_3d)
    cb = np.asarray(b.centroid_3d)
    return float(np.linalg.norm(ca - cb))
