"""
Dominant-plane (table / floor) removal via RANSAC.

Removing the support plane before measurement prevents the table surface from
inflating an object's oriented bounding box. Open3D's segment_plane is used
when available; otherwise a NumPy RANSAC implementation is used.
"""
from __future__ import annotations

import numpy as np

from .pointcloud import _HAS_O3D, to_o3d


def segment_plane(
    points: np.ndarray,
    dist_thresh: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the dominant plane and return (plane_model, inlier_mask).

    plane_model is (a, b, c, d) with a*x + b*y + c*z + d = 0 and unit normal.
    inlier_mask is a boolean array over `points` marking plane points.
    """
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    if n < ransac_n:
        return np.array([0, 0, 0, 0], dtype=np.float32), np.zeros(n, dtype=bool)

    if _HAS_O3D:
        pc = to_o3d(points)
        model, inliers = pc.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(inliers, dtype=int)] = True
        return np.asarray(model, dtype=np.float32), mask

    return _ransac_plane_numpy(points, dist_thresh, num_iterations, seed)


def _ransac_plane_numpy(
    points: np.ndarray,
    dist_thresh: float,
    num_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(points)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best_model = np.array([0, 0, 0, 0], dtype=np.float32)

    for _ in range(num_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        d = -float(normal @ p0)
        dist = np.abs(points @ normal + d)
        inliers = dist < dist_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_model = np.array([*normal, d], dtype=np.float32)

    return best_model, best_inliers


def remove_dominant_plane(
    points: np.ndarray,
    dist_thresh: float = 0.01,
    max_inlier_frac: float = 0.9,
) -> np.ndarray:
    """Return points with the dominant plane removed.

    If the detected plane would remove more than `max_inlier_frac` of points
    (likely not a support surface), the points are returned unchanged.
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 10:
        return points
    _, inliers = segment_plane(points, dist_thresh=dist_thresh)
    if inliers.mean() > max_inlier_frac:
        return points
    return points[~inliers]
