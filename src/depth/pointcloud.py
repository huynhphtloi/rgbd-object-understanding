"""
Point-cloud helpers built on top of depth_utils.

Open3D is used when available (faster, robust statistical outlier removal); a
pure-NumPy fallback is provided so the pipeline runs even without Open3D.
"""
from __future__ import annotations

import numpy as np

try:  # Open3D is optional.
    import open3d as o3d

    _HAS_O3D = True
except Exception:  # pragma: no cover - environment dependent
    o3d = None
    _HAS_O3D = False


def has_open3d() -> bool:
    return _HAS_O3D


def to_o3d(points: np.ndarray):
    """Wrap an (N, 3) array in an Open3D point cloud (requires Open3D)."""
    if not _HAS_O3D:
        raise RuntimeError("Open3D is not available")
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pc


def remove_statistical_outliers(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove statistical outliers from a point set.

    A point is dropped if its mean distance to its `nb_neighbors` nearest
    neighbours exceeds (global_mean + std_ratio * global_std).

    Returns the kept points and a boolean keep-mask over the input.
    """
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    if n <= nb_neighbors:
        return points, np.ones(n, dtype=bool)

    if _HAS_O3D:
        pc = to_o3d(points)
        _, idx = pc.remove_statistical_outlier(nb_neighbors, std_ratio)
        keep = np.zeros(n, dtype=bool)
        keep[np.asarray(idx, dtype=int)] = True
        return points[keep], keep

    # NumPy fallback: KD-tree mean kNN distance thresholding.
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    k = min(nb_neighbors + 1, n)
    dists, _ = tree.query(points, k=k)
    mean_knn = dists[:, 1:].mean(axis=1)  # exclude self (distance 0)
    thresh = mean_knn.mean() + std_ratio * mean_knn.std()
    keep = mean_knn <= thresh
    return points[keep], keep


def centroid(points: np.ndarray) -> np.ndarray:
    """3D centroid of a point set, or zeros if empty."""
    if len(points) == 0:
        return np.zeros(3, dtype=np.float32)
    return points.mean(axis=0).astype(np.float32)


def depth_stats(z_values: np.ndarray) -> dict:
    """Summary statistics of a 1D array of depth (z) values."""
    if len(z_values) == 0:
        return {"median": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "median": float(np.median(z_values)),
        "mean": float(np.mean(z_values)),
        "std": float(np.std(z_values)),
        "min": float(np.min(z_values)),
        "max": float(np.max(z_values)),
    }
