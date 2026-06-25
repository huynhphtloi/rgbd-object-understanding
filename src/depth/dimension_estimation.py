"""
Depth-based measurement.

Given a 2D instance mask + metric depth + intrinsics, estimate the object's
*visible* 3D extent by:
    mask -> masked depth -> back-projection -> outlier removal
         -> optional plane removal -> oriented bounding box (PCA) -> W/H/D.

We report quality indicators derived from the data (valid_depth_ratio,
n_points, depth_std, well-defined flag, visibility group) rather than an
invented confidence score.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .depth_utils import CameraIntrinsics, backproject, valid_depth_ratio
from .pointcloud import remove_statistical_outliers, centroid, depth_stats
from .plane_removal import remove_dominant_plane

M_TO_CM = 100.0


@dataclass
class MeasurementResult:
    object_id: int
    width_cm: float
    height_cm: float
    depth_cm: float
    valid_depth_ratio: float
    n_points: int
    depth_std_m: float
    median_depth_m: float
    centroid_3d: tuple  # (x, y, z) metres, camera frame
    obb_extent_well_defined: bool
    visibility_group: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["centroid_3d"] = [round(float(c), 4) for c in self.centroid_3d]
        for k in ("width_cm", "height_cm", "depth_cm", "depth_std_m",
                  "median_depth_m", "valid_depth_ratio"):
            d[k] = round(float(d[k]), 3)
        return d


def visibility_group(ratio: float, full: float = 0.7, partial: float = 0.3) -> str:
    if ratio >= full:
        return "fully_visible"
    if ratio >= partial:
        return "partially_occluded"
    return "heavily_occluded"


def oriented_extents(points: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the three OBB side lengths (metres, descending) via PCA.

    The second value flags whether the box is well defined (enough spread on
    all three principal axes).
    """
    if len(points) < 4:
        return np.zeros(3, dtype=np.float32), False

    mean = points.mean(axis=0)
    centred = points - mean
    cov = np.cov(centred, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Project onto principal axes and take peak-to-peak extent per axis.
    proj = centred @ eigvecs
    extents = proj.max(axis=0) - proj.min(axis=0)
    extents = np.sort(extents)[::-1].astype(np.float32)  # descending

    # Well defined if the smallest eigenvalue is not numerically degenerate.
    eigvals = np.clip(eigvals, 0, None)
    well_defined = bool(eigvals.min() > 1e-6 and extents.min() > 1e-3)
    return extents, well_defined


def measure_object(
    object_id: int,
    mask: np.ndarray,
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    remove_plane: bool = True,
    plane_dist_thresh_m: float = 0.01,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
) -> MeasurementResult:
    """Estimate visible 3D dimensions for a single masked object."""
    mask = mask.astype(bool)
    ratio = valid_depth_ratio(depth_m, mask)

    points, _ = backproject(depth_m, intr, pixel_mask=mask)
    z_vals = points[:, 2] if len(points) else np.array([])
    stats = depth_stats(z_vals)

    if len(points) >= 4:
        points, _ = remove_statistical_outliers(
            points, outlier_nb_neighbors, outlier_std_ratio
        )
    if remove_plane and len(points) >= 10:
        points = remove_dominant_plane(points, dist_thresh=plane_dist_thresh_m)

    extents, well_defined = oriented_extents(points)
    c3d = centroid(points)

    return MeasurementResult(
        object_id=int(object_id),
        width_cm=float(extents[0] * M_TO_CM),
        height_cm=float(extents[1] * M_TO_CM),
        depth_cm=float(extents[2] * M_TO_CM),
        valid_depth_ratio=float(ratio),
        n_points=int(len(points)),
        depth_std_m=float(stats["std"]),
        median_depth_m=float(stats["median"]),
        centroid_3d=tuple(float(v) for v in c3d),
        obb_extent_well_defined=bool(well_defined and len(points) >= 50),
        visibility_group=visibility_group(ratio),
    )
