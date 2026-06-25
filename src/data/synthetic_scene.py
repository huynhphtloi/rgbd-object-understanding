"""
Synthetic RGB-D scene generator.

Builds a small tabletop scene of axis-aligned cuboids with KNOWN dimensions and
renders an aligned RGB image + metric depth map. Each cuboid face (and the
table) is rasterized as a filled polygon; per-pixel depth is computed exactly by
ray-plane intersection and composited with a z-buffer. This yields hole-free,
tight instance masks.

Because ground-truth sizes, masks and relations are known, this lets us verify
the depth-measurement and spatial-relation modules without any real dataset.

Camera frame: +x right, +y down, +z forward (away from camera).
Depth is returned in millimetres (uint16) to mimic OCID/RealSense storage.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SyntheticObject:
    id: int
    center: tuple        # (x, y, z) metres, camera frame
    size: tuple          # (W, H, D) metres along (x, y, z)
    color: tuple         # BGR


DEFAULT_INTRINSICS = {
    "fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0,
    "width": 640, "height": 480,
}

TABLE_Y = 0.30          # table surface 30 cm below the optical axis (y is down)
TABLE_COLOR = (160, 160, 160)

# Cuboid corner offsets (unit cube, multiplied by half-size).
_CORNERS = np.array([
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
], dtype=np.float64)
# Six faces as corner-index quads.
_FACES = [
    (0, 1, 2, 3),  # -z (front)
    (4, 5, 6, 7),  # +z (back)
    (0, 1, 5, 4),  # -y (top, since y is down)
    (3, 2, 6, 7),  # +y (bottom)
    (0, 3, 7, 4),  # -x (left)
    (1, 2, 6, 5),  # +x (right)
]


def default_scene() -> list[SyntheticObject]:
    """A scene exercising measurement, occlusion and stacking.

    - obj1: standalone box, offset left (side+top visible -> good 3D extent)
    - obj2: wider box on the right
    - obj3: small box stacked ON TOP of obj2  -> likely_on_top_of
    - obj4: box closer to camera, overlapping obj1 in 2D -> in_front_of / occludes
    """
    objs = []
    objs.append(SyntheticObject(1, (-0.14, TABLE_Y - 0.08, 0.90), (0.12, 0.16, 0.10), (60, 180, 75)))
    objs.append(SyntheticObject(2, (0.13, TABLE_Y - 0.05, 1.05), (0.18, 0.10, 0.12), (230, 160, 50)))
    objs.append(SyntheticObject(3, (0.13, TABLE_Y - 0.13, 1.05), (0.06, 0.06, 0.06), (40, 70, 230)))
    objs.append(SyntheticObject(4, (-0.06, TABLE_Y - 0.06, 0.72), (0.10, 0.12, 0.08), (200, 80, 200)))
    return objs


def _project(points3d, intr):
    """Project (N,3) camera-frame points to (N,2) float pixel coords."""
    z = points3d[:, 2]
    u = intr["fx"] * points3d[:, 0] / z + intr["cx"]
    v = intr["fy"] * points3d[:, 1] / z + intr["cy"]
    return np.stack([u, v], axis=1)


def _rasterize_face(corners3d, color, obj_id, zbuf, idbuf, rgb, intr):
    """Rasterize one planar quad with exact per-pixel depth + z-buffer compositing."""
    H, W = zbuf.shape
    if np.any(corners3d[:, 2] <= 1e-6):
        return  # skip faces crossing/behind the camera

    pts2d = _project(corners3d, intr)
    poly = np.round(pts2d).astype(np.int32)

    # pixel coverage of the quad
    facemask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(facemask, poly, 1)
    ys, xs = np.nonzero(facemask)
    if len(xs) == 0:
        return

    # plane through the face: normal n, point p0
    p0 = corners3d[0]
    n = np.cross(corners3d[1] - corners3d[0], corners3d[2] - corners3d[0])
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return
    n = n / nn

    # ray-plane intersection: point = t*d, t = (n.p0)/(n.d); depth z == t
    d = np.stack([(xs - intr["cx"]) / intr["fx"],
                  (ys - intr["cy"]) / intr["fy"],
                  np.ones_like(xs, dtype=np.float64)], axis=1)
    denom = d @ n
    ok = np.abs(denom) > 1e-9
    t = np.full(len(xs), np.inf)
    t[ok] = (n @ p0) / denom[ok]
    z = t  # because d_z == 1

    valid = ok & (z > 1e-6)
    ys, xs, z = ys[valid], xs[valid], z[valid]

    closer = z < zbuf[ys, xs]
    ys, xs, z = ys[closer], xs[closer], z[closer]
    zbuf[ys, xs] = z
    idbuf[ys, xs] = obj_id
    rgb[ys, xs] = color


def render_scene(
    objects: list[SyntheticObject] | None = None,
    intrinsics: dict | None = None,
    depth_noise_std_m: float = 0.0,
    seed: int = 42,
):
    """Render the scene.

    Returns
    -------
    rgb      : (H, W, 3) uint8 (BGR)
    depth_mm : (H, W) uint16 metric depth in millimetres (0 = invalid)
    intr     : intrinsics dict
    gt       : list of {id, width_cm, height_cm, depth_cm, mask}
    """
    rng = np.random.default_rng(seed)
    objects = objects or default_scene()
    intr = intrinsics or dict(DEFAULT_INTRINSICS)
    H, W = intr["height"], intr["width"]

    zbuf = np.full((H, W), np.inf, dtype=np.float64)
    idbuf = np.zeros((H, W), dtype=np.int32)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # table as one large quad on the plane y = TABLE_Y
    table_quad = np.array([
        [-0.7, TABLE_Y, 0.45], [0.7, TABLE_Y, 0.45],
        [0.7, TABLE_Y, 1.8], [-0.7, TABLE_Y, 1.8],
    ], dtype=np.float64)
    _rasterize_face(table_quad, TABLE_COLOR, 0, zbuf, idbuf, rgb, intr)

    for obj in objects:
        c = np.asarray(obj.center)
        h = np.asarray(obj.size) / 2.0
        corners = c + _CORNERS * h
        for face in _FACES:
            _rasterize_face(corners[list(face)], obj.color, obj.id,
                            zbuf, idbuf, rgb, intr)

    depth_m = np.where(np.isfinite(zbuf), zbuf, 0.0)
    if depth_noise_std_m > 0:
        noise = rng.normal(0, depth_noise_std_m, size=depth_m.shape)
        depth_m = np.where(depth_m > 0, depth_m + noise, 0.0)
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

    gt = []
    for obj in objects:
        mask = idbuf == obj.id
        if mask.sum() == 0:
            continue
        gt.append({
            "id": obj.id,
            "width_cm": obj.size[0] * 100.0,
            "height_cm": obj.size[1] * 100.0,
            "depth_cm": obj.size[2] * 100.0,
            "mask": mask,
        })
    return rgb, depth_mm, intr, gt
