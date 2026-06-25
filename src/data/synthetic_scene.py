"""
Synthetic RGB-D scene generator.

Builds a small tabletop scene of axis-aligned cuboids with KNOWN dimensions and
renders an aligned RGB image + metric depth map via point-sampling + z-buffer.
Because ground-truth sizes, masks and relations are known, this lets us verify
the depth-measurement and spatial-relation modules without any real dataset.

Camera frame: +x right, +y down, +z forward (away from camera).
Depth is returned in millimetres (uint16) to mimic OCID/RealSense storage.
"""
from __future__ import annotations

from dataclasses import dataclass

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


def default_scene() -> list[SyntheticObject]:
    """A scene exercising measurement, occlusion and stacking.

    - obj1: standalone box, offset left (side+top visible -> good 3D extent)
    - obj2: wider box on the right
    - obj3: small box stacked ON TOP of obj2  -> likely_on_top_of
    - obj4: box closer to camera, overlapping obj1 in 2D -> in_front_of / occludes
    """
    objs = []
    # obj1: W12 H16 D10 cm, left side, mid depth
    objs.append(SyntheticObject(1, (-0.14, TABLE_Y - 0.08, 0.90), (0.12, 0.16, 0.10), (60, 180, 75)))
    # obj2: W18 H10 D12 cm, right side, farther
    objs.append(SyntheticObject(2, (0.13, TABLE_Y - 0.05, 1.05), (0.18, 0.10, 0.12), (230, 160, 50)))
    # obj3: W6 H6 D6 cm stacked on top of obj2 (obj2 top at y=TABLE_Y-0.10)
    objs.append(SyntheticObject(3, (0.13, TABLE_Y - 0.13, 1.05), (0.06, 0.06, 0.06), (40, 70, 230)))
    # obj4: W10 H12 D08 cm, closer + slightly left, overlaps obj1 in image
    objs.append(SyntheticObject(4, (-0.06, TABLE_Y - 0.06, 0.72), (0.10, 0.12, 0.08), (200, 80, 200)))
    return objs


def _sample_cuboid_surface(center, size, step=0.0025) -> np.ndarray:
    """Dense point sample over all six faces of an axis-aligned cuboid."""
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(size, dtype=np.float64) / 2.0
    pts = []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        n0 = max(2, int(2 * h[other[0]] / step))
        n1 = max(2, int(2 * h[other[1]] / step))
        g0 = np.linspace(c[other[0]] - h[other[0]], c[other[0]] + h[other[0]], n0)
        g1 = np.linspace(c[other[1]] - h[other[1]], c[other[1]] + h[other[1]], n1)
        gg0, gg1 = np.meshgrid(g0, g1)
        for sign in (-1, 1):
            face = np.zeros((gg0.size, 3))
            face[:, other[0]] = gg0.ravel()
            face[:, other[1]] = gg1.ravel()
            face[:, axis] = c[axis] + sign * h[axis]
            pts.append(face)
    return np.concatenate(pts, axis=0)


def _sample_table(step=0.005) -> np.ndarray:
    xs = np.arange(-0.6, 0.6, step)
    zs = np.arange(0.45, 1.7, step)
    xx, zz = np.meshgrid(xs, zs)
    pts = np.zeros((xx.size, 3))
    pts[:, 0] = xx.ravel()
    pts[:, 1] = TABLE_Y
    pts[:, 2] = zz.ravel()
    return pts


def render_scene(
    objects: list[SyntheticObject] | None = None,
    intrinsics: dict | None = None,
    depth_noise_std_m: float = 0.0,
    seed: int = 42,
):
    """Render the scene.

    Returns
    -------
    rgb        : (H, W, 3) uint8 (BGR)
    depth_mm   : (H, W) uint16 metric depth in millimetres (0 = invalid)
    intr       : intrinsics dict
    gt         : list of {id, width_cm, height_cm, depth_cm, mask}
    """
    rng = np.random.default_rng(seed)
    objects = objects or default_scene()
    intr = intrinsics or dict(DEFAULT_INTRINSICS)
    H, W = intr["height"], intr["width"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    zbuf = np.full((H, W), np.inf, dtype=np.float64)
    idbuf = np.zeros((H, W), dtype=np.int32)         # 0 = table/background
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    def splat(points, color, obj_id):
        z = points[:, 2]
        ok = z > 1e-6
        u = (fx * points[:, 0] / z + cx).astype(np.int64)
        v = (fy * points[:, 1] / z + cy).astype(np.int64)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z = u[inb], v[inb], z[inb]
        # z-buffer: keep nearest surface per pixel
        for ui, vi, zi in zip(u, v, z):
            if zi < zbuf[vi, ui]:
                zbuf[vi, ui] = zi
                idbuf[vi, ui] = obj_id
                rgb[vi, ui] = color

    # table first (farthest fallback), then objects
    splat(_sample_table(), TABLE_COLOR, 0)
    for obj in objects:
        splat(_sample_cuboid_surface(obj.center, obj.size), obj.color, obj.id)

    # Fill tiny z-buffer gaps with a 3x3 nearest pass for cleaner masks.
    _fill_holes(rgb, zbuf, idbuf)

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


def _fill_holes(rgb, zbuf, idbuf):
    """One-pass 3x3 nearest fill for unset pixels surrounded by surface."""
    H, W = zbuf.shape
    unset = ~np.isfinite(zbuf)
    ys, xs = np.nonzero(unset)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        nb = zbuf[y0:y1, x0:x1]
        if np.isfinite(nb).any():
            local = np.argmin(np.where(np.isfinite(nb), nb, np.inf))
            ry, rx = np.unravel_index(local, nb.shape)
            zbuf[y, x] = nb[ry, rx]
            idbuf[y, x] = idbuf[y0 + ry, x0 + rx]
            rgb[y, x] = rgb[y0 + ry, x0 + rx]
