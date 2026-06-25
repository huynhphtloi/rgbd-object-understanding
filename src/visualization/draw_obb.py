"""
Visualize estimated 3D extents projected back onto the image.

We don't have a full 6-DoF oriented box in image space (the OBB lives in 3D),
so for an interpretable overlay we draw the axis-aligned mask bbox annotated
with the estimated metric W/H/D. This keeps the visualization honest about
what is measured (visible extent) without faking a precise wireframe.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..reasoning.object_instance import ObjectInstance


def draw_dimension_boxes(rgb: np.ndarray, instances: list[ObjectInstance]) -> np.ndarray:
    out = rgb.copy()
    for inst in instances:
        if inst.measurement is None:
            continue
        m = inst.measurement
        x1, y1, x2, y2 = inst.bbox
        ok = m.obb_extent_well_defined
        color = (0, 200, 0) if ok else (0, 165, 255)  # green if reliable else orange
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        lines = [
            f"obj{inst.id} [{m.visibility_group}]",
            f"W {m.width_cm:.1f}  H {m.height_cm:.1f}  D {m.depth_cm:.1f} cm",
            f"vdr {m.valid_depth_ratio:.2f}  n {m.n_points}",
        ]
        _put_multiline(out, lines, (x1, y1), color)
    return out


def _put_multiline(img, lines, org, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, pad = 0.45, 1, 3
    x, y = org
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    lh = max(s[1] for s in sizes) + 5
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = lh * len(lines) + pad
    y0 = max(0, y - box_h)
    cv2.rectangle(img, (x, y0), (x + box_w, y0 + box_h), color, -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x + pad, y0 + lh * (i + 1) - 3),
                    font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return img
