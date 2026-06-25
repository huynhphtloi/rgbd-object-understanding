"""Overlay instance masks, boxes and labels on an RGB image."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from ..reasoning.object_instance import ObjectInstance


def distinct_colors(n: int) -> list[tuple]:
    """n visually distinct BGR colors."""
    colors = []
    for i in range(max(1, n)):
        h = (i / max(1, n)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def overlay_instances(
    rgb: np.ndarray,
    instances: list[ObjectInstance],
    alpha: float = 0.45,
    draw_labels: bool = True,
) -> np.ndarray:
    """Return a copy of `rgb` with colored masks, boxes and id/size labels."""
    out = rgb.copy()
    colors = distinct_colors(len(instances))

    for inst, color in zip(instances, colors):
        mask = inst.mask.astype(bool)
        colored = np.zeros_like(out)
        colored[mask] = color
        out = cv2.addWeighted(out, 1.0, colored, alpha, 0)

        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if draw_labels:
            label = f"obj{inst.id}"
            if inst.measurement is not None:
                m = inst.measurement
                label += f" {m.width_cm:.0f}x{m.height_cm:.0f}x{m.depth_cm:.0f}cm"
            _put_label(out, label, (x1, max(0, y1 - 6)), color)
    return out


def _put_label(img, text, org, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - th - base), (x + tw, y + base), color, -1)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
