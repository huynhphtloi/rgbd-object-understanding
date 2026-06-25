"""
Class-agnostic instance segmentation inference with YOLO-seg (Ultralytics).

Returns a list of `Detection` (binary mask at full image resolution, xyxy box,
score). If the trained class-agnostic weights are missing, falls back to a
pretrained COCO yolov8-seg model whose detections are treated class-agnostically
(every detected thing is just "object"). This keeps the demo runnable before
training finishes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    mask: np.ndarray   # (H, W) bool, full image resolution
    bbox: tuple        # (x1, y1, x2, y2) pixels
    score: float


class YoloSegmenter:
    def __init__(
        self,
        weights: str,
        fallback_weights: str = "yolov8n-seg.pt",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO

        path = weights if (weights and os.path.isfile(weights)) else fallback_weights
        self.using_fallback = path == fallback_weights and not (
            weights and os.path.isfile(weights)
        )
        self.model = YOLO(path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def predict(self, rgb: np.ndarray) -> list[Detection]:
        """Run segmentation on a BGR/RGB uint8 image -> list of Detection."""
        h, w = rgb.shape[:2]
        results = self.model.predict(
            rgb, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False
        )
        dets: list[Detection] = []
        if not results:
            return dets
        r = results[0]
        if r.masks is None:
            return dets

        masks = r.masks.data.cpu().numpy()           # (N, mh, mw) in [0,1]
        boxes = r.boxes.xyxy.cpu().numpy()           # (N, 4)
        scores = r.boxes.conf.cpu().numpy()          # (N,)

        for m, box, sc in zip(masks, boxes, scores):
            mask = (m > 0.5).astype(np.uint8)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            dets.append(Detection(mask.astype(bool), tuple(map(int, box)), float(sc)))
        return dets


def detections_from_masks(masks: list[np.ndarray], scores=None) -> list[Detection]:
    """Build Detection objects directly from binary masks (e.g. synthetic GT)."""
    out = []
    for i, m in enumerate(masks):
        m = m.astype(bool)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        sc = 1.0 if scores is None else float(scores[i])
        out.append(Detection(m, bbox, sc))
    return out
