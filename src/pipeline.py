"""
End-to-end RGB-D scene-understanding pipeline.

    detections -> counting filter -> object instances (+ measurement)
               -> depth-validity filter -> spatial relations -> scene graph

Shared by the CLI runner (run_pipeline.py) and the Gradio demo so both behave
identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .depth.depth_utils import CameraIntrinsics
from .reasoning.object_instance import ObjectInstance, build_instance
from .reasoning import relations as rel
from .reasoning.graph_builder import build_scene_dict
from .reasoning.rules import bbox_iou
from .segmentation.infer_yolo import Detection


@dataclass
class PipelineResult:
    instances: list[ObjectInstance]
    relations: list
    scene: dict
    counts: dict = field(default_factory=dict)


def _nms_dedup(dets: list[Detection], iou_thresh: float) -> list[Detection]:
    """Greedy NMS by score to drop near-duplicate masks (uses bbox IoU)."""
    order = sorted(range(len(dets)), key=lambda i: dets[i].score, reverse=True)
    keep: list[int] = []
    for i in order:
        if all(bbox_iou(dets[i].bbox, dets[j].bbox) <= iou_thresh for j in keep):
            keep.append(i)
    return [dets[i] for i in keep]


def filter_detections(
    dets: list[Detection],
    conf_thresh: float,
    min_mask_area: int,
    dup_iou_thresh: float,
) -> list[Detection]:
    """Counting stage 1: drop low-confidence, tiny, and duplicate masks."""
    kept = [
        d for d in dets
        if d.score >= conf_thresh and int(d.mask.sum()) >= min_mask_area
    ]
    return _nms_dedup(kept, dup_iou_thresh)


class Pipeline:
    def __init__(self, cfg: dict, segmenter=None):
        self.cfg = cfg
        self._segmenter = segmenter  # may be injected (or built lazily)

    # -- segmentation -------------------------------------------------------
    @property
    def segmenter(self):
        if self._segmenter is None:
            from .segmentation.infer_yolo import YoloSegmenter

            s = self.cfg["segmentation"]
            self._segmenter = YoloSegmenter(
                weights=s.get("weights", ""),
                fallback_weights=s.get("fallback_weights", "yolov8n-seg.pt"),
                conf=s.get("conf", 0.25),
                iou=s.get("iou", 0.7),
                imgsz=s.get("imgsz", 640),
            )
        return self._segmenter

    # -- main entry ---------------------------------------------------------
    def run(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intr: CameraIntrinsics,
        detections: list[Detection] | None = None,
        image_name: str | None = None,
    ) -> PipelineResult:
        seg = self.cfg["segmentation"]
        cnt = self.cfg["counting"]
        dep = self.cfg["depth"]

        # 1. segmentation (or use injected detections, e.g. synthetic GT)
        if detections is None:
            detections = self.segmenter.predict(rgb)
        n_raw = len(detections)

        # 2. counting stage 1: confidence / area / duplicate filtering
        dets = filter_detections(
            detections,
            conf_thresh=cnt.get("conf_thresh", seg.get("conf", 0.25)),
            min_mask_area=cnt.get("min_mask_area", seg.get("min_mask_area", 200)),
            dup_iou_thresh=cnt.get("dup_iou_thresh", 0.8),
        )
        n_rgb_only = len(dets)

        # 3. build instances with depth-based measurement
        instances: list[ObjectInstance] = []
        for i, d in enumerate(dets, start=1):
            inst = build_instance(
                i, d.mask, depth_m, intr, score=d.score, measure=True,
                remove_plane=dep.get("remove_plane", True),
                plane_dist_thresh_m=dep.get("plane_dist_thresh_m", 0.01),
                outlier_nb_neighbors=dep.get("outlier_nb_neighbors", 20),
                outlier_std_ratio=dep.get("outlier_std_ratio", 2.0),
            )
            instances.append(inst)

        # 4. counting stage 2: depth-validity filtering
        min_vdr = cnt.get("min_valid_depth_ratio", 0.0)
        min_pts = cnt.get("min_points", 0)
        depth_filtered = [
            inst for inst in instances
            if inst.valid_depth_ratio >= min_vdr and inst.n_points >= min_pts
        ]
        # renumber ids so output is contiguous
        for new_id, inst in enumerate(depth_filtered, start=1):
            inst.id = new_id
            if inst.measurement is not None:
                inst.measurement.object_id = new_id

        # 5. spatial relations
        relations = rel.infer_all(depth_filtered, self.cfg.get("relations"))

        # 6. scene graph dict
        scene = build_scene_dict(depth_filtered, relations, image_name=image_name)

        counts = {
            "raw_detections": n_raw,
            "rgb_only_count": n_rgb_only,
            "depth_filtered_count": len(depth_filtered),
        }
        scene["counts"] = counts
        return PipelineResult(depth_filtered, relations, scene, counts)
