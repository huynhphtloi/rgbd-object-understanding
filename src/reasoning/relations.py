"""
Rule-based pairwise spatial-relation reasoning.

Relations produced (directed unless noted):
    overlaps                (symmetric, emitted once)
    in_front_of / behind    (from median-depth ordering)
    occludes / partially_occluded_by
    likely_on_top_of        (heuristic, deliberately hedged)

Each record carries a `rule_margin`: a derived quantity (depth gap in metres,
overlap fraction, or 3D distance) that indicates how strongly the rule fired.
We do NOT invent a probability.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np

from .object_instance import ObjectInstance
from . import rules


@dataclass
class Relation:
    subject: int          # object id
    relation: str
    object: int           # object id
    rule_margin: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["subject"] = f"object_{self.subject}"
        d["object"] = f"object_{self.object}"
        d["rule_margin"] = round(float(self.rule_margin), 4)
        return d


DEFAULTS = {
    "overlap_iou_thresh": 0.05,
    "depth_margin_m": 0.02,
    "occlusion_overlap_thresh": 0.15,
    "on_top_vertical_frac": 0.4,
    "on_top_x_overlap_frac": 0.2,
    "on_top_depth_tol_m": 0.08,
    "on_top_gap_frac": 0.4,
}


def _x_overlap_frac(a: ObjectInstance, b: ObjectInstance) -> float:
    ax1, _, ax2, _ = a.bbox
    bx1, _, bx2, _ = b.bbox
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    denom = min(ax2 - ax1, bx2 - bx1)
    return inter / float(denom) if denom > 0 else 0.0


def infer_pair(a: ObjectInstance, b: ObjectInstance, cfg: dict | None = None) -> list[Relation]:
    """Infer relations between an unordered pair; returns directed records."""
    c = {**DEFAULTS, **(cfg or {})}
    out: list[Relation] = []

    iou = rules.mask_iou(a.mask, b.mask)
    bbiou = rules.bbox_iou(a.bbox, b.bbox)
    overlap = max(iou, bbiou)

    # --- overlaps (symmetric, emit once, lower id as subject) ---
    if overlap >= c["overlap_iou_thresh"]:
        s, o = (a, b) if a.id <= b.id else (b, a)
        out.append(Relation(s.id, "overlaps", o.id, overlap))

    # --- depth ordering -> in_front_of / behind ---
    front = back = None
    if a.has_depth and b.has_depth:
        gap = rules.depth_order(a, b)  # b.median - a.median
        if abs(gap) >= c["depth_margin_m"]:
            if gap > 0:
                front, back = a, b
            else:
                front, back = b, a
            out.append(Relation(front.id, "in_front_of", back.id, abs(gap)))
            out.append(Relation(back.id, "behind", front.id, abs(gap)))

    # --- occlusion (needs 2D overlap + known depth ordering) ---
    if front is not None and overlap >= c["overlap_iou_thresh"]:
        # fraction of the BACK object's mask hidden behind the FRONT object
        covered = rules.mask_overlap_fraction(back.mask, front.mask)
        if covered >= c["occlusion_overlap_thresh"]:
            out.append(Relation(front.id, "occludes", back.id, covered))
            out.append(Relation(back.id, "partially_occluded_by", front.id, covered))

    # --- likely_on_top_of (heuristic) ---
    rel = _maybe_on_top(a, b, c)
    if rel is not None:
        out.append(rel)

    return out


def _maybe_on_top(a: ObjectInstance, b: ObjectInstance, c: dict) -> Relation | None:
    """Decide whether one object is likely stacked on the other.

    Robust heuristic (deliberately hedged as "likely"). The upper object must:
      1. sit in the upper part of the union (smaller image-y),
      2. overlap the lower object horizontally,
      3. be at a SIMILAR depth (stacked objects share distance, unlike an
         in-front/behind pair), and
      4. be vertically adjacent in the image (its lower edge near the lower
         object's upper region, i.e. they touch rather than float apart).
    Uses median depth + image geometry rather than fragile 3D centroids.
    """
    if not (a.has_depth and b.has_depth):
        return None
    top, bottom = (a, b) if a.centroid_2d[1] <= b.centroid_2d[1] else (b, a)

    # (1) top object high in the union
    if rules.vertical_position_frac(top, bottom) > c["on_top_vertical_frac"]:
        return None
    # (2) horizontal overlap
    if _x_overlap_frac(top, bottom) < c["on_top_x_overlap_frac"]:
        return None
    # (3) similar depth (NOT an in-front/behind pair)
    ddepth = abs(top.median_depth - bottom.median_depth)
    if ddepth > c["on_top_depth_tol_m"]:
        return None
    # (4) vertical adjacency: bottom edge of top object near top object-region
    #     of the lower object. Gap measured relative to the lower object height.
    bottom_h = max(1.0, bottom.bbox[3] - bottom.bbox[1])
    gap = abs(top.bbox[3] - bottom.bbox[1])  # |top.y2 - bottom.y1|
    if gap > c["on_top_gap_frac"] * bottom_h:
        return None

    return Relation(top.id, "likely_on_top_of", bottom.id, ddepth)


def infer_all(instances: list[ObjectInstance], cfg: dict | None = None) -> list[Relation]:
    """Infer relations across every unordered pair of instances."""
    relations: list[Relation] = []
    for a, b in combinations(instances, 2):
        relations.extend(infer_pair(a, b, cfg))
    return relations
