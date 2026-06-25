"""
Assemble object instances + relations into a scene graph (JSON + networkx).
"""
from __future__ import annotations

import json

from .object_instance import ObjectInstance
from .relations import Relation


def build_scene_dict(
    instances: list[ObjectInstance],
    relations: list[Relation],
    image_name: str | None = None,
) -> dict:
    """Serializable scene description: objects, measurements, relations."""
    nodes = []
    for inst in instances:
        node = {
            "id": f"object_{inst.id}",
            "score": round(float(inst.score), 3),
            "area_px": inst.area_px,
            "bbox_xyxy": [int(v) for v in inst.bbox],
            "centroid_2d": [round(float(v), 1) for v in inst.centroid_2d],
            "median_depth_m": round(float(inst.median_depth), 3),
            "valid_depth_ratio": round(float(inst.valid_depth_ratio), 3),
        }
        if inst.measurement is not None:
            m = inst.measurement.to_dict()
            node["dimensions_cm"] = {
                "width": m["width_cm"],
                "height": m["height_cm"],
                "depth": m["depth_cm"],
            }
            node["obb_extent_well_defined"] = m["obb_extent_well_defined"]
            node["visibility_group"] = m["visibility_group"]
        nodes.append(node)

    return {
        "image": image_name,
        "object_count": len(instances),
        "objects": nodes,
        "relations": [r.to_dict() for r in relations],
    }


def save_scene_json(scene: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(scene, fh, indent=2)


def to_networkx(instances: list[ObjectInstance], relations: list[Relation]):
    """Build a networkx.DiGraph (requires networkx)."""
    import networkx as nx

    g = nx.MultiDiGraph()
    for inst in instances:
        label = f"obj{inst.id}"
        if inst.measurement is not None:
            m = inst.measurement
            label += f"\n{m.width_cm:.0f}x{m.height_cm:.0f}x{m.depth_cm:.0f}cm"
        g.add_node(inst.id, label=label, depth=inst.median_depth)
    for r in relations:
        g.add_edge(r.subject, r.object, key=r.relation,
                   relation=r.relation, margin=r.rule_margin)
    return g
