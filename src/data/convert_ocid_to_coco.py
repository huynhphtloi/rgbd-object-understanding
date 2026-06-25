"""
Convert OCID instance labels to COCO instance-segmentation JSON
(single category: object). Used for Mask R-CNN training and COCO-style mask AP
evaluation.

Writes data/processed/coco_annotations/<split>.json.

Usage
-----
    python3 -m src.data.convert_ocid_to_coco \
        --splits data/processed/splits.json --out data/processed \
        [--bg-labels 0 1] [--min-area 200]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .ocid_common import find_frames, load_label, instance_masks
from .convert_ocid_to_yolo import mask_to_polygons, _flat_name


def build_coco(frames_for_split, frames_index, bg_labels, min_area):
    images, annotations = [], []
    ann_id = 1
    for img_id, key in enumerate(frames_for_split, start=1):
        fr = frames_index.get(key)
        if fr is None:
            continue
        label = load_label(fr.label)
        h, w = label.shape[:2]
        images.append({
            "id": img_id,
            "file_name": _flat_name(key) + os.path.splitext(fr.rgb)[1],
            "width": int(w), "height": int(h),
            "ocid_key": key,
        })
        for _, mask in instance_masks(label, bg_labels):
            polys = mask_to_polygons(mask, min_area)
            if not polys:
                continue
            seg = [p.reshape(-1).astype(float).tolist() for p in polys]
            ys, xs = np.nonzero(mask)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": seg,
                "area": float(int(mask.sum())),
                "bbox": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                "iscrowd": 0,
            })
            ann_id += 1
    return {
        "info": {"description": "OCID class-agnostic (single class: object)"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
    }


def main():
    ap = argparse.ArgumentParser(description="OCID -> COCO")
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--bg-labels", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--min-area", type=int, default=200)
    args = ap.parse_args()

    with open(args.splits) as fh:
        split_data = json.load(fh)
    frames_index = {fr.key: fr for fr in find_frames(split_data["root"])}

    out_dir = os.path.join(args.out, "coco_annotations")
    os.makedirs(out_dir, exist_ok=True)

    for split, keys in split_data["frames"].items():
        coco = build_coco(keys, frames_index, args.bg_labels, args.min_area)
        path = os.path.join(out_dir, f"{split}.json")
        with open(path, "w") as fh:
            json.dump(coco, fh)
        print(f"[{split}] {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations -> {path}")


if __name__ == "__main__":
    main()
