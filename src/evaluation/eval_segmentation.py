"""
Segmentation evaluation.

Primary path uses Ultralytics' built-in validator on the converted dataset
(mask mAP50, mAP50-95, precision, recall). A COCO-based path is also provided
for comparing against Mask R-CNN style predictions.

Usage
-----
    python3 -m src.evaluation.eval_segmentation \
        --weights outputs/models/rgb_yolo_seg_baseline/weights/best.pt \
        --data config/dataset.yaml --split test
"""
from __future__ import annotations

import argparse
import json
import os


def eval_yolo(weights: str, data: str, split: str, imgsz: int = 640):
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data, split=split, imgsz=imgsz, verbose=True)
    seg = metrics.seg
    out = {
        "mask_mAP50": float(seg.map50),
        "mask_mAP75": float(seg.map75),
        "mask_mAP50_95": float(seg.map),
        "mask_precision": float(seg.mp),
        "mask_recall": float(seg.mr),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate YOLO-seg")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="config/dataset.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="outputs/metrics/segmentation_metrics.json")
    args = ap.parse_args()

    out = eval_yolo(args.weights, args.data, args.split, args.imgsz)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
