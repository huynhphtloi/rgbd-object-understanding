"""
Train a class-agnostic YOLO-seg model (single class: object) on the converted
OCID dataset.

Usage
-----
    python3 -m src.segmentation.train_yolo --config config/train_yolo.yaml

All hyperparameters come from the YAML; CLI flags override individual fields.
Equivalent Ultralytics CLI:
    yolo segment train model=yolov8n-seg.pt data=config/dataset.yaml \
        epochs=100 imgsz=640 batch=8
"""
from __future__ import annotations

import argparse

import yaml


def main():
    ap = argparse.ArgumentParser(description="Train YOLO-seg (class-agnostic)")
    ap.add_argument("--config", default="config/train_yolo.yaml")
    ap.add_argument("--model", help="override base weights, e.g. yolov8s-seg.pt")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--device", help="cuda device(s) or 'cpu' or 'mps'")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    for k in ("model", "epochs", "batch", "imgsz"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    from ultralytics import YOLO

    model = YOLO(cfg.pop("model"))
    train_kwargs = dict(cfg)
    if args.device:
        train_kwargs["device"] = args.device

    print("Training with:", train_kwargs)
    results = model.train(**train_kwargs)
    print("Done. Best weights:", getattr(results, "save_dir", "see outputs/models"))


if __name__ == "__main__":
    main()
