"""
Convert OCID instance labels to YOLO-seg format (single class: object).

For every frame in each split we copy the RGB image to
    data/processed/images/<split>/<flatname>.png
and write polygon labels to
    data/processed/labels/<split>/<flatname>.txt
(Ultralytics derives the label path by swapping `images` -> `labels`.)

Each label line:  0 x1 y1 x2 y2 ... xn yn   (class 0, normalised polygon).

Usage
-----
    python3 -m src.data.convert_ocid_to_yolo \
        --splits data/processed/splits.json --out data/processed \
        [--bg-labels 0 1] [--min-area 200] [--link]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import cv2
import numpy as np

from .ocid_common import find_frames, load_label, instance_masks


def _flat_name(key: str) -> str:
    stem = os.path.splitext(key)[0]
    return stem.replace(os.sep, "__").replace("/", "__")


def mask_to_polygons(mask: np.ndarray, min_area: int, eps_frac: float = 0.004):
    """Return list of polygons (each a flat [x1,y1,...] pixel list) for a mask."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        eps = eps_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        polys.append(approx.reshape(-1, 2))
    return polys


def main():
    ap = argparse.ArgumentParser(description="OCID -> YOLO-seg")
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--bg-labels", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--link", action="store_true",
                    help="symlink images instead of copying (saves disk)")
    args = ap.parse_args()

    with open(args.splits) as fh:
        split_data = json.load(fh)
    root = split_data["root"]

    # index frames by key for quick lookup
    frames = {fr.key: fr for fr in find_frames(root)}

    for split, keys in split_data["frames"].items():
        img_dir = os.path.join(args.out, "images", split)
        lbl_dir = os.path.join(args.out, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        n_img, n_inst = 0, 0
        for key in keys:
            fr = frames.get(key)
            if fr is None:
                continue
            label = load_label(fr.label)
            h, w = label.shape[:2]

            lines = []
            for _, mask in instance_masks(label, args.bg_labels):
                for poly in mask_to_polygons(mask, args.min_area):
                    norm = poly.astype(np.float64).copy()
                    norm[:, 0] /= w
                    norm[:, 1] /= h
                    norm = np.clip(norm, 0.0, 1.0)
                    coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
                    lines.append(f"0 {coords}")

            flat = _flat_name(key)
            dst_img = os.path.join(img_dir, flat + os.path.splitext(fr.rgb)[1])
            if args.link:
                if not os.path.exists(dst_img):
                    os.symlink(os.path.abspath(fr.rgb), dst_img)
            else:
                shutil.copy(fr.rgb, dst_img)

            with open(os.path.join(lbl_dir, flat + ".txt"), "w") as fh:
                fh.write("\n".join(lines))
            n_img += 1
            n_inst += len(lines)

        print(f"[{split}] {n_img} images, {n_inst} instance polygons")

    print(f"\nDone. Point config/dataset.yaml `path` at {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
