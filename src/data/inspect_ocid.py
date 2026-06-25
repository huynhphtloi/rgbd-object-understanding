"""
Inspect an extracted OCID tree and report the facts the pipeline depends on:
frame counts, image sizes, depth dtype + value range (to infer the unit),
label-id distribution (to identify the support plane) and object-count stats.

Usage
-----
    python3 -m src.data.inspect_ocid --root data/raw/ocid [--samples 50]
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import cv2
import numpy as np

from .ocid_common import find_frames, load_label, DEFAULT_BG_LABELS


def main():
    ap = argparse.ArgumentParser(description="Inspect OCID")
    ap.add_argument("--root", required=True)
    ap.add_argument("--samples", type=int, default=50,
                    help="how many frames to sample for depth/label stats")
    ap.add_argument("--out", default="outputs/metrics/ocid_inspection.json")
    args = ap.parse_args()

    frames = find_frames(args.root)
    if not frames:
        print(f"No RGB-D-label frames found under {args.root}. "
              f"Expected sibling rgb/ depth/ label/ folders.")
        return

    scenes = sorted({f.scene for f in frames})
    print(f"Frames: {len(frames)}   Scenes (sequences): {len(scenes)}")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(frames), size=min(args.samples, len(frames)), replace=False)

    depth_min, depth_max, depth_dtypes = np.inf, 0.0, set()
    sizes = collections.Counter()
    label_hist = collections.Counter()
    obj_counts = []

    for i in idx:
        fr = frames[int(i)]
        depth = cv2.imread(fr.depth, cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(fr.rgb, cv2.IMREAD_UNCHANGED)
        label = load_label(fr.label)
        if depth is None or rgb is None:
            continue
        depth_dtypes.add(str(depth.dtype))
        nz = depth[depth > 0]
        if nz.size:
            depth_min = min(depth_min, float(nz.min()))
            depth_max = max(depth_max, float(nz.max()))
        sizes[f"{rgb.shape[1]}x{rgb.shape[0]}"] += 1
        ids, counts = np.unique(label, return_counts=True)
        for v, c in zip(ids, counts):
            label_hist[int(v)] += int(c)
        n_obj = len([v for v in ids if int(v) not in set(DEFAULT_BG_LABELS)])
        obj_counts.append(n_obj)

    # depth-unit heuristic: max value in thousands -> millimetres
    likely_unit = "millimetres" if depth_max > 100 else "metres(?)"
    likely_scale = 0.001 if depth_max > 100 else 1.0

    print("\n--- Depth ---")
    print(f"  dtype(s): {sorted(depth_dtypes)}")
    print(f"  nonzero range: [{depth_min:.1f}, {depth_max:.1f}]")
    print(f"  likely unit: {likely_unit}  ->  recommended depth_scale = {likely_scale}")

    print("\n--- Image sizes ---")
    for s, c in sizes.most_common():
        print(f"  {s}: {c}")

    print("\n--- Top label ids (by pixel count) ---")
    for v, c in label_hist.most_common(12):
        tag = " (bg/plane?)" if v in DEFAULT_BG_LABELS else ""
        print(f"  id {v}: {c} px{tag}")
    print("  NOTE: confirm which low ids are background/support-plane vs objects,")
    print("        then pass --bg-labels to the conversion scripts accordingly.")

    if obj_counts:
        oc = np.array(obj_counts)
        print("\n--- Objects per frame (sampled) ---")
        print(f"  min {oc.min()}  mean {oc.mean():.1f}  max {oc.max()}")

    report = {
        "root": args.root,
        "n_frames": len(frames),
        "n_scenes": len(scenes),
        "depth_dtypes": sorted(depth_dtypes),
        "depth_nonzero_range": [depth_min, depth_max],
        "recommended_depth_scale": likely_scale,
        "image_sizes": dict(sizes),
        "label_pixel_hist_top": dict(label_hist.most_common(20)),
        "objects_per_frame": {
            "min": int(min(obj_counts)) if obj_counts else 0,
            "mean": float(np.mean(obj_counts)) if obj_counts else 0.0,
            "max": int(max(obj_counts)) if obj_counts else 0,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
