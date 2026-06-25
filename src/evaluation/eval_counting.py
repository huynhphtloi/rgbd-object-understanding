"""
Counting evaluation + RGB-only vs depth-filtered ablation.

Counting metrics: MAE, RMSE, exact-count accuracy, over-count rate,
under-count rate. Compares the RGB-only count (segmentation + NMS) against the
depth-validity-filtered count.

Input CSV must have columns: image, gt_count, rgb_only_count, depth_filtered_count
(produced by running the pipeline over a labeled split; see make_counting_csv).

Usage
-----
    python3 -m src.evaluation.eval_counting --pred outputs/metrics/counting.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def counting_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - gt
    return {
        "n": int(len(gt)),
        "MAE": float(np.mean(np.abs(err))) if len(gt) else 0.0,
        "RMSE": float(np.sqrt(np.mean(err ** 2))) if len(gt) else 0.0,
        "exact_accuracy": float(np.mean(err == 0)) if len(gt) else 0.0,
        "over_count_rate": float(np.mean(err > 0)) if len(gt) else 0.0,
        "under_count_rate": float(np.mean(err < 0)) if len(gt) else 0.0,
    }


def make_counting_csv(root_split_json, cfg_path, out_csv, bg_labels=(0, 1)):
    """Run the pipeline over a labeled split to produce the counting CSV.

    GT count = number of object instances in the OCID label image.
    """
    import yaml
    from ..data.ocid_common import find_frames, load_label, object_instance_ids
    from ..depth.depth_utils import CameraIntrinsics, to_metres, load_depth
    from ..data.ocid_common import DEFAULT_INTRINSICS
    from ..pipeline import Pipeline
    import cv2

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    with open(root_split_json) as fh:
        split_data = json.load(fh)

    frames = {fr.key: fr for fr in find_frames(split_data["root"])}
    pipe = Pipeline(cfg)
    intr = CameraIntrinsics.from_dict(DEFAULT_INTRINSICS)
    scale = cfg["depth"].get("depth_scale", 0.001)

    rows = []
    for key in split_data["frames"].get("test", []):
        fr = frames.get(key)
        if fr is None:
            continue
        rgb = cv2.imread(fr.rgb, cv2.IMREAD_COLOR)
        depth_m = to_metres(load_depth(fr.depth), scale,
                            cfg["depth"].get("min_depth_m", 0.0),
                            cfg["depth"].get("max_depth_m", np.inf))
        gt_count = len(object_instance_ids(load_label(fr.label), bg_labels))
        res = pipe.run(rgb, depth_m, intr, image_name=key)
        rows.append({
            "image": key, "gt_count": gt_count,
            "rgb_only_count": res.counts["rgb_only_count"],
            "depth_filtered_count": res.counts["depth_filtered_count"],
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Counting evaluation/ablation")
    ap.add_argument("--pred", required=True, help="CSV with gt/rgb/depth counts")
    ap.add_argument("--out", default="outputs/metrics/counting_ablation.json")
    args = ap.parse_args()

    df = pd.read_csv(args.pred)
    report = {
        "rgb_only": counting_metrics(df["gt_count"], df["rgb_only_count"]),
        "depth_filtered": counting_metrics(df["gt_count"], df["depth_filtered_count"]),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print("\nInterpretation: if depth filtering does not improve MAE/accuracy, "
          "report it as a null result -- RGB NMS already removed most duplicates "
          "and depth filtering mainly drops depth-invalid masks.")


if __name__ == "__main__":
    main()
