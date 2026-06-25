"""
Measurement evaluation against physically measured ground truth.

GT CSV columns (self-captured set):
    object_id, real_width_cm, real_height_cm, real_depth_cm, visibility
Pred CSV columns:
    object_id, width_cm, height_cm, depth_cm [, visibility_group]

Reports absolute + relative dimension error, overall and grouped by visibility.
Because the OBB axes are unordered, GT and predicted dimensions are compared
after sorting each triple descending.

Usage
-----
    python3 -m src.evaluation.eval_measurement \
        --gt data/samples/self_captured_measurement_gt/measurement_gt.csv \
        --pred outputs/metrics/measurement_results.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def _sorted_dims(row, cols):
    return np.sort(np.array([row[c] for c in cols], dtype=float))[::-1]


def measurement_errors(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    gt_cols = ["real_width_cm", "real_height_cm", "real_depth_cm"]
    pr_cols = ["width_cm", "height_cm", "depth_cm"]
    merged = gt.merge(pred, on="object_id", suffixes=("_gt", "_pred"))
    rows = []
    for _, r in merged.iterrows():
        g = _sorted_dims(r, gt_cols)
        p = _sorted_dims(r, pr_cols)
        abs_err = np.abs(g - p)
        rel_err = abs_err / np.clip(g, 1e-6, None) * 100
        rows.append({
            "object_id": r["object_id"],
            "visibility": r.get("visibility", r.get("visibility_group", "unknown")),
            "mean_abs_err_cm": float(abs_err.mean()),
            "max_abs_err_cm": float(abs_err.max()),
            "mean_rel_err_pct": float(rel_err.mean()),
        })
    return pd.DataFrame(rows)


def summarize(err_df: pd.DataFrame) -> dict:
    def agg(d):
        return {
            "n": int(len(d)),
            "mean_abs_err_cm": float(d["mean_abs_err_cm"].mean()),
            "mean_rel_err_pct": float(d["mean_rel_err_pct"].mean()),
        }

    report = {"overall": agg(err_df)}
    if "visibility" in err_df.columns:
        report["by_visibility"] = {
            str(v): agg(g) for v, g in err_df.groupby("visibility")
        }
    return report


def main():
    ap = argparse.ArgumentParser(description="Measurement evaluation")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="outputs/metrics/measurement_metrics.json")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)
    pred = pd.read_csv(args.pred)
    err = measurement_errors(gt, pred)
    report = summarize(err)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    err.to_csv(args.out.replace(".json", "_per_object.csv"), index=False)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
