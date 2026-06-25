"""
Run the full RGB-D pipeline on a synthetic scene or a real RGB-D sample.

Examples
--------
# Synthetic scene (no dataset needed) -> verifies measurement + relations
python3 -m src.run_pipeline --synthetic --out outputs/demo_results

# Real RGB-D sample
python3 -m src.run_pipeline \
    --rgb path/rgb.png --depth path/depth.png \
    --intrinsics path/intr.json --out outputs/demo_results
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import yaml

from .depth.depth_utils import CameraIntrinsics, to_metres, load_depth
from .pipeline import Pipeline
from .segmentation.infer_yolo import detections_from_masks
from .visualization.draw_masks import overlay_instances
from .visualization.draw_depth import colorize_depth
from .visualization.draw_obb import draw_dimension_boxes
from .visualization.draw_graph import render_relation_graph
from .reasoning.graph_builder import save_scene_json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _save(path, img):
    cv2.imwrite(path, img)
    return path


def evaluate_measurement(instances, gt, iou_match=0.3):
    """Match predicted instances to GT masks by IoU and tabulate dimension error."""
    from .reasoning.rules import mask_iou

    rows = []
    for g in gt:
        best, best_iou = None, 0.0
        for inst in instances:
            i = mask_iou(inst.mask, g["mask"])
            if i > best_iou:
                best, best_iou = inst, i
        if best is None or best_iou < iou_match or best.measurement is None:
            rows.append({"gt_id": g["id"], "matched": False})
            continue
        m = best.measurement
        # OBB axes are unordered; compare sorted GT dims vs sorted predicted dims.
        gt_dims = sorted([g["width_cm"], g["height_cm"], g["depth_cm"]], reverse=True)
        pr_dims = sorted([m.width_cm, m.height_cm, m.depth_cm], reverse=True)
        abs_err = [abs(a - b) for a, b in zip(gt_dims, pr_dims)]
        rel_err = [e / d * 100 for e, d in zip(abs_err, gt_dims)]
        rows.append({
            "gt_id": g["id"], "matched": True, "iou": round(best_iou, 3),
            "gt_dims_cm": [round(x, 1) for x in gt_dims],
            "pred_dims_cm": [round(x, 1) for x in pr_dims],
            "abs_err_cm": [round(x, 2) for x in abs_err],
            "rel_err_pct": [round(x, 1) for x in rel_err],
            "mean_abs_err_cm": round(float(np.mean(abs_err)), 2),
            "valid_depth_ratio": round(best.valid_depth_ratio, 2),
            "visibility": m.visibility_group,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="RGB-D scene understanding pipeline")
    ap.add_argument("--synthetic", action="store_true", help="use a synthetic scene")
    ap.add_argument("--rgb", help="RGB image path")
    ap.add_argument("--depth", help="depth image path (raw)")
    ap.add_argument("--intrinsics", help="JSON with fx,fy,cx,cy[,width,height]")
    ap.add_argument("--config", default=os.path.join(REPO, "config", "demo.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "demo_results"))
    ap.add_argument("--use-seg", action="store_true",
                    help="run YOLO-seg even on synthetic (default: use GT masks)")
    ap.add_argument("--depth-noise", type=float, default=0.0,
                    help="synthetic depth noise std in metres")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    gt = None
    detections = None

    if args.synthetic:
        from .data.synthetic_scene import render_scene

        rgb, depth_raw, intr_dict, gt = render_scene(depth_noise_std_m=args.depth_noise)
        depth_scale = 0.001  # synthetic depth stored in mm
        image_name = "synthetic_scene"
        if not args.use_seg:
            detections = detections_from_masks([g["mask"] for g in gt])
    else:
        if not (args.rgb and args.depth and args.intrinsics):
            ap.error("provide --rgb, --depth and --intrinsics (or use --synthetic)")
        rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
        depth_raw = load_depth(args.depth)
        with open(args.intrinsics) as fh:
            intr_dict = json.load(fh)
        intr_dict.setdefault("width", rgb.shape[1])
        intr_dict.setdefault("height", rgb.shape[0])
        depth_scale = cfg["depth"].get("depth_scale", 0.001)
        image_name = os.path.basename(args.rgb)

    intr = CameraIntrinsics.from_dict(intr_dict)
    depth_m = to_metres(
        depth_raw, depth_scale,
        min_depth_m=cfg["depth"].get("min_depth_m", 0.0),
        max_depth_m=cfg["depth"].get("max_depth_m", np.inf),
    )

    pipe = Pipeline(cfg)
    result = pipe.run(rgb, depth_m, intr, detections=detections, image_name=image_name)

    # ---- save visual + JSON outputs ----
    _save(os.path.join(args.out, "01_rgb.png"), rgb)
    _save(os.path.join(args.out, "02_depth.png"),
          colorize_depth(depth_m, cfg["depth"].get("max_depth_m")))
    _save(os.path.join(args.out, "03_masks.png"),
          overlay_instances(rgb, result.instances))
    _save(os.path.join(args.out, "04_dimensions.png"),
          draw_dimension_boxes(rgb, result.instances))
    render_relation_graph(result.instances, result.relations,
                          os.path.join(args.out, "05_relation_graph.png"))
    save_scene_json(result.scene, os.path.join(args.out, "scene.json"))

    # ---- console summary ----
    print("\n=== COUNTS ===")
    print(json.dumps(result.counts, indent=2))
    print("\n=== OBJECTS (visible dimensions) ===")
    for obj in result.scene["objects"]:
        dims = obj.get("dimensions_cm", {})
        print(f"  {obj['id']}: W{dims.get('width')} H{dims.get('height')} "
              f"D{dims.get('depth')} cm  depth={obj['median_depth_m']}m  "
              f"vdr={obj['valid_depth_ratio']}  [{obj.get('visibility_group')}]")
    print("\n=== RELATIONS ===")
    for r in result.scene["relations"]:
        print(f"  {r['subject']} --{r['relation']}--> {r['object']} "
              f"(margin {r['rule_margin']})")

    if gt is not None:
        rows = evaluate_measurement(result.instances, gt)
        with open(os.path.join(args.out, "measurement_eval.json"), "w") as fh:
            json.dump(rows, fh, indent=2)
        print("\n=== MEASUREMENT vs GROUND TRUTH (synthetic) ===")
        matched = [r for r in rows if r.get("matched")]
        for r in rows:
            if not r.get("matched"):
                print(f"  gt{r['gt_id']}: NOT MATCHED")
                continue
            print(f"  gt{r['gt_id']}: GT {r['gt_dims_cm']}  PRED {r['pred_dims_cm']}  "
                  f"absErr {r['abs_err_cm']}cm  meanAbs {r['mean_abs_err_cm']}cm  "
                  f"[{r['visibility']}]")
        if matched:
            mae = float(np.mean([r["mean_abs_err_cm"] for r in matched]))
            print(f"\n  Overall mean absolute dimension error: {mae:.2f} cm "
                  f"over {len(matched)}/{len(rows)} matched objects")

    print(f"\nOutputs written to: {args.out}")


if __name__ == "__main__":
    main()
