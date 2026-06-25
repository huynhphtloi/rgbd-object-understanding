"""
Gradio demo for the RGB-D scene-understanding pipeline.

Run:
    python3 -m src.app.gradio_app

Two modes:
  * "Synthetic sample"  - renders the built-in scene (no upload needed).
  * "Upload RGB-D"      - upload an RGB image + a 16-bit depth PNG + intrinsics.

Outputs: segmented masks, colorized depth, object count, a dimension table,
the spatial-relation graph and the raw JSON scene description.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
import yaml

from ..depth.depth_utils import CameraIntrinsics, to_metres, load_depth
from ..pipeline import Pipeline
from ..segmentation.infer_yolo import detections_from_masks
from ..visualization.draw_masks import overlay_instances
from ..visualization.draw_depth import colorize_depth
from ..visualization.draw_obb import draw_dimension_boxes
from ..visualization.draw_graph import render_relation_graph

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_CFG = os.path.join(REPO, "config", "demo.yaml")

_PIPELINE: Pipeline | None = None
_CFG: dict | None = None


def _load_cfg() -> dict:
    global _CFG
    if _CFG is None:
        with open(DEFAULT_CFG) as fh:
            _CFG = yaml.safe_load(fh)
    return _CFG


def _get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline(_load_cfg())
    return _PIPELINE


def _bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _dimension_table(scene: dict) -> pd.DataFrame:
    rows = []
    for o in scene["objects"]:
        d = o.get("dimensions_cm", {})
        rows.append({
            "object": o["id"],
            "W (cm)": d.get("width"),
            "H (cm)": d.get("height"),
            "D (cm)": d.get("depth"),
            "depth (m)": o["median_depth_m"],
            "valid_depth_ratio": o["valid_depth_ratio"],
            "visibility": o.get("visibility_group", "-"),
            "reliable": o.get("obb_extent_well_defined", False),
        })
    return pd.DataFrame(rows)


def _relation_table(scene: dict) -> pd.DataFrame:
    return pd.DataFrame(scene["relations"]) if scene["relations"] else pd.DataFrame(
        columns=["subject", "relation", "object", "rule_margin"]
    )


def _run(mode, rgb_in, depth_file, fx, fy, cx, cy, depth_scale, conf):
    cfg = _load_cfg()
    detections = None

    if mode == "Synthetic sample":
        from ..data.synthetic_scene import render_scene

        rgb, depth_raw, intr_dict, gt = render_scene()
        depth_scale = 0.001
        detections = detections_from_masks([g["mask"] for g in gt])
        image_name = "synthetic_scene"
    else:
        if rgb_in is None or depth_file is None:
            raise ValueError("Please provide both an RGB image and a depth file.")
        # gr.Image gives RGB uint8; pipeline/viz use BGR
        rgb = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2BGR)
        depth_raw = load_depth(depth_file)
        if depth_raw.shape[:2] != rgb.shape[:2]:
            depth_raw = cv2.resize(depth_raw, (rgb.shape[1], rgb.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        intr_dict = {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                     "width": rgb.shape[1], "height": rgb.shape[0]}
        image_name = "uploaded"

    cfg["segmentation"]["conf"] = float(conf)
    cfg["counting"]["conf_thresh"] = float(conf)

    intr = CameraIntrinsics.from_dict(intr_dict)
    depth_m = to_metres(
        depth_raw, float(depth_scale),
        min_depth_m=cfg["depth"].get("min_depth_m", 0.0),
        max_depth_m=cfg["depth"].get("max_depth_m", np.inf),
    )

    pipe = _get_pipeline()
    pipe.cfg = cfg
    result = pipe.run(rgb, depth_m, intr, detections=detections, image_name=image_name)

    masks_img = _bgr2rgb(overlay_instances(rgb, result.instances))
    depth_img = _bgr2rgb(colorize_depth(depth_m, cfg["depth"].get("max_depth_m")))
    dims_img = _bgr2rgb(draw_dimension_boxes(rgb, result.instances))
    graph_fig = render_relation_graph(result.instances, result.relations)

    count_md = (
        f"### Object count: **{result.counts['depth_filtered_count']}**\n"
        f"- raw detections: {result.counts['raw_detections']}\n"
        f"- after RGB filtering (NMS): {result.counts['rgb_only_count']}\n"
        f"- after depth-validity filtering: {result.counts['depth_filtered_count']}"
    )
    return (masks_img, depth_img, dims_img, graph_fig,
            count_md, _dimension_table(result.scene),
            _relation_table(result.scene), result.scene)


def build_app():
    import gradio as gr

    with gr.Blocks(title="RGB-D Scene Understanding") as demo:
        gr.Markdown(
            "# RGB-D Class-Agnostic Scene Understanding\n"
            "Segmentation + depth-based measurement + spatial-relation reasoning. "
            "Dimensions are **visible-extent estimates** (approximate metric)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["Synthetic sample", "Upload RGB-D"],
                                value="Synthetic sample", label="Input mode")
                rgb_in = gr.Image(label="RGB image", type="numpy")
                depth_file = gr.File(label="Depth (16-bit PNG, raw)", type="filepath")
                with gr.Row():
                    fx = gr.Number(value=525.0, label="fx")
                    fy = gr.Number(value=525.0, label="fy")
                with gr.Row():
                    cx = gr.Number(value=320.0, label="cx")
                    cy = gr.Number(value=240.0, label="cy")
                depth_scale = gr.Number(value=0.001, label="depth scale (raw->m)")
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05,
                                 label="segmentation confidence")
                run_btn = gr.Button("Run pipeline", variant="primary")
            with gr.Column(scale=2):
                count_md = gr.Markdown()
                with gr.Tab("Masks"):
                    masks_img = gr.Image(label="Instance masks")
                with gr.Tab("Depth"):
                    depth_img = gr.Image(label="Colorized depth")
                with gr.Tab("Dimensions"):
                    dims_img = gr.Image(label="Estimated dimensions")
                    dims_tbl = gr.Dataframe(label="Dimension table")
                with gr.Tab("Relations"):
                    graph_img = gr.Plot(label="Relation graph")
                    rel_tbl = gr.Dataframe(label="Relations")
                with gr.Tab("JSON"):
                    json_out = gr.JSON(label="Scene description")

        run_btn.click(
            _run,
            inputs=[mode, rgb_in, depth_file, fx, fy, cx, cy, depth_scale, conf],
            outputs=[masks_img, depth_img, dims_img, graph_img,
                     count_md, dims_tbl, rel_tbl, json_out],
        )
    return demo


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
