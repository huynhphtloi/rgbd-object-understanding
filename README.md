# RGB-D Object Understanding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Class-agnostic instance segmentation, depth-based measurement, and spatial-relation
reasoning for cluttered tabletop / household scenes.

Given an RGB image, an aligned depth map, and camera intrinsics, the pipeline produces:

1. Class-agnostic object instance masks (single foreground class: `object`)
2. An object count
3. Approximate **visible** 3D dimensions per object (from masked depth)
4. Pairwise spatial relations (`in_front_of`, `occludes`, `overlaps`, `likely_on_top_of`, ...)
5. A visual demo and a JSON scene graph

> **Scope.** The system measures the **visible extent** of each object. Under
> partial occlusion the depth sensor only sees visible surfaces, so dimensions are
> *approximate metric estimates*, not guaranteed full-object measurements.

## Pipeline

```
RGB + aligned depth + intrinsics
  → preprocessing (depth scale, invalid masking, optional plane removal)
  → class-agnostic YOLO-seg masks
  → counting (NMS + depth-validity filtering)
  → depth-based measurement (mask → point cloud → oriented bounding box)
  → rule-based spatial-relation reasoning (masks + depth ordering)
  → visual masks + dimension table + relation graph + JSON
```

## Quick start

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Run the full pipeline on a SYNTHETIC RGB-D scene (no dataset needed).
#    Objects have known dimensions, so you can sanity-check measurement error
#    before involving any real data.
python3 -m src.run_pipeline --synthetic --out outputs/demo_results

# 3. Launch the interactive demo
python3 -m src.app.gradio_app
```

Sample outputs from the synthetic run live in [outputs/demo_results/](outputs/demo_results/).

## Working with OCID (real data)

```bash
python3 -m src.data.download_ocid --out data/raw/ocid          # download + extract
python3 -m src.data.inspect_ocid  --root data/raw/ocid         # check depth unit + intrinsics
python3 -m src.data.split_by_scene --root data/raw/ocid        # scene-level split (no leakage)
python3 -m src.data.convert_ocid_to_yolo --root data/raw/ocid --out data/processed
python3 -m src.data.convert_ocid_to_coco --root data/raw/ocid --out data/processed

# Train class-agnostic YOLO-seg
yolo segment train model=yolov8n-seg.pt data=config/dataset.yaml epochs=100 imgsz=640 batch=8
```

## Repository layout

```
config/             YAML configs (dataset, training, demo, eval)
src/data/           OCID download / inspect / convert
src/segmentation/   YOLO-seg train + inference
src/depth/          back-projection, point clouds, plane removal, dimension estimation
src/reasoning/      object instances, relation rules, relation graph
src/visualization/  overlays for masks / depth / OBB / graph
src/evaluation/     segmentation, counting, measurement, relation metrics
src/app/            Gradio demo
notebooks/          end-to-end Colab walkthrough
data/               raw + processed datasets (gitignored)
outputs/            models, predictions, metrics (gitignored except demo_results/)
```

## Claims and limitations

**What it does:** class-agnostic instance segmentation; approximate metric
dimensions when depth coverage is sufficient; visible 3D extent under partial
occlusion; basic pairwise spatial relations.

**What it does not do:** it cannot recover hidden/occluded geometry, does not
guarantee full-object dimensions under heavy occlusion, and struggles with
transparent, reflective, dark, or deformable objects (depth sensors return
invalid values on those surfaces). These are inherent limitations, reported as
such rather than hidden.


## License

Released under the [MIT License](LICENSE).
