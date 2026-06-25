"""
Shared OCID helpers: dataset discovery, frame enumeration, label parsing and
default intrinsics.

OCID (Object Cluttered Indoor Dataset, TU Wien) is organized roughly as:

    OCID-dataset/<subset>/<floor|table>/<bottom|top>/<seqNN>/
        rgb/   *.png      (640x480 color)
        depth/ *.png      (16-bit, millimetres)
        label/ *.png      (per-pixel instance ids; 0 = background,
                           the support plane is usually a low id too)
        pcd/   *.pcd      (organized point cloud, one point per pixel)

We DON'T hard-code the directory tree: we walk for any folder that has sibling
`rgb/`, `depth/` and `label/` directories. Use inspect_ocid.py to verify depth
unit, intrinsics and which label ids correspond to the support plane.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np

# Default intrinsics for the OCID ASUS Xtion stream (640x480).
# VERIFY with inspect_ocid.py / the dataset's pcd files before trusting metric output.
DEFAULT_INTRINSICS = {
    "fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5,
    "width": 640, "height": 480,
}

# Label ids commonly NOT objects (background + support plane). Override per dataset.
DEFAULT_BG_LABELS = (0, 1)

IMG_EXTS = (".png", ".jpg", ".jpeg")


@dataclass
class Frame:
    rgb: str
    depth: str
    label: str
    scene: str       # sequence directory identifying the scene (for splitting)
    key: str         # unique id: <scene-rel>/<filename>


def find_frames(root: str) -> list[Frame]:
    """Walk `root` and return every RGB-D-label frame triple."""
    frames: list[Frame] = []
    for cur, dirs, _ in os.walk(root):
        rgb_dir = os.path.join(cur, "rgb")
        depth_dir = os.path.join(cur, "depth")
        label_dir = os.path.join(cur, "label")
        if not (os.path.isdir(rgb_dir) and os.path.isdir(depth_dir)
                and os.path.isdir(label_dir)):
            continue
        scene_rel = os.path.relpath(cur, root)
        for rgb_path in sorted(_list_images(rgb_dir)):
            name = os.path.basename(rgb_path)
            depth_path = _match(depth_dir, name)
            label_path = _match(label_dir, name)
            if depth_path is None or label_path is None:
                continue
            frames.append(Frame(
                rgb=rgb_path, depth=depth_path, label=label_path,
                scene=cur, key=os.path.join(scene_rel, name),
            ))
    return frames


def _list_images(d: str) -> list[str]:
    out = []
    for ext in IMG_EXTS:
        out.extend(glob.glob(os.path.join(d, f"*{ext}")))
        out.extend(glob.glob(os.path.join(d, f"*{ext.upper()}")))
    return sorted(set(out))


def _match(directory: str, name: str) -> str | None:
    """Find the file in `directory` matching `name` (allowing ext differences)."""
    cand = os.path.join(directory, name)
    if os.path.isfile(cand):
        return cand
    stem = os.path.splitext(name)[0]
    for ext in IMG_EXTS:
        p = os.path.join(directory, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def load_label(path: str) -> np.ndarray:
    """Load an instance-label image as int32 (pixel value = instance id)."""
    lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(path)
    if lab.ndim == 3:
        lab = lab[..., 0]
    return lab.astype(np.int32)


def object_instance_ids(label: np.ndarray, bg_labels=DEFAULT_BG_LABELS) -> list[int]:
    """Return sorted object instance ids present in a label image."""
    ids = np.unique(label)
    return [int(i) for i in ids if int(i) not in set(bg_labels)]


def instance_masks(label: np.ndarray, bg_labels=DEFAULT_BG_LABELS):
    """Yield (instance_id, boolean_mask) for each object instance."""
    for i in object_instance_ids(label, bg_labels):
        yield i, (label == i)
