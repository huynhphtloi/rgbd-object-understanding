"""
Create a SCENE-LEVEL train/val/test split for OCID.

Splitting by scene (sequence) rather than by frame is essential: OCID sequences
are built incrementally (objects added one by one), so random frame splits leak
near-identical frames across train/test and inflate scores.

Writes data/processed/splits.json mapping each split to a list of frame keys.

Usage
-----
    python3 -m src.data.split_by_scene --root data/raw/ocid \
        --out data/processed/splits.json --ratios 0.7 0.15 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from .ocid_common import find_frames


def main():
    ap = argparse.ArgumentParser(description="Scene-level OCID split")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/processed/splits.json")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames = find_frames(args.root)
    if not frames:
        print(f"No frames found under {args.root}")
        return

    by_scene = defaultdict(list)
    for fr in frames:
        by_scene[fr.scene].append(fr.key)

    scenes = sorted(by_scene)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(scenes)

    n = len(scenes)
    n_train = int(round(args.ratios[0] * n))
    n_val = int(round(args.ratios[1] * n))
    split_scenes = {
        "train": scenes[:n_train],
        "val": scenes[n_train:n_train + n_val],
        "test": scenes[n_train + n_val:],
    }

    splits = {s: [] for s in split_scenes}
    for split, scene_list in split_scenes.items():
        for sc in scene_list:
            splits[split].extend(by_scene[sc])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "root": os.path.abspath(args.root),
            "n_scenes": n,
            "scenes": split_scenes,
            "frames": splits,
        }, fh, indent=2)

    print(f"Scenes  -> train {len(split_scenes['train'])}, "
          f"val {len(split_scenes['val'])}, test {len(split_scenes['test'])}")
    print(f"Frames  -> train {len(splits['train'])}, "
          f"val {len(splits['val'])}, test {len(splits['test'])}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
