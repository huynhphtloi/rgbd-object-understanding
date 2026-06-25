"""
Spatial-relation evaluation.

Compares predicted pairwise relations against a held-out annotated set and
reports per-class precision / recall / F1, a confusion matrix, and (optionally)
inter-annotator Cohen's kappa.

CSV format (gt and pred): columns subject, object, relation
Pairs absent from a file are treated as relation "none".

Annotation protocol reminder: label FIRST, tune thresholds LATER, and keep the
evaluation set disjoint from any data used to tune the rules.

Usage
-----
    python3 -m src.evaluation.eval_relations \
        --gt data/samples/relation_labels.csv \
        --pred outputs/metrics/relation_predictions.csv \
        [--annotator2 data/samples/relation_labels_annotator2.csv]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CLASSES = [
    "in_front_of", "behind", "overlaps", "occludes",
    "partially_occluded_by", "likely_on_top_of", "none",
]


def _pair_map(df: pd.DataFrame) -> dict:
    return {(str(r["subject"]), str(r["object"])): str(r["relation"])
            for _, r in df.iterrows()}


def align(gt: pd.DataFrame, pred: pd.DataFrame):
    """Align gt/pred on the union of (subject, object) pairs -> (y_true, y_pred)."""
    g, p = _pair_map(gt), _pair_map(pred)
    pairs = sorted(set(g) | set(p))
    y_true = [g.get(k, "none") for k in pairs]
    y_pred = [p.get(k, "none") for k in pairs]
    return y_true, y_pred, pairs


def evaluate(gt: pd.DataFrame, pred: pd.DataFrame, classes=None) -> dict:
    from sklearn.metrics import (precision_recall_fscore_support,
                                 classification_report, confusion_matrix)

    classes = classes or DEFAULT_CLASSES
    y_true, y_pred, _ = align(gt, pred)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    per_class = {
        c: {"precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f1[i]), "support": int(sup[i])}
        for i, c in enumerate(classes)
    }
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0)
    return {
        "per_class": per_class,
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]),
                  "f1": float(macro[2])},
        "report_text": classification_report(
            y_true, y_pred, labels=classes, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classes": classes,
    }


def cohen_kappa(a1: pd.DataFrame, a2: pd.DataFrame) -> float:
    from sklearn.metrics import cohen_kappa_score

    m1, m2 = _pair_map(a1), _pair_map(a2)
    pairs = sorted(set(m1) | set(m2))
    y1 = [m1.get(k, "none") for k in pairs]
    y2 = [m2.get(k, "none") for k in pairs]
    return float(cohen_kappa_score(y1, y2))


def plot_confusion(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Relation confusion matrix")
    thr = np.max(cm) / 2 if np.max(cm) > 0 else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > thr else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Relation evaluation")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--annotator2", help="second annotator CSV for Cohen's kappa")
    ap.add_argument("--out", default="outputs/metrics/relation_metrics.json")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)
    pred = pd.read_csv(args.pred)
    report = evaluate(gt, pred)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_confusion(np.array(report["confusion_matrix"]), report["classes"],
                   args.out.replace(".json", "_confusion.png"))

    if args.annotator2:
        report["cohen_kappa"] = cohen_kappa(gt, pd.read_csv(args.annotator2))

    print(report["report_text"])
    print("Macro F1:", round(report["macro"]["f1"], 4))
    if "cohen_kappa" in report:
        print("Cohen's kappa (annotators):", round(report["cohen_kappa"], 4))

    # report_text isn't JSON-friendly to re-parse; keep it but also store rest
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
