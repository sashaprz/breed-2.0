#!/usr/bin/env python3
"""
Optimize the metal / non-metal decision threshold on validation data.

IMPROVEMENT #5: rather than the default 0.5 (or a fixed gap cutoff), sweep the
classifier's P(metal) threshold over the validation set and pick the value that
maximizes a chosen metric -- balanced accuracy by default (robust to the heavy
metal/nonmetal imbalance in MP), F1 optional.

Reads  : models/classifier_val_probs.json  (written by train_classifier.py)
Writes : models/threshold.json  ->  {"threshold": t, "metric": ..., "value": ...}

Usage
-----
    python tune_threshold.py                     # balanced accuracy
    python tune_threshold.py --metric f1
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import config


def balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # metal == positive (1)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr)


def f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


METRICS = {"balanced_accuracy": balanced_accuracy, "f1": f1}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=list(METRICS), default="balanced_accuracy")
    ap.add_argument("--steps", type=int, default=199, help="threshold grid resolution")
    args = ap.parse_args()

    probs_file = config.MODELS_DIR / "classifier_val_probs.json"
    if not probs_file.exists():
        raise SystemExit(f"{probs_file} not found -- run train_classifier.py first.")

    data = json.loads(probs_file.read_text())
    p_metal = np.asarray(data["p_metal"], dtype=float)
    y_true = np.asarray(data["is_metal"], dtype=int)
    metric_fn = METRICS[args.metric]

    grid = np.linspace(0.01, 0.99, args.steps)
    best_t, best_v = 0.5, -1.0
    for t in grid:
        y_pred = (p_metal >= t).astype(int)
        v = metric_fn(y_true, y_pred)
        if v > best_v:
            best_v, best_t = v, float(t)

    # Report at the chosen threshold.
    y_pred = (p_metal >= best_t).astype(int)
    out = {
        "threshold": best_t,
        "metric": args.metric,
        "value": float(best_v),
        "n_val": int(len(y_true)),
        "default_0.5_value": float(metric_fn(y_true, (p_metal >= 0.5).astype(int))),
    }
    with open(config.THRESHOLD_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"=> best threshold P(metal) >= {best_t:.3f}  "
          f"({args.metric} {best_v:.4f}, vs {out['default_0.5_value']:.4f} at 0.5)")
    print(f"=> wrote {config.THRESHOLD_JSON}")


if __name__ == "__main__":
    main()
