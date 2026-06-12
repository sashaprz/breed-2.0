#!/usr/bin/env python3
"""
Composition-grouped train/val/test split.

IMPROVEMENT #2: split by composition group so polymorphs or near-duplicate
compositions cannot leak between train and test. We group on
`reduced_formula` (e.g. all TiO2 polymorphs share one group) and use
sklearn.GroupShuffleSplit so an entire group lands wholly in one split.

Reads  : data/labels.csv  (from fetch_mp_training_data.py)
Writes : splits/train_ids.txt, splits/val_ids.txt, splits/test_ids.txt
         splits/split_meta.json  (counts + leakage assertion result)

Usage
-----
    python make_splits.py                       # default 80/10/10, seed 42
    python make_splits.py --val 0.1 --test 0.1 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import config


def load_labels():
    ids, groups = [], []
    with open(config.LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["material_id"])
            groups.append(row["reduced_formula"] or row["formula"])
    return np.array(ids), np.array(groups)


def grouped_split(ids, groups, val_frac, test_frac, seed):
    """Return (train_ids, val_ids, test_ids) with no group spanning two splits."""
    idx = np.arange(len(ids))

    # First carve out the test set by group.
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(idx, groups=groups))

    # Then carve val out of the remaining train+val, again by group.
    # val_frac is expressed as a fraction of the WHOLE dataset, so rescale.
    rel_val = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tv_groups = groups[trainval_idx]
    tr_rel, val_rel = next(gss2.split(trainval_idx, groups=tv_groups))
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[val_rel]

    return ids[train_idx], ids[val_idx], ids[test_idx]


def assert_no_leak(ids, groups, train_ids, val_ids, test_ids):
    g = dict(zip(ids, groups))
    gtr = {g[i] for i in train_ids}
    gva = {g[i] for i in val_ids}
    gte = {g[i] for i in test_ids}
    overlaps = {
        "train_val": gtr & gva,
        "train_test": gtr & gte,
        "val_test": gva & gte,
    }
    leaked = {k: len(v) for k, v in overlaps.items() if v}
    return leaked


def write_ids(path, ids):
    with open(path, "w") as f:
        f.write("\n".join(ids) + ("\n" if len(ids) else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=float, default=config.VAL_FRAC)
    ap.add_argument("--test", type=float, default=config.TEST_FRAC)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.ensure_dirs()
    if not config.LABELS_CSV.exists():
        raise SystemExit(f"{config.LABELS_CSV} not found -- run fetch_mp_training_data.py first.")

    ids, groups = load_labels()
    print(f"=> {len(ids)} materials, {len(set(groups))} unique composition groups")

    train_ids, val_ids, test_ids = grouped_split(
        ids, groups, args.val, args.test, args.seed)

    leaked = assert_no_leak(ids, groups, train_ids, val_ids, test_ids)
    if leaked:
        raise SystemExit(f"Composition leakage detected: {leaked} -- aborting.")

    write_ids(config.TRAIN_IDS, list(train_ids))
    write_ids(config.VAL_IDS, list(val_ids))
    write_ids(config.TEST_IDS, list(test_ids))

    meta = {
        "n_total": len(ids),
        "n_groups": len(set(groups)),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "val_frac": args.val,
        "test_frac": args.test,
        "seed": args.seed,
        "leakage": "none",
    }
    with open(config.SPLIT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")
    print(f"   no composition group spans two splits  [OK]")
    print(f"=> wrote {config.SPLITS_DIR}")


if __name__ == "__main__":
    main()
