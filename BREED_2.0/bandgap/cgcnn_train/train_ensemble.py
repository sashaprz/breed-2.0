#!/usr/bin/env python3
"""
Train the full improved model: one metal/non-metal classifier + an ensemble of N
band-gap regressors with different random seeds.

IMPROVEMENT #8: the spread (std) across the N regressors' predictions is used at
inference time (predict.py) as an uncertainty estimate.

Each member is trained in its own subprocess so GPU memory is released cleanly
between runs and a single failure doesn't kill the others. After training, run
tune_threshold.py then predict.py.

Usage
-----
    python train_ensemble.py                       # 1 classifier + 5 regressors
    python train_ensemble.py --n-models 3 --epochs 60 --weight-scheme inverse
    python train_ensemble.py --skip-classifier     # regressors only (re-run)
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import config


def run(cmd):
    print("\n$ " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f"step failed ({res.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-models", type=int, default=config.N_MODELS,
                    help="number of regressors in the ensemble (3-5 recommended)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--weight-scheme", choices=["config", "inverse", "none"],
                    default="config")
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--skip-classifier", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    common_flags = ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                    "--lr", str(args.lr)]
    if args.no_cache:
        common_flags.append("--no-cache")

    if not args.skip_classifier:
        run([py, "train_classifier.py", "--seed", "0", *common_flags])

    for seed in range(args.n_models):
        cmd = [py, "train_regressor.py", "--seed", str(seed),
               "--weight-scheme", args.weight_scheme, *common_flags]
        if args.warm_start:
            cmd.append("--warm-start")
        run(cmd)

    print("\n=> ensemble training complete.")
    print("   next:  python tune_threshold.py   then   python predict.py --cif <file>")


if __name__ == "__main__":
    main()
