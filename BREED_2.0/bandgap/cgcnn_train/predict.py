#!/usr/bin/env python3
"""
Full inference with the improved model:

    classifier  ->  P(metal)  ->  if metal (P >= tuned threshold): gap = 0 eV
                                  else: gap = mean of N regressors
                                        uncertainty = std of N regressors  (#8)

A material predicted to be a metal is pinned to exactly 0 eV (removing the old
single-regressor's ~0.4 eV floor). Non-metals get the band-gap ensemble mean, and
the ensemble spread is reported as the uncertainty.

Inputs (one of):
    --cif PATH            a single .cif file
    --cif-dir DIR         a directory of .cif files
    --test                predict over the held-out test split (data/cifs)

Usage
-----
    python predict.py --cif candidate.cif
    python predict.py --cif-dir ./my_cifs --out preds.csv
    python predict.py --test --out test_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import config
import common
from loaders import GraphDataset
from cgcnn.data import collate_pool
from torch.utils.data import DataLoader


def load_classifier():
    ckpt = torch.load(config.CLASSIFIER_CKPT, map_location=config.DEVICE)
    dims = ckpt["graph_dims"]
    model = common.build_model(dims[0], dims[1], classification=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_regressors():
    members = []
    for ckpt_path in sorted(config.MODELS_DIR.glob("regressor_seed*.pth.tar")):
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
        dims = ckpt["graph_dims"]
        model = common.build_model(dims[0], dims[1], classification=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        norm = common.Normalizer(torch.zeros(1))
        norm.load_state_dict(ckpt["normalizer"])
        members.append((model, norm))
    if not members:
        raise SystemExit("No regressor checkpoints found -- run train_ensemble.py first.")
    return members


def load_threshold():
    if config.THRESHOLD_JSON.exists():
        return float(json.loads(config.THRESHOLD_JSON.read_text())["threshold"])
    print("[warn] no threshold.json (run tune_threshold.py); defaulting to 0.5")
    return 0.5


def resolve_inputs(args):
    """Return (cif_dir, list_of_ids)."""
    if args.cif:
        p = Path(args.cif)
        return p.parent, [p.stem]
    if args.cif_dir:
        d = Path(args.cif_dir)
        return d, sorted(c.stem for c in d.glob("*.cif"))
    if args.test:
        return config.CIF_DIR, common.load_split(config.TEST_IDS)
    raise SystemExit("Provide one of --cif, --cif-dir, or --test.")


@torch.no_grad()
def predict(args):
    threshold = load_threshold()
    classifier = load_classifier()
    regressors = load_regressors()
    print(f"=> classifier + {len(regressors)} regressors on {config.DEVICE}, "
          f"metal threshold P>={threshold:.3f}")

    cif_dir, ids = resolve_inputs(args)
    use_cache = args.test  # the cache only exists for the fetched dataset
    dataset = GraphDataset([(i, 0.0) for i in ids], cif_dir=cif_dir, use_cache=use_cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_pool,
                        pin_memory=(config.DEVICE.type == "cuda"))

    results = []
    for inp, _, batch_ids in loader:
        inp = common.move_input(inp)
        with torch.autocast(device_type=config.AMP_DEVICE_TYPE, enabled=config.USE_AMP):
            clf_out = classifier(*inp)
            p_metal = torch.exp(clf_out.float())[:, 1].cpu().numpy()

            # all regressors over the same batch. denorm on the normalizer's own
            # device (checkpoint tensors were moved to DEVICE by map_location), then
            # bring the result to CPU -- avoids cpu/cuda tensor mixing.
            member_preds = []
            for model, norm in regressors:
                out = model(*inp).detach().float()
                member_preds.append(norm.denorm(out).view(-1).cpu().numpy())
        member_preds = np.stack(member_preds, axis=0)  # (n_members, B)

        for k, mid in enumerate(batch_ids):
            is_metal = bool(p_metal[k] >= threshold)
            if is_metal:
                gap, unc = 0.0, 0.0
            else:
                gap = float(member_preds[:, k].mean())
                gap = max(0.0, gap)
                unc = float(member_preds[:, k].std())
            results.append({
                "material_id": mid,
                "predicted_band_gap_eV": round(gap, 4),
                "uncertainty_eV": round(unc, 4),
                "p_metal": round(float(p_metal[k]), 4),
                "classified_metal": is_metal,
            })
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cif", help="single .cif file")
    g.add_argument("--cif-dir", help="directory of .cif files")
    g.add_argument("--test", action="store_true", help="held-out test split")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--out", default=None, help="write results to this CSV")
    args = ap.parse_args()

    results = predict(args)

    # If we predicted over the test split, also report MAE on non-metals.
    if args.test:
        labels = common.load_labels()
        t, p = [], []
        for r in results:
            lab = labels.get(r["material_id"])
            if lab and not lab["is_metal"]:
                t.append(lab["band_gap"])
                p.append(r["predicted_band_gap_eV"])
        if t:
            by_range = common.mae_by_range(t, p)
            overall = float(np.mean(np.abs(np.array(p) - np.array(t))))
            print(f"\nTEST nonmetal MAE = {overall:.4f} eV  (n={len(t)})")
            for lab in config.METRIC_BIN_LABELS:
                v = by_range[lab]
                m = f"{v['mae']:.4f}" if v["mae"] is not None else "  n/a"
                print(f"    {lab:<8} n={v['n']:<5} MAE={m}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"=> wrote {args.out}")
    else:
        for r in results[:50]:
            print(f"  {r['material_id']:<16} gap={r['predicted_band_gap_eV']:>7.3f} "
                  f"+/- {r['uncertainty_eV']:.3f} eV  "
                  f"{'METAL' if r['classified_metal'] else ''}")
        if len(results) > 50:
            print(f"  ... ({len(results)} total; use --out to write all)")


if __name__ == "__main__":
    main()
