#!/usr/bin/env python3
"""
Benchmark the new ensemble (classifier + N regressors) against Materials
Project PBE band gaps.

This is the apples-to-apples counterpart to
`BREED/env/bandgap/benchmark/benchmark_bandgap.py`, which benchmarks the old
single-model `band-gap.pth.tar`. Same MP sampling strategy (gap-stratified,
seed-reproducible) and the same scoring metrics, so the two MAE numbers can be
compared directly.

Inference here uses the full pipeline:

    classifier -> P(metal) -> if P >= tuned threshold: gap = 0 eV
                              else: gap = mean of N regressors

(same as `predict.py`), so the reported MAE/classification accuracy reflect
what downstream BREED screening would actually see.

Usage
-----
    python benchmark_mp.py --smoke                 # 30 materials
    python benchmark_mp.py --n 300                 # full run (default)
    python benchmark_mp.py --n 300 --min-mpid 1000000   # bias toward newer/held-out materials
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import common
from loaders import GraphDataset
from cgcnn.data import collate_pool
from predict import load_classifier, load_regressors, load_threshold


THIS_DIR = Path(__file__).resolve().parent
# fall back to the old benchmark's gitignored API key file if no key is given
OLD_KEYFILE = THIS_DIR.parents[1] / "BREED" / "env" / "bandgap" / "benchmark" / ".mp_api_key"

# Same gap-stratified bins as the old benchmark, for a directly comparable sample.
DEFAULT_BINS = [(0.0, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 6.0), (6.0, 12.0)]
SCORE_RANGES = [(0, 0.3), (0.3, 1), (1, 2), (2, 3.5), (3.5, 6), (6, 99)]

# Reference numbers from the old single-model benchmark (n=300, seed=42, 2026-06-09).
OLD_CGCNN_MAE_EV = 0.70
OLD_CGCNN_METAL_ACC = 0.86


# ------------------------------------------------------------------------------
# MP sampling (mirrors benchmark_bandgap.py)
# ------------------------------------------------------------------------------
def resolve_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key.strip()
    env = os.environ.get("MP_API_KEY")
    if env:
        return env.strip()
    if OLD_KEYFILE.exists():
        return OLD_KEYFILE.read_text().strip()
    raise SystemExit(
        "No Materials Project API key. Pass --api-key, set $MP_API_KEY, or ensure "
        f"{OLD_KEYFILE} exists. Get a free key at https://materialsproject.org/api"
    )


def _mpid_num(mpid: str) -> int:
    try:
        return int(mpid.split("-")[1])
    except (IndexError, ValueError):
        return 0


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def pull_mp_sample(api_key: str, n: int, seed: int, min_mpid: int,
                    bins=DEFAULT_BINS, pool_per_bin: int = 1000):
    """Gap-stratified sample: {material_id, formula, band_gap, is_metal, structure}."""
    from mp_api.client import MPRester

    rng = random.Random(seed)
    per_bin = max(1, n // len(bins))
    chosen_ids: list[str] = []

    with MPRester(api_key) as mpr:
        for lo, hi in bins:
            docs = mpr.materials.summary.search(
                band_gap=(lo, hi),
                fields=["material_id", "band_gap"],
                num_chunks=1,
                chunk_size=min(pool_per_bin, 1000),
            )
            ids = [str(d.material_id) for d in docs]
            if min_mpid > 0:
                ids = [i for i in ids if _mpid_num(i) >= min_mpid]
            rng.shuffle(ids)
            picked = ids[:per_bin]
            chosen_ids.extend(picked)
            print(f"  bin [{lo:>4}, {hi:>5}) eV: pool={len(ids):5d}  picked={len(picked)}")

        chosen_ids = list(dict.fromkeys(chosen_ids))
        print(f"=> fetching structures for {len(chosen_ids)} materials ...")

        rows = []
        for batch in _chunks(chosen_ids, 200):
            docs = mpr.materials.summary.search(
                material_ids=batch,
                fields=["material_id", "formula_pretty", "band_gap", "is_metal", "structure"],
            )
            for d in docs:
                if d.structure is None:
                    continue
                rows.append({
                    "material_id": str(d.material_id),
                    "formula": d.formula_pretty,
                    "band_gap": float(d.band_gap),
                    "is_metal": bool(d.is_metal),
                    "structure": d.structure,
                })
    print(f"=> retrieved {len(rows)} materials with structures")
    return rows


# ------------------------------------------------------------------------------
# CIFs + graph validation
# ------------------------------------------------------------------------------
def write_and_validate_cifs(rows, work_dir: Path):
    """Write <id>.cif for each row, drop any that can't form a valid crystal graph."""
    from pymatgen.io.cif import CifWriter

    work_dir.mkdir(parents=True, exist_ok=True)
    valid = []
    for r in rows:
        cif_path = work_dir / f"{r['material_id']}.cif"
        try:
            CifWriter(r["structure"]).write_file(str(cif_path))
        except Exception as e:  # noqa: BLE001
            print(f"    [skip] {r['material_id']}: CIF write failed ({e})")
            continue
        valid.append(r)

    # validate via the same graph builder used at train/inference time
    probe = GraphDataset([(r["material_id"], 0.0) for r in valid], cif_dir=work_dir, use_cache=False)
    ok_rows = []
    for i, r in enumerate(valid):
        try:
            probe[i]
        except Exception as e:  # noqa: BLE001
            print(f"    [skip] {r['material_id']}: graph build failed ({e})")
            (work_dir / f"{r['material_id']}.cif").unlink(missing_ok=True)
            continue
        ok_rows.append(r)

    print(f"=> {len(ok_rows)} / {len(rows)} materials produced valid crystal graphs")
    return ok_rows


# ------------------------------------------------------------------------------
# Ensemble inference (mirrors predict.py)
# ------------------------------------------------------------------------------
@torch.no_grad()
def run_ensemble(rows, cif_dir: Path, batch_size: int):
    threshold = load_threshold()
    classifier = load_classifier()
    regressors = load_regressors()
    print(f"=> classifier + {len(regressors)} regressors on {config.DEVICE}, "
          f"metal threshold P>={threshold:.3f}")

    ids = [r["material_id"] for r in rows]
    dataset = GraphDataset([(i, 0.0) for i in ids], cif_dir=cif_dir, use_cache=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_pool,
                         pin_memory=(config.DEVICE.type == "cuda"))

    preds = {}
    for inp, _, batch_ids in loader:
        inp = common.move_input(inp)
        with torch.autocast(device_type=config.AMP_DEVICE_TYPE, enabled=config.USE_AMP):
            clf_out = classifier(*inp)
            p_metal = torch.exp(clf_out.float())[:, 1].cpu().numpy()

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
            preds[mid] = {
                "predicted_band_gap_eV": gap,
                "uncertainty_eV": unc,
                "p_metal": float(p_metal[k]),
                "classified_metal": is_metal,
            }
    return preds


# ------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------
def score(rows, preds: dict, out_dir: Path):
    records = []
    for r in rows:
        pid = r["material_id"]
        if pid not in preds:
            continue
        p = preds[pid]
        actual = r["band_gap"]
        pred = p["predicted_band_gap_eV"]
        records.append({
            "material_id": pid,
            "formula": r["formula"],
            "mp_pbe_band_gap": actual,
            "predicted_band_gap_eV": pred,
            "uncertainty_eV": p["uncertainty_eV"],
            "error": pred - actual,
            "abs_error": abs(pred - actual),
            "mp_is_metal": r["is_metal"],
            "p_metal": p["p_metal"],
            "classified_metal": p["classified_metal"],
        })

    if not records:
        raise RuntimeError("No predictions matched sampled materials -- nothing to score.")

    actual = np.array([x["mp_pbe_band_gap"] for x in records])
    pred = np.array([x["predicted_band_gap_eV"] for x in records])
    err = pred - actual

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 1 else float("nan")

    # metal classification: our classifier+threshold vs MP ground truth
    is_metal_true = np.array([x["mp_is_metal"] for x in records])
    is_metal_pred = np.array([x["classified_metal"] for x in records])
    cls_acc = float(np.mean(is_metal_true == is_metal_pred))

    # MAE by gap range (same bins as the old benchmark)
    by_range = {}
    for lo, hi in SCORE_RANGES:
        m = (actual >= lo) & (actual < hi)
        if m.sum():
            by_range[f"[{lo}, {hi}) eV"] = {"n": int(m.sum()), "mae": float(np.mean(np.abs(err[m])))}

    # Nonmetal-only MAE using config.METRIC_BINS, comparable to the training test MAE
    nonmetal_mask = ~is_metal_true
    nonmetal_by_range = {}
    if nonmetal_mask.sum():
        nm_actual, nm_err = actual[nonmetal_mask], err[nonmetal_mask]
        nonmetal_mae = float(np.mean(np.abs(nm_err)))
        for (lo, hi), lab in zip(config.METRIC_BINS, config.METRIC_BIN_LABELS):
            m = (nm_actual >= lo) & (nm_actual < hi)
            nonmetal_by_range[lab] = {"n": int(m.sum()),
                                       "mae": float(np.mean(np.abs(nm_err[m]))) if m.any() else None}
    else:
        nonmetal_mae = float("nan")

    summary = {
        "n_materials": len(records),
        "mae_eV": mae,
        "rmse_eV": rmse,
        "bias_eV_pred_minus_actual": bias,
        "r2": r2,
        "pearson_r": pearson,
        "metal_classification_accuracy": cls_acc,
        "mae_by_gap_range": by_range,
        "nonmetal_only": {
            "n": int(nonmetal_mask.sum()),
            "mae_eV": nonmetal_mae,
            "mae_by_range": nonmetal_by_range,
        },
        "reference_old_cgcnn": {
            "mae_eV": OLD_CGCNN_MAE_EV,
            "metal_classification_accuracy": OLD_CGCNN_METAL_ACC,
            "note": "BREED/env/bandgap/benchmark results, n=300 seed=42 2026-06-09",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    with open(out_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot(actual, pred, out_dir / "benchmark_scatter.png", mae, r2)
    _print_summary(summary, csv_path, out_dir)
    return summary


def _plot(actual, pred, path: Path, mae, r2):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skipping plot, matplotlib unavailable: {e})")
        return
    lim = float(max(actual.max(), pred.max())) * 1.05 + 0.2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred, s=14, alpha=0.5, edgecolors="none")
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (perfect)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Materials Project PBE band gap (eV)")
    ax.set_ylabel("Predicted band gap (eV) -- ensemble")
    ax.set_title(f"New ensemble vs MP PBE\nMAE = {mae:.3f} eV   R² = {r2:.3f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"=> wrote {path}")


def _print_summary(s, csv_path, out_dir):
    print("\n" + "=" * 60)
    print("BANDGAP BENCHMARK SUMMARY (new ensemble vs MP PBE)")
    print("=" * 60)
    print(f"  materials scored      : {s['n_materials']}")
    print(f"  MAE                   : {s['mae_eV']:.3f} eV")
    print(f"  RMSE                  : {s['rmse_eV']:.3f} eV")
    print(f"  bias (pred - actual)  : {s['bias_eV_pred_minus_actual']:+.3f} eV")
    print(f"  R^2                   : {s['r2']:.3f}")
    print(f"  Pearson r             : {s['pearson_r']:.3f}")
    print(f"  metal class. accuracy : {s['metal_classification_accuracy']:.3f}")
    print("  MAE by gap range:")
    for rng, v in s["mae_by_gap_range"].items():
        print(f"      {rng:<12} n={v['n']:<4} MAE={v['mae']:.3f} eV")
    nm = s["nonmetal_only"]
    print(f"  nonmetal-only MAE      : {nm['mae_eV']:.3f} eV  (n={nm['n']})")
    for lab, v in nm["mae_by_range"].items():
        m = f"{v['mae']:.3f}" if v["mae"] is not None else "  n/a"
        print(f"      {lab:<8} n={v['n']:<4} MAE={m} eV")
    ref = s["reference_old_cgcnn"]
    print("-" * 60)
    print(f"  old single-model CGCNN : MAE={ref['mae_eV']:.3f} eV, "
          f"metal acc={ref['metal_classification_accuracy']:.3f}  ({ref['note']})")
    print("-" * 60)
    print(f"  per-material CSV : {csv_path}")
    print(f"  summary JSON     : {out_dir / 'benchmark_summary.json'}")
    print(f"  scatter plot     : {out_dir / 'benchmark_scatter.png'}")
    print("=" * 60)


# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", default=None, help="Materials Project API key")
    ap.add_argument("--n", type=int, default=300, help="target number of materials")
    ap.add_argument("--smoke", action="store_true", help="quick 30-material run")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (reproducible)")
    ap.add_argument("--min-mpid", type=int, default=0,
                    help="only sample mp-ids >= this number (bias toward newer / "
                         "more likely held-out materials)")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--out-dir", default=str(THIS_DIR.parent / "benchmark_results"))
    ap.add_argument("--keep-cifs", action="store_true",
                    help="keep the temporary CIF working directory")
    args = ap.parse_args()

    if args.smoke:
        args.n = 30

    api_key = resolve_api_key(args.api_key)
    out_dir = Path(args.out_dir)

    print(f"=> sampling ~{args.n} materials from Materials Project (seed={args.seed}) ...")
    rows = pull_mp_sample(api_key, args.n, args.seed, args.min_mpid)
    if not rows:
        raise SystemExit("No materials pulled from MP -- check API key / filters.")

    work_dir = Path(tempfile.mkdtemp(prefix="bandgap_bench_mp_"))
    try:
        valid = write_and_validate_cifs(rows, work_dir)
        if not valid:
            raise SystemExit("No valid crystal graphs were built.")
        print("=> running ensemble predictions ...")
        preds = run_ensemble(valid, work_dir, args.batch_size)
        score(valid, preds, out_dir)
    finally:
        if args.keep_cifs:
            print(f"=> CIF working dir kept at {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
