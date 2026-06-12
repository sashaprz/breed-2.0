#!/usr/bin/env python3
"""
Fetch a Materials Project training set for the improved CGCNN band-gap model.

IMPROVEMENT #1: retrain on *current* Materials Project data (not the ~2018 CGCNN
snapshot) for better chemistry coverage and wide-gap representation.

What it does
------------
1. Metadata pass (cheap): walk a set of band-gap bins and pull, for every
   non-deprecated material, its `material_id, band_gap, is_metal,
   energy_above_hull, formula_pretty` -- ids only, no structures. By default this
   is ALL suitable materials in MP (~150k+), so wide-gap insulators (rare in MP)
   are fully represented. Optional `--max-ehull` stability filter and `--max-n`
   cap (for smoke tests) narrow it.
2. Structure pass (heavy, RESUMABLE): for each chosen id whose CIF is not already
   on disk, fetch the structure, write `data/cifs/<id>.cif`, and validate that it
   builds a CGCNN crystal graph (drop the ones that don't). Re-running picks up
   where it left off.
3. Write `data/labels.csv` with columns:
       material_id, formula, reduced_formula, band_gap, is_metal, e_above_hull
   (reduced_formula is the composition group key used later by make_splits.py).

Runs ON the training machine -- it downloads the data where it is needed.

API key resolution order:  --api-key  ->  $MP_API_KEY  ->  ./.mp_api_key  (gitignored)

Usage
-----
    python fetch_mp_training_data.py                 # everything (full run)
    python fetch_mp_training_data.py --smoke         # ~40 materials, quick wiring test
    python fetch_mp_training_data.py --max-ehull 0.1 # only near-stable materials
    python fetch_mp_training_data.py --max-n 30000   # cap total (random, seeded)
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import config


# ---------------------------------------------------------------------------
# API key (reused pattern from benchmark/benchmark_bandgap.py:resolve_api_key)
# ---------------------------------------------------------------------------
def resolve_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key.strip()
    env = os.environ.get("MP_API_KEY")
    if env:
        return env.strip()
    keyfile = config.PKG_DIR / ".mp_api_key"
    if keyfile.exists():
        return keyfile.read_text().strip()
    raise SystemExit(
        "No Materials Project API key. Pass --api-key, set $MP_API_KEY, or create "
        f"{keyfile} (one line, just the key). Free key: https://materialsproject.org/api"
    )


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------------------
# Metadata pass: collect every candidate material (no structures yet)
# ---------------------------------------------------------------------------
def pull_metadata(mpr, bins, max_ehull):
    """Return {material_id: {band_gap, is_metal, e_above_hull, formula}}."""
    meta: dict[str, dict] = {}
    fields = ["material_id", "band_gap", "is_metal",
              "energy_above_hull", "formula_pretty"]
    for lo, hi in bins:
        kwargs = dict(band_gap=(lo, hi), fields=fields, chunk_size=1000)
        # Push the stability filter server-side when possible (cheaper).
        if max_ehull is not None:
            kwargs["energy_above_hull"] = (0.0, max_ehull)
        docs = mpr.materials.summary.search(**kwargs)
        added = 0
        for d in docs:
            mid = str(d.material_id)
            if mid in meta:
                continue
            ehull = getattr(d, "energy_above_hull", None)
            if max_ehull is not None and ehull is not None and ehull > max_ehull:
                continue
            meta[mid] = {
                "band_gap": float(d.band_gap) if d.band_gap is not None else 0.0,
                "is_metal": bool(d.is_metal),
                "e_above_hull": float(ehull) if ehull is not None else "",
                "formula": d.formula_pretty,
            }
            added += 1
        print(f"  bin [{lo:>6}, {hi:>6}) eV: +{added:6d}  (running total {len(meta)})")
    return meta


# ---------------------------------------------------------------------------
# Structure pass: download CIFs + validate graphs (resumable)
# ---------------------------------------------------------------------------
def reduced_formula(formula: str) -> str:
    try:
        from pymatgen.core import Composition
        return Composition(formula).reduced_formula
    except Exception:
        return formula


def already_have_cif(mid: str) -> bool:
    return (config.CIF_DIR / f"{mid}.cif").exists()


def validate_graph(cif_path: Path, ari, gdf) -> bool:
    """True if the structure converts to a CGCNN crystal graph."""
    from loaders import build_graph
    try:
        build_graph(cif_path, ari, gdf,
                    max_num_nbr=config.MAX_NUM_NBR, radius=config.RADIUS)
        return True
    except Exception:
        return False


def fetch_structures(mpr, ids, meta, batch_size=400):
    """Download + write + validate CIFs for `ids` not already on disk.

    Returns list of validated material_ids (including ones already present)."""
    from pymatgen.io.cif import CifWriter
    from cgcnn.data import AtomCustomJSONInitializer, GaussianDistance

    ari = AtomCustomJSONInitializer(str(config.ATOM_INIT_FILE))
    gdf = GaussianDistance(dmin=config.GAUSS_DMIN, dmax=config.RADIUS,
                           step=config.GAUSS_STEP)

    valid: list[str] = []
    todo = [mid for mid in ids if not already_have_cif(mid)]
    have = [mid for mid in ids if already_have_cif(mid)]
    # Trust CIFs already on disk from a previous (validated) run.
    valid.extend(have)
    if have:
        print(f"  resuming: {len(have)} CIFs already present, {len(todo)} to fetch")

    n_written = n_dropped = 0
    for batch in _chunks(todo, batch_size):
        docs = mpr.materials.summary.search(
            material_ids=batch, fields=["material_id", "structure"])
        for d in docs:
            mid = str(d.material_id)
            if d.structure is None:
                n_dropped += 1
                continue
            cif_path = config.CIF_DIR / f"{mid}.cif"
            try:
                CifWriter(d.structure).write_file(str(cif_path))
            except Exception as e:  # noqa: BLE001
                print(f"    [skip] {mid}: CIF write failed ({e})")
                n_dropped += 1
                continue
            if validate_graph(cif_path, ari, gdf):
                valid.append(mid)
                n_written += 1
            else:
                cif_path.unlink(missing_ok=True)
                n_dropped += 1
        print(f"  fetched {n_written:6d} ok / {n_dropped:5d} dropped "
              f"(of {len(todo)} to fetch)")
    return valid


def write_labels(valid_ids, meta):
    rows = []
    for mid in valid_ids:
        m = meta.get(mid)
        if m is None:
            # CIF present from a prior run but metadata absent this run; skip from
            # labels (it will simply not be trained on). Rare.
            continue
        rows.append({
            "material_id": mid,
            "formula": m["formula"],
            "reduced_formula": reduced_formula(m["formula"]),
            "band_gap": m["band_gap"],
            "is_metal": int(m["is_metal"]),
            "e_above_hull": m["e_above_hull"],
        })
    rows.sort(key=lambda r: r["material_id"])
    with open(config.LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["material_id", "formula",
                                          "reduced_formula", "band_gap",
                                          "is_metal", "e_above_hull"])
        w.writeheader()
        w.writerows(rows)
    return rows


def summarize(rows):
    n = len(rows)
    if not n:
        print("No materials written.")
        return
    metals = sum(1 for r in rows if r["is_metal"])
    print("\n" + "=" * 56)
    print("TRAINING SET SUMMARY")
    print("=" * 56)
    print(f"  total materials : {n}")
    print(f"  metals          : {metals} ({metals / n * 100:.1f}%)")
    print(f"  nonmetals       : {n - metals} ({(n - metals) / n * 100:.1f}%)")
    print("  nonmetal gap distribution:")
    for (lo, hi), lab in zip(config.METRIC_BINS, config.METRIC_BIN_LABELS):
        c = sum(1 for r in rows if not r["is_metal"]
                and lo <= float(r["band_gap"]) < hi)
        print(f"      {lab:<8} n={c}")
    print(f"  labels.csv      : {config.LABELS_CSV}")
    print(f"  cifs            : {config.CIF_DIR}")
    print("=" * 56)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-ehull", type=float, default=None,
                    help="only keep materials with energy_above_hull <= this "
                         "(eV/atom). Default: no stability filter (keep all).")
    ap.add_argument("--max-n", type=int, default=None,
                    help="cap total materials (random, seeded). Default: no cap.")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny ~40-material run to validate the pipeline wiring")
    args = ap.parse_args()

    config.ensure_dirs()
    api_key = resolve_api_key(args.api_key)

    bins = config.FETCH_BINS
    max_n = args.max_n
    if args.smoke:
        max_n = 40
        print("[smoke] capping at ~40 materials")

    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        print("=> metadata pass (ids + gap + is_metal, no structures yet) ...")
        meta = pull_metadata(mpr, bins, args.max_ehull)
        ids = list(meta.keys())
        print(f"=> {len(ids)} candidate materials after metadata pass")
        if not ids:
            raise SystemExit("No materials matched -- check API key / filters.")

        if max_n is not None and len(ids) > max_n:
            random.Random(args.seed).shuffle(ids)
            ids = ids[:max_n]
            print(f"=> capped to {len(ids)} (seed={args.seed})")

        print("=> structure pass (download CIFs + validate graphs, resumable) ...")
        valid_ids = fetch_structures(mpr, ids, meta)

    rows = write_labels(valid_ids, meta)
    summarize(rows)


if __name__ == "__main__":
    main()
