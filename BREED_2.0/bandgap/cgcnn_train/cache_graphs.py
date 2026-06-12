#!/usr/bin/env python3
"""
Precompute crystal graphs to disk so multi-seed ensemble training doesn't rebuild
them N times (graph construction is the CPU bottleneck).

Writes data/graphs/<material_id>.pt = (atom_fea, nbr_fea, nbr_fea_idx) for every
material in data/labels.csv. RESUMABLE: skips ids already cached. Optional but
recommended before train_ensemble.py on the full dataset.

Usage
-----
    python cache_graphs.py
"""
from __future__ import annotations

import csv

import torch

import config
from cgcnn.data import AtomCustomJSONInitializer, GaussianDistance
from loaders import build_graph


def main():
    config.ensure_dirs()
    config.GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not config.LABELS_CSV.exists():
        raise SystemExit(f"{config.LABELS_CSV} not found -- run fetch_mp_training_data.py first.")

    ari = AtomCustomJSONInitializer(str(config.ATOM_INIT_FILE))
    gdf = GaussianDistance(dmin=config.GAUSS_DMIN, dmax=config.RADIUS, step=config.GAUSS_STEP)

    with open(config.LABELS_CSV, newline="") as f:
        ids = [row["material_id"] for row in csv.DictReader(f)]

    done = built = failed = 0
    for mid in ids:
        out = config.GRAPH_CACHE_DIR / f"{mid}.pt"
        if out.exists():
            done += 1
            continue
        cif = config.CIF_DIR / f"{mid}.cif"
        try:
            g = build_graph(cif, ari, gdf)
            torch.save(g, out)
            built += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [skip] {mid}: {e}")
        if (built + done) % 1000 == 0:
            print(f"  cached {built} new / {done} already present / {failed} failed")

    print(f"=> graph cache: {built} built, {done} already present, {failed} failed "
          f"-> {config.GRAPH_CACHE_DIR}")


if __name__ == "__main__":
    main()
