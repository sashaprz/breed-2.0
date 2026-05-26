# compute_obelix_bvse.py
#
# Runs bvlain BVSE on every CIF in cifs/ that isn't already in
# bvse_features_combined.csv, then appends the new rows to that file.
#
# Features computed per CIF:
#   barrier_1d/2d/3d     — percolation energy thresholds (eV)
#   dimensionality        — highest percolating dimension (0–3)
#   bottleneck_radius     — min free-sphere radius for 3-D void percolation (Å)
#   accessible_fraction   — fraction of grid points with E < E_min + 1 eV
#   li_site_count         — total Li atoms per unit cell

import os
import glob
import traceback
import numpy as np
import pandas as pd
from bvlain import Lain
from pymatgen.core import Element
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

CIF_DIR  = "cifs"
BVSE_CSV = "bvse_features_combined.csv"

ALL_COLS = ['cif_id', 'barrier_1d', 'barrier_2d', 'barrier_3d',
            'dimensionality', 'bottleneck_radius', 'accessible_fraction',
            'li_site_count', 'max_shift']

# ── Load existing BVSE results ────────────────────────────────────────────────
if os.path.exists(BVSE_CSV):
    existing = pd.read_csv(BVSE_CSV)
    # Add any new columns that didn't exist in older versions of the file
    for col in ALL_COLS:
        if col not in existing.columns:
            existing[col] = np.nan
    done_ids = set(existing['cif_id'].astype(str))
else:
    existing = pd.DataFrame(columns=ALL_COLS)
    done_ids = set()

all_cifs = sorted(glob.glob(os.path.join(CIF_DIR, "*.cif")))
todo     = [f for f in all_cifs if os.path.splitext(os.path.basename(f))[0] not in done_ids]

print(f"CIFs on disk    : {len(all_cifs)}")
print(f"Already computed: {len(done_ids)}")
print(f"To compute      : {len(todo)}")

# ── Batch BVSE ───────────────────────────────────────────────────────────────
new_rows = []

for i, cif_path in enumerate(todo):
    cif_id = os.path.splitext(os.path.basename(cif_path))[0]
    print(f"  [{i+1}/{len(todo)}] {cif_id} ... ", end="", flush=True)
    try:
        calc = Lain(verbose=False)
        calc.read_file(cif_path)

        # ── Energy landscape: barriers + accessible fraction ──────────────────
        calc.bvse_distribution(mobile_ion="Li1+", r_cut=10.0, resolution=0.2)
        b    = calc.percolation_barriers()
        E_min = float(calc.data.min())
        accessible_fraction = float(np.mean(calc.data < (E_min + 1.0)))

        # Dimensionality from barriers
        if b['E_3D'] < 9.99:
            dim = 3
        elif b['E_2D'] < 9.99:
            dim = 2
        elif b['E_1D'] < 9.99:
            dim = 1
        else:
            dim = 0

        # ── Void/geometric landscape: bottleneck radius ───────────────────────
        try:
            calc.void_distribution(mobile_ion="Li1+", r_cut=10.0, resolution=0.2)
            radii = calc.percolation_radii()
            bottleneck_radius = radii['r_3D'] if radii['r_3D'] > 0 else radii['r_2D']
        except Exception:
            bottleneck_radius = np.nan

        # ── Li site count + CIF distortion (max atomic shift from ideal symmetry) ──
        max_shift = np.nan
        li_count  = np.nan
        try:
            struct   = CifParser(cif_path).get_structures(primitive=False)[0]
            li_count = float(struct.composition[Element('Li')])
            p0 = np.array([s.frac_coords for s in struct])
            refined = SpacegroupAnalyzer(struct, symprec=0.1).get_refined_structure()
            if len(refined) == len(struct):
                p1 = np.array([s.frac_coords for s in refined])
                max_shift = float(np.max(np.linalg.norm(p0 - p1, axis=1)))
        except Exception:
            pass

        new_rows.append({
            'cif_id':             cif_id,
            'barrier_1d':         b['E_1D'],
            'barrier_2d':         b['E_2D'],
            'barrier_3d':         b['E_3D'],
            'dimensionality':     dim,
            'bottleneck_radius':  bottleneck_radius,
            'accessible_fraction': accessible_fraction,
            'li_site_count':      li_count,
            'max_shift':          max_shift,
        })
        neck_s = f"{bottleneck_radius:.3f}" if not np.isnan(bottleneck_radius) else "nan"
        li_s   = f"{li_count:.0f}"          if not np.isnan(li_count)          else "nan"
        ms_s   = f"{max_shift:.4f}"         if not np.isnan(max_shift)         else "nan"
        print(f"1D={b['E_1D']:.3f}  3D={b['E_3D']:.3f}  dim={dim}"
              f"  neck={neck_s}  acc={accessible_fraction:.3f}"
              f"  Li={li_s}  shift={ms_s}")
    except Exception:
        print("FAILED")
        traceback.print_exc()

# ── Append and save ───────────────────────────────────────────────────────────
if new_rows:
    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
else:
    updated = existing  # may have gained new columns (NaN) from the header update above

updated.to_csv(BVSE_CSV, index=False)
if new_rows:
    print(f"\nAppended {len(new_rows)} new rows → {BVSE_CSV}  (total {len(updated)})")
else:
    print(f"\nNo new rows — column schema updated and saved → {BVSE_CSV}")
