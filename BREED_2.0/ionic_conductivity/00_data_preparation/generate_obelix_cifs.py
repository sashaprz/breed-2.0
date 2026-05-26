# generate_obelix_cifs.py
#
# Creates CIF files for OBELiX training entries that don't already have one.
# Strategy:
#   1. Integer-stoichiometry compositions → search MP by exact formula.
#   2. Non-integer / partial-occupancy → search MP by chemical system (chemsys),
#      then filter by space-group number.
#   3. Among candidates, score by lattice-parameter similarity to the experimental
#      values (a, b, c, alpha, beta, gamma) stored in obelix_train.csv.
#   4. Save the best match as cifs/{ID}.cif using the same naming convention as
#      the existing cif library.
#
# Run this before computing BVSE on the expanded set.

import os
import glob
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester

# ── Config ────────────────────────────────────────────────────────────────────
CIF_DIR   = "cifs"
CSV_PATH  = os.path.join("raw_data", "obelix_train.csv")

MP_API_KEY = os.environ.get("MP_API_KEY")
if MP_API_KEY is None:
    raise EnvironmentError(
        "Set MP_API_KEY environment variable.\n"
        "Get your key at https://materialsproject.org/api"
    )

os.makedirs(CIF_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
existing = {os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(CIF_DIR, "*.cif"))}

needs_cif = df[~df['ID'].isin(existing)].reset_index(drop=True)
print(f"OBELiX entries  : {len(df)}")
print(f"Already have CIF: {len(existing.intersection(set(df['ID'])))}")
print(f"Need CIF from MP: {len(needs_cif)}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_formula(formula):
    """Return (mp_formula_or_None, chemsys_string)."""
    try:
        comp = Composition(formula)
        reduced = comp.reduced_composition
        is_int = all(abs(v - round(v)) < 0.01 for v in reduced.values())
        chemsys = "-".join(sorted(str(e) for e in comp.elements))
        return (reduced.to_pretty_string() if is_int else None), chemsys
    except Exception:
        return None, None


def get_candidates(mpr, mp_formula, chemsys, formula, has_sg, sg_num):
    """
    Try exact formula, full chemsys, then progressively relaxed chemsys by dropping
    minor-fraction elements.  Returns (candidates, query_used, source_type).

    source_type:
      'exact_formula'  — matched by exact reduced formula on MP
      'exact_chemsys'  — matched by full elemental system
      'proxy_parent'   — matched after dropping minor dopant elements
    """
    queries = []  # (query_type, query_value, source_type)
    if mp_formula:
        queries.append(('formula', mp_formula, 'exact_formula'))

    try:
        comp    = Composition(formula)
        total   = comp.num_atoms
        by_frac = sorted(comp.elements, key=lambda el: comp[el] / total)
        symbols = [str(el) for el in by_frac]  # ascending abundance
    except Exception:
        symbols = chemsys.split('-')

    first_chemsys = True
    while len(symbols) >= 2:
        cs  = '-'.join(sorted(symbols))
        src = 'exact_chemsys' if first_chemsys else 'proxy_parent'
        queries.append(('chemsys', cs, src))
        symbols      = symbols[1:]  # drop least abundant
        first_chemsys = False

    seen = set()
    for qtype, qval, src in queries:
        if qval in seen:
            continue
        seen.add(qval)
        if qtype == 'formula':
            docs = mpr.materials.search(formula=qval, fields=["material_id", "structure", "symmetry"])
        else:
            docs = mpr.materials.search(chemsys=qval, fields=["material_id", "structure", "symmetry"])
        if not docs:
            continue
        if has_sg:
            matched = [d for d in docs if sg_matches(d, sg_num)]
            return (matched if matched else docs), qval, src
        return docs, qval, src

    return [], None, None


def lattice_score(struct, row):
    """
    Normalised distance between experimental (row) and MP lattice.
    Lengths normalised by experimental value; angles normalised by 90°.
    Lower = better match.
    """
    a0, b0, c0 = float(row['a']), float(row['b']), float(row['c'])
    al0, be0, ga0 = float(row['alpha']), float(row['beta']), float(row['gamma'])
    if a0 <= 0 or b0 <= 0 or c0 <= 0:
        return float('inf')
    l = struct.lattice
    len_err = ((l.a - a0)/a0)**2 + ((l.b - b0)/b0)**2 + ((l.c - c0)/c0)**2
    ang_err = ((l.alpha - al0)/90)**2 + ((l.beta - be0)/90)**2 + ((l.gamma - ga0)/90)**2
    return float(np.sqrt(len_err + ang_err))


def sg_matches(doc, sg_num):
    try:
        return doc.symmetry.number == int(sg_num)
    except Exception:
        return False


# ── Load / initialise CIF metadata (source + lattice quality per CIF) ─────────
META_CSV  = "cif_metadata.csv"
META_COLS = ['cif_id', 'source_type', 'lattice_score', 'mp_material_id']
if os.path.exists(META_CSV):
    meta_df  = pd.read_csv(META_CSV)
    meta_ids = set(meta_df['cif_id'].astype(str))
else:
    meta_df  = pd.DataFrame(columns=META_COLS)
    meta_ids = set()

new_meta_rows = []

# ── Main loop ─────────────────────────────────────────────────────────────────
stats = {'created': 0, 'not_found': 0, 'failed': 0, 'no_sg_match_fallback': 0}

with MPRester(MP_API_KEY) as mpr:
    for i, row in needs_cif.iterrows():
        entry_id = row['ID']
        formula  = row['Reduced Composition']
        sg_num   = row['Space group #']
        has_sg   = pd.notna(sg_num) and float(sg_num) > 0

        mp_formula, chemsys = parse_formula(formula)
        if chemsys is None:
            print(f"  [{i+1}/{len(needs_cif)}] SKIP (parse fail): {entry_id} {formula}")
            stats['failed'] += 1
            continue

        try:
            candidates, query_used, source_type = get_candidates(
                mpr, mp_formula, chemsys, formula, has_sg, sg_num
            )

            if not candidates:
                print(f"  [{i+1}/{len(needs_cif)}] NOT FOUND: {entry_id} ({formula})")
                stats['not_found'] += 1
                continue

            # ── Pick best lattice match and save CIF ──────────────────────────
            best  = min(candidates, key=lambda d: lattice_score(d.structure, row))
            score = lattice_score(best.structure, row)

            cif_path = os.path.join(CIF_DIR, f"{entry_id}.cif")
            CifWriter(best.structure).write_file(cif_path)
            stats['created'] += 1

            if entry_id not in meta_ids:
                new_meta_rows.append({
                    'cif_id':        entry_id,
                    'source_type':   source_type,
                    'lattice_score': score,
                    'mp_material_id': best.material_id,
                })

            print(f"  [{i+1}/{len(needs_cif)}] OK  {entry_id:6s}  {formula:<30s}"
                  f"  → {best.material_id}  SG={best.symmetry.number}"
                  f"  src={source_type}  lattice_score={score:.3f}")

        except Exception as e:
            stats['failed'] += 1
            print(f"  [{i+1}/{len(needs_cif)}] FAILED: {entry_id} ({formula}): {e}")

# ── Save CIF metadata ─────────────────────────────────────────────────────────
if new_meta_rows:
    updated_meta = pd.concat([meta_df, pd.DataFrame(new_meta_rows)], ignore_index=True)
    updated_meta.to_csv(META_CSV, index=False)
    print(f"\nAppended {len(new_meta_rows)} rows → {META_CSV}")

# ── Summary ───────────────────────────────────────────────────────────────────
total_cifs = len(glob.glob(os.path.join(CIF_DIR, "*.cif")))
print(f"\n{'='*55}")
print(f"  CIF generation complete")
print(f"{'='*55}")
print(f"  Created this run      : {stats['created']}")
print(f"  Not found on MP       : {stats['not_found']}")
print(f"  Failed (parse/API)    : {stats['failed']}")
print(f"  SG fallback (no match): {stats['no_sg_match_fallback']}")
print(f"  Total CIFs in cifs/   : {total_cifs}")
print(f"\nNext step: run compute_obelix_bvse.py to compute BVSE for new CIFs,")
print(f"then re-run GBT_predict_LLZO.py.")
