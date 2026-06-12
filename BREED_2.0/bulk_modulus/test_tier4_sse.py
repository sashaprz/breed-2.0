#!/usr/bin/env python3
"""
Tier 4 -- SSE benchmark set (needs an MLIP).

Runs the full static pipeline on a handful of solid electrolytes with MP
elastic data spanning oxides and sulfides, via the built-in Phase-5 validate()
gate, and asserts:

  * mean abs % error vs MP k_vrh is in a reasonable band (target ~10-15%),
  * the QUALITATIVE ranking holds: oxides come out stiffer than sulfides
    (the hard-to-fake bar -- S-P bonds are far more compliant than O bonds).

A curated small-cell subset is used by default so it finishes in a sensible
time; pass --full to run the whole VALIDATION_MP_IDS list.

ASCII-only output (Windows cp1252 safe).
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from physics_bulk_modulus import validate, VALIDATION_MP_IDS

# Curated small-cell subset spanning oxide / sulfide / halide so the oxide-vs-
# sulfide ranking is testable without the big LLZO / LGPS / argyrodite cells.
CURATED = [
    ("mp-5840",  "Li2O",     "oxide"),
    ("mp-3834",  "Li3PO4",   "oxide"),
    ("mp-1153",  "Li2S",     "sulfide"),
    ("mp-22905", "LiCl",     "halide"),
]


def main():
    mp_ids = VALIDATION_MP_IDS if "--full" in sys.argv else CURATED
    print(f"Tier 4: benchmarking {len(mp_ids)} SSEs "
          f"({'full set' if mp_ids is VALIDATION_MP_IDS else 'curated small-cell subset'})")

    out = validate(model="auto", mp_ids=mp_ids)

    passed = []

    def check(name, cond, detail=""):
        tag = "PASS" if cond else "FAIL"
        passed.append(bool(cond))
        print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))

    print("\n  Tier 4 assertions:")
    mae = out["mae_pct"]
    # Single-MLIP MACE-MP on a tiny set: allow a looser band than the 15% target
    # but report the exact MAE so regressions are visible.
    check("MAE vs MP is reported and < 30%",
          not np.isnan(mae) and mae < 30.0, f"MAE={mae:.1f}% (target ~10-15%)")
    check("qualitative ranking OK (oxide stiffer than sulfide)",
          out["ranking_ok"], "oxide vs sulfide")

    n_pass, n_tot = sum(passed), len(passed)
    print(f"\n  TIER 4 RESULT: {n_pass}/{n_tot} assertions passed "
          f"(gate.passed={out['passed']})")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
