#!/usr/bin/env python3
"""
Tier 3 -- Disorder front end for physics_bulk_modulus.py

Exercises order_structure() / _apply_occupancy_tolerance() WITHOUT an MLIP
(calc=None disables only the MLIP re-ranking of Ewald candidates; the Ewald
ordering itself is pure pymatgen). For each disordered input we confirm the
output is a usable ordered crystal:

  * every site has a single species at occupancy 1.0 (integer occupancy),
  * the composition ratio is preserved,
  * charge neutrality holds for the ordered cell,
  * the cell did not blow past the 2x2x2 hard ceiling.

Plus the near-integer (0.98) refinement-noise path (must snap to occ 1.0 with
no cell growth) and the designed clear-error path for a minimal disordered
primitive whose TOTAL composition is fractional.

ASCII-only console output (Windows cp1252 safe).
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pymatgen.core import Lattice, Structure, Composition

from physics_bulk_modulus import order_structure, _apply_occupancy_tolerance

_results = []


def check(name, condition, detail=""):
    tag = "PASS" if condition else "FAIL"
    _results.append(bool(condition))
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


def assert_integer_occupancy(struct, name):
    bad = []
    for i, site in enumerate(struct):
        occ = dict(site.species.items())
        if len(occ) != 1 or abs(next(iter(occ.values())) - 1.0) > 1e-9:
            bad.append((i, occ))
    check(f"{name}: integer occupancy on all sites",
          struct.is_ordered and not bad,
          f"{len(struct)} sites" if struct.is_ordered else f"bad={bad[:3]}")


def reduced_ratio(comp: Composition) -> dict:
    total = sum(comp.values())
    return {str(el): amt / total for el, amt in comp.items()}


def assert_composition_preserved(before, after, name):
    rb, ra = reduced_ratio(before), reduced_ratio(after)
    same = set(rb) == set(ra) and all(abs(rb[e] - ra[e]) < 1e-6 for e in rb)
    check(f"{name}: composition ratio preserved",
          same, f"{after.reduced_formula}")


def assert_charge_neutral(struct, name):
    s = struct.copy()
    try:
        s.add_oxidation_state_by_guess()
        total = sum(getattr(sp, "oxi_state", 0) * amt
                    for sp, amt in s.composition.items())
        check(f"{name}: charge neutral", abs(total) < 1e-6, f"sum q={total:+.3f}")
    except Exception as exc:
        check(f"{name}: charge neutral", False, f"oxi guess failed: {exc}")


def assert_reasonable_cell(before_n, after, name, max_mult=8):
    mult = len(after) / before_n
    check(f"{name}: cell within 2x2x2 ceiling",
          mult <= max_mult + 1e-9, f"{before_n} -> {len(after)} sites ({mult:.0f}x)")


# ------------------------------------------------------------- NaLiCl2 ------ #

def make_nalicl2():
    """Conventional rocksalt cell, cation site 50/50 Na/Li -> total Na2Li2Cl4.

    Built with an INTEGER total composition (the realistic case): a minimal
    2-site primitive would have a fractional total and is handled separately
    in test E.
    """
    lat = Lattice.cubic(5.5)
    cat = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    ani = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]]
    sp = [{"Na": 0.5, "Li": 0.5}] * 4 + ["Cl"] * 4
    return Structure(lat, sp, cat + ani)


def test_nalicl2():
    print("\nTest A: NaLiCl2 (50/50 Na/Li cation disorder, integer total)")
    s = make_nalicl2()
    print(f"    input: {s.composition.reduced_formula}, "
          f"disordered={not s.is_ordered}, {len(s)} sites")
    check("input is genuinely disordered", not s.is_ordered)
    ordered = order_structure(s, method="ewald", calc=None)
    print(f"    ordered: {ordered.composition.reduced_formula}, {len(ordered)} sites")
    assert_integer_occupancy(ordered, "NaLiCl2")
    assert_composition_preserved(s.composition, ordered.composition, "NaLiCl2")
    assert_charge_neutral(ordered, "NaLiCl2")
    assert_reasonable_cell(len(s), ordered, "NaLiCl2")


# ---------------------------------------- Li2O with Li-sublattice disorder -- #

def make_li2o_disordered():
    """Antifluorite-style Li2O proxy: 8 tetrahedral Li sites at occ 0.5 (-> Li4)
    over an O sublattice (-> O2). Total Li4O2 = Li2O: integer and charge neutral
    (4*+1 + 2*-2 = 0). Stand-in for the LLZO-style fractional-Li resolution
    (true 192-site LLZO needs a large supercell; MP's mp-942733 already ships an
    ORDERED 96-site approximant -- see test D)."""
    lat = Lattice.cubic(4.6)
    tet = [[0.25, 0.25, 0.25], [0.75, 0.25, 0.25], [0.25, 0.75, 0.25],
           [0.25, 0.25, 0.75], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75],
           [0.25, 0.75, 0.75], [0.75, 0.75, 0.75]]
    o = [[0, 0, 0], [0.5, 0.5, 0.5]]
    sp = [{"Li": 0.5}] * 8 + ["O"] * 2
    return Structure(lat, sp, tet + o)


def test_li2o_disordered():
    print("\nTest B: Li2O with fractional Li sublattice (LLZO-style proxy)")
    s = make_li2o_disordered()
    print(f"    input: {s.composition.reduced_formula}, "
          f"disordered={not s.is_ordered}, {len(s)} sites")
    check("input is genuinely disordered", not s.is_ordered)
    ordered = order_structure(s, method="ewald", calc=None)
    print(f"    ordered: {ordered.composition.reduced_formula}, {len(ordered)} sites")
    assert_integer_occupancy(ordered, "Li2O")
    assert_composition_preserved(s.composition, ordered.composition, "Li2O")
    assert_charge_neutral(ordered, "Li2O")
    assert_reasonable_cell(len(s), ordered, "Li2O")
    n_li = int(round(ordered.composition.get("Li", 0)))
    check("Li2O: resolved to integer Li count (4 per cell)",
          n_li == 4, f"n_Li={n_li}")


# -------------------------------------------- real LLZO from MP (network) --- #

def test_llzo_from_mp():
    print("\nTest D: real LLZO mp-942733 through the ordering path")
    key = os.environ.get("MP_API_KEY")
    if not key:
        print("    SKIP: MP_API_KEY not set")
        return
    try:
        from mp_api.client import MPRester
        with MPRester(key) as m:
            s = m.get_structure_by_material_id("mp-942733")
    except Exception as exc:
        print(f"    SKIP: MP fetch failed ({type(exc).__name__}: {exc})")
        return
    print(f"    MP returns: {s.composition.reduced_formula}, {len(s)} sites, "
          f"ordered={s.is_ordered}")
    ordered = order_structure(s, method="ewald", calc=None)
    assert_integer_occupancy(ordered, "LLZO(MP)")
    check("LLZO(MP): composition unchanged by ordering path",
          ordered.composition.reduced_formula == s.composition.reduced_formula,
          ordered.composition.reduced_formula)


# ------------------------------------------ near-integer (0.98) snap -------- #

def test_near_integer_snap():
    print("\nTest C: near-integer occupancy (0.98) is refinement noise -> snap")
    lat = Lattice.cubic(4.0)
    s = Structure(lat, [{"Mg": 0.98}, {"O": 0.98}],
                  [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    snapped, is_disordered = _apply_occupancy_tolerance(s, tol=0.02)
    check("0.98 treated as noise (not disorder)", not is_disordered)
    assert_integer_occupancy(snapped, "snap-0.98")
    check("snap did not enlarge the cell", len(snapped) == len(s),
          f"{len(s)} -> {len(snapped)} sites")
    ordered = order_structure(s.copy(), method="ewald", calc=None)
    check("order_structure returns ordered cell for 0.98 input",
          ordered.is_ordered and len(ordered) == len(s), f"{len(ordered)} sites")
    # A genuine 0.5/0.5 two-species site must NOT be snapped.
    s2 = Structure(lat, [{"Na": 0.5, "Li": 0.5}, "Cl"],
                   [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    _, is_dis2 = _apply_occupancy_tolerance(s2, tol=0.02)
    check("genuine 50/50 site is NOT snapped (still disordered)", is_dis2)


# ------------------- minimal fractional-total primitive: clear error -------- #

def test_fractional_total_primitive():
    print("\nTest E: minimal disordered primitive (fractional total) -> clear error")
    # Na0.5Li0.5Cl1: total composition is non-integer, so pymatgen cannot guess
    # oxidation states. The pipeline must raise a CLEAR RuntimeError pointing to
    # the fix (explicit oxi states / enumerate), not crash with a raw ValueError.
    lat = Lattice.cubic(5.5)
    s = Structure(lat, [{"Na": 0.5, "Li": 0.5}, "Cl"],
                  [[0, 0, 0], [0.5, 0.5, 0.5]])
    try:
        order_structure(s, method="ewald", calc=None)
        check("fractional-total primitive raises a clear error", False,
              "no error raised")
    except RuntimeError as exc:
        msg = str(exc)
        good = ("oxidation states" in msg.lower() and
                ("enumerate" in msg.lower() or "explicit" in msg.lower()))
        check("fractional-total primitive raises a clear, actionable RuntimeError",
              good, msg.split(".")[0])
    except Exception as exc:
        check("fractional-total primitive raises a clear RuntimeError (not raw)",
              False, f"{type(exc).__name__}: {exc}")


def main():
    print("=" * 70)
    print("  TIER 3 -- DISORDER FRONT END (pymatgen Ewald, no MLIP re-rank)")
    print("=" * 70)
    test_nalicl2()
    test_li2o_disordered()
    test_near_integer_snap()
    test_llzo_from_mp()
    test_fractional_total_primitive()
    print("-" * 70)
    n_pass, n_tot = sum(_results), len(_results)
    print(f"  TIER 3 RESULT: {n_pass}/{n_tot} checks passed")
    print("=" * 70)
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
