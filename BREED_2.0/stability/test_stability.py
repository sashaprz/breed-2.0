#!/usr/bin/env python3
"""
test_stability.py
===================

Offline unit tests for the phase-diagram logic in ``stability.py``.

These tests use a small synthetic Li-P-S system built from ``ComputedEntry``
objects with hand-picked energies -- no Materials Project API, no MLIP, and
no spglib (``stability.py``'s Phase-0/relaxation/MP-fetch code paths are not
exercised here; only ``existence_check``, ``electrochemical_window`` and
``interfacial_reactivity``, which only touch
``pymatgen.analysis.phase_diagram`` / ``interface_reactions``).

Toy system
----------
Elements Li, P, S at 0 eV/atom. Compounds (energy/atom):

    Li2S  -1.5     Li3P  -1.0     P2S5  -0.8

Candidate "Li3PS4" at -2.0 eV/atom is the deepest point in the system, so it
sits exactly on the convex hull (e_above_hull = 0) by construction.

All expected numbers below were computed by running the same pymatgen calls
directly against this toy system -- they are not "physical", just
self-consistent fixed points for regression testing.

Run with::

    python test_stability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stability import (  # noqa: E402
    DEFAULT_TOL_EV_ATOM,
    existence_check,
    electrochemical_window,
    interfacial_reactivity,
)

_PASS = "PASS"
_FAIL = "FAIL"
_results = []


def check(name, condition, detail=""):
    tag = _PASS if condition else _FAIL
    _results.append(bool(condition))
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ----------------------------------------------------------------------------- #
#  Toy Li-P-S system                                                             #
# ----------------------------------------------------------------------------- #

def _entry(formula: str, e_per_atom: float):
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry
    comp = Composition(formula)
    return ComputedEntry(comp, e_per_atom * comp.num_atoms)


def _competitors():
    return [
        _entry("Li", 0.0),
        _entry("P", 0.0),
        _entry("S", 0.0),
        _entry("Li2S", -1.5),
        _entry("Li3P", -1.0),
        _entry("P2S5", -0.8),
    ]


def _stable_candidate():
    """Li3PS4 at -2.0 eV/atom -- deepest point in the system, on the hull."""
    return _entry("Li3PS4", -2.0)


# ----------------------------------------------------------------------------- #
#  Calc 1 -- existence_check                                                     #
# ----------------------------------------------------------------------------- #

def test_existence_on_hull():
    res = existence_check(_stable_candidate(), _competitors(), identifier="Li3PS4")
    check("existence: ok", res.ok, res.error or "")
    check("existence: e_above_hull == 0", close(res.e_above_hull_ev_atom, 0.0),
          f"e_above_hull={res.e_above_hull_ev_atom}")
    check("existence: exists == True", res.exists is True)
    check("existence: decomposes_into == [Li3PS4]", res.decomposes_into == ["Li3PS4"],
          str(res.decomposes_into))


def test_existence_decomposes():
    # Li2S at -1.4 eV/atom is 0.1 eV/atom above the competitor Li2S (-1.5).
    competitors = _competitors()
    unstable = _entry("Li2S", -1.4)
    res = existence_check(unstable, competitors, tol=DEFAULT_TOL_EV_ATOM,
                          identifier="Li2S(meta)")
    check("decomposes: ok", res.ok, res.error or "")
    check("decomposes: e_above_hull == 0.1", close(res.e_above_hull_ev_atom, 0.1),
          f"e_above_hull={res.e_above_hull_ev_atom}")
    check("decomposes: exists == False (0.1 > default tol 0.05)", res.exists is False)
    check("decomposes: decomposes_into == [Li2S]", res.decomposes_into == ["Li2S"],
          str(res.decomposes_into))


def test_existence_within_tolerance():
    # Same 0.1 eV/atom case, but with a looser tolerance it should pass.
    unstable = _entry("Li2S", -1.4)
    res = existence_check(unstable, _competitors(), tol=0.15, identifier="Li2S(meta)")
    check("tolerance: ok", res.ok, res.error or "")
    check("tolerance: exists == True with tol=0.15", res.exists is True)


# ----------------------------------------------------------------------------- #
#  Calc 2 -- electrochemical_window                                              #
# ----------------------------------------------------------------------------- #

def test_electrochemical_window():
    res = electrochemical_window(_stable_candidate(), _competitors(), mu_min=-6.0,
                                  step=0.1, identifier="Li3PS4")
    check("ESW: ok", res.ok, res.error or "")

    # Stable window is [0.8, 4.4] V -- does NOT include the 0 V anode point,
    # i.e. a hard pass/fail gate at the anode would (wrongly) kill this
    # candidate even though it's perfectly fine at typical cathode voltages.
    check("ESW: v_reduction_limit == 0.8", close(res.v_reduction_limit, 0.8),
          f"got {res.v_reduction_limit}")
    check("ESW: v_oxidation_limit == 4.4", close(res.v_oxidation_limit, 4.4),
          f"got {res.v_oxidation_limit}")
    check("ESW: window_width_v == 3.6", close(res.window_width_v, 3.6),
          f"got {res.window_width_v}")
    check("ESW: reduction_products == [Li3PS4]", res.reduction_products == ["Li3PS4"],
          str(res.reduction_products))
    # At the oxidation edge e_above_hull ~ 0 but the hull there is degenerate
    # (multiple decompositions tie within float precision); pymatgen reports
    # the P2S5+S combination rather than Li3PS4 itself.
    check("ESW: oxidation_products == [P2S5, S]", res.oxidation_products == ["P2S5", "S"],
          str(res.oxidation_products))

    # Soft penalties at the operating voltages, computed regardless of the
    # pass/fail window above.
    check("ESW: anode_v == 0.0", close(res.anode_v, 0.0))
    check("ESW: cathode_v == 4.0", close(res.cathode_v, 4.0))
    check("ESW: anode_penalty == 1.2 eV/atom (unstable at V=0)",
          close(res.anode_penalty_ev_atom, 1.2), f"got {res.anode_penalty_ev_atom}")
    check("ESW: anode_products == [Li2S, Li3P]",
          res.anode_products == ["Li2S", "Li3P"], str(res.anode_products))
    check("ESW: cathode_penalty == 0.0 eV/atom (stable at V=4)",
          close(res.cathode_penalty_ev_atom, 0.0), f"got {res.cathode_penalty_ev_atom}")
    check("ESW: cathode_products == [Li3PS4]",
          res.cathode_products == ["Li3PS4"], str(res.cathode_products))


def test_electrochemical_window_requires_li():
    no_li = _entry("P2S5", -0.8)
    res = electrochemical_window(no_li, _competitors(), identifier="P2S5")
    check("ESW: no-Li candidate errors out", not res.ok)
    check("ESW: no-Li error mentions 'Li reservoir'",
          "Li reservoir" in (res.error or ""), res.error or "")


# ----------------------------------------------------------------------------- #
#  Calc 3 -- interfacial_reactivity                                              #
# ----------------------------------------------------------------------------- #

def test_interfacial_anode_pure_li():
    """Pure-Li electrode branch (avoids the GrandPotentialInterfacialReactivity
    ZeroDivisionError -- see module docstring)."""
    res = interfacial_reactivity(_stable_candidate(), _competitors(), electrode_comp="Li",
                                  mu_li=0.0, identifier="Li3PS4")
    check("interfacial(anode): ok", res.ok, res.error or "")
    check("interfacial(anode): electrode_formula == Li", res.electrode_formula == "Li")
    check("interfacial(anode): min_reaction_x == 0.5", close(res.min_reaction_x, 0.5),
          f"got {res.min_reaction_x}")
    check("interfacial(anode): min_reaction_energy == -0.375 eV/atom",
          close(res.min_reaction_energy_ev_atom, -0.375),
          f"got {res.min_reaction_energy_ev_atom}")
    check("interfacial(anode): 3 reaction kinks", len(res.reactions) == 3,
          f"got {len(res.reactions)}")


def test_interfacial_cathode_grand_potential():
    """Non-Li electrode at mu_Li=0 (V=0): GrandPotentialInterfacialReactivity
    branch. Li3PS4 reacts strongly with P2S5 under Li-rich conditions."""
    res = interfacial_reactivity(_stable_candidate(), _competitors(), electrode_comp="P2S5",
                                  mu_li=0.0, identifier="Li3PS4")
    check("interfacial(cathode, V=0): ok", res.ok, res.error or "")
    check("interfacial(cathode, V=0): electrode_formula == P2S5",
          res.electrode_formula == "P2S5")
    check("interfacial(cathode, V=0): min_reaction_x == 0.0",
          close(res.min_reaction_x, 0.0), f"got {res.min_reaction_x}")
    check("interfacial(cathode, V=0): min_reaction_energy == -3.5571 eV/atom",
          close(res.min_reaction_energy_ev_atom, -3.557142857142858),
          f"got {res.min_reaction_energy_ev_atom}")
    check("interfacial(cathode, V=0): a reaction kink mentions Li3P",
          any("Li3P" in r["reaction"] for r in res.reactions))


def test_interfacial_cathode_no_reaction_at_operating_voltage():
    """At V=4 (mu_Li=-4), Li3PS4 and P2S5 are both already on the grand-potential
    hull -- essentially no driving force to react."""
    res = interfacial_reactivity(_stable_candidate(), _competitors(), electrode_comp="P2S5",
                                  mu_li=-4.0, identifier="Li3PS4")
    check("interfacial(cathode, V=4): ok", res.ok, res.error or "")
    check("interfacial(cathode, V=4): min_reaction_energy ~ 0",
          close(res.min_reaction_energy_ev_atom, 0.0, tol=1e-6),
          f"got {res.min_reaction_energy_ev_atom}")


# ----------------------------------------------------------------------------- #
#  Driver                                                                        #
# ----------------------------------------------------------------------------- #

def main():
    print("=" * 70)
    print("  STABILITY -- offline phase-diagram tests (no MLIP, no MP API)")
    print("=" * 70)
    print("\nCalc 1 -- existence_check")
    test_existence_on_hull()
    test_existence_decomposes()
    test_existence_within_tolerance()

    print("\nCalc 2 -- electrochemical_window")
    test_electrochemical_window()
    test_electrochemical_window_requires_li()

    print("\nCalc 3 -- interfacial_reactivity")
    test_interfacial_anode_pure_li()
    test_interfacial_cathode_grand_potential()
    test_interfacial_cathode_no_reaction_at_operating_voltage()

    print("-" * 70)
    n_pass = sum(_results)
    n_tot = len(_results)
    print(f"  RESULT: {n_pass}/{n_tot} checks passed")
    print("=" * 70)
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
