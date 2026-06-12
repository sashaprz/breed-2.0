#!/usr/bin/env python3
r"""
physics_bulk_modulus.py
=======================

A *physics-based* bulk-modulus predictor for solid-state electrolytes (SSEs).

This module computes K from first(ish)-principles energetics: a single MLIP
energy surface is sampled by a constrained energy minimization
(volume clamped, every internal coordinate relaxed) at several volumes, and the
resulting E(V) curve is fit to a third-order Birch-Murnaghan equation of state.
The bulk modulus is the curvature of that curve at the minimum.

The whole pipeline is one idea -- constrained energy minimization sampled at
several volumes, fit to a physically-shaped curve, behind a validation gate you
don't cross until the method has proven itself on known answers.

Phases (each is a function/method below, and the docstrings name the traps):

  Phase 0  load_calculator()        Lock one MLIP + float64, used identically everywhere.
  Phase 1  acquire_structure()      Pull DFT-relaxed structure; order any disorder
           order_structure()         (Ewald or enumerate) BEFORE the MLIP sees it;
                                     reduce to primitive.
  Phase 2  relax_structure()        Relax positions-then-cell with FrechetCellFilter to
                                     find the MLIP's own V0; re-symmetrize; check vs MP.
  Phase 3  volume_scan()            Scale V0 to +/-5% (9 pts), relax internals at FIXED cell.
  Phase 4  fit_birch_murnaghan()    3rd-order BM fit; K0' and V0 alarms; stress cross-check.
  Phase 5  validate()              Run the full pipeline on known SSEs from MP and compare.

  Phase 3b volume_scan_qha()        Per-volume phonons (phonopy + MLIP forces) -> F_vib(V,T).
  Phase 4b run_qha()               Assemble F(V,T)=E_static+F_vib; BM fit per T -> K(T), V(T).
           compute_qha()            Orchestrates the finite-T path end to end.
           validate_phonons()       Gate the MLIP's phonons on Si/MgO first.

The static phases 1-4 are a 0 K calculation: they ignore thermal softening. The
QHA phases 3b/4b lift that -- ``compute_qha()`` returns K(T), V(T), thermal
expansion and Grueneisen parameters, so you can compare to room-temperature
experiment. (Still validate the MLIP's phonons on a known crystal first; a model
with great forces can still give garbage force constants.) And for SSEs the
decision-relevant stiffness for dendrite suppression (Monroe-Newman) is the
*shear* modulus, not K -- see ``compute_elastic_tensor()`` at the bottom, which
reuses the same strain machinery to get the full Voigt-Reuss-Hill K and G.

Requires one MLIP backend (and phonopy for the QHA path). None is required to
import this file, but the compute methods fail clearly until one is installed::

    pip install mace-torch          # preferred (set default_dtype=float64)
    pip install sevenn              # SevenNet, good for systematic-bias fixes
    pip install chgnet             # note: float32 only -- noisier 2nd derivative
    pip install matgl              # M3GNet
    pip install phonopy            # required for --qha / --validate-phonons

Usage::

    python physics_bulk_modulus.py --mp-id mp-1234            # static K (0 K)
    python physics_bulk_modulus.py --cif path/to/structure.cif
    python physics_bulk_modulus.py --validate                 # Phase-5 gate
    python physics_bulk_modulus.py --validate-phonons Si       # phonon gate first
    python physics_bulk_modulus.py --mp-id mp-1234 --qha       # K(T), finite-T
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ----------------------------------------------------------------------------- #
#  Constants                                                                     #
# ----------------------------------------------------------------------------- #

# 1 eV/Angstrom^3 = 160.21766208 GPa.  Applied once, at the very end.
EV_A3_TO_GPA = 160.21766208

# Effective receptive field of a default MACE-MP model (cutoff ~5-6 A times a
# couple of message-passing layers).  Cells smaller than this in any direction
# let atoms interact with their own periodic images and corrupt the energy.
DEFAULT_RECEPTIVE_FIELD = 12.0  # Angstrom

SCRIPT_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- #
#  Result container                                                              #
# ----------------------------------------------------------------------------- #

@dataclass
class BulkModulusResult:
    """Everything the pipeline learned about one material.

    ``bulk_modulus_gpa`` is the answer; the rest are the built-in diagnostics
    the spec insists you keep -- they are how you know whether to trust it.
    """
    identifier: str
    bulk_modulus_gpa: Optional[float] = None        # K0 from the BM fit
    k0_prime: Optional[float] = None                # pressure derivative (sanity: ~3.5-4.5)
    v0_fit_a3: Optional[float] = None               # BM-fit equilibrium volume
    v0_relaxed_a3: Optional[float] = None           # Phase-2 relaxed volume
    e0_ev: Optional[float] = None
    bm_residual_rms_mev: Optional[float] = None     # fit quality, meV/atom
    stress_bulk_modulus_gpa: Optional[float] = None # independent -V dP/dV check
    n_scan_points: int = 0
    volumes_a3: list = field(default_factory=list)
    energies_ev: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    ok: bool = False
    error: Optional[str] = None

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {self.identifier}: {msg}")

    def summary(self) -> str:
        if not self.ok:
            return f"{self.identifier}: FAILED ({self.error})"
        agree = ""
        if self.stress_bulk_modulus_gpa is not None and self.bulk_modulus_gpa:
            d = abs(self.stress_bulk_modulus_gpa - self.bulk_modulus_gpa)
            agree = (f"  stress-check={self.stress_bulk_modulus_gpa:.1f} GPa "
                     f"(d={100*d/self.bulk_modulus_gpa:.1f}%)")
        return (f"{self.identifier}: K0={self.bulk_modulus_gpa:.1f} GPa  "
                f"K0'={self.k0_prime:.2f}  "
                f"V0_fit/V0_relaxed={self.v0_fit_a3:.2f}/{self.v0_relaxed_a3:.2f} A^3  "
                f"fit_rms={self.bm_residual_rms_mev:.2f} meV/atom{agree}")


# ============================================================================= #
#  PHASE 0 -- Environment and model, locked down once.                          #
# ============================================================================= #

def assert_frechet_available() -> Callable:
    """Confirm FrechetCellFilter exists (it replaced the less-stable ExpCellFilter).

    Older ASE puts it in ``ase.constraints``; newer in ``ase.filters``.
    """
    try:
        from ase.filters import FrechetCellFilter
        return FrechetCellFilter
    except ImportError:
        pass
    try:
        from ase.constraints import FrechetCellFilter  # older ASE
        return FrechetCellFilter
    except ImportError as exc:
        raise RuntimeError(
            "FrechetCellFilter not found in this ASE. Upgrade ASE "
            "(`pip install -U ase`); ExpCellFilter is intentionally not used "
            "here because it is less numerically stable for cell relaxation."
        ) from exc


def load_calculator(model: str = "auto", dtype: str = "float64"):
    """Build ONE MLIP calculator and return ``(calc, model_label)``.

    The single most important rule of the whole pipeline: use the *identical*
    calculator and settings for the initial relaxation and for every scan point.
    Mixing models, or relaxing in float32 and scanning in float64, puts the EOS
    curve on a different energy surface than the minimum you found, and the
    curvature -- a second derivative, so it amplifies noise twice -- becomes
    meaningless.  float32 energy noise is ~meV-scale: invisible for relaxation,
    fatal for K.  Hence ``dtype="float64"`` for anything feeding the EOS.

    ``model="auto"`` tries MACE -> SevenNet -> CHGNet -> M3GNet, first hit wins.
    Pass an explicit name to pin it.  Call this ONCE and thread the result
    through every phase.
    """
    order = (["mace", "sevennet", "chgnet", "m3gnet"]
             if model == "auto" else [model.lower()])
    errors = []

    for name in order:
        try:
            if name == "mace":
                from mace.calculators import mace_mp
                # dispersion=False to match the plain training surface; if you
                # ever switch to the +D3 variant you must use it for BOTH the
                # relaxation and the scan, never one of each.
                calc = mace_mp(model="medium", default_dtype=dtype,
                               dispersion=False)
                return calc, f"mace-medium/{dtype}"

            if name == "sevennet":
                from sevenn.calculator import SevenNetCalculator
                calc = SevenNetCalculator(model="7net-0")
                return calc, "sevennet-7net-0"

            if name == "chgnet":
                from chgnet.model.dynamics import CHGNetCalculator
                if dtype == "float64":
                    warnings.warn(
                        "CHGNet runs in float32; its meV-scale energy noise "
                        "degrades the second derivative. Prefer MACE/SevenNet "
                        "in float64 for production K values.")
                calc = CHGNetCalculator()
                return calc, "chgnet/float32"

            if name == "m3gnet":
                import matgl
                from matgl.ext.ase import PESCalculator
                pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
                calc = PESCalculator(pot)
                return calc, "m3gnet-MP-2021.2.8"

            errors.append(f"{name}: unknown backend name")
        except Exception as exc:  # ImportError or load failure
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No MLIP backend available. Install one of:\n"
        "  pip install mace-torch   (preferred, float64)\n"
        "  pip install sevenn\n"
        "  pip install chgnet\n"
        "  pip install matgl\n"
        "Detection log:\n  " + "\n  ".join(errors))


# ============================================================================= #
#  PHASE 1 -- Structure acquisition, where SSEs specifically betray you.        #
# ============================================================================= #

_OCC_SNAP_TOL = 0.02  # occupancy within this of 1.0 treated as refinement noise


def _apply_occupancy_tolerance(structure, tol: float = _OCC_SNAP_TOL,
                                res: Optional[BulkModulusResult] = None):
    """Snap near-full single-species sites to occ 1.0; return (struct, is_disordered).

    Refinement software emits 0.97-0.99 for fully occupied sites; is_ordered
    rejects any sub-1.0 occupancy as disorder. This pre-pass separates noise
    from real multi-species mixing before the ordering branch executes.
    A site with 0.5 Li / 0.5 Na (two distinct species) is left untouched.
    """
    to_snap = []
    for i, site in enumerate(structure):
        sp_occ = dict(site.species.items())
        if len(sp_occ) == 1:
            el, frac = next(iter(sp_occ.items()))
            # +1e-9 so the documented boundary value snaps despite IEEE-754
            # round-off: e.g. 1.0 - 0.98 == 0.020000000000000018 > 0.02 exactly.
            if frac < 1.0 and (1.0 - frac) <= tol + 1e-9:
                to_snap.append((i, el))
    if not to_snap:
        return structure, not structure.is_ordered
    work = structure.copy()
    for i, el in to_snap:
        work.replace(i, el)  # same species, occupancy snapped to 1.0
    if res is not None:
        res.warn(
            f"Snapped {len(to_snap)} site(s) with |1-occ| <= {tol} to occ 1.0 "
            "(refinement noise).")
    return work, not work.is_ordered


def _detect_solid_solution(structure, res: Optional[BulkModulusResult],
                            eq_tol: float = 0.15):
    """Warn when site disorder has near-equal multi-species fractions (solid solution).

    Ewald ordering picks one arrangement and misrepresents random-alloy physics.
    SQS is not implemented here, so this is a warning only.
    """
    for site in structure:
        sp_occ = dict(site.species.items())
        if len(sp_occ) < 2:
            continue
        fracs = list(sp_occ.values())
        if max(fracs) - min(fracs) < eq_tol and max(fracs) < 0.85:
            desc = ", ".join(f"{el}:{f:.2f}" for el, f in sp_occ.items())
            if res is not None:
                res.warn(
                    f"Site [{desc}] looks like a solid solution (near-equal "
                    "fractions of distinct species). Ewald ordering picks ONE "
                    "arrangement and misrepresents random-alloy physics; an SQS "
                    "cell would be more accurate (not implemented here). K from "
                    "this approximant should be treated with extra scepticism.")
            return


def _mlip_rerank(ranked_list, calc, res: Optional[BulkModulusResult] = None):
    """Re-rank Ewald-ordered candidates by MLIP single-point energy.

    Ewald electrostatics assumes formal charges and ignores covalency, so its
    minimum can differ from the MLIP minimum -- particularly for sulfides and
    covalent SSEs. A wrong ground-state ordering feeds the wrong geometry into
    the EOS scan, producing a silently wrong K. Evaluating the top-N Ewald
    candidates with a quick MLIP single-point corrects this.

    Returns the decorated (oxidation states intact) structure with the lowest
    MLIP single-point energy.
    """
    best_struct = ranked_list[0]["structure"]
    best_energy = float("inf")
    ewald_winner_energy: Optional[float] = None
    n_failed = 0

    for i, entry in enumerate(ranked_list):
        try:
            atoms = _to_ase(entry["structure"])  # strips oxi states for MLIP
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            if i == 0:
                ewald_winner_energy = e
            if e < best_energy:
                best_energy = e
                best_struct = entry["structure"]
        except Exception as exc:
            n_failed += 1
            if res is not None:
                res.warn(f"MLIP single-point failed for ordering candidate {i}: {exc}")

    if n_failed == len(ranked_list):
        if res is not None:
            res.warn("All MLIP single-points failed; keeping Ewald-best ordering.")
        return ranked_list[0]["structure"]

    if (ewald_winner_energy is not None and
            best_energy < ewald_winner_energy - 0.05 and
            res is not None):
        delta = ewald_winner_energy - best_energy
        res.warn(
            f"MLIP re-ranking changed the best ordering (Ewald-best was NOT the "
            f"MLIP ground state; delta={delta:.3f} eV across {len(ranked_list)} "
            "candidates). Likely covalent/sulfide bonding. K from this ordering "
            "should be treated with extra care.")

    return best_struct


def order_structure(structure, method: str = "ewald", calc=None, n_top: int = 3,
                    res: Optional[BulkModulusResult] = None):
    """Hand the rest of the pipeline a structure with DEFINITE atoms.

    This runs once, before anything else; its only job is to remove partial site
    occupancies before the MLIP ever sees them. An MLIP needs a definite atom at
    every site -- feed it fractional occupancy and you either crash or silently
    get a nonsense guess. Superionic conductors are intrinsically disordered
    (LLZO has fractional Li sites, argyrodites have anion mixing, Na-beta-alumina
    is *defined* by it), so this matters constantly for SSEs.

    Pre-pass: _apply_occupancy_tolerance separates refinement noise (0.98, 0.99
    on a single species) from real multi-species disorder before is_ordered runs.

    Two ordering entry points:
      ``method="ewald"`` (default) -- OrderDisorderedStructureTransformation:
          enlarges the cell to make occupancies integer, ranks by Ewald, then
          re-ranks the top-N by MLIP single-point (when ``calc`` is provided)
          to correct Ewald's formal-charge assumption for covalent/sulfide SSEs.
      ``method="enumerate"`` -- EnumerateStructureTransformation:
          enumerates symmetry-distinct orderings; needs enumlib binaries on PATH.

    Either way the result is an *approximant*, so distrust K accordingly.
    """
    # Fix 1: occupancy-tolerance pre-pass -- separates noise from real disorder.
    structure, is_truly_disordered = _apply_occupancy_tolerance(structure, res=res)
    if not is_truly_disordered:
        return structure

    # Fix 7: warn when disorder looks like a solid solution (SQS territory).
    _detect_solid_solution(structure, res)

    if res is not None:
        res.warn(
            f"disordered -> ordered approximant via '{method}' "
            "(distrust K accordingly; consider a DFT spot-check).")

    n_candidates = n_top if calc is not None else 1

    if method == "enumerate":
        from pymatgen.transformations.advanced_transformations import (
            EnumerateStructureTransformation)
        work = structure.copy()
        # Fix 3: guess oxidation states once; hard error rather than silent skip.
        try:
            work.add_oxidation_state_by_guess()
        except Exception as exc:
            raise RuntimeError(
                f"Oxidation state guess failed for enumerate ordering: {exc}. "
                "Supply explicit oxidation states or use method='ewald'.") from exc
        try:
            est = EnumerateStructureTransformation(max_cell_size=2)
            ranked = est.apply_transformation(work, return_ranked_list=n_candidates)
            if not isinstance(ranked, list):
                ranked = [{"structure": ranked, "energy": 0.0}]
        except Exception as exc:
            raise RuntimeError(
                "EnumerateStructureTransformation failed (needs enumlib "
                f"`enum.x`/`makestr.x` on PATH). Try method='ewald'. "
                f"Error: {exc}") from exc
        # Fix 2: MLIP re-rank to correct Ewald != covalent ground state.
        ordered = (_mlip_rerank(ranked, calc, res=res)
                   if calc is not None and len(ranked) > 1
                   else ranked[0]["structure"])
        # Fix 6: strip oxi states only here; they were needed through ranking.
        ordered.remove_oxidation_states()
        return ordered

    # --- ewald (default) ---
    # Fix 3: guess oxidation states ONCE on the original composition; hard-fail
    # so mixed-valence compounds surface immediately rather than producing
    # inconsistent guesses across supercell sizes.
    oxi_struct = structure.copy()
    try:
        oxi_struct.add_oxidation_state_by_guess()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot assign oxidation states for Ewald ordering: {exc}. "
            "For mixed-valence compositions supply explicit oxidation states "
            "or use method='enumerate'.") from exc
    oxi_map: dict = {}
    for sp in oxi_struct.composition.elements:
        sym = sp.symbol
        oxi = getattr(sp, "oxi_state", 0)
        if sym not in oxi_map:  # keep first (dominant) valence per element
            oxi_map[sym] = oxi

    from pymatgen.transformations.standard_transformations import (
        OrderDisorderedStructureTransformation)
    odt = OrderDisorderedStructureTransformation()

    # Fix 4: hard ceiling at 2x2x2 with explicit RuntimeError if nothing resolves.
    scalings = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                (2, 2, 1), (2, 1, 2), (1, 2, 2), (2, 2, 2)]
    last_exc = None
    for sc in scalings:
        work = structure.copy()
        if sc != (1, 1, 1):
            work.make_supercell(list(sc))
        # Fix 3: apply pre-computed oxi map to each supercell (no re-guessing).
        try:
            work.add_oxidation_state_by_element(oxi_map)
        except Exception as exc:
            last_exc = f"oxi-state assignment failed at supercell {sc}: {exc}"
            continue
        try:
            # Fix 2: top-N Ewald candidates for MLIP re-ranking.
            ranked = odt.apply_transformation(work, return_ranked_list=n_candidates)
            if not isinstance(ranked, list):
                ranked = [{"structure": ranked, "energy": 0.0}]
        except ValueError as exc:
            last_exc = exc  # occupancies not integer-consistent at this supercell
            continue

        # Fix 2 + Fix 6: MLIP re-rank with oxi states still on the candidates;
        # _to_ase (inside _mlip_rerank) strips them for energy evals only.
        ordered = (_mlip_rerank(ranked, calc, res=res)
                   if calc is not None and len(ranked) > 1
                   else ranked[0]["structure"])
        ordered.remove_oxidation_states()
        return ordered

    raise RuntimeError(
        "Could not order disordered structure within a 2x2x2 supercell "
        "(hard cell-size ceiling reached). Occupancies may require a larger "
        f"cell or an SQS representation. Last error: {last_exc}")


def _flag_solvent(structure, res: Optional[BulkModulusResult]):
    """Flag molecular fragments / solvent -- a red flag for a co-crystal/solvate
    rather than a crystalline SSE framework."""
    if res is None:
        return
    elements = {str(e) for e in structure.composition.elements}
    if {"H", "C"} <= elements and "N" not in elements:
        res.warn("structure contains both H and C -- check for solvent/organic "
                 "fragments before trusting the elastic response.")


def to_primitive(structure, symprec: float = 1e-3):
    """Reduce to the primitive cell (fewer atoms, faster, same physics)."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        prim = sga.get_primitive_standard_structure()
        if prim is not None and len(prim) > 0:
            return prim
    except Exception:
        pass
    return structure


def acquire_structure(mp_id: Optional[str] = None,
                      cif_path: Optional[str] = None,
                      structure=None,
                      api_key: Optional[str] = None,
                      res: Optional[BulkModulusResult] = None,
                      order_method: str = "ewald",
                      reduce_primitive: bool = True,
                      calc=None):
    """Phase 1: obtain a clean, ordered, primitive pymatgen ``Structure``.

    Exactly one of ``mp_id`` / ``cif_path`` / ``structure`` should be given.
    Disordered inputs are ordered up front by ``order_structure`` (the cell may
    grow to make occupancies integer); ``order_method`` selects ewald|enumerate.
    Pass ``calc`` to enable MLIP re-ranking of Ewald ordering candidates (fix 2).
    """
    from pymatgen.core import Structure

    if res is None:
        res = BulkModulusResult(identifier=str(mp_id or cif_path or "structure"))

    if structure is not None:
        struct = structure.copy()
    elif cif_path is not None:
        struct = Structure.from_file(cif_path)
    elif mp_id is not None:
        from mp_api.client import MPRester
        key = api_key or os.environ.get("MP_API_KEY")
        if not key:
            raise RuntimeError("No Materials Project API key (set MP_API_KEY or "
                               "pass api_key=...).")
        with MPRester(key) as mpr:
            struct = mpr.get_structure_by_material_id(mp_id)
    else:
        raise ValueError("Provide one of mp_id, cif_path, or structure.")

    # Order before the MLIP ever sees the structure; branch is inside.
    was_ordered = struct.is_ordered
    struct = order_structure(struct, method=order_method, calc=calc, res=res)
    _flag_solvent(struct, res)
    # Only reduce to primitive when the input was already ordered: an ordered
    # approximant we just built is a specific supercell arrangement we don't want
    # to re-standardize away.
    if reduce_primitive and was_ordered:
        struct = to_primitive(struct)
    return struct, res


# ============================================================================= #
#  ASE <-> pymatgen plumbing + supercell guard                                  #
# ============================================================================= #

def _to_ase(structure):
    from pymatgen.io.ase import AseAtomsAdaptor
    s = structure.copy()
    s.remove_oxidation_states()  # fix 6: strip at ASE handoff, not earlier
    return AseAtomsAdaptor.get_atoms(s)


def _to_pmg(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_structure(atoms)


def ensure_supercell(atoms, receptive_field: float = DEFAULT_RECEPTIVE_FIELD,
                     res: Optional[BulkModulusResult] = None):
    """Expand tiny cells so no atom sees its own periodic image.

    If the primitive cell is smaller than the model's effective receptive field
    (cutoff x message-passing layers, ~10-12 A for default MACE) along any
    direction, atoms interact with their own images and the energy is corrupted.
    Most SSE cells are big enough; small high-symmetry ones (a 2-atom primitive)
    are not. We replicate until every lattice vector's perpendicular width meets
    the field.
    """
    cell = atoms.get_cell()
    # Perpendicular widths (distance between opposite faces), not vector norms.
    vol = abs(np.linalg.det(np.array(cell)))
    widths = []
    cell_arr = np.array(cell)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cross = np.cross(cell_arr[j], cell_arr[k])
        widths.append(vol / (np.linalg.norm(cross) + 1e-12))
    reps = [max(1, int(math.ceil(receptive_field / w))) for w in widths]
    if any(r > 1 for r in reps):
        if res is not None:
            res.warn(f"small cell (widths {[f'{w:.1f}' for w in widths]} A) < "
                     f"receptive field {receptive_field} A -> supercell {reps}")
        atoms = atoms.repeat(reps)
    return atoms


# ============================================================================= #
#  PHASE 2 -- Initial relaxation, finding the MLIP's own V0.                     #
# ============================================================================= #

def relax_structure(atoms, calc, fmax: float = 0.02, steps: int = 500,
                    mp_volume: Optional[float] = None,
                    mp_n_atoms: Optional[int] = None,
                    res: Optional[BulkModulusResult] = None,
                    resymmetrize: bool = True):
    """Phase 2: relax positions then cell to the MLIP's own equilibrium.

    Strategy that avoids the coupled cell+position oscillation trap: relax
    internal positions at fixed cell first, *then* enable cell relaxation with
    FrechetCellFilter. We ASSERT fmax was actually reached rather than that the
    step cap was hit -- a relaxation that merely ran out of steps has residual
    forces that become scatter on the E(V) curve.

    A large volume disagreement (>10%) with the MP DFT input is the earliest
    warning that this MLIP is unreliable on this chemistry; we record it and you
    should distrust the eventual K.
    """
    from ase.optimize import BFGS
    FrechetCellFilter = assert_frechet_available()

    atoms = atoms.copy()
    atoms.calc = calc

    # Stage 1: internal coordinates only, fixed cell (stabilizes the start).
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)

    # Stage 2: cell + positions together.
    flt = FrechetCellFilter(atoms)
    opt = BFGS(flt, logfile=None)
    opt.run(fmax=fmax, steps=steps)

    final_fmax = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max()))
    if final_fmax > fmax * 1.5:
        if res is not None:
            res.warn(f"relaxation did not reach fmax: {final_fmax:.3f} > {fmax} "
                     f"eV/A (hit step cap). Curvature will be noisy.")

    if mp_volume is not None:
        # Compare volume PER ATOM. ensure_supercell may have replicated the cell
        # (e.g. 4x4x4 on a 2-atom Si primitive), so raw cell volumes are NOT
        # comparable to MP's -- doing so reports nonsense like "6330% disagreement"
        # on every supercelled cell. mp_n_atoms lets us normalize both sides; if
        # it is absent we fall back to the raw comparison (valid only when the
        # caller guarantees the same cell size).
        if mp_n_atoms:
            v_mlip = atoms.get_volume() / max(1, len(atoms))
            v_mp = mp_volume / mp_n_atoms
        else:
            v_mlip, v_mp = atoms.get_volume(), mp_volume
        dv = abs(v_mlip - v_mp) / v_mp
        if dv > 0.10 and res is not None:
            res.warn(f"MLIP relaxed volume disagrees with MP by {100*dv:.1f}% "
                     f"(>10%): MLIP may be unreliable on this chemistry.")

    # Re-symmetrize so tiny numerical symmetry-breaking doesn't seed a crooked scan.
    # Fix 5: after snapping to the refined/primitive cell, re-settle positions
    # (short fixed-cell relax) so spglib snapping doesn't leave residual forces.
    # Also warn rather than silently swallow failure so the caller knows the true
    # symmetry of the cell that goes into the scan.
    if resymmetrize:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            struct = _to_pmg(atoms)
            refined = SpacegroupAnalyzer(struct, symprec=1e-2).get_refined_structure()
            refined = to_primitive(refined)
            new_atoms = _to_ase(refined)
            new_atoms.calc = calc
            # Re-settle positions after symmetry snapping (positions-only, fixed cell).
            from ase.optimize import BFGS
            BFGS(new_atoms, logfile=None).run(fmax=fmax, steps=100)
            atoms = new_atoms
        except Exception as exc:
            if res is not None:
                res.warn(
                    f"Re-symmetrization failed ({type(exc).__name__}: {exc}); "
                    "continuing with un-symmetrized cell.")

    return atoms


# ============================================================================= #
#  PHASE 3 -- The volume scan, the heart of it.                                 #
# ============================================================================= #

def _scale_to_volume(atoms, target_volume: float):
    """Return a copy scaled isotropically (f^(1/3) on the cell) to target_volume,
    carrying the atoms with it (scale_atoms=True)."""
    a = atoms.copy()
    f = (target_volume / a.get_volume()) ** (1.0 / 3.0)
    a.set_cell(np.array(a.get_cell()) * f, scale_atoms=True)
    return a


def _relax_internal_fixed_cell(atoms, calc, fmax: float, steps: int):
    """Relax ONLY internal coordinates with the cell frozen.

    The trap, stated plainly: do NOT wrap this in any cell filter. A cell filter
    lets the volume drift back toward V0, collapsing the curvature to garbage.
    The cell is frozen; only fractional coordinates move. Skipping this step
    entirely measures frustrated atoms and biases K too stiff -- worst exactly on
    the low-symmetry framework anions SSEs care about.
    """
    from ase.optimize import BFGS
    a = atoms.copy()
    a.calc = calc
    opt = BFGS(a, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return a


def _relax_shape_const_volume(atoms, calc, fmax: float, steps: int,
                              res: Optional["BulkModulusResult"] = None,
                              label: str = ""):
    """Relax cell SHAPE and ions while holding VOLUME constant.

    For QHA only the volume should be clamped, not the cell shape. If the shape
    is frozen rigid (as in the static scan), a low-symmetry ordered approximant
    carries spurious shear stress straight into the force constants, biasing
    K(T). FrechetCellFilter(constant_volume=True) relaxes the shape and ions but
    holds V fixed -- exactly the constraint we want here.
    """
    from ase.optimize import BFGS
    FrechetCellFilter = assert_frechet_available()
    a = atoms.copy()
    a.calc = calc
    v0 = a.get_volume()
    flt = FrechetCellFilter(a, constant_volume=True)
    BFGS(flt, logfile=None).run(fmax=fmax, steps=steps)
    drift = abs(a.get_volume() - v0) / v0
    if res is not None and drift > 1e-3:
        res.warn(f"constant-volume shape relax drifted V by {100*drift:.2f}% "
                 f"at {label or 'a scan point'} (should be ~0).")
    return a


def volume_scan(relaxed_atoms, calc, strain: float = 0.05, n_points: int = 9,
                fmax: float = 0.02, steps: int = 300,
                res: Optional[BulkModulusResult] = None):
    """Phase 3: sample E(V) on a grid of volumes around the relaxed V0.

    Each point is scaled INDEPENDENTLY from the same V0 structure (not marched
    point-to-point), which is parallelizable and immune to path-dependent
    hysteresis. The window is +/-5% volume: wider pulls in anharmonicity or
    crosses a phase boundary; narrower lets relaxation noise dominate.

    After collecting the curve we check it is smooth and convex. A kink means an
    internal relaxation dropped into a different minimum at that volume and the
    structure changed character -- such points are flagged so you can trim them
    or narrow the window.
    """
    v0 = relaxed_atoms.get_volume()
    factors = np.linspace(1.0 - strain, 1.0 + strain, n_points)
    volumes, energies = [], []

    for fct in factors:
        scaled = _scale_to_volume(relaxed_atoms, v0 * fct)
        relaxed_pt = _relax_internal_fixed_cell(scaled, calc, fmax=fmax, steps=steps)
        volumes.append(relaxed_pt.get_volume())
        energies.append(relaxed_pt.get_potential_energy())

    volumes = np.array(volumes)
    energies = np.array(energies)
    order = np.argsort(volumes)
    volumes, energies = volumes[order], energies[order]

    _check_curve_smoothness(volumes, energies, res)
    return volumes, energies


def _check_curve_smoothness(volumes, energies, res: Optional[BulkModulusResult]):
    """Flag non-convexity / kinks in E(V) (the plot-it-every-time discipline)."""
    if res is None or len(volumes) < 5:
        return
    # Discrete second derivative; for a convex well it should be > 0 everywhere.
    d2 = np.gradient(np.gradient(energies, volumes), volumes)
    if np.any(d2 <= 0):
        bad = volumes[d2 <= 0]
        res.warn(f"E(V) is not convex at V~{np.round(bad, 2).tolist()} A^3 "
                 f"(d2E/dV2 <= 0): possible kink / internal-minimum switch. "
                 f"Inspect the curve; trim those points or narrow the window.")


# ============================================================================= #
#  PHASE 4 -- The EOS fit and its built-in alarms.                              #
# ============================================================================= #

def birch_murnaghan_energy(V, E0, V0, B0, B0p):
    """Third-order Birch-Murnaghan E(V). B0 in eV/Å³, V in Å³."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0) ** 3 * B0p + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))


def fit_birch_murnaghan(volumes, energies, n_atoms: int = 1,
                        v0_relaxed: Optional[float] = None,
                        res: Optional[BulkModulusResult] = None):
    """Phase 4: fit 3rd-order BM and read off K0, plus its two free diagnostics.

    K0 is the answer. The fit also hands you alarms for free:
      * V0_fit should match the Phase-2 relaxed volume; drift means the scan
        didn't straddle the true minimum and the fit is extrapolating.
      * K0' (pressure derivative) should land near 3.5-4.5 for nearly all
        solids; a wild value means non-convex or noisy data, not a real material.

    Returns ``(K0_GPa, K0', V0_fit, E0, rms_residual_meV_per_atom)``.
    """
    from scipy.optimize import curve_fit

    volumes = np.asarray(volumes, float)
    energies = np.asarray(energies, float)

    # Physically-motivated initial guesses.
    e0_0 = float(energies.min())
    v0_0 = float(volumes[np.argmin(energies)]) if v0_relaxed is None else v0_relaxed
    b0_0 = 0.5   # eV/Å³ ~ 80 GPa, a soft-ish SSE starting point
    b0p_0 = 4.0

    popt, _ = curve_fit(birch_murnaghan_energy, volumes, energies,
                        p0=[e0_0, v0_0, b0_0, b0p_0], maxfev=20000)
    e0, v0_fit, b0, b0p = popt
    k0_gpa = b0 * EV_A3_TO_GPA

    resid = energies - birch_murnaghan_energy(volumes, *popt)
    rms_mev = 1000.0 * float(np.sqrt(np.mean(resid ** 2))) / max(1, n_atoms)

    if res is not None:
        if v0_relaxed and abs(v0_fit - v0_relaxed) / v0_relaxed > 0.02:
            res.warn(f"BM V0 ({v0_fit:.2f}) drifts from relaxed V0 "
                     f"({v0_relaxed:.2f}) by "
                     f"{100*abs(v0_fit-v0_relaxed)/v0_relaxed:.1f}% (>2%): scan "
                     f"may not straddle the true minimum; re-center and rerun.")
        if not (3.0 <= b0p <= 5.0):
            res.warn(f"K0'={b0p:.2f} is outside the physical 3.5-4.5 window: "
                     f"data is likely non-convex or noisy, not a real material.")

    return k0_gpa, b0p, v0_fit, e0, rms_mev


def stress_bulk_modulus(relaxed_atoms, calc, delta: float = 0.01,
                        fmax: float = 0.02, steps: int = 300):
    """Independent two-point cross-check: K = -V0 (dP/dV) from the stress tensor.

    Compute pressure from the stress tensor at +/-1% volume (internals relaxed,
    cell fixed) and apply your own -V(ΔP/ΔV). It should agree with the BM K0
    within a few percent. Disagreement localizes the problem: if the two-point
    and nine-point methods diverge, the energies are noisy or the curve has
    structure the BM fit is smearing over.
    """
    v0 = relaxed_atoms.get_volume()
    data = []
    for s in (-delta, +delta):
        scaled = _scale_to_volume(relaxed_atoms, v0 * (1.0 + s))
        a = _relax_internal_fixed_cell(scaled, calc, fmax=fmax, steps=steps)
        stress = a.get_stress(voigt=True)            # eV/Å³, ASE sign convention
        pressure = -(stress[0] + stress[1] + stress[2]) / 3.0
        data.append((a.get_volume(), pressure))
    (v_m, p_m), (v_p, p_p) = data
    k_ev_a3 = -v0 * (p_p - p_m) / (v_p - v_m)
    return k_ev_a3 * EV_A3_TO_GPA


# ============================================================================= #
#  PHASE 3b/4b -- Quasi-harmonic approximation: finite-temperature K(T).        #
#                                                                               #
#  At 0 K the static EOS above ignores thermal softening. The QHA fixes that:   #
#  at each scanned volume, add a phonon calculation on the relaxed cell to get   #
#  the vibrational free energy F_vib(V,T); then F(V,T) = E_static + F_vib(V,T),  #
#  and a Birch-Murnaghan fit to F-vs-V at each T gives V(T) (the minimum) and    #
#  K(T) (the curvature). The MLIP makes the displacement-force evaluations       #
#  nearly free -- this is the step that is ruinous with DFT and routine here.    #
# ============================================================================= #

@dataclass
class QHAResult:
    """Finite-temperature output: K(T), V(T), thermal expansion, Grueneisen.

    Trust K(T) (from the per-T Birch-Murnaghan fit) over the thermal_expansion
    and gruneisen fields: those come from numerical T- and V-differentiation,
    which magnifies the per-volume F_vib noise. They are reported for context,
    not as primary deliverables.
    """
    identifier: str
    temperatures: list = field(default_factory=list)
    bulk_modulus_gpa: list = field(default_factory=list)   # K(T) -- the deliverable
    volume_a3: list = field(default_factory=list)          # V(T)
    thermal_expansion: list = field(default_factory=list)  # alpha(T), 1/K (noisy)
    gruneisen: Optional[list] = None                       # (noisy; see docstring)
    k0_static_gpa: Optional[float] = None
    n_volumes: int = 0
    n_dropped: int = 0
    from_disorder_approximant: bool = False
    warnings: list = field(default_factory=list)
    ok: bool = False
    error: Optional[str] = None

    def k_at(self, temperature: float) -> Optional[float]:
        if not self.temperatures:
            return None
        return float(np.interp(temperature, self.temperatures, self.bulk_modulus_gpa))

    def summary(self) -> str:
        if not self.ok:
            return f"{self.identifier}: QHA FAILED ({self.error})"
        k0 = self.bulk_modulus_gpa[0] if self.bulk_modulus_gpa else None
        k300 = self.k_at(300.0)
        k0s = f"{k0:.1f}" if k0 is not None else "N/A"
        k3s = f"{k300:.1f}" if k300 is not None else "N/A"
        s = (f"{self.identifier}: K({self.temperatures[0]:.0f}K)={k0s} GPa  "
             f"K(300K)={k3s} GPa  "
             f"[{self.n_volumes} volumes, {self.n_dropped} dropped]")
        if self.from_disorder_approximant:
            s += ("  | ordering-approximant: every phonon used ONE ordering "
                  "snapshot -- K(T) is no more trustworthy than that choice")
        return s


def _require_phonopy():
    """Confirm phonopy is importable; return (Phonopy, PhonopyQHA)."""
    try:
        from phonopy import Phonopy, PhonopyQHA
        return Phonopy, PhonopyQHA
    except ImportError as exc:
        raise RuntimeError(
            "phonopy not installed -- needed for the QHA finite-temperature path. "
            "Install it with `pip install phonopy`.") from exc


def phonon_supercell_matrix(atoms, min_length: float = 14.0,
                            res: Optional[BulkModulusResult] = None,
                            label: str = ""):
    """Diagonal phonon supercell so forces decay before wrapping (~10-15 A/side).

    Check BEFORE you multiply: the phonon supercell is built on the ALREADY
    enlarged ordered cell, so disorder and QHA supercells multiply -- a 2x
    disorder supercell times a phonon supercell gets large fast. If the ordered
    cell is already big it may satisfy the cutoff at multiplier 1, and this
    returns the identity.
    """
    cell = np.array(atoms.get_cell())
    vol = abs(np.linalg.det(cell))
    widths = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cross = np.cross(cell[j], cell[k])
        widths.append(vol / (np.linalg.norm(cross) + 1e-12))
    reps = [max(1, int(math.ceil(min_length / w))) for w in widths]
    if res is not None and any(r > 1 for r in reps):
        n_cells = reps[0] * reps[1] * reps[2]
        res.warn(f"phonon supercell {reps} ({n_cells}x) on {label or 'cell'} "
                 f"(widths {[f'{w:.1f}' for w in widths]} A, target {min_length} A); "
                 f"disorder x phonon supercells multiply -- watch total atom count.")
    return np.diag(reps).tolist()


def phonon_thermal_properties(unit_atoms, calc, *, min_supercell_length: float = 14.0,
                              displacement: float = 0.01, mesh: float = 50.0,
                              t_min: float = 0.0, t_max: float = 1000.0,
                              t_step: float = 10.0, imaginary_tol: float = 0.1,
                              acoustic_tol: float = 0.3, symmetrize_fc: bool = True,
                              res: Optional[BulkModulusResult] = None,
                              label: str = "") -> dict:
    """Phonons for ONE (fixed-cell, internally relaxed) structure -> F_vib(T).

    Builds the phonon supercell, lets phonopy generate the symmetry-distinct
    displacements, evaluates forces for each with the MLIP, and hands them back
    for force constants and thermal properties. Returns temperatures, the
    Helmholtz free energy (kJ/mol), Cv and entropy (J/K/mol), and the min/max
    frequency so callers can police imaginary modes.

    Failure mode this guards: MLIPs are trained on energies and forces, not
    second derivatives, so a model can give great forces yet garbage force
    constants -- which surface as imaginary frequencies. See validate_phonons().
    """
    Phonopy, _ = _require_phonopy()
    from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

    ph_unit = get_phonopy_structure(_to_pmg(unit_atoms))
    smat = phonon_supercell_matrix(unit_atoms, min_supercell_length, res=res, label=label)
    phonon = Phonopy(ph_unit, supercell_matrix=smat)
    phonon.generate_displacements(distance=displacement)

    force_sets = []
    for sc in phonon.supercells_with_displacements:
        if sc is None:
            continue
        ase_sc = _to_ase(get_pmg_structure(sc))
        ase_sc.calc = calc
        force_sets.append(np.asarray(ase_sc.get_forces()))
    phonon.forces = force_sets
    phonon.produce_force_constants()

    # Acoustic sum rule: numerical force noise otherwise gives nonzero/imaginary
    # acoustic modes at Gamma, which dominate F_vib and bias K(T). Symmetrizing
    # the force constants enforces permutation symmetry AND translational
    # invariance (the ASR).
    if symmetrize_fc:
        try:
            phonon.symmetrize_force_constants()
        except Exception:
            pass

    # Verify the 3 acoustic branches go to ~0 at Gamma.
    phonon.run_qpoints([[0.0, 0.0, 0.0]])
    gamma_freqs = np.sort(np.asarray(phonon.get_qpoints_dict()["frequencies"])[0])
    gamma_acoustic_max = float(np.max(np.abs(gamma_freqs[:3])))
    if res is not None and gamma_acoustic_max > acoustic_tol:
        res.warn(f"acoustic modes at Gamma not ~0 ({gamma_acoustic_max:.2f} THz > "
                 f"{acoustic_tol} THz) at {label or 'a scan point'}: acoustic sum "
                 f"rule weakly satisfied; F_vib and K(T) biased by low-freq noise.")

    phonon.run_mesh(mesh)
    freqs = np.asarray(phonon.get_mesh_dict()["frequencies"])  # THz, (nq, nbands)
    min_freq = float(freqs.min())
    max_freq = float(freqs.max())

    phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    tp = phonon.get_thermal_properties_dict()
    return {
        "temperatures": np.asarray(tp["temperatures"]),
        "free_energy": np.asarray(tp["free_energy"]),       # kJ/mol per unit cell
        "entropy": np.asarray(tp["entropy"]),               # J/K/mol
        "heat_capacity": np.asarray(tp["heat_capacity"]),   # J/K/mol
        "min_frequency_thz": min_freq,
        "max_frequency_thz": max_freq,
        "gamma_acoustic_max_thz": gamma_acoustic_max,
        "has_imaginary": min_freq < -abs(imaginary_tol),
        "n_unitcell_atoms": len(unit_atoms),
        "supercell_matrix": smat,
    }


def volume_scan_qha(relaxed_atoms, calc, *, strain_low: float = 0.02,
                    strain_high: float = 0.06, n_points: int = 9,
                    fmax: float = 0.003, steps: int = 800,
                    min_supercell_length: float = 14.0, displacement: float = 0.01,
                    mesh: float = 50.0, t_min: float = 0.0, t_max: float = 1000.0,
                    t_step: float = 10.0, res: Optional[BulkModulusResult] = None):
    """Phase 3b: at each volume relax internals TIGHT, then compute phonons.

    Three things differ from the static scan and all matter:
      * fmax is tightened to ~0.001-0.005 eV/A (default 0.003). Force constants
        are second derivatives, so residual forces become spurious -- often
        imaginary -- phonons.
      * only the VOLUME is clamped; the cell SHAPE and ions relax
        (constant-volume FrechetCellFilter). Freezing the shape rigid would feed
        spurious shear stress into the force constants of low-symmetry cells.
      * the volume grid is biased toward expansion (default -2% to +6%) so V(T)
        at the top temperature stays inside the grid; if it lands outside, the
        free-energy minimum is an extrapolation.

    Imaginary-frequency policy:
      * COMPRESSED volume (factor < 1): drop that volume from the QHA set.
      * EQUILIBRIUM (point nearest factor 1.0): raise -- the ordering or
        relaxation is wrong; go back to the front end, do not paper over it.

    Returns (volumes, static_energies, thermal_props, n_dropped) with the three
    arrays aligned and the dropped points removed.
    """
    v0 = relaxed_atoms.get_volume()
    factors = np.linspace(1.0 - strain_low, 1.0 + strain_high, n_points)
    eq_index = int(np.argmin(np.abs(factors - 1.0)))

    volumes, energies, thermals = [], [], []
    n_dropped = 0
    for i, fct in enumerate(factors):
        scaled = _scale_to_volume(relaxed_atoms, v0 * fct)
        pt = _relax_shape_const_volume(scaled, calc, fmax=fmax, steps=steps,
                                       res=res, label=f"factor {fct:.3f}")
        tp = phonon_thermal_properties(
            pt, calc, min_supercell_length=min_supercell_length,
            displacement=displacement, mesh=mesh, t_min=t_min, t_max=t_max,
            t_step=t_step, res=res,
            label=f"V={pt.get_volume():.1f}A^3 (factor {fct:.3f})")

        if tp["has_imaginary"]:
            if i == eq_index:
                raise RuntimeError(
                    f"imaginary phonons at equilibrium volume (min freq "
                    f"{tp['min_frequency_thz']:.2f} THz): the ordering or "
                    f"relaxation is wrong -- fix the front end before QHA.")
            if fct < 1.0:
                if res is not None:
                    res.warn(f"imaginary phonons at compressed V (factor {fct:.3f}, "
                             f"min freq {tp['min_frequency_thz']:.2f} THz) -> "
                             f"dropping this volume from the QHA set.")
                n_dropped += 1
                continue
            if res is not None:
                res.warn(f"imaginary phonons at expanded V (factor {fct:.3f}, "
                         f"min freq {tp['min_frequency_thz']:.2f} THz) -> kept; "
                         f"inspect K(T) near the grid edge before trusting it.")

        volumes.append(pt.get_volume())
        energies.append(pt.get_potential_energy())
        thermals.append(tp)

    if len(volumes) < 5:
        raise RuntimeError(f"only {len(volumes)} usable volumes after dropping "
                           f"imaginary-mode points; QHA needs >= 5. Widen/shift "
                           f"the grid or fix the model.")
    return np.array(volumes), np.array(energies), thermals, n_dropped


def run_qha(volumes, static_energies, thermal_props, *, eos: str = "birch_murnaghan",
            t_max: Optional[float] = None) -> dict:
    """Phase 4b: assemble F(V,T) and fit BM at each T -> K(T), V(T), expansion.

    This is exactly what phonopy-qha automates: feed it the static energies and
    the per-volume thermal properties and it emits K(T), thermal expansion, and
    Grueneisen parameters. PhonopyQHA returns the bulk modulus directly in GPa.
    """
    _, PhonopyQHA = _require_phonopy()

    temperatures = np.asarray(thermal_props[0]["temperatures"])
    n_t, n_v = len(temperatures), len(volumes)
    free_energy = np.zeros((n_t, n_v))
    cv = np.zeros((n_t, n_v))
    entropy = np.zeros((n_t, n_v))
    for j, tp in enumerate(thermal_props):
        free_energy[:, j] = tp["free_energy"]
        cv[:, j] = tp["heat_capacity"]
        entropy[:, j] = tp["entropy"]

    # PhonopyQHA wants volumes increasing.
    order = np.argsort(volumes)
    volumes = np.asarray(volumes)[order]
    static_energies = np.asarray(static_energies)[order]
    free_energy = free_energy[:, order]
    cv = cv[:, order]
    entropy = entropy[:, order]

    if t_max is None:
        t_max = float(temperatures.max())

    qha = PhonopyQHA(volumes=volumes, electronic_energies=static_energies,
                     temperatures=temperatures, free_energy=free_energy,
                     cv=cv, entropy=entropy, eos=eos, t_max=t_max)

    def _get(attr, method):
        if hasattr(qha, attr):
            val = getattr(qha, attr)
            return val() if callable(val) else val
        return getattr(qha, method)()

    k_t = np.asarray(_get("bulk_modulus_temperature", "get_bulk_modulus_temperature"))
    v_t = np.asarray(_get("volume_temperature", "get_volume_temperature"))
    alpha = np.asarray(_get("thermal_expansion", "get_thermal_expansion"))
    try:
        grun = np.asarray(_get("gruneisen_temperature", "get_gruneisen_temperature"))
    except Exception:
        grun = None
    try:
        k0_static = float(_get("bulk_modulus", "get_bulk_modulus"))
    except Exception:
        k0_static = None

    # QHA differentiates numerically, so K(T)/alpha(T) drop the last temperature(s);
    # align everything to the shortest array.
    n = min(len(k_t), len(v_t), len(alpha))
    return {
        "temperatures": temperatures[:n].tolist(),
        "bulk_modulus_gpa": k_t[:n].tolist(),
        "volume_a3": v_t[:n].tolist(),
        "thermal_expansion": alpha[:n].tolist(),
        "gruneisen": (grun[:n].tolist() if grun is not None else None),
        "k0_static_gpa": k0_static,
    }


# Validation crystals. cv300_per_atom is the experimental isochoric heat
# capacity at 300 K in J/(K.mol) PER ATOM (~19-20 for both, near 3R); it gates on
# the low-frequency region the way omega_max alone cannot. max_thz is kept only
# as context, not as the pass/fail criterion.
_PHONON_REFERENCE = {
    "Si":  {"a": 5.43, "proto": "diamond",  "max_thz": 15.5,
            "cv300_per_atom": 19.8, "cv_tol": 3.0},
    "MgO": {"a": 4.21, "proto": "rocksalt", "max_thz": 21.0,
            "cv300_per_atom": 18.9, "cv_tol": 3.0},
}


def _value_at(tp: dict, key: str, temperature: float = 300.0) -> float:
    return float(np.interp(temperature, tp["temperatures"], tp[key]))


def validate_phonons(engine, material: str = "Si", min_supercell_length: float = 14.0,
                     mesh: float = 50.0, displacements=(0.01, 0.03),
                     acoustic_tol: float = 0.3) -> dict:
    """Validate the MLIP's phonons against a known crystal BEFORE trusting SSEs.

    MLIPs are trained on energies and forces, not second derivatives, so a model
    can give great forces and still produce garbage force constants. Run this on
    Si or MgO first. The gate is deliberately NOT omega_max alone (a model can
    nail the top optical mode and miss the low-frequency branches that drive
    F_vib); it is:
      * no imaginary modes,
      * acoustic branches ~0 at Gamma (acoustic sum rule actually satisfied),
      * Cv(300 K) per atom near the reference (probes the low-frequency region).

    It also reports two convergence diagnostics you should eyeball once per
    chemistry: sensitivity to the finite-displacement amplitude (bumpy MLIP
    forces make a single ~0.01 A step noisy) and q-mesh convergence of F_vib.
    """
    from ase.build import bulk
    ref = _PHONON_REFERENCE[material]
    print(f"\n  Phonon validation on {material} (proto {ref['proto']}):")
    atoms = bulk(material, ref["proto"], a=ref["a"])
    atoms = ensure_supercell(atoms, engine.receptive_field)
    relaxed = relax_structure(atoms, engine.calc, fmax=0.003)

    def _run(disp, msh):
        return phonon_thermal_properties(
            relaxed, engine.calc, min_supercell_length=min_supercell_length,
            displacement=disp, mesh=msh, t_max=300.0, t_step=50.0,
            acoustic_tol=acoustic_tol)

    base = _run(displacements[0], mesh)
    cv300 = _value_at(base, "heat_capacity") / base["n_unitcell_atoms"]

    no_imaginary = not base["has_imaginary"]
    acoustic_ok = base["gamma_acoustic_max_thz"] <= acoustic_tol
    cv_ok = abs(cv300 - ref["cv300_per_atom"]) <= ref["cv_tol"]
    passed = no_imaginary and acoustic_ok and cv_ok

    print(f"    min freq        = {base['min_frequency_thz']:.2f} THz "
          f"({'no imaginary' if no_imaginary else 'IMAGINARY MODES'})")
    print(f"    Gamma acoustic  = {base['gamma_acoustic_max_thz']:.2f} THz "
          f"(<= {acoustic_tol}: {'OK' if acoustic_ok else 'ASR VIOLATED'})")
    print(f"    Cv(300K)/atom   = {cv300:.1f} J/K/mol "
          f"(ref ~{ref['cv300_per_atom']} +/- {ref['cv_tol']}: "
          f"{'OK' if cv_ok else 'OFF -- low-freq region wrong'})")
    print(f"    max freq        = {base['max_frequency_thz']:.2f} THz "
          f"(context only, ref ~{ref['max_thz']})")

    # Displacement-amplitude sensitivity.
    disp_note = ""
    if len(displacements) > 1:
        alt = _run(displacements[1], mesh)
        cv_alt = _value_at(alt, "heat_capacity") / alt["n_unitcell_atoms"]
        d_max = abs(alt["max_frequency_thz"] - base["max_frequency_thz"])
        d_cv = abs(cv_alt - cv300)
        disp_note = ("OK" if (d_max < 0.5 and d_cv < 1.0)
                     else "SENSITIVE -- forces noisy; average amplitudes")
        print(f"    disp {displacements[0]}->{displacements[1]} A: "
              f"d(max freq)={d_max:.2f} THz, d(Cv300)={d_cv:.2f} -> {disp_note}")

    # q-mesh convergence of F_vib(300 K).
    fine = _run(displacements[0], mesh * 2.0)
    f_base = _value_at(base, "free_energy")
    f_fine = _value_at(fine, "free_energy")
    d_f = abs(f_fine - f_base)
    mesh_note = "OK" if d_f < 0.5 else "UNDER-CONVERGED -- use a finer mesh"
    print(f"    mesh {mesh:.0f}->{mesh*2:.0f}: dF_vib(300K)={d_f:.3f} kJ/mol "
          f"-> {mesh_note}")

    print(f"    PHONON GATE: {'PASSED' if passed else 'NOT PASSED -- fix the model'}")
    return {"material": material, "passed": passed,
            "min_frequency_thz": base["min_frequency_thz"],
            "gamma_acoustic_max_thz": base["gamma_acoustic_max_thz"],
            "cv300_per_atom": cv300, "max_frequency_thz": base["max_frequency_thz"]}


# ============================================================================= #
#  Orchestrator                                                                  #
# ============================================================================= #

class PhysicsBulkModulus:
    """One locked-down MLIP, threaded through phases 1-4 for any structure.

    Build it once (loads the calculator once), then call ``compute(...)`` per
    material so every structure sees the identical energy surface.
    """

    def __init__(self, model: str = "auto", dtype: str = "float64",
                 strain: float = 0.05, n_points: int = 9,
                 fmax: float = 0.02, receptive_field: float = DEFAULT_RECEPTIVE_FIELD,
                 api_key: Optional[str] = None):
        self.strain = strain
        self.n_points = n_points
        self.fmax = fmax
        self.receptive_field = receptive_field
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        self.calc, self.model_label = load_calculator(model=model, dtype=dtype)
        print(f"[physics_bulk_modulus] MLIP locked: {self.model_label}")

    def compute(self, mp_id: Optional[str] = None, cif_path: Optional[str] = None,
                structure=None, mp_volume: Optional[float] = None,
                mp_n_atoms: Optional[int] = None,
                order_method: str = "ewald",
                do_stress_check: bool = True) -> BulkModulusResult:
        ident = str(mp_id or cif_path or
                    (structure.composition.reduced_formula if structure else "structure"))
        res = BulkModulusResult(identifier=ident)
        try:
            # Phase 1 -- pass calc so order_structure can MLIP-rerank candidates.
            struct, res = acquire_structure(
                mp_id=mp_id, cif_path=cif_path, structure=structure,
                api_key=self.api_key, res=res,
                order_method=order_method, calc=self.calc)

            atoms = _to_ase(struct)
            atoms = ensure_supercell(atoms, self.receptive_field, res)

            # Phase 2
            relaxed = relax_structure(atoms, self.calc, fmax=self.fmax,
                                      mp_volume=mp_volume, mp_n_atoms=mp_n_atoms,
                                      res=res)
            res.v0_relaxed_a3 = float(relaxed.get_volume())
            n_atoms = len(relaxed)

            # Phase 3
            volumes, energies = volume_scan(
                relaxed, self.calc, strain=self.strain, n_points=self.n_points,
                fmax=self.fmax, res=res)
            res.volumes_a3 = volumes.tolist()
            res.energies_ev = energies.tolist()
            res.n_scan_points = len(volumes)

            # Phase 4
            k0, k0p, v0_fit, e0, rms = fit_birch_murnaghan(
                volumes, energies, n_atoms=n_atoms,
                v0_relaxed=res.v0_relaxed_a3, res=res)
            res.bulk_modulus_gpa = k0
            res.k0_prime = k0p
            res.v0_fit_a3 = v0_fit
            res.e0_ev = e0
            res.bm_residual_rms_mev = rms

            if do_stress_check:
                try:
                    res.stress_bulk_modulus_gpa = stress_bulk_modulus(
                        relaxed, self.calc, fmax=self.fmax)
                    if res.bulk_modulus_gpa:
                        disagree = abs(res.stress_bulk_modulus_gpa -
                                       res.bulk_modulus_gpa) / res.bulk_modulus_gpa
                        if disagree > 0.10:
                            res.warn(f"stress check ({res.stress_bulk_modulus_gpa:.1f} "
                                     f"GPa) disagrees with BM ({res.bulk_modulus_gpa:.1f} "
                                     f"GPa) by {100*disagree:.0f}%: noisy energies "
                                     f"or curve structure the fit smears over.")
                except Exception as exc:
                    res.warn(f"stress cross-check failed: {exc}")

            res.ok = True
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            res.ok = False
        return res

    def compute_qha(self, mp_id: Optional[str] = None, cif_path: Optional[str] = None,
                    structure=None, order_method: str = "ewald",
                    phonon_fmax: float = 0.003,
                    strain_low: float = 0.02, strain_high: float = 0.06,
                    min_supercell_length: float = 14.0, mesh: float = 50.0,
                    displacement: float = 0.01, t_min: float = 0.0,
                    t_max: float = 1000.0, t_step: float = 10.0,
                    eos: str = "birch_murnaghan") -> QHAResult:
        """Finite-temperature K(T) via the quasi-harmonic approximation.

        Same front end as compute() (phases 1-2), but the per-volume relaxation
        is tighter, the volume grid is biased toward expansion, and each volume
        also gets a phonon calculation. Returns a QHAResult.
        """
        ident = str(mp_id or cif_path or
                    (structure.composition.reduced_formula if structure else "structure"))
        qres = QHAResult(identifier=ident)
        bres = BulkModulusResult(identifier=ident)  # collects phase-1/2/scan warnings
        try:
            # Phase 1 -- pass calc so order_structure can MLIP-rerank candidates.
            struct, bres = acquire_structure(
                mp_id=mp_id, cif_path=cif_path, structure=structure,
                api_key=self.api_key, res=bres, order_method=order_method,
                calc=self.calc)
            atoms = ensure_supercell(_to_ase(struct), self.receptive_field, bres)

            # Phase 2 -- relax tight (force constants are second derivatives).
            relaxed = relax_structure(atoms, self.calc, fmax=phonon_fmax, res=bres)

            # Phase 3b -- volume scan with phonons.
            volumes, energies, thermals, n_dropped = volume_scan_qha(
                relaxed, self.calc, strain_low=strain_low, strain_high=strain_high,
                n_points=self.n_points, fmax=phonon_fmax,
                min_supercell_length=min_supercell_length, displacement=displacement,
                mesh=mesh, t_min=t_min, t_max=t_max, t_step=t_step, res=bres)

            # Phase 4b -- QHA assembly.
            out = run_qha(volumes, energies, thermals, eos=eos, t_max=t_max)
            qres.temperatures = out["temperatures"]
            qres.bulk_modulus_gpa = out["bulk_modulus_gpa"]
            qres.volume_a3 = out["volume_a3"]
            qres.thermal_expansion = out["thermal_expansion"]
            qres.gruneisen = out["gruneisen"]
            qres.k0_static_gpa = out["k0_static_gpa"]
            qres.n_volumes = len(volumes)
            qres.n_dropped = n_dropped

            # Recheck the grid still BRACKETS the minimum at every T (not just
            # that >=5 volumes survived). Dropping compressed points can leave
            # V(T) at low T unbracketed; V(T) at high T can exceed the grid.
            vmin, vmax = float(np.min(volumes)), float(np.max(volumes))
            vt = np.asarray(qres.volume_a3)
            if vt.size and (vt.min() <= vmin + 1e-6 or vt.max() >= vmax - 1e-6):
                temps = np.asarray(qres.temperatures)
                edge_T = temps[(vt <= vmin + 1e-6) | (vt >= vmax - 1e-6)]
                bres.warn(
                    f"V(T) reaches the volume-grid edge "
                    f"(grid [{vmin:.1f}, {vmax:.1f}] A^3, V(T) in "
                    f"[{vt.min():.1f}, {vt.max():.1f}]) at T~"
                    f"{np.round(edge_T[[0, -1]], 0).tolist()} K: the free-energy "
                    f"minimum is not bracketed there, so K(T) is extrapolated. "
                    f"Extend the grid (esp. on the side where points were dropped).")

            # Fix-8: surface the disorder-approximant caveat (set during Phase 1).
            qres.from_disorder_approximant = any(
                "ordered approximant" in w for w in bres.warnings)

            # Fix-7: numerical V/T differentiation amplifies F_vib noise.
            if qres.gruneisen is not None:
                bres.warn("thermal_expansion / gruneisen come from numerical V- "
                          "and T-differentiation and amplify per-volume F_vib "
                          "noise; trust K(T) from the BM fit over them.")
            qres.ok = True
        except Exception as exc:
            qres.error = f"{type(exc).__name__}: {exc}"
            qres.ok = False
        qres.warnings = bres.warnings
        return qres


# ============================================================================= #
#  PHASE 5 -- The validation gate. Do not skip, do not screen before passing.   #
# ============================================================================= #

# Known SSE-relevant materials with MP elastic data. The qualitative ranking
# (oxides stiff ~100 GPa, sulfides soft ~tens of GPa) is the hard-to-fake bar.
VALIDATION_MP_IDS = [
    # mp-id,         label,                    family
    ("mp-942733",   "Li7La3Zr2O12 (LLZO)",     "oxide"),
    ("mp-3834",     "Li3PO4",                  "oxide"),
    ("mp-5840",     "Li2O",                    "oxide"),
    ("mp-696128",   "Li10GeP2S12 (LGPS)",      "sulfide"),
    ("mp-1185319",  "Li6PS5Cl (argyrodite)",   "sulfide"),
    ("mp-1153",     "Li2S",                    "sulfide"),
    ("mp-22905",    "LiCl",                    "halide"),
    ("mp-23268",    "Li3InCl6-like",           "halide"),
]


def fetch_mp_bulk_modulus(mp_id: str, api_key: Optional[str] = None):
    """Pull the MP DFT VRH bulk modulus yourself rather than trusting memory.

    Returns ``(k_vrh_GPa, formula, dft_volume_A3, n_atoms)`` (any field may be
    None). n_atoms pairs with dft_volume_A3 so callers can compare volume PER
    ATOM against an MLIP supercell.
    """
    from mp_api.client import MPRester
    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("No MP_API_KEY for validation.")
    with MPRester(key) as mpr:
        try:
            docs = mpr.materials.elasticity.search(
                material_ids=[mp_id],
                fields=["material_id", "formula_pretty", "bulk_modulus"])
            k = None
            formula = None
            if docs and docs[0].bulk_modulus is not None:
                k = docs[0].bulk_modulus.vrh
                formula = docs[0].formula_pretty
        except Exception:
            k, formula = None, None
        vol = None
        n_atoms = None
        try:
            struct = mpr.get_structure_by_material_id(mp_id)
            vol = struct.volume
            n_atoms = len(struct)
            formula = formula or struct.composition.reduced_formula
        except Exception:
            pass
    return k, formula, vol, n_atoms


def validate(model: str = "auto", mp_ids=None, api_key: Optional[str] = None,
             **kwargs) -> dict:
    """Phase 5: run the full pipeline on known SSEs and compare to MP DFT.

    Quantitative bar: ~10-15% MAE against MP VRH K (pulled live, not from
    memory). Qualitative bar (more important, harder to fake): oxides must come
    out stiff (~100 GPa) and sulfides soft (tens of GPa), because S-P bonds are
    far more compliant than O bonds. If a sulfide ever reports stiffer than an
    oxide, the model is BROKEN -- stop and fix it before trusting one novel
    prediction.
    """
    entries = mp_ids or VALIDATION_MP_IDS
    engine = PhysicsBulkModulus(model=model, api_key=api_key, **kwargs)
    rows = []

    print("\n" + "=" * 78)
    print("  PHASE 5 VALIDATION GATE")
    print("=" * 78)

    family_means = {}
    for mp_id, label, family in entries:
        k_ref, formula, mp_vol, mp_n = fetch_mp_bulk_modulus(mp_id, api_key=engine.api_key)
        res = engine.compute(mp_id=mp_id, mp_volume=mp_vol, mp_n_atoms=mp_n)
        k_pred = res.bulk_modulus_gpa if res.ok else None
        err_pct = (100 * abs(k_pred - k_ref) / k_ref
                   if (k_pred is not None and k_ref) else None)
        rows.append({"mp_id": mp_id, "label": label, "family": family,
                     "k_ref": k_ref, "k_pred": k_pred, "err_pct": err_pct,
                     "result": res})
        if k_pred is not None:
            family_means.setdefault(family, []).append(k_pred)
        print(f"  {label:28s} ref={_fmt(k_ref)} pred={_fmt(k_pred)} "
              f"err={_fmt(err_pct, '%')}")
        if not res.ok:
            print(f"      -> {res.error}")

    # Quantitative bar
    valid = [r for r in rows if r["err_pct"] is not None]
    mae_pct = np.mean([r["err_pct"] for r in valid]) if valid else float("nan")
    print("-" * 78)
    print(f"  Mean abs % error vs MP: {mae_pct:.1f}%  (target ~10-15%)  "
          f"[{len(valid)}/{len(rows)} computed]")

    # Qualitative bar: sulfide must be softer than oxide
    ok_ranking = True
    if "oxide" in family_means and "sulfide" in family_means:
        ox = np.mean(family_means["oxide"])
        su = np.mean(family_means["sulfide"])
        print(f"  Mean K: oxide={ox:.1f} GPa, sulfide={su:.1f} GPa")
        if su >= ox:
            ok_ranking = False
            print("  *** BROKEN: sulfide predicted stiffer than oxide. "
                  "STOP and fix the model (calibrate or switch to SevenNet) "
                  "before any novel prediction. ***")
        else:
            print("  Qualitative ranking OK (sulfide softer than oxide).")

    passed = (not math.isnan(mae_pct)) and mae_pct <= 15.0 and ok_ranking
    print(f"  GATE: {'PASSED' if passed else 'NOT PASSED'}")
    print("=" * 78)
    return {"rows": rows, "mae_pct": mae_pct, "ranking_ok": ok_ranking,
            "passed": passed}


def _fmt(x, suffix="") -> str:
    return "  N/A " if x is None else f"{x:6.1f}{suffix}"


# ============================================================================= #
#  Beyond K: the full elastic tensor (K AND the shear modulus G).               #
# ============================================================================= #

def compute_elastic_tensor(relaxed_atoms, calc, strain_magnitude: float = 0.005,
                           fmax: float = 0.02):
    """Extend the same strain machinery to the full elastic tensor -> K and G (VRH).

    For SSE dendrite resistance the decision-relevant stiffness is the SHEAR
    modulus, not K -- the Monroe-Newman criterion wants electrolyte G roughly 2x
    lithium's, and K barely speaks to it. So if mechanical screening is the real
    goal, validate the tooling with the bulk-modulus pipeline first (above), then
    use this: apply the six independent strains, read the stress response,
    assemble C_ij, and take the Voigt-Reuss-Hill average for both K and G.

    Returns a dict with ``K_vrh`` and ``G_vrh`` in GPa (and the 6x6 C in GPa).
    Uses ASE's ElasticTensor machinery if available, else a finite-difference
    fallback over the six Voigt strains.
    """
    try:
        from matcalc import ElasticityCalc  # optional convenience path
        ec = ElasticityCalc(calc, fmax=fmax, relax_structure=False)
        out = ec.calc(_to_pmg(relaxed_atoms))
        return {"K_vrh": out["bulk_modulus_vrh"] * EV_A3_TO_GPA,
                "G_vrh": out["shear_modulus_vrh"] * EV_A3_TO_GPA,
                "C_gpa": np.array(out["elastic_tensor"]) * EV_A3_TO_GPA}
    except Exception:
        pass

    # Finite-difference fallback: 6 independent Voigt strains, +/- each.
    base = relaxed_atoms.copy()
    base.calc = calc
    cell0 = np.array(base.get_cell())
    C = np.zeros((6, 6))
    voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    for j, (p, q) in enumerate(voigt):
        stresses = {}
        for sgn in (-1, +1):
            eps = np.zeros((3, 3))
            d = sgn * strain_magnitude
            if p == q:
                eps[p, p] = d
            else:
                eps[p, q] = eps[q, p] = d / 2.0
            defo = np.eye(3) + eps
            a = base.copy()
            a.calc = calc
            a.set_cell(cell0 @ defo.T, scale_atoms=True)
            from ase.optimize import BFGS
            BFGS(a, logfile=None).run(fmax=fmax, steps=200)
            stresses[sgn] = a.get_stress(voigt=True)  # eV/Å³
        dstress = (stresses[1] - stresses[-1]) / (2.0 * strain_magnitude)
        C[:, j] = dstress

    C = 0.5 * (C + C.T) * EV_A3_TO_GPA  # symmetrize, to GPa

    # Voigt-Reuss-Hill averages.
    K_v = ((C[0, 0] + C[1, 1] + C[2, 2]) +
           2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    G_v = ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[0, 2] + C[1, 2]) +
           3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    try:
        S = np.linalg.inv(C)
        K_r = 1.0 / ((S[0, 0] + S[1, 1] + S[2, 2]) +
                     2 * (S[0, 1] + S[0, 2] + S[1, 2]))
        G_r = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) -
                      4 * (S[0, 1] + S[0, 2] + S[1, 2]) +
                      3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    except np.linalg.LinAlgError:
        K_r, G_r = K_v, G_v
    return {"K_vrh": 0.5 * (K_v + K_r), "G_vrh": 0.5 * (G_v + G_r), "C_gpa": C}


# ============================================================================= #
#  CLI                                                                            #
# ============================================================================= #

def _main():
    p = argparse.ArgumentParser(
        description="Physics-based (MLIP + Birch-Murnaghan EOS) bulk modulus.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mp-id", help="Materials Project id, e.g. mp-1234")
    p.add_argument("--cif", help="Path to a CIF file")
    p.add_argument("--validate", action="store_true",
                   help="Run the Phase-5 validation gate on known SSEs")
    p.add_argument("--model", default="auto",
                   help="MLIP backend: auto|mace|sevennet|chgnet|m3gnet")
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--strain", type=float, default=0.05, help="+/- volume window")
    p.add_argument("--points", type=int, default=9, help="number of scan points")
    p.add_argument("--fmax", type=float, default=0.02, help="force convergence eV/A")
    p.add_argument("--api-key", default=None, help="MP API key (or set MP_API_KEY)")
    p.add_argument("--order-method", default="ewald", choices=["ewald", "enumerate"],
                   help="how to order disordered inputs (ewald=cheap default)")
    p.add_argument("--shear", action="store_true",
                   help="Also compute the full elastic tensor (K and G, VRH)")
    # --- QHA (finite-temperature) options ---
    p.add_argument("--qha", action="store_true",
                   help="Quasi-harmonic K(T): per-volume phonons + F(V,T) fit")
    p.add_argument("--validate-phonons", choices=["Si", "MgO"], default=None,
                   help="Validate MLIP phonons on Si/MgO before trusting SSEs")
    p.add_argument("--tmin", type=float, default=0.0, help="QHA min temperature K")
    p.add_argument("--tmax", type=float, default=1000.0, help="QHA max temperature K")
    p.add_argument("--tstep", type=float, default=10.0, help="QHA temperature step K")
    p.add_argument("--strain-low", type=float, default=0.02,
                   help="QHA grid lower bound (compression), default 2%%")
    p.add_argument("--strain-high", type=float, default=0.06,
                   help="QHA grid upper bound (expansion), default 6%%")
    p.add_argument("--phonon-fmax", type=float, default=0.003,
                   help="tight per-volume fmax for QHA (eV/A), default 0.003")
    p.add_argument("--supercell-length", type=float, default=14.0,
                   help="min phonon supercell side (A), default 14")
    p.add_argument("--mesh", type=float, default=50.0, help="phonon q-mesh length")
    args = p.parse_args()

    if args.validate:
        validate(model=args.model, api_key=args.api_key, dtype=args.dtype,
                 strain=args.strain, n_points=args.points, fmax=args.fmax)
        return

    if not (args.mp_id or args.cif or args.validate_phonons):
        p.error("Provide --mp-id, --cif, --validate, or --validate-phonons.")

    # Build the MLIP engine once and reuse it for every requested step.
    engine = PhysicsBulkModulus(model=args.model, dtype=args.dtype,
                                strain=args.strain, n_points=args.points,
                                fmax=args.fmax, api_key=args.api_key)

    if args.validate_phonons:
        validate_phonons(engine, material=args.validate_phonons,
                         min_supercell_length=args.supercell_length, mesh=args.mesh)
        if not (args.mp_id or args.cif):
            return

    if args.qha:
        print("\nQuasi-harmonic K(T) (per-volume phonons + F(V,T) fit)...")
        q = engine.compute_qha(
            mp_id=args.mp_id, cif_path=args.cif, order_method=args.order_method,
            phonon_fmax=args.phonon_fmax, strain_low=args.strain_low,
            strain_high=args.strain_high, min_supercell_length=args.supercell_length,
            mesh=args.mesh, t_min=args.tmin, t_max=args.tmax, t_step=args.tstep)
        print("\n" + q.summary())
        if q.ok:
            print(f"  {'T (K)':>8} {'K (GPa)':>10} {'V (A^3)':>10}")
            temps = np.asarray(q.temperatures)
            for tk in (0.0, 300.0, 600.0, 900.0):
                if temps.size and temps.min() <= tk <= temps.max():
                    k = float(np.interp(tk, temps, q.bulk_modulus_gpa))
                    v = float(np.interp(tk, temps, q.volume_a3))
                    print(f"  {tk:>8.0f} {k:>10.1f} {v:>10.2f}")
        for w in q.warnings:
            print(f"  warning: {w}")
        return

    res = engine.compute(mp_id=args.mp_id, cif_path=args.cif,
                         order_method=args.order_method)
    print("\n" + res.summary())
    for w in res.warnings:
        print(f"  warning: {w}")

    if args.shear and res.ok:
        print("\nComputing full elastic tensor (K and G, VRH)...")
        struct, _ = acquire_structure(mp_id=args.mp_id, cif_path=args.cif,
                                      api_key=engine.api_key,
                                      order_method=args.order_method,
                                      calc=engine.calc)
        atoms = ensure_supercell(_to_ase(struct), engine.receptive_field)
        relaxed = relax_structure(atoms, engine.calc, fmax=args.fmax)
        el = compute_elastic_tensor(relaxed, engine.calc)
        print(f"  K_vrh = {el['K_vrh']:.1f} GPa")
        print(f"  G_vrh = {el['G_vrh']:.1f} GPa  "
              f"(shear is the dendrite-relevant modulus, Monroe-Newman)")


if __name__ == "__main__":
    _main()
