#!/usr/bin/env python3
r"""
stability.py
=============

Thermodynamic stability screening for SSE candidates via grand-canonical
phase diagrams (pymatgen + Materials Project data).

Three calculations, one physics (grand-canonical thermodynamics), different
reservoirs:

  Calc 1  existence_check()         Does this composition hold together at all?
                                     Closed system, no reservoir.
  Calc 2  electrochemical_window()  Across what voltages does it survive?
                                     Open to a Li reservoir; voltage-dependent.
  Calc 3  interfacial_reactivity()  Does it react with the electrode it touches?
                                     Open to Li, two phases; voltage-dependent.

All three build a ``PhaseDiagram`` / ``GrandPotentialPhaseDiagram`` from
Materials Project entries plus ONE relaxed energy for the candidate
(``candidate_to_entry``, Shared Step 0 -- run once, feeds all three calcs).

------------------------------------------------------------------------------
Section 0 -- energy-scale consistency (read this before changing ``--model``)
------------------------------------------------------------------------------

All three calcs compare the candidate's energy to MP reference phases. MP
entries carry MP2020 corrections (anion corrections, GGA/GGA+U compatibility
shifts) -- real offsets of tens to hundreds of meV/atom. If the candidate's
energy isn't on the same scale, hull distances are WRONG WITH NO ERROR RAISED.

  * CHGNet and M3GNet (matgl) were trained on *corrected* MP energies ->
    their output is ~directly comparable to MP entries pulled "as is".
  * MACE-MP-0 / SevenNet were trained on *uncorrected* energies -> using them
    here without a manual post-hoc correction silently shifts every hull
    distance by tens of meV/atom.

``load_calculator()`` therefore only auto-selects CHGNet or M3GNet. This is
the "(A) pragmatic" strategy from the implementation guide: trust
CHGNet/M3GNet ~= MP-corrected scale, pull MP entries directly as competitors,
relax the candidate, compare. It introduces ~20-40 meV/atom of cross-method
noise -- do not set ``tol`` to a razor-thin value (50 meV/atom is the default
and is itself a deliberately generous starting point, see Calc 1).

Strategy "(B) self-consistent" (relax every competitor with the same MLIP too,
build the hull entirely from MLIP energies) removes that cross-method offset
at the cost of N extra relaxations per chemsys. Not implemented here; if you
need it, relax each competitor's structure with ``candidate_to_entry`` and
substitute those entries for the MP competitors returned by
``StabilityScreen.competitors()``.

------------------------------------------------------------------------------
GA integration
------------------------------------------------------------------------------

    For each candidate:
      1. relax (candidate_to_entry)        -> kill if not converged
      2. Gate A (HARD): existence_check     -> kill if e_above_hull > tol
      3. Gate B: bandgap, bulk modulus (other modules)
      4. ionic conductivity (other module) -- the optimization objective
      5. electrochemical_window +
         interfacial_reactivity             -> SOFT penalty on fitness, ranked
                                                by decomposition-energy magnitude
                                                at operating voltages (NOT pass/fail)

``screen_population()`` implements steps 1, 2 and 5 (the stability-specific
gates); see its docstring.

Requires one MLIP backend::

    pip install chgnet      # preferred -- matches the reference code below
    pip install matgl       # M3GNet, also MP2020-corrected scale
    pip install mp-api pymatgen

Usage::

    python stability.py --cif candidate.cif --all
    python stability.py --cif candidate.cif --existence --tol 0.05
    python stability.py --cif candidate.cif --esw
    python stability.py --cif candidate.cif --interfacial --electrode Li
    python stability.py --cif candidate.cif --interfacial --electrode LiCoO2 --mu-li -4.0
    python stability.py --mp-id mp-1234 --all
    python stability.py --validate

Citations
---------
  Ong, Wang, Kang, Ceder, "Li-Fe-P-O2 Phase Diagram from First Principles",
    Chem. Mater. 2008 -- grand potential phase diagram construction.
  Richards, Miara, Wang, Kim, Ceder, "Interface Stability in Solid-State
    Batteries", Chem. Mater. 2016 -- interfacial reactivity method.
  Zhu, He, Mo, "Origin of Outstanding Stability in Li Solid Electrolytes",
    ACS AMI 2015 -- ESW via grand potential + the kinetic-stabilization
    argument (do not binarize the ESW gate, see electrochemical_window()).
  Sun et al., "The thermodynamic scale of inorganic crystalline metastability",
    Sci. Adv. 2016 -- justifies the ~50 meV/atom default tolerance.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ----------------------------------------------------------------------------- #
#  Constants                                                                     #
# ----------------------------------------------------------------------------- #

# Metastability tolerance, eV/atom. Sun et al. (2016) found most synthesized
# metastable oxides sit within ~36 meV/atom of the hull; 50 meV/atom is a
# common generous cut. This is the GA's explore/exploit knob -- tune on
# purpose, don't leave it at the default without thinking about it.
DEFAULT_TOL_EV_ATOM = 0.050

# Pin to GGA/GGA+U so MP entries stay on the same functional as CHGNet/M3GNet
# training data. Letting r2SCAN entries in alongside CHGNet-scale candidate
# energies reintroduces the Section-0 problem.
DEFAULT_THERMO_TYPES = ["GGA_GGA+U"]

# Same receptive-field guard as physics_bulk_modulus.py: cells smaller than
# this in any direction let atoms interact with their own periodic images.
DEFAULT_RECEPTIVE_FIELD = 12.0  # Angstrom

# MLIPs whose total energies are ~directly comparable to MP2020-corrected
# entries (see module docstring, Section 0).
ENERGY_SCALE_COMPATIBLE_MODELS = ("chgnet", "m3gnet")

# Default operating voltages for the ESW soft-penalty (anode ~ 0 V vs Li/Li+,
# a typical cathode ~ 4 V vs Li/Li+).
DEFAULT_VOLTAGES_OF_INTEREST = (0.0, 4.0)


# ============================================================================= #
#  Result containers                                                            #
# ============================================================================= #

@dataclass
class ExistenceResult:
    """Calc 1 output. ``e_above_hull_ev_atom == 0`` means on the convex hull."""
    identifier: str
    tolerance_ev_atom: float = DEFAULT_TOL_EV_ATOM
    e_above_hull_ev_atom: Optional[float] = None
    exists: Optional[bool] = None
    decomposes_into: list = field(default_factory=list)
    n_competitors: int = 0
    warnings: list = field(default_factory=list)
    ok: bool = False
    error: Optional[str] = None

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {self.identifier}: {msg}")

    def summary(self) -> str:
        if not self.ok:
            return f"{self.identifier}: existence check FAILED ({self.error})"
        verdict = "EXISTS" if self.exists else "DECOMPOSES"
        return (f"{self.identifier}: e_above_hull="
                f"{1000 * self.e_above_hull_ev_atom:.1f} meV/atom "
                f"(tol={1000 * self.tolerance_ev_atom:.0f}) -> {verdict}; "
                f"decomposes_into={self.decomposes_into}")


@dataclass
class ESWResult:
    """Calc 2 output.

    ``v_reduction_limit``/``v_oxidation_limit`` are the contiguous voltage
    window (V vs Li/Li+) over which the candidate sits on the grand-potential
    hull. ``anode_penalty_ev_atom`` / ``cathode_penalty_ev_atom`` are the
    e_above_hull at ``anode_v`` / ``cathode_v`` regardless of pass/fail --
    THIS is what should feed the GA's soft penalty (see module docstring;
    do not binarize this gate, LGPS itself fails a hard window and works via
    kinetic passivation).
    """
    identifier: str
    tolerance_ev_atom: float = DEFAULT_TOL_EV_ATOM
    v_reduction_limit: Optional[float] = None
    v_oxidation_limit: Optional[float] = None
    window_width_v: Optional[float] = None
    reduction_products: list = field(default_factory=list)
    oxidation_products: list = field(default_factory=list)
    anode_v: float = DEFAULT_VOLTAGES_OF_INTEREST[0]
    cathode_v: float = DEFAULT_VOLTAGES_OF_INTEREST[1]
    anode_penalty_ev_atom: Optional[float] = None
    cathode_penalty_ev_atom: Optional[float] = None
    anode_products: list = field(default_factory=list)
    cathode_products: list = field(default_factory=list)
    n_competitors: int = 0
    warnings: list = field(default_factory=list)
    ok: bool = False
    error: Optional[str] = None

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {self.identifier}: {msg}")

    def summary(self) -> str:
        if not self.ok:
            return f"{self.identifier}: ESW FAILED ({self.error})"
        if self.v_reduction_limit is None:
            window = "unstable across full mu_Li sweep"
        else:
            window = f"[{self.v_reduction_limit:.2f}, {self.v_oxidation_limit:.2f}] V"
        return (f"{self.identifier}: ESW={window}  "
                f"anode(@{self.anode_v:.1f}V) penalty="
                f"{1000 * self.anode_penalty_ev_atom:.1f} meV/atom "
                f"-> {self.anode_products}  "
                f"cathode(@{self.cathode_v:.1f}V) penalty="
                f"{1000 * self.cathode_penalty_ev_atom:.1f} meV/atom "
                f"-> {self.cathode_products}")


@dataclass
class InterfacialResult:
    """Calc 3 output. ``min_reaction_energy_ev_atom`` is the worst (most
    negative) reaction energy along the candidate<->electrode tie-line --
    large negative = aggressive interfacial decomposition (bad)."""
    identifier: str
    electrode_formula: str
    mu_li_ev: float
    min_reaction_x: Optional[float] = None
    min_reaction_energy_ev_atom: Optional[float] = None
    reactions: list = field(default_factory=list)
    n_competitors: int = 0
    warnings: list = field(default_factory=list)
    ok: bool = False
    error: Optional[str] = None

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {self.identifier}: {msg}")

    def summary(self) -> str:
        if not self.ok:
            return (f"{self.identifier} vs {self.electrode_formula}: "
                    f"interfacial reactivity FAILED ({self.error})")
        return (f"{self.identifier} vs {self.electrode_formula} "
                f"(mu_Li={self.mu_li_ev:.2f} eV): "
                f"min reaction energy = "
                f"{self.min_reaction_energy_ev_atom:.3f} eV/atom "
                f"at x={self.min_reaction_x:.2f}")


@dataclass
class CandidateStabilityResult:
    """Bundles Shared-Step-0 relaxation + all requested calcs for one candidate."""
    identifier: str
    model_label: str = ""
    converged: bool = False
    existence: Optional[ExistenceResult] = None
    esw: Optional[ESWResult] = None
    interfacial: dict = field(default_factory=dict)  # electrode label -> InterfacialResult
    error: Optional[str] = None

    def summary(self) -> str:
        if self.error:
            return f"{self.identifier}: {self.error}"
        lines = [f"{self.identifier} [{self.model_label}, converged={self.converged}]"]
        if self.existence is not None:
            lines.append("  " + self.existence.summary())
        if self.esw is not None:
            lines.append("  " + self.esw.summary())
        for ir in self.interfacial.values():
            lines.append("  " + ir.summary())
        return "\n".join(lines)


# ============================================================================= #
#  Phase 0 -- one MLIP, locked down, plus candidate -> ComputedStructureEntry.  #
# ============================================================================= #

def assert_frechet_available() -> Callable:
    """Confirm FrechetCellFilter exists (older ASE puts it in ase.constraints)."""
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
            "(`pip install -U ase`).") from exc


def load_calculator(model: str = "auto"):
    """Build ONE MLIP calculator on the MP2020-corrected energy scale.

    See module docstring Section 0 for why this matters. ``model="auto"``
    tries CHGNet then M3GNet. Passing ``model="mace"``/``"sevennet"``
    raises immediately -- those models are off the MP-corrected scale and
    using them here without a manual correction step gives silently-wrong
    hull distances.
    """
    order = list(ENERGY_SCALE_COMPATIBLE_MODELS) if model == "auto" else [model.lower()]
    errors = []

    for name in order:
        if name not in ENERGY_SCALE_COMPATIBLE_MODELS:
            errors.append(
                f"{name}: not on the MP2020-corrected energy scale (see "
                "stability.py module docstring, Section 0). MACE-MP-0 and "
                "SevenNet were trained on UNCORRECTED MP energies; using "
                "them here without a manual post-hoc correction silently "
                "shifts every hull distance. Use 'chgnet' or 'm3gnet'.")
            continue
        try:
            if name == "chgnet":
                from chgnet.model.dynamics import CHGNetCalculator
                return CHGNetCalculator(), "chgnet/float32"
            if name == "m3gnet":
                import matgl
                from matgl.ext.ase import PESCalculator
                pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
                return PESCalculator(pot), "m3gnet-MP-2021.2.8"
        except Exception as exc:  # ImportError or load failure
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No MP2020-corrected-scale MLIP backend available. Install one of:\n"
        "  pip install chgnet   (preferred)\n"
        "  pip install matgl    (M3GNet)\n"
        "Detection log:\n  " + "\n  ".join(errors))


def _to_ase(structure):
    from pymatgen.io.ase import AseAtomsAdaptor
    s = structure.copy()
    s.remove_oxidation_states()
    return AseAtomsAdaptor.get_atoms(s)


def _to_pmg(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_structure(atoms)


def ensure_supercell(atoms, receptive_field: float = DEFAULT_RECEPTIVE_FIELD):
    """Expand tiny cells so no atom sees its own periodic image.

    Same guard as physics_bulk_modulus.py: a candidate's total energy is
    compared against MP hull entries, so a self-interaction artifact here is
    not just noise -- it can move the candidate on or off the hull.
    """
    cell_arr = np.array(atoms.get_cell())
    vol = abs(np.linalg.det(cell_arr))
    widths = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cross = np.cross(cell_arr[j], cell_arr[k])
        widths.append(vol / (np.linalg.norm(cross) + 1e-12))
    reps = [max(1, int(math.ceil(receptive_field / w))) for w in widths]
    if any(r > 1 for r in reps):
        atoms = atoms.repeat(reps)
    return atoms


def candidate_to_entry(structure, calc, fmax: float = 0.05, steps: int = 500,
                       receptive_field: float = DEFAULT_RECEPTIVE_FIELD):
    """Shared Step 0: relax ``structure`` with ``calc``, return
    ``(ComputedStructureEntry, converged_bool)``.

    This single relaxed energy feeds all three calcs -- compute it ONCE per
    candidate. Relaxes positions then cell+positions (avoids the coupled
    cell/position oscillation trap). Non-convergence is treated as an
    automatic kill: don't trust the energy of a relaxation that merely ran
    out of steps.
    """
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    from ase.optimize import BFGS

    try:
        atoms = _to_ase(structure)
        atoms = ensure_supercell(atoms, receptive_field)
        atoms.calc = calc

        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)

        FrechetCellFilter = assert_frechet_available()
        flt = FrechetCellFilter(atoms)
        BFGS(flt, logfile=None).run(fmax=fmax, steps=steps)

        final_fmax = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max()))
        if final_fmax > fmax * 1.5:
            return None, False

        energy = float(atoms.get_potential_energy())
        if not np.isfinite(energy):
            return None, False

        relaxed = _to_pmg(atoms)
        entry = ComputedStructureEntry(relaxed, energy)
        return entry, True
    except Exception:
        return None, False


# ============================================================================= #
#  Structure acquisition                                                        #
# ============================================================================= #

def acquire_structure(mp_id: Optional[str] = None, cif_path: Optional[str] = None,
                      structure=None, api_key: Optional[str] = None):
    """Return an ordered pymatgen ``Structure`` from exactly one of
    ``mp_id`` / ``cif_path`` / ``structure``.

    GA candidates (PyXtal/crossover output) are always ordered. Disordered
    inputs (e.g. raw LLZO from MP) are rejected with a pointer to
    ``bulk_modulus.physics_bulk_modulus.order_structure`` rather than
    re-implementing that machinery here.
    """
    from pymatgen.core import Structure

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

    if not struct.is_ordered:
        raise RuntimeError(
            "Structure is disordered (partial site occupancies). stability.py "
            "expects ordered structures. Order it first with "
            "bulk_modulus.physics_bulk_modulus.order_structure(), then pass "
            "the result as `structure=`.")
    return struct


# ============================================================================= #
#  Calc 1 -- existence check (closed system, energy above hull)                #
# ============================================================================= #

def existence_check(entry, competitors: list, tol: float = DEFAULT_TOL_EV_ATOM,
                    identifier: Optional[str] = None) -> ExistenceResult:
    """Is this a real compound, or does it want to phase-separate?

    ``competitors`` should come from ``StabilityScreen.competitors()`` (all
    entries in the candidate's chemsys, pinned to ``DEFAULT_THERMO_TYPES`` --
    a missing competing phase makes a candidate look falsely stable).
    """
    from pymatgen.analysis.phase_diagram import PhaseDiagram

    ident = identifier or entry.composition.reduced_formula
    res = ExistenceResult(identifier=ident, tolerance_ev_atom=tol,
                          n_competitors=len(competitors))
    try:
        pd = PhaseDiagram(competitors + [entry])
        decomp, e_above_hull = pd.get_decomp_and_e_above_hull(entry)
        res.e_above_hull_ev_atom = float(e_above_hull)
        res.exists = bool(e_above_hull <= tol)
        res.decomposes_into = sorted({e.composition.reduced_formula for e in decomp})
        res.ok = True
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
    return res


# ============================================================================= #
#  Calc 2 -- electrochemical stability window (open Li reservoir)              #
# ============================================================================= #

def _decomp_formulas(decomp) -> list:
    out = set()
    for e in decomp:
        orig = getattr(e, "original_entry", e)
        out.add(orig.composition.reduced_formula)
    return sorted(out)


def electrochemical_window(entry, competitors: list, mu_min: float = -6.0,
                           step: float = 0.05, tol: float = DEFAULT_TOL_EV_ATOM,
                           voltages_of_interest=DEFAULT_VOLTAGES_OF_INTEREST,
                           identifier: Optional[str] = None) -> ESWResult:
    """Sweep mu_Li and find the contiguous voltage window where the candidate
    sits on the grand-potential hull, PLUS soft-penalty decomposition
    energies at ``voltages_of_interest`` (computed regardless of pass/fail).

    Voltage convention: mu_Li = mu_Li^0 - eV, referenced to Li metal
    (mu_Li^0 = 0). Sweeping mu_Li from 0 -> mu_min corresponds to
    V = 0 -> -mu_min. More negative mu_Li = higher voltage = oxidizing.
    """
    from pymatgen.core import Element
    from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, GrandPotPDEntry

    ident = identifier or entry.composition.reduced_formula
    res = ESWResult(identifier=ident, tolerance_ev_atom=tol,
                    n_competitors=len(competitors),
                    anode_v=voltages_of_interest[0], cathode_v=voltages_of_interest[1])

    if Element("Li") not in entry.composition.elements:
        res.error = "ESW is defined w.r.t. a Li reservoir; candidate contains no Li."
        return res

    try:
        all_entries = competitors + [entry]

        mus = np.arange(0.0, mu_min, -step)
        stable_mask, decomps = [], []
        for mu in mus:
            gpd = GrandPotentialPhaseDiagram(all_entries, {Element("Li"): float(mu)})
            gentry = GrandPotPDEntry(entry, {Element("Li"): float(mu)})
            decomp, e_above = gpd.get_decomp_and_e_above_hull(gentry)
            stable_mask.append(e_above <= tol)
            decomps.append(decomp)

        V = -mus
        if any(stable_mask):
            idx = np.where(stable_mask)[0]
            res.v_reduction_limit = float(V[idx.min()])
            res.v_oxidation_limit = float(V[idx.max()])
            res.window_width_v = res.v_oxidation_limit - res.v_reduction_limit
            res.reduction_products = _decomp_formulas(decomps[idx.min()])
            res.oxidation_products = _decomp_formulas(decomps[idx.max()])
        else:
            res.warn(f"unstable across the full mu_Li sweep (V in "
                     f"[0, {-mu_min:.1f}]); see anode/cathode penalties below.")

        # Soft-penalty: e_above_hull at the operating voltages, independent of
        # the pass/fail window above. THIS is what should drive the GA fitness
        # -- a hard ESW gate deletes kinetically-stabilized winners (LGPS).
        for v, e_attr, p_attr in (
            (voltages_of_interest[0], "anode_penalty_ev_atom", "anode_products"),
            (voltages_of_interest[1], "cathode_penalty_ev_atom", "cathode_products"),
        ):
            mu = -float(v)
            gpd = GrandPotentialPhaseDiagram(all_entries, {Element("Li"): mu})
            gentry = GrandPotPDEntry(entry, {Element("Li"): mu})
            decomp, e_above = gpd.get_decomp_and_e_above_hull(gentry)
            setattr(res, e_attr, float(e_above))
            setattr(res, p_attr, _decomp_formulas(decomp))

        res.ok = True
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
    return res


# ============================================================================= #
#  Calc 3 -- interfacial reactivity (candidate vs. a specific electrode)       #
# ============================================================================= #

def interfacial_reactivity(entry, competitors: list, electrode_comp: str = "Li",
                           mu_li: float = 0.0,
                           identifier: Optional[str] = None) -> InterfacialResult:
    """Reaction of ``entry`` against ``electrode_comp`` at chemical potential
    ``mu_li``. ``mu_li=0.0`` -> Li metal anode. For a cathode, pass
    ``mu_li = -V_cathode``.

    Pure-Li electrode special case: ``GrandPotentialInterfacialReactivity``
    strips chempot elements from both compositions before normalizing; if
    the electrode IS the chempot element (Li metal), its stripped composition
    has zero atoms and pymatgen raises ZeroDivisionError. Li metal directly
    DEFINES mu_Li=0 (it doesn't need a grand-potential reservoir abstraction
    layered on top of itself), so this case uses the plain (non-grand)
    ``InterfacialReactivity`` instead -- verified working pattern.
    """
    from pymatgen.core import Composition, Element
    from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram
    from pymatgen.analysis.interface_reactions import (
        InterfacialReactivity, GrandPotentialInterfacialReactivity)

    ident = identifier or entry.composition.reduced_formula
    electrode_composition = Composition(electrode_comp)
    res = InterfacialResult(identifier=ident,
                            electrode_formula=electrode_composition.reduced_formula,
                            mu_li_ev=mu_li, n_competitors=len(competitors))
    try:
        all_entries = competitors + [entry]
        pd = PhaseDiagram(all_entries)

        if set(electrode_composition.elements) <= {Element("Li")}:
            if abs(mu_li) > 1e-9:
                res.warn(f"mu_li={mu_li} ignored for pure-Li electrode "
                         "(Li metal defines mu_Li=0 directly).")
            ir = InterfacialReactivity(c1=entry.composition, c2=electrode_composition,
                                       pd=pd, norm=True, use_hull_energy=True)
        else:
            gpd = GrandPotentialPhaseDiagram(all_entries, {Element("Li"): float(mu_li)})
            ir = GrandPotentialInterfacialReactivity(
                c1=entry.composition, c2=electrode_composition,
                grand_pd=gpd, pd_non_grand=pd, norm=True,
                include_no_mixing_energy=True)

        x_min, e_min = ir.minimum
        res.min_reaction_x = float(x_min)
        res.min_reaction_energy_ev_atom = float(e_min)
        res.reactions = [
            {"atomic_fraction": round(float(x), 4), "reaction": str(rxn),
             "reaction_energy_ev_atom": round(float(e), 4)}
            for _, x, e, rxn, _ in ir.get_kinks()
        ]
        res.ok = True
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
    return res


# ============================================================================= #
#  Orchestrator -- one MLIP + cached MP competitor lookups, GA-facing          #
# ============================================================================= #

class StabilityScreen:
    """One locked-down MLIP + cached per-chemsys MP competitor lists.

    Build once (loads the calculator once), then call ``screen(...)`` or the
    individual ``existence_check`` / ``electrochemical_window`` /
    ``interfacial_reactivity`` methods per candidate. ``get_entries_in_chemsys``
    is the dominant cost of all three calcs; GA candidates sharing a chemsys
    fetch competitors once and reuse them.
    """

    def __init__(self, model: str = "auto", api_key: Optional[str] = None,
                 tol: float = DEFAULT_TOL_EV_ATOM,
                 thermo_types=DEFAULT_THERMO_TYPES,
                 receptive_field: float = DEFAULT_RECEPTIVE_FIELD):
        self.calc, self.model_label = load_calculator(model=model)
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        self.tol = tol
        self.thermo_types = list(thermo_types)
        self.receptive_field = receptive_field
        self._competitor_cache: dict[frozenset, list] = {}
        print(f"[stability] MLIP locked: {self.model_label} "
              f"(MP2020-corrected energy scale, thermo_types={self.thermo_types})")

    def competitors(self, elements) -> list:
        """All MP entries in the chemsys of ``elements``, cached."""
        key = frozenset(str(e) for e in elements)
        if key not in self._competitor_cache:
            from mp_api.client import MPRester
            if not self.api_key:
                raise RuntimeError("No MP API key (set MP_API_KEY or pass api_key=...).")
            with MPRester(self.api_key) as mpr:
                entries = mpr.get_entries_in_chemsys(
                    sorted(key), additional_criteria={"thermo_types": self.thermo_types})
            self._competitor_cache[key] = entries
            print(f"  [stability] fetched {len(entries)} competitor entries "
                  f"for chemsys {'-'.join(sorted(key))}")
        return self._competitor_cache[key]

    def relax_candidate(self, structure, fmax: float = 0.05, steps: int = 500):
        """Shared Step 0. Returns ``(ComputedStructureEntry | None, converged)``."""
        return candidate_to_entry(structure, self.calc, fmax=fmax, steps=steps,
                                  receptive_field=self.receptive_field)

    def existence_check(self, entry, tol: Optional[float] = None,
                        identifier: Optional[str] = None) -> ExistenceResult:
        competitors = self.competitors(entry.composition.elements)
        return existence_check(entry, competitors,
                               tol=self.tol if tol is None else tol,
                               identifier=identifier)

    def electrochemical_window(self, entry, mu_min: float = -6.0, step: float = 0.05,
                               tol: Optional[float] = None,
                               voltages_of_interest=DEFAULT_VOLTAGES_OF_INTEREST,
                               identifier: Optional[str] = None) -> ESWResult:
        competitors = self.competitors(entry.composition.elements)
        return electrochemical_window(entry, competitors, mu_min=mu_min, step=step,
                                       tol=self.tol if tol is None else tol,
                                       voltages_of_interest=voltages_of_interest,
                                       identifier=identifier)

    def interfacial_reactivity(self, entry, electrode_comp: str = "Li",
                               mu_li: float = 0.0,
                               identifier: Optional[str] = None) -> InterfacialResult:
        from pymatgen.core import Composition
        elements = set(entry.composition.elements) | set(Composition(electrode_comp).elements)
        competitors = self.competitors(elements)
        return interfacial_reactivity(entry, competitors, electrode_comp=electrode_comp,
                                      mu_li=mu_li, identifier=identifier)

    def screen(self, identifier: str, structure, electrodes=(("Li", 0.0),),
              fmax: float = 0.05, steps: int = 500,
              esw_kwargs: Optional[dict] = None) -> CandidateStabilityResult:
        """Full pipeline for one candidate: relax -> existence (hard gate) ->
        ESW + interfacial vs each ``electrodes`` entry (soft penalties).

        Stops after the existence check if it fails or errors -- no point
        computing voltage-dependent penalties for a composition that
        decomposes regardless of any battery.
        """
        result = CandidateStabilityResult(identifier=identifier, model_label=self.model_label)
        entry, converged = self.relax_candidate(structure, fmax=fmax, steps=steps)
        result.converged = converged
        if entry is None:
            result.error = "relaxation failed or did not converge"
            return result

        result.existence = self.existence_check(entry, identifier=identifier)
        if not result.existence.ok or not result.existence.exists:
            return result

        result.esw = self.electrochemical_window(entry, identifier=identifier,
                                                  **(esw_kwargs or {}))
        for electrode_comp, mu_li in electrodes:
            result.interfacial[electrode_comp] = self.interfacial_reactivity(
                entry, electrode_comp=electrode_comp, mu_li=mu_li, identifier=identifier)
        return result


# ============================================================================= #
#  GA-facing population screen (Gate A + soft penalties)                       #
# ============================================================================= #

def screen_population(screen: StabilityScreen, candidates: list,
                      tol: Optional[float] = None, verbose: bool = True) -> list:
    """Apply the existence hard gate, then attach ESW/interfacial soft
    penalties for survivors.

    ``candidates`` is a list of ``(identifier, pymatgen Structure)``. Returns
    a list of per-candidate dicts:

      {"id", "converged", "e_above_hull_ev_atom", "decomposes_into",
       "passes_existence",
       # only present if passes_existence:
       "esw_window_v", "anode_penalty_ev_atom", "cathode_penalty_ev_atom",
       "interfacial_li_ev_atom"}

    Merge these into the GA's per-candidate record: ``passes_existence``
    feeds Gate A (kill); ``anode_penalty_ev_atom`` / ``cathode_penalty_ev_atom``
    / ``interfacial_li_ev_atom`` feed the fitness soft penalty.
    """
    records = []
    for ident, structure in candidates:
        rec = {"id": ident}
        entry, converged = screen.relax_candidate(structure)
        rec["converged"] = converged
        if entry is None:
            rec["passes_existence"] = False
            rec["error"] = "relaxation failed or did not converge"
            records.append(rec)
            continue

        existence = screen.existence_check(entry, tol=tol, identifier=ident)
        rec["e_above_hull_ev_atom"] = existence.e_above_hull_ev_atom
        rec["decomposes_into"] = existence.decomposes_into
        rec["passes_existence"] = bool(existence.ok and existence.exists)
        if not existence.ok:
            rec["error"] = existence.error

        if rec["passes_existence"]:
            esw = screen.electrochemical_window(entry, identifier=ident)
            rec["esw_window_v"] = (esw.v_reduction_limit, esw.v_oxidation_limit)
            rec["anode_penalty_ev_atom"] = esw.anode_penalty_ev_atom
            rec["cathode_penalty_ev_atom"] = esw.cathode_penalty_ev_atom

            interf = screen.interfacial_reactivity(entry, electrode_comp="Li",
                                                    mu_li=0.0, identifier=ident)
            rec["interfacial_li_ev_atom"] = interf.min_reaction_energy_ev_atom

        records.append(rec)

    if verbose:
        n_pass = sum(r["passes_existence"] for r in records)
        eff_tol = screen.tol if tol is None else tol
        print(f"\n  Stability Gate A: {n_pass}/{len(records)} candidates "
              f"survive the existence check (tol={1000 * eff_tol:.0f} meV/atom)")

    return records


# ============================================================================= #
#  Validation -- existence-check plumbing sanity, no MLIP required             #
# ============================================================================= #

# Known-stable SSEs. Each material's OWN MP entry is used as the "candidate";
# compared against its own chemsys competitors (itself included), a stable
# phase must sit at e_above_hull == 0 by construction. This validates chemsys
# completeness, thermo_type filtering and MP connectivity WITHOUT an MLIP.
VALIDATION_MP_IDS = [
    ("mp-942733", "Li7La3Zr2O12 (LLZO)"),
    ("mp-696128", "Li10GeP2S12 (LGPS)"),
    ("mp-1185319", "Li6PS5Cl (argyrodite)"),
    ("mp-22905", "LiCl"),
    ("mp-1153", "Li2S"),
]


def validate(api_key: Optional[str] = None, tol: float = DEFAULT_TOL_EV_ATOM,
            thermo_types=DEFAULT_THERMO_TYPES) -> bool:
    """Phase-5-ish gate: confirm existence_check reports e_above_hull == 0 for
    known-stable SSEs using their own MP entries (no MLIP needed)."""
    from mp_api.client import MPRester

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("No MP API key (set MP_API_KEY or pass api_key=...).")

    print("Stability validation -- existence check on known-stable SSEs from MP\n")
    n_ok = 0
    with MPRester(key) as mpr:
        for mp_id, label in VALIDATION_MP_IDS:
            ident = f"{label} ({mp_id})"
            try:
                matches = mpr.get_entries(mp_id, additional_criteria={
                    "thermo_types": list(thermo_types)})
                entry = next(e for e in matches if e.entry_id == mp_id)
                elements = [str(e) for e in entry.composition.elements]
                competitors = mpr.get_entries_in_chemsys(
                    elements, additional_criteria={"thermo_types": list(thermo_types)})
            except Exception as exc:
                print(f"  [FAIL] {ident}: {type(exc).__name__}: {exc}")
                continue

            res = existence_check(entry, competitors, tol=tol, identifier=ident)
            ok = res.ok and abs(res.e_above_hull_ev_atom) <= 1e-6
            n_ok += int(ok)
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {res.summary() if res.ok else res.error}")

    print(f"\n  {n_ok}/{len(VALIDATION_MP_IDS)} known-stable materials confirmed "
          "on the MP hull (e_above_hull == 0).")
    return n_ok == len(VALIDATION_MP_IDS)


# ============================================================================= #
#  CLI                                                                          #
# ============================================================================= #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cif", help="path to a candidate CIF")
    parser.add_argument("--mp-id", help="Materials Project ID (e.g. mp-1234)")
    parser.add_argument("--model", default="auto", help="'auto', 'chgnet', or 'm3gnet'")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL_EV_ATOM,
                        help="metastability tolerance, eV/atom (default 0.050)")
    parser.add_argument("--existence", action="store_true", help="run Calc 1 only")
    parser.add_argument("--esw", action="store_true", help="run Calc 2 only")
    parser.add_argument("--interfacial", action="store_true", help="run Calc 3 only")
    parser.add_argument("--electrode", default="Li",
                        help="electrode composition for --interfacial (default Li)")
    parser.add_argument("--mu-li", type=float, default=0.0,
                        help="Li chemical potential for --interfacial "
                             "(0.0 = Li metal anode; -V_cathode for a cathode)")
    parser.add_argument("--all", action="store_true", help="run all three calcs")
    parser.add_argument("--validate", action="store_true",
                        help="run the Phase-5 plumbing sanity gate (no MLIP needed)")
    parser.add_argument("--api-key", default=None, help="MP API key (default: MP_API_KEY env var)")
    args = parser.parse_args()

    if args.validate:
        ok = validate(api_key=args.api_key, tol=args.tol)
        raise SystemExit(0 if ok else 1)

    if not args.cif and not args.mp_id:
        parser.error("provide --cif or --mp-id (or --validate)")

    structure = acquire_structure(mp_id=args.mp_id, cif_path=args.cif, api_key=args.api_key)
    screen = StabilityScreen(model=args.model, api_key=args.api_key, tol=args.tol)
    ident = args.mp_id or Path(args.cif).stem

    entry, converged = screen.relax_candidate(structure)
    print(f"[{ident}] relaxed with {screen.model_label}; converged={converged}")
    if entry is None:
        raise SystemExit("relaxation failed or did not converge")
    print(f"  E = {entry.energy:.4f} eV  ({entry.composition.reduced_formula}, "
          f"{len(entry.composition)} elements, {int(entry.composition.num_atoms)} atoms)")

    any_specific = args.existence or args.esw or args.interfacial
    if args.existence or args.all or not any_specific:
        res = screen.existence_check(entry, identifier=ident)
        print(res.summary() if res.ok else f"  existence FAILED: {res.error}")

    if args.esw or args.all:
        res = screen.electrochemical_window(entry, identifier=ident)
        print(res.summary() if res.ok else f"  ESW FAILED: {res.error}")

    if args.interfacial or args.all:
        res = screen.interfacial_reactivity(entry, electrode_comp=args.electrode,
                                            mu_li=args.mu_li, identifier=ident)
        print(res.summary() if res.ok else f"  interfacial FAILED: {res.error}")


if __name__ == "__main__":
    main()
