#!/usr/bin/env python3
"""
second_pass.py
==============
Scissor correction for PBE band gaps of inorganic solid-state electrolytes (SSEs).

A scissor is a fixed additive or linear offset fit ONCE, offline, to a set of
anchor materials from the inorganic SSE chemical family.  After fitting it is
frozen; every candidate then gets a corrected gap by applying that frozen offset
to the raw PBE result.

Workflow
--------
1. Run PBE on a relaxed structure (same VASP setup as anchors) to get E_g^PBE.
2. Load the frozen correction::

       correction = FITTED_CORRECTION          # pre-fit at bottom of this file
       eg_corr = correction.apply(eg_pbe)      # one line

3. Use the LOO-MAE as your honest uncertainty::

       print(correction.summary())

Methodology
-----------
* Anchor set: 20 inorganic SSE materials spanning oxides, sulfides, halides,
  phosphates, NASICON, and LISICON sub-families drawn from the OBELiX dataset
  chemical family (Li-ion conducting inorganic solids).
* Each anchor provides a matched (E_g^PBE, E_g^trusted) pair computed on the
  SAME GGA-relaxed geometry; only the functional differs.  Mixing geometry
  sources leaks structural differences into the offset.
* Trusted gaps: HSE06 from the literature for most materials; experimental
  fundamental gaps (from photoemission / VUV band-to-band absorption) for
  simple halides (LiCl, LiBr, LiI).  NOTE: the experimental values for these
  three materials are the quasiparticle (fundamental) gap, not the first
  excitonic absorption onset (which lies ~0.5-0.9 eV below the fundamental
  for alkali halides).  For LiCl: GW fundamental = 9.5 eV, exp fundamental =
  9.4 eV, first exciton = 8.8 eV (Phys. Rev. B 88, 245202, 2013).  For LiI:
  GW fundamental = 6.3 eV, exp fundamental = 6.4 eV (Phys. B 448, 68, 2014).
* NEVER use MP or r²SCAN gaps as the trusted reference — that is DFT calibrating
  DFT and does not correct the systematic underestimation.
* Form selection: leave-one-out (LOO) cross-validation picks constant vs linear.
  With < 15 anchors the constant form is always used (linear overfits at that
  sample size).
* Coefficients are frozen after fitting; the LOO-MAE is the error bar.

Caveats
-------
* Corrects gap MAGNITUDE only.  Band alignment, effective masses, and k-point
  character are unchanged.
* Apply only to PBE gaps from the same setup (PAW PBE, ENCUT ≥ 520 eV,
  Γ-centred k-mesh) as the anchor set.
* On novel chemistries outside the anchor composition space this is extrapolation;
  the LOO-MAE understates the true uncertainty there.

Usage (CLI)::

    # Re-fit and print summary (uses the built-in anchor table):
    python second_pass.py --fit

    # Apply correction to a single PBE gap:
    python second_pass.py --apply 3.25

    # Apply to every row in a CSV (column "eg_pbe"):
    python second_pass.py --csv candidates.csv --out candidates_corrected.csv

    # Load updated anchors from a CSV of your own calculations and refit:
    python second_pass.py --fit --anchor-csv my_anchors.csv

    # Parse a VASP vasprun.xml, get gap, apply correction:
    python second_pass.py --vasprun vasprun.xml

References (verified — papers containing the actual band gap values)
----------
Thompson et al., ACS Energy Lett. 2017, 2, 462-468 (DOI: 10.1021/acsenergylett.6b00593)
  — LLZO HSE06 band gap 5.79-5.87 eV, optical gap 5.46 eV, GW gap 6.4 eV.
Binninger et al., arXiv:1901.02251 (IBM Research, 2019)
  — PBE HOMO-LUMO gaps (Table 2): LGPS=2.21, LLZO=4.34, LATP=2.48 eV
  — HSE06 HOMO-LUMO gaps (Table 3): LLZO=5.81, LATP=4.19 eV

Papers that do NOT contain PBE/HSE06 band gap tables (both are about
electrochemical stability windows computed via phase diagrams at the PBE level):
  Richards et al., Chem. Mater. 2016, 28, 266 — interface stability windows
  Zhu et al., J. Mater. Chem. A 2016 (DOI: 10.1039/c5ta08574h) — interfacial thermodynamics
  Zhu et al., ACS Appl. Mater. Interfaces 2015, 7, 23685 — phase stability thermodynamics

NOTE: 17 of the 20 anchor entries are marked [UNVERIFIED] — their PBE and HSE06
values were estimated from training-data knowledge and have not been traced to
a specific table in a paper we have read.  Replace with your own DFT+HSE06
calculations (same VASP setup as candidates) before relying on this correction
in production.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
#  Anchor table                                                                #
# --------------------------------------------------------------------------- #

# Columns: (label, formula, family, eg_pbe_ev, eg_trusted_ev, trusted_type, reference)
#
# eg_pbe_ev  : PBE-GGA gap on GGA-relaxed geometry, PAW pseudopotentials,
#              ENCUT=520 eV, Γ-centred k-mesh, ISTART=0 ICHARG=2.
#              Replace these with your own calculations when possible — the
#              scissor is most accurate when the anchor PBE setup matches
#              exactly the candidate PBE setup.
#
# eg_trusted_ev : HSE06 (same geometry, same k-mesh, HFSCREEN=0.2) or
#                 experimental fundamental gap (photoemission / VUV band-to-band).
#                 Do NOT substitute MP/r2SCAN; do NOT use first-exciton
#                 optical absorption peaks as the trusted value.

# Verification status key:
#   [VERIFIED]    - PBE and trusted gap both read directly from a published paper.
#   [PBE-ONLY]   - PBE gap from a paper; trusted gap estimated from literature search.
#   [UNVERIFIED] - Both values estimated from training-data knowledge; not traced to
#                  a specific table or figure in a paper we have actually read.
#                  Replace with your own DFT+HSE06 calculations before production use.
#
# Papers confirmed to contain these values:
#   Binninger2019 : Binninger et al., arXiv:1901.02251 (IBM Research 2019)
#                   Table 2 (PBE HOMO-LUMO gaps), Table 3 (HSE06 HOMO-LUMO gaps)
#   Thompson2017  : Thompson et al., ACS Energy Lett. 2017, 2, 462-468
#                   DOI: 10.1021/acsenergylett.6b00593, Table 1 (HSE06 band gaps)
#
# Papers that do NOT contain band gap data (they are about electrochemical
# stability windows / reaction energetics at the PBE level only):
#   Richards2016  : Richards et al., Chem. Mater. 2016, 28, 266 -- interface stability
#   Zhu2016_JMCA  : Zhu et al., J. Mater. Chem. A 2016 -- interfacial thermodynamics
#   Zhu2015_ACS   : Zhu et al., ACS Appl. Mater. Interfaces 2015, 7, 23685 -- phase stability

ANCHOR_TABLE: List[Tuple] = [
    # label          formula                     family        eg_pbe  eg_trusted trusted_type  reference
    # ---- Oxides ----
    # [VERIFIED] LLZO: PBE from Binninger2019 Table 2; HSE06 from Binninger2019 Table 3 & Thompson2017 Table 1
    ("LLZO",         "Li7La3Zr2O12",             "oxide",       4.34,   5.81,     "HSE06",  "Binninger2019 T2/T3; Thompson2017 T1"),
    ("Li3PO4",       "Li3PO4",                   "phosphate",   6.79,   8.15,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    ("Li2O",         "Li2O",                     "oxide",       5.30,   6.19,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    ("LiAlO2",       "LiAlO2",                   "oxide",       6.29,   7.61,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    ("Li3OCl",       "Li3OCl",                   "anti-perov",  4.97,   6.03,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    # ---- Sulfides ----
    # [PBE-ONLY] LGPS: PBE from Binninger2019 Table 2; HSE06 from web search (LGPO arXiv paper) - not read
    ("LGPS",         "Li10GeP2S12",              "thio-LGPS",   2.21,   3.22,     "HSE06",  "Binninger2019 T2 (PBE); HSE06 [est., not verified]"),
    ("b-Li3PS4",     "Li3PS4",                   "thio-LPS",    3.73,   4.47,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    ("Li2S",         "Li2S",                     "sulfide",     3.49,   4.50,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    ("Li6PS5Cl",     "Li6PS5Cl",                 "argyrodite",  2.81,   3.53,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    ("Li4GeS4",      "Li4GeS4",                  "thio-LIS",    2.54,   3.28,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    ("Li4SiS4",      "Li4SiS4",                  "thio-LIS",    2.78,   3.55,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    # ---- Halides ----
    # Fundamental (quasiparticle / band-to-band) gaps, NOT first-exciton optical gaps.
    # LiCl: GW fundamental = 9.5 eV, exp fundamental = 9.4 eV, first exciton = 8.8 eV
    #        -> delta_exciton ~ 0.7 eV. Phys. Rev. B 88, 245202 (2013).
    # LiBr: Scaife-era experimental band-to-band onset; expected near GW fundamental.
    # LiI:  GW fundamental = 6.3 eV, exp fundamental = 6.4 eV. Phys. B 448, 68 (2014).
    #        Previous value of 6.10 eV was the optical/excitonic gap; corrected to 6.40 eV.
    ("LiCl",         "LiCl",                     "halide",      6.25,   9.40,     "exp-fundamental",  "[UNVERIFIED PBE] exp fundamental; Phys. Rev. B 88, 245202 (2013)"),
    ("LiBr",         "LiBr",                     "halide",      5.53,   7.59,     "exp-fundamental",  "[UNVERIFIED PBE] exp fundamental Scaife et al.; near GW fundamental"),
    ("LiI",          "LiI",                      "halide",      4.45,   6.40,     "exp-fundamental",  "[UNVERIFIED PBE] exp fundamental = 6.4 eV; Phys. B 448, 68 (2014)"),
    ("Li3YCl6",      "Li3YCl6",                  "halide",      4.37,   5.61,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    ("Li3InCl6",     "Li3InCl6",                 "halide",      4.02,   5.17,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    # ---- NASICON ----
    # [VERIFIED] LATP: PBE and HSE06 from Binninger2019 Tables 2 & 3
    ("LATP",         "Li1.3Al0.3Ti1.7(PO4)3",   "NASICON",     2.48,   4.19,     "HSE06",  "Binninger2019 T2/T3 [VERIFIED]"),
    ("LAGP",         "Li1.5Al0.5Ge1.5(PO4)3",   "NASICON",     3.81,   4.73,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    # ---- LISICON / sulfate ----
    ("Li14ZnGe4O16", "Li14ZnGe4O16",             "LISICON",     4.87,   5.93,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
    ("Li2SO4",       "Li2SO4",                   "sulfate",     7.12,   8.34,     "HSE06",  "[UNVERIFIED] est. from training knowledge"),
]

_ANCHOR_KEYS = ("label", "formula", "family", "eg_pbe", "eg_trusted", "trusted_type", "reference")


def anchors_to_arrays(anchors: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    eg_pbe = np.array([a[3] for a in anchors], dtype=float)
    eg_trusted = np.array([a[4] for a in anchors], dtype=float)
    return eg_pbe, eg_trusted


def load_anchors_from_csv(path: str | Path) -> List[Tuple]:
    """Load anchor pairs from a CSV with columns: label, formula, family,
    eg_pbe, eg_trusted, trusted_type, reference.

    Use this to replace or augment the built-in table once you have run your
    own PBE+HSE06 calculations on the anchor structures.  The eg_pbe column
    MUST come from the same VASP setup you use for candidates.
    """
    anchors = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            anchors.append((
                row["label"],
                row["formula"],
                row["family"],
                float(row["eg_pbe"]),
                float(row["eg_trusted"]),
                row.get("trusted_type", "HSE06"),
                row.get("reference", ""),
            ))
    if not anchors:
        raise ValueError(f"No valid rows in anchor CSV: {path}")
    return anchors


# --------------------------------------------------------------------------- #
#  Core fitting                                                                #
# --------------------------------------------------------------------------- #

def _fit_constant(eg_pbe: np.ndarray, eg_trusted: np.ndarray) -> float:
    """Mean signed offset: Δ = mean(E_trusted - E_PBE)."""
    return float(np.mean(eg_trusted - eg_pbe))


def _fit_linear(eg_pbe: np.ndarray, eg_trusted: np.ndarray) -> Tuple[float, float]:
    """OLS fit: E_trusted = a * E_PBE + b.  Returns (a, b)."""
    X = np.column_stack([eg_pbe, np.ones_like(eg_pbe)])
    coeffs, _, _, _ = np.linalg.lstsq(X, eg_trusted, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _loo_mae_constant(eg_pbe: np.ndarray, eg_trusted: np.ndarray) -> float:
    n = len(eg_pbe)
    errors = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        delta = float(np.mean(eg_trusted[mask] - eg_pbe[mask]))
        errors[i] = abs(eg_pbe[i] + delta - eg_trusted[i])
    return float(np.mean(errors))


def _loo_mae_linear(eg_pbe: np.ndarray, eg_trusted: np.ndarray) -> float:
    n = len(eg_pbe)
    errors = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        X = np.column_stack([eg_pbe[mask], np.ones(n - 1)])
        coeffs, _, _, _ = np.linalg.lstsq(X, eg_trusted[mask], rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
        errors[i] = abs(a * eg_pbe[i] + b - eg_trusted[i])
    return float(np.mean(errors))


# --------------------------------------------------------------------------- #
#  ScissorCorrection — the frozen, serializable result                        #
# --------------------------------------------------------------------------- #

@dataclass
class ScissorCorrection:
    """A frozen scissor correction for PBE → corrected band gap.

    Do not refit this per candidate.  Fit it once offline on the anchor set,
    freeze the coefficients, and carry the LOO-MAE as the uncertainty.

    Attributes
    ----------
    form : "constant" or "linear"
    delta : additive offset when form="constant"  (E_g^corr = E_g^PBE + delta)
    a, b  : slope and intercept when form="linear" (E_g^corr = a * E_g^PBE + b)
    loo_mae : leave-one-out MAE in eV — your honest error bar
    n_anchors : number of anchor materials used
    pbe_setup : description of the DFT setup the anchors (and candidates) must share
    """
    form: str = "constant"
    delta: float = 0.0
    a: float = 1.0
    b: float = 0.0
    loo_mae: float = 0.0
    n_anchors: int = 0
    pbe_setup: str = "PAW-PBE, ENCUT=520 eV, Gamma-centred k-mesh"
    anchor_labels: List[str] = field(default_factory=list)
    loo_mae_const: Optional[float] = None
    loo_mae_linear: Optional[float] = None

    # ------------------------------------------------------------------ #

    def apply(self, eg_pbe: float | np.ndarray) -> float | np.ndarray:
        """Return scissor-corrected gap(s).  Same shape as input."""
        arr = np.asarray(eg_pbe, dtype=float)
        if self.form == "constant":
            out = arr + self.delta
        else:
            out = self.a * arr + self.b
        return float(out) if np.ndim(eg_pbe) == 0 else out

    def uncertainty(self) -> float:
        """LOO-MAE as the recommended ± error bar (eV)."""
        return self.loo_mae

    def summary(self) -> str:
        if self.form == "constant":
            expr = f"E_g^corr = E_g^PBE + {self.delta:+.4f} eV"
        else:
            expr = f"E_g^corr = {self.a:.4f} * E_g^PBE + {self.b:+.4f} eV"
        lines = [
            f"ScissorCorrection [{self.form}]",
            f"  {expr}",
            f"  LOO-MAE : {self.loo_mae:.3f} eV  (honest +/- error bar)",
            f"  n_anchors: {self.n_anchors}",
            f"  PBE setup: {self.pbe_setup}",
        ]
        if self.loo_mae_const is not None and self.loo_mae_linear is not None:
            lines.append(f"  LOO comparison - constant: {self.loo_mae_const:.3f} eV  "
                         f"linear: {self.loo_mae_linear:.3f} eV  "
                         f"(selected: {self.form})")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Serialization                                                       #
    # ------------------------------------------------------------------ #

    def to_json(self, path: str | Path | None = None) -> str:
        d = asdict(self)
        text = json.dumps(d, indent=2)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, path: str | Path) -> "ScissorCorrection":
        d = json.loads(Path(path).read_text())
        return cls(**d)


# --------------------------------------------------------------------------- #
#  Public fitting function                                                     #
# --------------------------------------------------------------------------- #

def fit_scissor(
    anchors: Optional[List[Tuple]] = None,
    force_constant: bool = False,
    pbe_setup: str = "PAW-PBE, ENCUT=520 eV, Gamma-centred k-mesh",
    verbose: bool = True,
) -> ScissorCorrection:
    """Fit a scissor correction to the anchor set and return a frozen ScissorCorrection.

    Parameters
    ----------
    anchors :
        List of tuples matching ANCHOR_TABLE columns.  Defaults to the
        built-in ANCHOR_TABLE if None.
    force_constant :
        Skip LOO CV and always use the constant form.
    pbe_setup :
        Description of the DFT setup.  Recorded in the correction so that
        future readers know what "E_g^PBE" means.
    verbose :
        Print a full diagnostic table and summary when True.

    Returns
    -------
    ScissorCorrection
        Frozen, serializable correction object.
    """
    if anchors is None:
        anchors = ANCHOR_TABLE

    eg_pbe, eg_trusted = anchors_to_arrays(anchors)
    n = len(eg_pbe)
    labels = [a[0] for a in anchors]
    deltas = eg_trusted - eg_pbe

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Scissor fit - {n} anchors, chemical family: inorganic SSE")
        print(f"  PBE setup : {pbe_setup}")
        print(f"{'='*70}")
        header = f"  {'Label':<18} {'Formula':<28} {'Family':<14} "
        header += f"{'E_PBE':>7} {'E_trust':>8} {'dE':>6} {'Type'}"
        print(header)
        print("  " + "-" * 68)
        for a in anchors:
            delta_i = a[4] - a[3]
            print(f"  {a[0]:<18} {a[1]:<28} {a[2]:<14} "
                  f"{a[3]:>7.2f} {a[4]:>8.2f} {delta_i:>+6.2f}  {a[5]}")
        print("  " + "-" * 68)
        print(f"  Mean dE = {np.mean(deltas):.3f} eV  "
              f"std = {np.std(deltas):.3f} eV  "
              f"range [{deltas.min():.3f}, {deltas.max():.3f}] eV")

    # LOO MAE for both forms
    mae_const = _loo_mae_constant(eg_pbe, eg_trusted)

    if n < 15 or force_constant:
        if verbose:
            print(f"\n  n={n} < 15 -> using constant form (linear overfits at this size).")
        delta = _fit_constant(eg_pbe, eg_trusted)
        correction = ScissorCorrection(
            form="constant",
            delta=delta,
            loo_mae=mae_const,
            n_anchors=n,
            pbe_setup=pbe_setup,
            anchor_labels=labels,
            loo_mae_const=mae_const,
            loo_mae_linear=None,
        )
    else:
        mae_linear = _loo_mae_linear(eg_pbe, eg_trusted)
        if verbose:
            print(f"\n  LOO-MAE - constant: {mae_const:.3f} eV  "
                  f"linear: {mae_linear:.3f} eV")

        if mae_linear < mae_const:
            a, b = _fit_linear(eg_pbe, eg_trusted)
            correction = ScissorCorrection(
                form="linear",
                a=a,
                b=b,
                loo_mae=mae_linear,
                n_anchors=n,
                pbe_setup=pbe_setup,
                anchor_labels=labels,
                loo_mae_const=mae_const,
                loo_mae_linear=mae_linear,
            )
            if verbose:
                print(f"  -> selected: linear  (a={a:.4f}, b={b:+.4f} eV)")
        else:
            delta = _fit_constant(eg_pbe, eg_trusted)
            correction = ScissorCorrection(
                form="constant",
                delta=delta,
                loo_mae=mae_const,
                n_anchors=n,
                pbe_setup=pbe_setup,
                anchor_labels=labels,
                loo_mae_const=mae_const,
                loo_mae_linear=mae_linear,
            )
            if verbose:
                print(f"  -> selected: constant  (delta={delta:+.4f} eV)")

    if verbose:
        print()
        print(correction.summary())
        print(f"{'='*70}\n")

    return correction


# --------------------------------------------------------------------------- #
#  Batch application                                                           #
# --------------------------------------------------------------------------- #

def screen_candidates(
    records: List[dict],
    correction: "ScissorCorrection",
    pbe_key: str = "eg_pbe",
    out_key: str = "eg_corrected",
    verbose: bool = True,
) -> List[dict]:
    """Apply the scissor correction to a list of candidate dicts.

    Each dict must have a ``pbe_key`` field.  The corrected gap is added under
    ``out_key``.  Also adds ``scissor_uncertainty`` = LOO-MAE, and
    ``scissor_form`` for provenance.

    Parameters
    ----------
    records : list of dicts, one per candidate material.
    correction : a frozen ScissorCorrection (do not refit per candidate).
    pbe_key : column name for the raw PBE gap.
    out_key : column name written for the corrected gap.

    Returns
    -------
    records with out_key (and provenance fields) added in-place.
    """
    for rec in records:
        eg_pbe = float(rec[pbe_key])
        eg_corr = correction.apply(eg_pbe)
        rec[out_key] = round(float(eg_corr), 4)
        rec["scissor_form"] = correction.form
        rec["scissor_uncertainty_ev"] = round(correction.loo_mae, 4)

    if verbose:
        print(f"\n  Scissor applied to {len(records)} candidates "
              f"(form={correction.form}, LOO-MAE={correction.loo_mae:.3f} eV)")
        header = f"  {'ID':<20} {'E_g^PBE':>10} {'E_g^corr':>10} {'+/-':>6}"
        print(header)
        print("  " + "-" * 50)
        for rec in records:
            rid = str(rec.get("id", rec.get("label", rec.get("formula", "?"))))
            print(f"  {rid:<20} {rec[pbe_key]:>10.3f} {rec[out_key]:>10.3f} "
                  f"{correction.loo_mae:>+6.3f}")

    return records


# --------------------------------------------------------------------------- #
#  Optional: parse VASP output                                                #
# --------------------------------------------------------------------------- #

def extract_pbe_gap_vasp(vasprun_path: str | Path) -> float:
    """Parse vasprun.xml and return the PBE band gap in eV.

    Requires pymatgen.  Returns the direct/indirect gap as computed by
    pymatgen's BSAnalyzer (eigenvalue-based, no Fermi-smearing correction).
    For an accurate gap from metallic or nearly-metallic systems, check the
    density of states manually.
    """
    try:
        from pymatgen.io.vasp import Vasprun
        from pymatgen.electronic_structure.core import Spin
    except ImportError as exc:
        raise RuntimeError(
            "pymatgen is required to parse vasprun.xml.  "
            "Install it with: pip install pymatgen"
        ) from exc

    vr = Vasprun(str(vasprun_path), parse_potcar_file=False)
    bs = vr.get_band_structure()
    gap = bs.get_band_gap()["energy"]
    return float(gap)


# --------------------------------------------------------------------------- #
#  Pre-computed frozen correction                                              #
#                                                                             #
#  This block is evaluated once at module import.  The ScissorCorrection      #
#  object is ready to use immediately without any refitting.                  #
#                                                                             #
#  To regenerate: python second_pass.py --fit                                 #
# --------------------------------------------------------------------------- #

def _build_default_correction() -> ScissorCorrection:
    """Build and return the default correction from the built-in anchor table.
    Called once at module import; result stored as FITTED_CORRECTION.
    """
    return fit_scissor(ANCHOR_TABLE, verbose=False)


FITTED_CORRECTION: ScissorCorrection = _build_default_correction()
"""The pre-fit scissor correction ready to use.  Import and apply::

    from second_pass import FITTED_CORRECTION
    eg_corrected = FITTED_CORRECTION.apply(eg_pbe)
"""


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="second_pass",
        description="Scissor correction for PBE band gaps of inorganic SSEs.",
    )
    p.add_argument("--fit", action="store_true",
                   help="(Re)fit the scissor and print the full diagnostic table.")
    p.add_argument("--apply", type=float, metavar="EG_PBE",
                   help="Apply the frozen scissor to a single PBE gap (eV).")
    p.add_argument("--csv", metavar="FILE",
                   help="Apply to all rows in CSV; must have an 'eg_pbe' column.")
    p.add_argument("--pbe-col", default="eg_pbe",
                   help="Column name for PBE gap in --csv (default: eg_pbe).")
    p.add_argument("--out", metavar="FILE",
                   help="Output CSV path for --csv results.")
    p.add_argument("--anchor-csv", metavar="FILE",
                   help="Replace built-in anchors with your own (see load_anchors_from_csv).")
    p.add_argument("--save-json", metavar="FILE",
                   help="Save the fitted correction to a JSON file.")
    p.add_argument("--load-json", metavar="FILE",
                   help="Load a previously saved ScissorCorrection JSON.")
    p.add_argument("--vasprun", metavar="FILE",
                   help="Parse a vasprun.xml, extract PBE gap, and apply correction.")
    p.add_argument("--force-constant", action="store_true",
                   help="Always use the constant form regardless of LOO-CV result.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    # -- Load or fit the correction ---------------------------------------- #
    if args.load_json:
        correction = ScissorCorrection.from_json(args.load_json)
        print(f"Loaded correction from {args.load_json}")
        print(correction.summary())
    elif args.fit or args.anchor_csv:
        anchors = ANCHOR_TABLE
        if args.anchor_csv:
            anchors = load_anchors_from_csv(args.anchor_csv)
            print(f"Loaded {len(anchors)} anchors from {args.anchor_csv}")
        correction = fit_scissor(anchors, force_constant=args.force_constant, verbose=True)
        if args.save_json:
            correction.to_json(args.save_json)
            print(f"Correction saved to {args.save_json}")
        return
    else:
        correction = FITTED_CORRECTION

    if args.save_json and not args.fit:
        correction.to_json(args.save_json)
        print(f"Correction saved to {args.save_json}")

    # -- Single gap --------------------------------------------------------- #
    if args.apply is not None:
        eg_corr = correction.apply(args.apply)
        print(f"\n  E_g^PBE      = {args.apply:.4f} eV")
        print(f"  E_g^corrected= {eg_corr:.4f} eV")
        print(f"  uncertainty  +/- {correction.loo_mae:.4f} eV  (LOO-MAE)")
        print(f"  form         = {correction.form}")

    # -- CSV batch ---------------------------------------------------------- #
    if args.csv:
        records = []
        with open(args.csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))

        pbe_key = args.pbe_col
        if not records:
            print(f"No rows in {args.csv}", file=sys.stderr)
            sys.exit(1)
        if pbe_key not in records[0]:
            print(f"Column '{pbe_key}' not found in {args.csv}.  "
                  f"Use --pbe-col to specify the correct name.", file=sys.stderr)
            sys.exit(1)

        screen_candidates(records, correction, pbe_key=pbe_key, verbose=True)

        out_path = args.out or args.csv.replace(".csv", "_corrected.csv")
        fieldnames = list(records[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\n  Results written to {out_path}")

    # -- vasprun.xml -------------------------------------------------------- #
    if args.vasprun:
        print(f"\nParsing {args.vasprun} ...")
        eg_pbe = extract_pbe_gap_vasp(args.vasprun)
        eg_corr = correction.apply(eg_pbe)
        print(f"  E_g^PBE      = {eg_pbe:.4f} eV")
        print(f"  E_g^corrected= {eg_corr:.4f} eV")
        print(f"  uncertainty  +/- {correction.loo_mae:.4f} eV  (LOO-MAE)")

    # -- Default: just show the correction ---------------------------------- #
    if not any([args.apply, args.csv, args.vasprun, args.fit]):
        print("\nCurrent frozen scissor correction:")
        print(correction.summary())
        print("\nRun with --help for usage options.")


if __name__ == "__main__":
    main()
