# BVSE_test.py
#
# NOTE: BVSE (Bond Valence Site Energy) is not fully accurate. It misses lattice
# vibrations and entropy effects, and the effect of defects on ion transport.
# Treat results as semi-quantitative estimates only.

import os
import numpy as np
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from bvlain import Lain

# ── Materials Project API key ────────────────────────────────────────────────
MP_API_KEY = os.environ.get("MP_API_KEY")
if MP_API_KEY is None:
    raise EnvironmentError(
        "Set the MP_API_KEY environment variable before running this script.\n"
        "Get your key at https://materialsproject.org/api"
    )

# ── 1. Fetch the most-stable LLZO structure from Materials Project ────────────
FORMULA = "Li7La3Zr2O12"
print(f"Fetching {FORMULA} from Materials Project...")

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.search(
        formula=FORMULA,
        fields=["material_id", "structure", "entries"],
    )

if not docs:
    raise RuntimeError(f"No entries found for '{FORMULA}' on Materials Project.")

# Sort by energy above hull, extracted from the GGA(+U) computed entry
def _e_above_hull(doc):
    try:
        return doc.entries["GGA+U"].energy_above_hull
    except (KeyError, AttributeError, TypeError):
        try:
            return next(iter(doc.entries.values())).energy_above_hull
        except Exception:
            return float("inf")

docs.sort(key=_e_above_hull)
best = docs[0]
e_hull = _e_above_hull(best)
print(f"  material_id : {best.material_id}")
print(f"  E above hull: {e_hull:.4f} eV/atom" if e_hull < float("inf") else f"  material_id : {best.material_id}")

# ── 2. Write CIF ──────────────────────────────────────────────────────────────
cif_path = "LLZO_mp.cif"
CifWriter(best.structure).write_file(cif_path)
print(f"  CIF written : {cif_path}")

# ── 3. BVSE calculation ───────────────────────────────────────────────────────
print("\nRunning BVSE for Li1+ ...")
calc = Lain(verbose=True)
calc.read_file(cif_path)
calc.bvse_distribution(mobile_ion="Li1+", r_cut=10.0, resolution=0.2)

# Activation energy: Ea = E_max - E_min
# percolation_barriers() normalises internally, so E_3D is already (E_saddle - E_min)
barriers = calc.percolation_barriers()
E_min = float(calc.data.min())        # absolute minimum energy of Li in landscape (eV)
Ea    = barriers["E_3D"]              # 3-D percolation threshold = migration barrier (eV)
E_max = E_min + Ea

print(f"\n  E_min  = {E_min:.4f} eV   (equilibrium Li site)")
print(f"  E_max  = {E_max:.4f} eV   (3-D percolation saddle point)")
print(f"  Ea     = {Ea:.4f} eV   (migration barrier)")

# ── 4. Arrhenius ionic conductivity ──────────────────────────────────────────
# σ = σ_0 · exp(−Ea / kB·T)
#
# σ_0 is empirically calibrated for cubic garnet LLZO from the literature
# (Murugan et al. 2007: σ(300 K) ≈ 2.4×10⁻⁴ S/cm, Ea ≈ 0.26 eV → σ_0 ≈ 7 S/cm).
# BVSE tends to overestimate Ea (no defects / cooperative hopping), so the
# absolute conductivity should be treated as a lower-bound estimate.
kB      = 8.617333e-5   # eV / K  (Boltzmann constant)
T       = 300.0          # K — standard room temperature
sigma_0 = 7.0            # S/cm — pre-exponential, calibrated for cubic garnet LLZO

sigma = sigma_0 * np.exp(-Ea / (kB * T))

print(f"\n{'='*57}")
print(f"  Predicted ionic conductivity for {FORMULA}")
print(f"{'='*57}")
print(f"  T            = {T:.0f} K   (standard room temperature)")
print(f"  Ea  (BVSE)   = {Ea:.4f} eV")
print(f"  σ_0          = {sigma_0:.1f} S/cm   (garnet LLZO literature prefactor)")
print(f"  σ (300 K)    = {sigma:.4e} S/cm")
print(f"\n  Experimental reference for cubic LLZO: ~10⁻⁴ – 10⁻³ S/cm")
print(f"{'='*57}")
