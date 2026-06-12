#!/usr/bin/env python3
"""
Tier 0 -- EOS sanity check for physics_bulk_modulus.py

Pure-math validation of the Birch-Murnaghan fitter and unit handling. Needs
NO MLIP backend, no network: it feeds the fitter synthetic E(V) data whose
true bulk modulus we already know and checks the recovered value.

Tests:
  1. Recover a known K0 (100 GPa) from clean synthetic BM data.
  2. Stay stable when small float64-scale noise is added.
  3. Unit conversion eV/A^3 -> GPa is the x160.21766 factor, applied once.
  4. A pure parabola in E(V) recovers K0' ~ 4.

All console output is ASCII only (Windows cp1252 console safe).
"""

import numpy as np

from physics_bulk_modulus import (
    birch_murnaghan_energy,
    fit_birch_murnaghan,
    EV_A3_TO_GPA,
)

# Reference truth used by several tests.
V0_TRUE = 40.0        # A^3
E0_TRUE = -10.0       # eV
K0P_TRUE = 4.0        # pressure derivative


def _make_bm_curve(k0_gpa, v0=V0_TRUE, e0=E0_TRUE, k0p=K0P_TRUE,
                   strain=0.05, n=9):
    """Synthetic 3rd-order BM E(V) with an exactly-known K0 (in GPa)."""
    b0_ev = k0_gpa / EV_A3_TO_GPA            # GPa -> eV/A^3 (the inverse factor)
    volumes = np.linspace(v0 * (1 - strain), v0 * (1 + strain), n)
    energies = birch_murnaghan_energy(volumes, e0, v0, b0_ev, k0p)
    return volumes, energies


# ---------------------------------------------------------------- helpers --- #

_PASS = "PASS"
_FAIL = "FAIL"
_results = []


def check(name, condition, detail=""):
    tag = _PASS if condition else _FAIL
    _results.append(bool(condition))
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


# ------------------------------------------------------------------ test 1 -- #

def test_recover_known_k0():
    print("\nTest 1: recover known K0 = 100 GPa from clean BM data")
    volumes, energies = _make_bm_curve(100.0)
    k0, k0p, v0_fit, e0, rms = fit_birch_murnaghan(volumes, energies, n_atoms=1)
    print(f"    fitted: K0={k0:.4f} GPa  K0'={k0p:.4f}  "
          f"V0={v0_fit:.4f}  rms={rms:.4e} meV/atom")
    check("K0 recovered to <0.1%", abs(k0 - 100.0) / 100.0 < 1e-3,
          f"K0={k0:.4f} GPa")
    check("V0 recovered to <0.1%", abs(v0_fit - V0_TRUE) / V0_TRUE < 1e-3,
          f"V0={v0_fit:.4f} A^3")
    check("K0' recovered to <1%", abs(k0p - K0P_TRUE) / K0P_TRUE < 1e-2,
          f"K0'={k0p:.4f}")
    check("residual ~ 0 on noiseless data", rms < 1e-3,
          f"rms={rms:.2e} meV/atom")


# ------------------------------------------------------------------ test 2 -- #

def test_noise_stability():
    print("\nTest 2: stability under small float64-scale energy noise")
    volumes, energies = _make_bm_curve(100.0)
    # ~0.5 meV per point of Gaussian noise -- the scale of float64 MLIP energy
    # jitter. Seeded so the test is deterministic.
    rng = np.random.default_rng(12345)
    k_vals = []
    for trial in range(20):
        noisy = energies + rng.normal(0.0, 5e-4, size=energies.shape)
        k0, k0p, v0_fit, e0, rms = fit_birch_murnaghan(volumes, noisy, n_atoms=1)
        k_vals.append(k0)
    k_vals = np.array(k_vals)
    spread = k_vals.std()
    bias = abs(k_vals.mean() - 100.0)
    print(f"    over 20 noisy trials: mean K0={k_vals.mean():.2f} GPa  "
          f"std={spread:.2f} GPa  max bias={bias:.2f} GPa")
    check("mean K0 within 5% of truth under noise",
          bias / 100.0 < 0.05, f"bias={bias:.2f} GPa")
    check("K0 spread under noise < 10 GPa",
          spread < 10.0, f"std={spread:.2f} GPa")


# ------------------------------------------------------------------ test 3 -- #

def test_unit_conversion():
    print("\nTest 3: unit conversion eV/A^3 -> GPa (the x160.21766 factor)")
    check("constant equals 160.21766208",
          abs(EV_A3_TO_GPA - 160.21766208) < 1e-6, f"{EV_A3_TO_GPA}")

    # Build a curve whose true B0 is exactly 1.0 eV/A^3 -> must fit to 160.2 GPa.
    volumes = np.linspace(V0_TRUE * 0.95, V0_TRUE * 1.05, 9)
    energies = birch_murnaghan_energy(volumes, E0_TRUE, V0_TRUE, 1.0, K0P_TRUE)
    k0, *_ = fit_birch_murnaghan(volumes, energies, n_atoms=1)
    print(f"    B0 = 1.0 eV/A^3 fits to K0 = {k0:.4f} GPa")
    check("1 eV/A^3 fits to ~160.22 GPa",
          abs(k0 - EV_A3_TO_GPA) < 0.1, f"K0={k0:.4f} GPa")
    # Demonstrate the classic bug: forgetting the factor would report ~0.62 GPa
    # (a self-evidently nonsense bulk modulus for any solid).
    forgotten = k0 / EV_A3_TO_GPA
    check("without the factor K0 would be a nonsense ~0.6 GPa",
          forgotten < 1.0, f"would report {forgotten:.4f} GPa")


# ------------------------------------------------------------------ test 4 -- #

def test_parabola_k0prime():
    print("\nTest 4: a pure parabola recovers K0' ~ 4")
    # The textbook 'parabola -> K0'~4' result is for a parabola in EULERIAN
    # STRAIN f = 1/2[(V0/V)^(2/3) - 1], because BM3 in strain is
    #   E = E0 + (9/2) V0 B0 f^2 [1 + (K0'-4) f + ...]
    # so the f^3 term -- and all asymmetry -- vanishes exactly at K0' = 4.
    # (A parabola in VOLUME is a different curve; see the bonus check below.)
    k0_target_gpa = 80.0
    b0_ev = k0_target_gpa / EV_A3_TO_GPA
    volumes = np.linspace(V0_TRUE * 0.95, V0_TRUE * 1.05, 9)
    f = 0.5 * ((V0_TRUE / volumes) ** (2.0 / 3.0) - 1.0)
    energies = E0_TRUE + (9.0 / 2.0) * V0_TRUE * b0_ev * f ** 2
    k0, k0p, v0_fit, e0, rms = fit_birch_murnaghan(volumes, energies, n_atoms=1)
    print(f"    parabola in Eulerian strain (K0~{k0_target_gpa} GPa) fits: "
          f"K0={k0:.2f} GPa  K0'={k0p:.4f}  V0={v0_fit:.3f}  rms={rms:.2e}")
    check("strain-parabola gives K0' within [3.5, 4.5]",
          3.5 <= k0p <= 4.5, f"K0'={k0p:.4f}")
    check("strain-parabola K0 within 1% of target",
          abs(k0 - k0_target_gpa) / k0_target_gpa < 0.01, f"K0={k0:.2f} GPa")

    # Bonus consistency check: a parabola in VOLUME has a zero cubic term in
    # x=(V-V0)/V0, and the BM3 cubic coefficient is B0*V0*(-K0'-1)/6, which is
    # zero at K0' = -1 (verified symbolically). The fitter must reproduce that.
    a = (k0_target_gpa / EV_A3_TO_GPA) / V0_TRUE
    e_volpar = E0_TRUE + 0.5 * a * (volumes - V0_TRUE) ** 2
    _, k0p_v, _, _, _ = fit_birch_murnaghan(volumes, e_volpar, n_atoms=1)
    print(f"    parabola in volume fits: K0'={k0p_v:.4f} (theory: -1)")
    check("volume-parabola gives K0' ~ -1 (matches BM3 cubic coeff)",
          abs(k0p_v - (-1.0)) < 0.1, f"K0'={k0p_v:.4f}")


# ----------------------------------------------------------------- driver --- #

def main():
    print("=" * 70)
    print("  TIER 0 -- EOS SANITY CHECK (no MLIP, pure math)")
    print("=" * 70)
    test_recover_known_k0()
    test_noise_stability()
    test_unit_conversion()
    test_parabola_k0prime()
    print("-" * 70)
    n_pass = sum(_results)
    n_tot = len(_results)
    print(f"  TIER 0 RESULT: {n_pass}/{n_tot} checks passed")
    print("=" * 70)
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
