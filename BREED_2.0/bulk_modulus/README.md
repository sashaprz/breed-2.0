# bulk_modulus

Bulk modulus predictor for inorganic solid-state electrolytes (SSEs).

---

## Final model: `physics_bulk_modulus.py`

**Method:** MLIP + Birch-Murnaghan equation of state (3rd-order BM fit).

A machine-learned interatomic potential (MACE-MP, SevenNet, CHGNet, or M3GNet)
samples the E(V) curve by relaxing internal coordinates at a fixed cell volume
over nine scan points spanning ±5% of the equilibrium volume. The bulk modulus
K is extracted from the curvature of the BM fit at the minimum.

**Why this approach wins over composition/structure ML regressors:**
- Operates on first-principles energetics, not learned property correlations.
- Gives a physically-shaped E(V) curve rather than an unconstrained prediction.
- Naturally produces K0' (pressure derivative) and a stress cross-check as built-in
  diagnostics — you know whether to trust the result before using it.

**MLIP priority (auto-selected):** MACE-MP (float64) > SevenNet > CHGNet > M3GNet.
Use MACE-MP in float64 for production; the second derivative amplifies noise and
float32 (CHGNet) degrades K reproducibility.

### Quick start

```bash
pip install mace-torch            # preferred backend

# Static K at 0 K from an MP entry:
python physics_bulk_modulus.py --mp-id mp-1234

# From a CIF file:
python physics_bulk_modulus.py --cif path/to/structure.cif

# Run the Phase-5 validation gate (known SSEs from MP):
python physics_bulk_modulus.py --validate

# Finite-temperature K(T) via quasi-harmonic approximation:
python physics_bulk_modulus.py --mp-id mp-1234 --qha
```

### Pipeline phases

| Phase | Function | What it does |
|-------|----------|-------------|
| 0 | `load_calculator()` | Lock one MLIP + float64 for all subsequent calls |
| 1 | `acquire_structure()` / `order_structure()` | Pull DFT-relaxed structure; order disorder via Ewald before MLIP sees it |
| 2 | `relax_structure()` | Full cell + positions relaxation; re-symmetrize; volume sanity check vs MP |
| 3 | `volume_scan()` | Scale V0 ±5% (9 pts); relax internals at fixed cell each point |
| 4 | `fit_birch_murnaghan()` | 3rd-order BM fit; K0' alarm (physical range 3.5–4.5); stress cross-check |
| 5 | `validate()` | Phase-5 gate: known SSEs vs MP experimental K values |
| 3b | `volume_scan_qha()` | Per-volume phonons (phonopy + MLIP forces) → F_vib(V,T) |
| 4b | `run_qha()` / `compute_qha()` | BM fit per temperature → K(T), V(T), thermal expansion, Gruneisen |

---

## Accuracy (benchmark run 2026-06-03, MACE-medium/float64)

Validated on 11 materials spanning the full stiffness range (soft halides ~24 GPa to diamond 435 GPa), with reference values from Materials Project DFT VRH.

**Aggregate:**

| Metric | Value |
|--------|-------|
| n | 11 |
| MAE | **5.67 GPa** |
| MAPE | **5.75%** |
| RMSE | 7.14 GPa |
| Bias | +0.51 GPa (near-zero) |
| Median % error | 3.1% |
| R² | 0.996 |
| Pearson r | 0.998 |
| Spearman r | 0.991 |
| Qualitative rank correct | yes (oxides > halides/sulfides) |

**Per-material:**

| Material | Family | Ref (GPa) | Pred (GPa) | Error % |
|----------|--------|-----------|-----------|---------|
| Si | other | 88.9 | 74.6 | -16.1% |
| TiO2 | oxide | 208.5 | 209.4 | +0.4% |
| Li2O | oxide | 118.8 | 130.9 | +10.2% |
| Li3PO4 | oxide | 148.8 | 153.7 | +3.2% |
| Li2S | sulfide | 40.5 | 34.5 | -14.7% |
| LiCl | halide | 32.2 | 33.2 | +3.1% |
| C (diamond) | other | 435.2 | 445.2 | +2.3% |
| MgO | oxide | 151.5 | 146.9 | -3.0% |
| Al2O3 | oxide | 231.0 | 227.5 | -1.5% |
| NaCl | halide | 23.8 | 25.4 | +7.0% |
| SiC | other | 212.7 | 216.2 | +1.7% |

9 of 11 within 10%; 6 of 11 within 4%. The two larger errors (Si and Li2S) are consistent with known MACE-MP behaviour: Si is a covalent semiconductor with a different electronic character from the oxide/sulfide training distribution, and Li2S is at the soft edge where second-derivative noise is largest.

To reproduce: run `python benchmark_accuracy.py --set standard` from `BREED/env/bulk_modulus/`.

---

## Other models (in `BREED/env/bulk_modulus/`)

Two earlier ML regression models were developed before the physics approach and live in the dev environment:

| File | Approach | Notes |
|------|----------|-------|
| `composition_bulk_modulus_predictor.py` | Random forest on composition features | Can't distinguish polymorphs; fast, no CIF needed |
| `bulk_modulus_structure.py` | Random forest on structure + composition features | Requires CIF; trained on MP elastic data |

The physics model (`physics_bulk_modulus.py`) supersedes both — it requires no labeled K training data and extrapolates more reliably to novel SSE compositions outside the MP training distribution.

---

## Test suite

| File | What it tests |
|------|--------------|
| `test_tier0_eos.py` | Phase 0–4 on a simple elemental EOS (sanity baseline) |
| `test_tier12_si_tio2.py` | Phase 1–2 structure acquisition and relaxation on Si/TiO2 |
| `test_tier3_disorder.py` | Phase 1 disorder handling (Ewald ordering, occupancy snapping) |
| `test_tier4_sse.py` | Phase 3–4 on SSE candidates from the OBELiX chemical family |
| `test_tier5_phonons_qha.py` | Phase 3b/4b phonon gate and QHA finite-T path |
| `test_volume_check_fix.py` | Regression test for the MLIP V0 vs MP V0 sanity check |

Run in order of ascending cost: tier 0 first, tier 5 last (requires phonopy and is slow).

---

## Dependencies

```
mace-torch    # or sevenn / chgnet / matgl
phonopy       # for --qha / --validate-phonons only
pymatgen      # structure I/O and MP API
ase           # ASE >= 3.23 for FrechetCellFilter
```
