# 05 — Ensemble Model & Benchmark Comparison

## Why an ensemble?

The fundamental tension: only the OBELiX entries (~478) had CIF files, which meant BVSE migration barriers could only be computed for that subset. The Liverpool entries (~348) had no CIFs — composition-only. Two naive options were both bad:

- **Train on OBELiX-only with BVSE:** richer features but throw away 42% of the data
- **Train on everything with composition-only:** more data but no structural physics

The ensemble solves this by training both models and combining them. Model 1 captures the BVSE physics for the entries where it's available; Model 2 uses all the data. The weighted average means Liverpool data contributes rather than being discarded, and the BVSE signal is preserved for the OBELiX entries. Weights are determined automatically by inverse CV MAE — the better-performing model gets higher weight, no hand-tuning.

---

## Model 1: BVSE + composition (OBELiX-only, 478 samples)

**16 Tier 1 composition features** (same as `01_composition_only/`):
`li_count`, `mean_electronegativity`, `anion_electronegativity`, `cation_anion_en_diff`, `mean_ionic_radius`, `anion_ionic_radius`, `radius_std`, `anion_polarizability`, `mean_atomic_mass`, `n_elements`, `mean_atomic_number`, `anion_halide`, `anion_mixed`, `anion_other`, `anion_oxide`, `anion_sulfide`

**3 defect proxy features** (computed from formula via pymatgen charge-balance):
| Feature | Description |
|---|---|
| `li_stoich_deviation` | \|Li_actual − Li_ideal\| where ideal is the charge-balanced stoichiometry |
| `dopant_presence` | Count of elements with non-integer reduced stoichiometry (>0.05 threshold) |
| `defect_strength` | li_stoich_deviation + dopant_presence combined |

**9 BVSE features** (from bvlain + CIF files):
| Feature | Description |
|---|---|
| `bvse_energy` | 3D percolation barrier in eV — primary migration barrier (Eₐ proxy) |
| `topology_score` | Dimensionality / 3 — normalised pathway connectivity (0=none, 1=3D network) |
| `bvse_anisotropy` | max − min of 1D/2D/3D barriers — spread across transport directions |
| `bvse_bottleneck` | Void bottleneck radius from void distribution (Å) |
| `bvse_accessible` | Fraction of grid points within 1 eV of global minimum |
| `bvse_li_count` | Li atoms per unit cell (from CIF) |
| `bvse_available` | 0/1 flag: whether real or proxy BVSE data exists (vs. filled with zeros) |
| `bvse_energy_inv` | 1 / (barrier_3d + 1e-6) — amplifies differences at low barriers |
| `li_mobility_capacity` | bvse_accessible × bvse_li_count — combined accessibility × carrier count |

**Total: 28 features.** Sample weights: real BVSE = 1.0, proxy = 0.5, guessed (no CIF) = 0.1.

CV MAE: **0.945 ± 0.099**

---

## Model 2: Composition-only (OBELiX + Liverpool, 826 samples)

Same 16 Tier 1 features + 3 defect proxy features = **19 features total**. No BVSE. Uniform sample weights.

CV MAE: **0.851 ± 0.080**

---

## Ensemble

```
weight_i = (1 / CV_MAE_i) / sum(1 / CV_MAE_j)

Model 1 weight: 0.474    Model 2 weight: 0.526
log₁₀(σ) = 0.474 × Model1 + 0.526 × Model2
```

---

## Benchmark

| Model | Train N | CV MAE |
|---|---|---|
| OBELiX RF ([published baseline](https://arxiv.org/abs/2502.14234)) | 478 | 1.531 |
| Our GBT (composition, 478 pts) | 478 | 1.440 |
| Our GBT (composition, 826 pts) | 826 | 1.192 |
| **Final ensemble** | 478 + 826 | **~0.90** |

Earlier scripts in this folder (`GBT_predict_LLZO.py`, `XGB_predict_LLZO.py`) are predecessors to the current `XGB_ensemble_LLZO.py` at root.
