# XGB Ensemble — Design & Methodology

Script: `XGB_ensemble_LLZO.py`  
Current target compound: **Li₆PS₅Cl** (argyrodite)  
Experimental reference: ~10⁻³ S/cm

---

## What this model is (and isn't)

This model gives meaningful quantitative conductivity estimates and reliably identifies the correct order of magnitude — it has strong screening ability. It is **not** rank-only.

However, exact σ values are limited by factors that no model of this kind can fully overcome:

- **Experimental heterogeneity** — reported conductivities for nominally identical compositions vary by 0.5–1 log unit across labs due to sintering conditions, grain boundaries, pellet density, and measurement protocol. This noise is in the training labels, not a modelling artefact.
- **Regime mixing** — the training set blends oxides, sulfides, halides, and disordered conductors into a single model. These material classes follow qualitatively different transport physics, so a global model makes compromises that a class-specific model would not. Performance could be improved with specialised models per structure type.
- **Unmodeled physics** — grain boundary resistance, processing history, finite-temperature dynamics, and many-body effects are absent from the feature set.

As a result, **errors of ~0.8–0.95 log units are expected**, corresponding roughly to a factor of 6–9× on the linear conductivity scale. This is largely a ceiling imposed by the data, not just model weakness.

The model provides **reliable screening-level predictions**, typically accurate to within ~0.8–1.0 log₁₀ units on held-out experimental data. It is best interpreted as a **ranking and order-of-magnitude estimator across chemical space**, rather than a precise physical simulator of σ for individual compounds.

---

## Overview

Two XGBoost regressors are trained independently and combined via an inverse-MAE weighted average to predict log₁₀(σ) in S/cm at 300 K.

```
final_prediction = w₁ · model1(X_bvse+comp) + w₂ · model2(X_comp)

where  wᵢ = (1/MAEᵢ) / (1/MAE₁ + 1/MAE₂)
```

The model with lower 5-fold CV MAE automatically receives higher weight. No weight is hand-tuned.

---

## Training Data

| Dataset | Source | ~N samples | Used by |
|---|---|---|---|
| OBELiX | Experimental database | ~478 | Model 1 (BVSE+comp) |
| OBELiX + Liverpool | Combined | ~826 | Model 2 (comp-only) |

- IDs are loaded from `comp_train.csv` (column `source == 'obelix'` identifies OBELiX entries)
- Features are loaded from `comp_train_features.csv`
- BVSE features are loaded from `bvse_features_combined.csv`
- BVSE source metadata (CIF matching quality) is loaded from `cif_metadata.csv` if present

---

## Model 1: BVSE + Composition (OBELiX-only)

**Training set:** OBELiX entries only (~478 samples after NaN filter)  
**Params cache:** `xgb_best_params.json`  
**Sample weights:** BVSE data confidence (see below)

### Feature set

All features are physically inspired descriptors used in a statistical model — they encode chemistry and approximate structural information, but the model learns purely from empirical correlations with measured conductivity.

**Composition features** (from `comp_train_features.csv`, minus meta columns):
| Feature | Description |
|---|---|
| `li_count` | Absolute Li count in formula unit |
| `mean_electronegativity` | Composition-weighted Pauling EN |
| `anion_electronegativity` | EN of most electronegative anion |
| `cation_anion_en_diff` | Anion EN − weighted cation EN |
| `mean_ionic_radius` | Composition-weighted ionic radius |
| `anion_ionic_radius` | Ionic radius of dominant anion |
| `radius_std` | Std dev of per-atom ionic radii |
| `anion_polarizability` | Weighted anion polarizability (Å³) |
| `mean_atomic_mass` | Composition-weighted atomic mass |
| `n_elements` | Number of distinct elements |
| `mean_atomic_number` | Composition-weighted atomic number |
| `anion_oxide` / `anion_sulfide` / `anion_halide` / `anion_mixed` / `anion_other` | One-hot anion type |

**Defect proxy features** (computed from formula via pymatgen):
| Feature | Description |
|---|---|
| `li_stoich_deviation` | \|Li_actual − Li_ideal\| where ideal is charge-balanced |
| `charge_compensation_proxy` | Sum of per-site oxidation state deviation from expected |
| `dopant_presence` | Count of elements with non-integer reduced stoichiometry (>0.05 threshold) |
| `defect_strength` | `li_stoich_deviation + dopant_presence` |

> **Note on defect features:** In practice, many of these features have near-zero variance for stoichiometric compositions, which dominate the dataset. Their contribution to model performance is generally weak. They are retained because they add signal for doped/off-stoichiometric entries.

**BVSE features** (from `bvse_features_combined.csv` + CIF parsing):
| Feature | Source column | Description |
|---|---|---|
| `bvse_energy` | `barrier_3d` | 3D percolation barrier (eV) — primary BVSE descriptor |
| `topology_score` | `dimensionality / 3` | Normalised Li-pathway dimensionality (0–1) |
| `bvse_anisotropy` | max − min of 1D/2D/3D barriers | Spread of directional barriers |
| `bvse_bottleneck` | `bottleneck_radius` | Void bottleneck radius from void distribution (Å) |
| `bvse_accessible` | `accessible_fraction` | Fraction of grid points within 1 eV of minimum |
| `bvse_li_count` | CIF Li site count | Number of Li atoms per unit cell |
| `bvse_available` | 0/1 flag | Whether real/proxy BVSE data exists (not guessed) |
| `bvse_energy_inv` | 1 / (barrier_3d + 1e-6) | Reciprocal barrier — amplifies differences at low Ea |
| `li_mobility_capacity` | `bvse_accessible × bvse_li_count` | Combined accessibility × Li count proxy |

> **Note on BVSE:** These features are approximate proxies for migration barriers derived from a static, classical energy landscape. They capture broad trends in Li hopping feasibility but ignore disorder, many-body effects, and finite-temperature dynamics. BVSE is not ground-truth physics — it is a computationally cheap structural descriptor that correlates with conductivity in practice.

**Total Model 1 features:** len(comp_feat_cols) + 4 defect + 9 BVSE

### BVSE source confidence & sample weights

Each training sample is weighted by how reliably its BVSE barriers were obtained. These weights express **confidence in the input data**, not a correction for underlying physical bias:

| `bvse_source` | Weight | Condition |
|---|---|---|
| `real` | 1.0 | `barrier_3d` is present and `source_type` is NaN or `exact_formula`/`exact_chemsys` |
| `proxy` | exp(−3 × lattice_score), clipped [0.05, 0.75] | `source_type == 'proxy_parent'` |
| `guessed` | 0.1 | `barrier_3d` is NaN (BVSE unavailable, features filled with 0) |

---

## Model 2: Composition-Only (OBELiX + Liverpool)

**Training set:** Full dataset (~826 samples after NaN filter)  
**Params cache:** `xgb_comp_only_best_params.json`  
**Sample weights:** Uniform (None)

### Feature set

Same composition features + defect proxy features as Model 1.  
No BVSE features.

**Total Model 2 features:** len(comp_feat_cols) + 4 defect

---

## Training Procedure

Both models use the same pipeline:

### Hyperparameter tuning (Optuna)
- 200 trials, TPE sampler (seed=42)
- Objective: 5-fold CV MAE on log₁₀(σ)
- Search space:

| Hyperparameter | Range |
|---|---|
| `n_estimators` | 100 – 1000 |
| `max_depth` | 2 – 8 |
| `learning_rate` | 0.005 – 0.3 (log scale) |
| `subsample` | 0.4 – 1.0 |
| `colsample_bytree` | 0.3 – 1.0 |
| `min_child_weight` | 1 – 20 |
| `reg_alpha` | 1e-8 – 10 (log scale) |
| `reg_lambda` | 1e-8 – 10 (log scale) |
| `gamma` | 0 – 5 |

Results are cached to JSON. On subsequent runs the cached params are loaded directly, skipping Optuna.

### Cross-validation (MAE evaluation)
- 5-fold CV, `KFold(shuffle=True, random_state=42)`
- Each fold: inner 85/15 split of the training portion for early stopping
- `early_stopping_rounds=30`
- Reported MAE = mean across 5 folds; also reports ± std

### Final model training
- 85/15 train/validation split of the full dataset (random_state=42) for early stopping
- Same `early_stopping_rounds=30`

---

## BVSE Pipeline for Target Compound

This runs once at the start of the script to generate BVSE features for the query compound.

### Step 1 — Fetch structure from Materials Project
- Queries MP by `FORMULA`
- Selects the entry with the lowest energy above hull (GGA+U preferred, falls back to first available entry)
- Writes the structure to `<formula>_mp.cif` for reference

### Step 2 — Manual oxidation state assignment
Oxidation states are assigned from a hardcoded lookup table (`OXIDATION_STATES`) covering common solid-electrolyte elements:

```
Li/Na/K: +1 | Mg/Ca/Ba/Zn/Cd: +2 | Al/Ga/In/B/La/Y: +3
Si/Ge/Sn/Ti/Zr/Hf: +4 | P/As/Nb/Ta: +5
O/S/Se/Te: -2 | F/Cl/Br/I: -1
```

For Li₆PS₅Cl: Li=+1, P=+5, S=−2, Cl=−1 → charge sum = 6(+1) + 1(+5) + 5(−2) + 1(−1) = **0** ✓

Any element not in the table triggers a pymatgen BVAnalyzer fallback with a warning printed. Charge neutrality is verified and printed before proceeding.

**Why manual assignment?**  
`bvlain` uses `BVAnalyzer` internally when reading a CIF (`oxi_check=True` default). For complex sulfide electrolytes, BVAnalyzer can assign incorrect oxidation states (e.g., S=−1 for LGPS) because bond lengths at mixed/disordered sites don't match expected BV sums. Manual assignment bypasses this by calling `calc.read_structure(structure, oxi_check=False)` with the already-decorated pymatgen Structure.

### Step 3 — BVSE calculation (bvlain)
```python
calc.bvse_distribution(mobile_ion="Li1+", r_cut=10.0, resolution=0.2)
calc.void_distribution(mobile_ion="Li1+", r_cut=10.0, resolution=0.2)
```

Extracted quantities:
| Variable | Description |
|---|---|
| `barrier_1d/2d/3d` | Percolation barriers along each dimensionality (eV) |
| `dimensionality` | Highest dimension with finite barrier (0–3) |
| `anisotropy` | max(finite barriers) − min(finite barriers) |
| `accessible_fraction` | Fraction of grid within 1 eV of global minimum |
| `bottleneck_radius` | Void bottleneck radius: r_3D if >0, else r_2D (Å) |
| `li_site_count` | Li atoms per unit cell from MP structure composition |

---

## Ensemble Combination

```python
inv1 = 1.0 / cv_mae1
inv2 = 1.0 / cv_mae2
w1   = inv1 / (inv1 + inv2)
w2   = inv2 / (inv1 + inv2)

log_sigma_ens = w1 * log_sigma1 + w2 * log_sigma2
```

Better CV performance (lower MAE) → higher weight. No manual tuning.

### Uncertainty estimate

```python
ens_mae = w1 * cv_mae1 + w2 * cv_mae2
sigma_low  = 10^(log_sigma_ens - ens_mae)
sigma_high = 10^(log_sigma_ens + ens_mae)
```

This interval is an **empirical error proxy**, not a statistically rigorous prediction interval. It reports ±1 CV-MAE in log space, which approximates the typical magnitude of error on held-out data but makes no distributional assumptions and does not have a formal coverage guarantee (e.g., it is not a 68% or 95% confidence interval). Treat it as an order-of-magnitude error band.

---

## Limitations Summary

| Limitation | Impact |
|---|---|
| Experimental label noise (~0.5–1 log unit) | Hard floor on achievable MAE; not a modelling failure |
| Regime mixing (oxides, sulfides, halides in one model) | Global model makes class-level compromises; per-class models would perform better |
| BVSE is a static, classical proxy | Ignores disorder, many-body effects, finite-T dynamics; correlated with Ea but not ground truth |
| Defect features low-variance for stoichiometric data | Weak signal in practice; relevant mainly for doped entries |
| Uncertainty interval is empirical, not statistical | ±1 MAE band is an error proxy, not a confidence interval |
| Sample weights reflect data confidence, not physical bias | Weighting improves training stability but does not correct for regime differences |

---

## Changing the Target Compound

1. Update `FORMULA` at the top of the script
2. Check that all elements are in `OXIDATION_STATES`; add any that are missing
3. Verify the charge sum printed at runtime is 0
4. Rename the CIF output path if desired (cosmetic only)
5. Variable prefixes (`lpscl_`) are internal — no functional impact

---

## Files

| File | Purpose |
|---|---|
| `comp_train.csv` | Source labels per sample (`obelix` / `liverpool`) |
| `comp_train_features.csv` | Precomputed composition features for all training samples |
| `bvse_features_combined.csv` | Precomputed BVSE barriers for OBELiX CIFs |
| `cif_metadata.csv` | CIF matching metadata (source_type, lattice_score) — optional |
| `cifs/` | CIF files used to compute li_site_count per training sample |
| `xgb_best_params.json` | Cached Optuna params for Model 1 |
| `xgb_comp_only_best_params.json` | Cached Optuna params for Model 2 |
| `<formula>_mp.cif` | CIF written from MP for the query compound |
