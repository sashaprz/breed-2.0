# BREED 2.0 — Ionic Conductivity Predictor

## Goal: build a ML model to predict ionic conductivity

For lithium solid electrolytes (log₁₀ σ in S/cm at 300 K), as part of the BREED genetic algorithm. The target is to beat the [published OBELiX RF baseline](https://arxiv.org/abs/2502.14234) (MAE 1.531)

The broader context: ionic conductivity is one of the most important properties for electorlytes, as an electrolyte's main job is to move ions. 
---

## The overarching approach

Five things physically determine ionic conductivity: **anion framework geometry, bottleneck size, carrier concentration/disorder, lattice dynamics, and pathway dimensionality.** The strategy was to encode these progressively through three tiers of features, adding complexity one tier at a time and measuring whether each tier justified the cost in data (CIFs) or compute.

**Tier 1 — Composition only** (no CIF needed)
Chemical formula descriptors: anion type, ionic radii, electronegativity, Li count, n_elements, polarizability. Captures *why* sulfides beat oxides (larger anions, lower EN, weaker Li binding) but can't distinguish between two garnets or two argyrodites.

**Tier 2 — Local structure** (CIF required)
Coordination environment around Li sites: nearest-neighbor count, Li–anion bond distances (mean + variance), polyhedral volume/distortion, tetrahedral vs octahedral fraction, min Li–Li distance. Captures *where atoms are* but not how Li moves between them.

**Tier 3 — Transport physics** (CIF required, harder)
Two approaches:
- **Zeo++** — geometric void analysis: bottleneck diameter, probe-occupiable volume (POAV). Designed for porous materials; turned out to be a poor fit for dense solid electrolytes.
- **BVSE** — Bond Valence Site Energy: maps the energy landscape Li⁺ experiences as it moves through the crystal, extracts the migration barrier (Eₐ). This directly approximates the Arrhenius activation energy, σ ∝ exp(−Eₐ/kT).

---

## What we learned

| Observation | Implication |
|---|---|
| Adding 348 Liverpool points → −17% MAE | More data beats better features at this scale |
| Tier 2 structural features → −6% MAE, but CIF restriction → +20% MAE | Data loss from requiring CIFs dominates any feature gain |
| Zeo++ POAV: only 22/404 structures had connected channels | Li hops, doesn't flow — wrong tool for dense electrolytes |
| BVSE barrier_3d: Spearman ρ = −0.455 on ordered structures | Strong physical signal, but cuts 60% of data (disordered structures) |
| Composition-only model wins globally | It pattern-matches chemistry groups, not transport physics |

**Key tension:** every feature tier that adds physical information also requires a CIF file (shrinking the dataset) and often ordered structures (shrinking the dataset even more). At 800 samples, data quantity beats feature quality.

Initially the composition only model was best simply because it had more data. Eventually I figured out how to add more physics-based features AND simultaneously use all the data. This was in my ensemble model. 

---

## Current best model

Two XGBoost regressors, inverse-MAE weighted ensemble:

| Model | Features | Train N | CV MAE |
|---|---|---|---|
| Model 1 (BVSE + composition) | Tier 1 + BVSE barriers | 478 (OBELiX, has CIFs) | 0.945 ± 0.099 |
| Model 2 (composition-only) | Tier 1 | 826 (OBELiX + Liverpool) | 0.851 ± 0.080 |
| **Ensemble** | weighted average | — | **~0.90** |

Ensemble weights: Model 1 = 0.474, Model 2 = 0.526 (inverse-MAE, no hand-tuning).

**Noise floor:** experimental MAD ≈ 0.41 log units. Measurement variability (bulk vs total conductivity, sintering, grain boundaries) sets an irreducible noise ceiling. We got a MAE of 0.85–0.95 means we're not far above it.

**Validated on known electrolytes:**
| Material | Predicted σ | Experimental | Error |
|---|---|---|---|
| Li₆PS₅Cl (argyrodite) | 5.3×10⁻⁴ S/cm | ~10⁻³ S/cm | 2× |
| LGPS (Li₁₀GeP₂S₁₂) | 2.5×10⁻³ S/cm | ~10⁻² S/cm | 4× |
| LLZO (Li₇La₃Zr₂O₁₂) | 6.3×10⁻⁵ S/cm | ~10⁻⁴–10⁻³ S/cm | in range |

All within ~1 log unit of experiment. Relative ordering is correct: Li₆PS₅Cl > LGPS > LLZO. Yay the model did good! 

---

## Folder structure

| Folder | What it contains |
|---|---|
| `00_data_preparation/` | Dataset merging, CIF matching, train/test split |
| `01_composition_only/` | Tier 1 features, GBT/XGB baseline |
| `02_tier2_structural/` | Tier 2 local structure features |
| `03_tier3_zeo/` | Zeo++ geometric features (POAV, bottleneck) |
| `04_tier3_bvse/` | BVSE migration barriers |
| `05_ensemble_benchmark/` | Benchmark comparison, earlier scripts |
| `cifs/` | All CIF files + Zeo++ outputs (.res, .volpo) |
| `cifs_cleaned/` | Option 1 cleaned CIFs (dominant species, drop low-occupancy) |
| `cifs_ordered/` | pymatgen OrderDisorderedStructure CIFs |
| `raw_data/` | Original downloaded datasets |
| `scratch/` | Scratch notebooks and editor artifacts |

**Production files (root):**
- `XGB_ensemble_LLZO.py` — prediction script (set FORMULA at top, run to predict)
- `FINAL_ensemble.py` — ensemble training + evaluation
- `FINAL_explainer.md` — full methodology documentation

---

## Setup

```
pip install -r requirements.txt
$env:MP_API_KEY = "your_key"   # materialsproject.org/api
python XGB_ensemble_LLZO.py
```
