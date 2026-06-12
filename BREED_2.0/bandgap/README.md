# Band-Gap Screening

This directory predicts band-gap in **two stages**. Main issue: **DFT-PBE systematically underestimates band gaps** by
30–60% relative to HSE06/experiment. So, first there is a CGCNN that will predict PBE bandgap (trained on Materials Project 
data), and then there is a scissor correction that will apply an offset to a PBE bandgap to estimate HSE bandgap. 

**Stage 1** (`cgcnn_train/`) is an ML surrogate trained to *reproduce* Materials
Project's PBE `band_gap` field from a CIF — it stands in for a PBE-DFT
calculation so the GA can triage thousands of candidates cheaply. Its accuracy
(§ Part 1, "Accuracy") is measured against real MP PBE values.

**Stage 2** (`second_pass.py`) takes an *actual* PBE band gap (computed
for the smaller set of candidates that survive to DFT) and applies a frozen
**scissor correction** — a fixed additive offset calibrated against real
HSE06/experimental gaps for SSE materials — to estimate the true gap before
the final electronic-stability filter is applied.

In short: Stage 1 predicts what PBE-DFT *would* say, cheaply; Stage 2 corrects
what PBE-DFT *actually* said, toward HSE. Neither stage should be skipped —
a candidate's gap is only meaningfully comparable to an HSE/experimental
threshold after Stage 2.


## Part 1 — First Pass: CGCNN Ensemble Predictor

A retrained CGCNN-based predictor that estimates a candidate's **PBE band gap
directly from its crystal structure** (CIF). This is the fast ML surrogate
that runs on every GA-generated candidate before any DFT is performed.

### How it works

```
CIF  ──►  CGCNN crystal graph  ──►  classifier ──► P(metal)
            (92-dim atom feats,                 │
             8 Å radius, 12 nbrs)               ├─ P(metal) >= 0.480 ──► gap = 0 eV
                                                 │
                                                 └─ else ──► gap = mean(5 regressors)
                                                             uncertainty = std(5 regressors)
```

- **Classifier**: predicts metal vs. non-metal. A material classed as a metal
  is pinned to exactly **0 eV** (no "0.4 eV floor" like the old vendored model).
- **Regressor ensemble**: 5 independently-seeded CGCNN regressors, trained
  **only on non-metals**, so they specialize on predicting finite gaps. The
  ensemble mean is the prediction; the spread across the 5 members is reported
  as an uncertainty.
- Both heads share the same graph representation (`cgcnn/model.py`,
  `CrystalGraphConvNet`, `atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1`) —
  the same architecture as the vendored `band-gap.pth.tar`, so the original
  weights could be used as a warm start.

### How it was trained

Code lives in `cgcnn_train/` (see `cgcnn_train/README_TRAINING.md` for the full
step-by-step pipeline). Trained 2026-06-09/10 on `mariana.matter.sandbox`
(CPU, via SLURM).

| # | Improvement over the vendored `band-gap.pth.tar` |
|---|---|
| 1 | Retrained on **current** Materials Project data (not the ~2018 snapshot) |
| 2 | **Composition-grouped** train/val/test split — no polymorph leakage |
| 3 | Separate **metal / non-metal classifier** |
| 4 | Regressor ensemble trained **on non-metals only** |
| 5 | **Tuned** metal decision threshold (val-optimized, not a fixed 0.5) |
| 6 | **Band-gap-range-weighted** loss (wide gaps up-weighted) |
| 7 | **Per-range MAE** tracked every epoch (0–1 / 1–3 / 3–6 / >6 eV) |
| 8 | **Ensemble of 5 seeds**; prediction std = uncertainty |

**Dataset**: 154,373 materials fetched from MP, split by composition group
(104,734 groups, zero leakage between splits):

| Split | Materials |
|-------|-----------|
| Train | 123,865 |
| Val   | 15,232 |
| Test  | 15,276 |

**Loss weighting** (`WEIGHT_BY_BIN = [1.0, 1.0, 2.0, 4.0]`): the 3–6 eV and
>6 eV bins are up-weighted 2x/4x in the regressor loss because they're rare in
MP's natural distribution.

**Training**: 60 epochs, Adam, lr=0.01 with step decay at epochs 40/55, batch
size 256. 5 regressors (seeds 0–4) + 1 classifier.

### Accuracy

#### 1. Held-out MP test split (the model's own 15,276-material test set)

Classifier — balanced accuracy at the tuned threshold (0.480):

| Metric | Value |
|---|---|
| Balanced accuracy | 0.873 (n_val=15,232) |
| (default 0.5 threshold) | 0.872 |

Regressor ensemble — non-metal test MAE (n=8,333):

| Range | n | MAE (eV) |
|---|---|---|
| 0–1 eV | 2,639 | 0.473 |
| 1–3 eV | 3,349 | 0.449 |
| 3–6 eV | 2,230 | 0.579 |
| >6 eV  | 115   | 0.847 |
| **Overall** | **8,333** | **0.497** |

#### 2. Independent benchmark vs. Materials Project (gap-stratified, n=300, seed=42)

This is a fresh sample (`cgcnn_train/benchmark_mp.py`), scored against MP's
PBE `band_gap` field, run through the *full* pipeline (classifier + ensemble,
metals pinned to 0). Same sampling strategy/seed as the old single-model
benchmark, so the two are directly comparable:

| Metric | New ensemble | Old vendored CGCNN |
|---|---|---|
| MAE | **0.403 eV** | 0.70 eV |
| RMSE | **0.786 eV** | 1.20 eV |
| R² | **0.890** | 0.74 |
| Pearson r | **0.944** | 0.87 |
| Bias (pred − actual) | **−0.087 eV** | +0.31 eV |
| Metal classification accuracy | **0.893** | 0.86 |

MAE by gap range (new ensemble):

| Range | n | MAE (eV) |
|---|---|---|
| [0, 0.3) eV | 50 | 0.203 |
| [0.3, 1) eV | 50 | 0.457 |
| [1, 2) eV   | 50 | 0.400 |
| [2, 3.5) eV | 50 | 0.424 |
| [3.5, 6) eV | 50 | 0.291 |
| [6, 99) eV  | 50 | 0.642 |

Non-metal-only MAE on this benchmark: **0.437 eV** (n=258), consistent with the
0.497 eV seen on the model's own held-out test split.

**Bottom line**: roughly a **40% MAE reduction** and a **3.5x reduction in
prediction bias** over the old vendored model, on the same evaluation
methodology.

Full results (per-material CSV, scatter plot, summary JSON) are in
`cgcnn_train/benchmark_results/`.

### Usage

```bash
cd cgcnn_train

# Single structure
python predict.py --cif candidate.cif

# Directory of candidates
python predict.py --cif-dir ./my_candidates --out preds.csv

# Re-run the model's own held-out test split
python predict.py --test --out test_predictions.csv

# Re-run the independent MP benchmark
python benchmark_mp.py --n 300 --out-dir benchmark_results
python benchmark_mp.py --smoke   # quick 30-material check
```

`predict.py` returns, per material:
`{material_id, predicted_band_gap_eV, uncertainty_eV, p_metal, classified_metal}`.

### File reference (Part 1)

| File / dir | Contents |
|---|---|
| `cgcnn_train/` | Full training pipeline (see `README_TRAINING.md`) |
| `cgcnn_train/models/` | Trained checkpoints — classifier, 5 regressors, `threshold.json`, per-seed histories |
| `cgcnn_train/splits/` | Train/val/test material-id lists + `split_meta.json` |
| `cgcnn_train/predict.py` | Inference: classifier + ensemble |
| `cgcnn_train/benchmark_mp.py` | Independent MP accuracy benchmark (§2 above) |
| `cgcnn_train/test_predictions.csv` | Predictions over the held-out test split |
| `cgcnn_train/benchmark_results/` | Benchmark outputs (CSV, scatter plot, summary JSON) |

### Caveats

- **PBE-level only.** This predicts the same quantity Materials Project
  reports (PBE/GGA(+U)), which systematically *underestimates* true gaps by
  30–60%. The Stage-2 scissor correction (Part 2 below) handles the PBE→HSE
  step for candidates that reach DFT — this model's output should not be
  compared directly to an HSE/experimental threshold.
- **>6 eV bin is the weakest** (MAE 0.85 eV on the held-out test set, 0.64 eV
  on the MP benchmark), despite 4x loss up-weighting — wide-gap insulators are
  rare in MP, so this remains the highest-uncertainty region. Treat
  ensemble `uncertainty_eV` as a signal here.
- **In-distribution caveat.** Both the held-out test split and the MP
  benchmark sample are drawn from Materials Project. Genuinely novel
  GA-generated structures may fall outside this distribution; use
  `--min-mpid` on `benchmark_mp.py` to bias the benchmark toward newer
  (more likely held-out) materials for a stricter estimate.
- **Misclassified metals are the biggest single-material error mode**: a true
  non-metal misclassified as a metal is forced to 0 eV regardless of its real
  gap, which is a much larger error than the regressor's own MAE.

---

## Part 2 — Second Pass: Scissor Correction (PBE → HSE)

`second_pass.py` implements a **scissor correction** that post-processes the
**raw PBE band gap from a VASP calculation** (the Stage-1 output above is
*not* an input here — this stage corrects real DFT results) and converts it to
a more accurate estimate before any band-gap filter is applied to SSE
candidates.

### Why a Scissor Correction?

DFT-PBE systematically underestimates band gaps by 30–60% in inorganic solids.
A candidate screened on its raw PBE gap will be judged too favourably (the gap
looks narrow, making the material appear less stable electronically than it really
is).  The scissor shifts the PBE gap toward the true gap using a single offset or
linear relation fit once to a set of anchor materials.

The correction is calibrated on **inorganic solid-state electrolytes (SSEs)** —
the same chemical family as the candidates — so it captures family-specific
trends rather than a generic PBE error.

### Methodology

#### 1. Anchor Set

Twenty inorganic SSEs spanning the sub-families found in the OBELiX dataset:

| Sub-family      | Representatives                                 |
|-----------------|------------------------------------------------|
| Oxide           | LLZO, Li2O, LiAlO2                              |
| Phosphate       | Li3PO4                                          |
| Anti-perovskite | Li3OCl                                          |
| Sulfide         | LGPS, beta-Li3PS4, Li2S, Li4GeS4, Li4SiS4      |
| Argyrodite      | Li6PS5Cl                                        |
| Halide          | LiCl, LiBr, LiI, Li3YCl6, Li3InCl6             |
| NASICON         | LATP, LAGP                                      |
| LISICON         | Li14ZnGe4O16                                    |
| Sulfate         | Li2SO4                                          |

Each anchor provides a matched `(E_g^PBE, E_g^trusted)` pair computed on the
**same GGA-relaxed geometry**.  Mixing geometry sources (e.g., using an
MP-relaxed structure for PBE and an experimentally determined structure for HSE06)
leaks structural differences into the offset.

**Trusted reference:**
- HSE06 hybrid functional for most materials (same geometry, same k-mesh,
  HFSCREEN = 0.2 bohr^-1).
- Experimental **fundamental** gaps (from photoemission / VUV band-to-band
  absorption) for simple halides (LiCl, LiBr, LiI). These are the quasiparticle
  gaps, NOT the first excitonic absorption onset. For alkali halides the first
  exciton lies 0.5-0.9 eV below the fundamental: LiCl fundamental = 9.4 eV,
  first exciton = 8.8 eV (Phys. Rev. B 88, 245202, 2013); LiI fundamental =
  6.4 eV (Phys. B 448, 68, 2014). The scissor targets a single consistent
  physical quantity (quasiparticle / fundamental gap) across all anchors.
- **Never** MP energies, r2SCAN, or any other DFT-level calculation — that is DFT
  calibrating DFT and does not correct the systematic underestimation.

**Required PBE setup (must match when computing candidates):**

    PAW PBE pseudopotentials, ENCUT >= 520 eV, Gamma-centred k-mesh,
    ISTART=0, ICHARG=2, no spin-orbit unless standard for the material.

#### 2. Form Selection via LOO Cross-Validation

Two forms are considered:

- **Constant**: `E_g^corr = E_g^PBE + delta`
- **Linear**: `E_g^corr = a * E_g^PBE + b`

Leave-one-out (LOO) cross-validation computes the MAE for each form across all
anchors.  The form with lower MAE is selected.  If fewer than 15 anchors are
available, the constant form is always used (linear overfits at that sample size).

**Current result (20 anchors, after optical→fundamental fix for LiI):**

    Constant LOO-MAE : 0.429 eV   <-- SELECTED
    Linear   LOO-MAE : 0.438 eV
    Frozen correction: E_g^corr = E_g^PBE + 1.278 eV

The constant form was selected; the correction is a fixed +1.28 eV additive
offset.  The LOO-MAE of **0.43 eV** is the honest uncertainty to report alongside
any corrected gap.

**Note on the LiCl outlier:** LiCl has delta = +3.15 eV vs the mean +1.28 eV.
This is not a measurement-type mismatch — the 9.40 eV anchor value is already
the fundamental (quasiparticle) gap, confirmed by GW calculations (Phys. Rev. B
88, 245202, 2013) that give fundamental = 9.5 eV vs first exciton at 8.8 eV.
The outlier is genuine PBE underestimation: alkali halides have large exciton
binding energies AND large absolute PBE errors. Cook's D for LiCl is >10 in the
constant fit, meaning it dominates the offset; true epistemic uncertainty is
±0.4–0.5 eV, not the 0.43 eV LOO-MAE alone.

#### 3. Freezing the Correction

Coefficients are frozen after fitting.  Every candidate gets the same frozen
offset applied — no per-candidate refitting that would leak information.

### Usage

```bash
# Show the current frozen correction:
python second_pass.py

# Re-fit and print full diagnostic table:
python second_pass.py --fit

# Apply to a single PBE gap:
python second_pass.py --apply 3.25
# -> E_g^corrected = 4.51 eV  (+/- 0.41 eV)

# Apply to every row in a CSV (must have an 'eg_pbe' column):
python second_pass.py --csv candidates.csv --out candidates_corrected.csv

# Parse a vasprun.xml, extract gap, apply correction (requires pymatgen):
python second_pass.py --vasprun path/to/vasprun.xml

# Load your own anchor CSV and refit:
python second_pass.py --fit --anchor-csv my_anchors.csv

# Force constant form regardless of LOO result:
python second_pass.py --fit --force-constant

# Save/load frozen correction as JSON:
python second_pass.py --fit --save-json correction.json
python second_pass.py --load-json correction.json --apply 2.5
```

#### Python API

```python
from second_pass import FITTED_CORRECTION

# Apply the frozen scissor to one gap:
eg_corrected = FITTED_CORRECTION.apply(eg_pbe)

# Full provenance:
print(FITTED_CORRECTION.summary())

# Batch screening:
from second_pass import screen_candidates
records = [{"id": "mat1", "eg_pbe": 3.1}, {"id": "mat2", "eg_pbe": 2.4}]
screen_candidates(records, FITTED_CORRECTION)
```

### Sources

#### [SOURCE 1] Thompson et al. — LLZO electrochemical window (VERIFIED, READ)

> T. Thompson, S. Yu, L. Williams, R. D. Schmidt, R. Garcia-Mendez, J. Wolfenstine,
> J. L. Allen, E. Kioupakis, D. J. Siegel, J. Sakamoto.
> "Electrochemical Window of the Li-Ion Solid Electrolyte Li7La3Zr2O12."
> *ACS Energy Letters* **2017**, *2*, 462–468.
> DOI: 10.1021/acsenergylett.6b00593

**What it contains:**
- LLZO (Li7La3Zr2O12) band gaps computed at three levels of theory (Table 1):
  - HSE06: **5.79–5.87 eV** (insensitive to Ta/Al doping level)
  - PBE+G0W0: 6.07 eV
  - HSE06+G0W0: 6.42 eV (authors' best estimate: ~6.4 eV)
- Experimental optical gap (Tauc plot, Al-doped LLZO): **5.46 eV**
- Electronic conductivity measurements confirming LLZO is an excellent insulator
- DFT setup: VASP, PAW, PBE (GGA) + HSE06 + G0W0; 4-formula-unit primitive cell

**What it does NOT contain:** PBE-only band gap for LLZO (the PBE gap is implicit
in the GW starting point but not reported as a standalone number). No data on any
other SSE material.

**Used for:** LLZO HSE06 = 5.81 eV in the anchor table (consistent with 5.79 eV
reported here; Binninger 2019 independently reports 5.81 eV).

---

#### [SOURCE 2] Binninger et al. — Comparison of SSE stability window methods (VERIFIED, READ)

> T. Binninger, A. Marcolongo, M. Mottet, V. Weber, T. Laino.
> "Comparison of computational methods for the electrochemical stability window
> of solid-state electrolyte materials."
> *arXiv:1901.02251* (IBM Research – Zurich, 2019). Published in *J. Mater. Chem. A*.

**What it contains:**
- PBE HOMO–LUMO gaps for seven SSE materials (Table 2, computed with Quantum
  ESPRESSO, PBE, SSSP Efficiency pseudopotentials, Γ-point only):

  | Material | PBE gap (eV) |
  |----------|-------------|
  | LGPS     | 2.21        |
  | LIPON    | 5.13        |
  | LLZO     | 4.34        |
  | LLTO     | 2.56        |
  | LATP     | 2.48        |
  | LISICON  | 3.63        |
  | NASICON  | 4.34        |

- HSE06 HOMO–LUMO gaps for three of those materials (Table 3, SG15 ONCV
  pseudopotentials, wavefunction cutoff 60–100 Ry):

  | Material | HSE06 gap (eV) |
  |----------|---------------|
  | LLZO     | 5.81          |
  | LLTO     | 4.28          |
  | LATP     | 4.19          |

- Stoichiometry stability windows (Li-insertion/extraction potentials) and phase
  stability windows for LGPS, LIPON, LLZO, LLTO, LATP, LISICON, NASICON.

**What it does NOT contain:** Band gap data for Li3PS4, Li2S, Li6PS5Cl, any halide
SSE, Li3OCl, Li3PO4, Li2O, LiAlO2, LAGP, Li14ZnGe4O16, Li2SO4.

**Used for:** LLZO PBE = 4.34 eV, LLZO HSE06 = 5.81 eV; LATP PBE = 2.48 eV,
LATP HSE06 = 4.19 eV; LGPS PBE = 2.21 eV.

---

#### [SOURCE 3] Richards et al. — Interface stability in solid-state batteries (READ, NO BAND GAP DATA)

> W. D. Richards, L. J. Miara, Y. Wang, J. C. Kim, G. Ceder.
> "Interface Stability in Solid-State Batteries."
> *Chemistry of Materials* **2016**, *28*, 266–273.
> DOI: 10.1021/acs.chemmater.5b04082

**What it contains:** Grand-canonical phase diagrams for SSE–electrode interfaces.
Electrochemical stability *windows* expressed in V vs Li/Li+, computed from reaction
free energies at the PBE level (VASP, PAW, 520 eV cutoff, ≥500 k-points/atom).
Covers LGPS, LLZO, Li3PS4, Li6PS5Cl, and related materials.

**What it does NOT contain:** No PBE band gaps. No HSE06 calculations. No band
gap table of any kind. The stability "windows" are voltage ranges (e.g. 1.7–3.9 V),
not band gaps in eV.

**Status:** Previously cited incorrectly in this anchor table as a source for HSE06
band gaps. **Those citations have been removed.**

---

#### [SOURCE 4] Zhu, He, Mo — Origin of stability in lithium solid electrolytes (READ, NO BAND GAP DATA)

> Y. Zhu, X. He, Y. Mo.
> "First principles study on electrochemical and chemical compatibility of
> solid electrolytes with electrodes in all-solid-state Li-ion batteries."
> *Journal of Materials Chemistry A* **2016**.
> DOI: 10.1039/c5ta08574h

**What it contains:** Electrochemical and chemical stability of SSE–electrode
interfaces computed from PBE-GGA phase diagrams (Materials Project parameters:
PAW, PBE-GGA, 520 eV, ≥500 k-points/atom). Reports decomposition energies and
voltage stability windows (Table 2) for LGPS, LLZO, Li3PS4, Li6PS5Cl, and others.

**What it does NOT contain:** No HSE06 calculations. No band gap table.
The "stability windows" are electrochemical voltage ranges, not band gaps.

**Status:** Previously cited incorrectly in this anchor table as a source for HSE06
band gaps. **Those citations have been removed.**

---

#### Anchor table verification summary

Of the 20 entries in `ANCHOR_TABLE`:

| Status | Count | Materials |
|--------|-------|-----------|
| **VERIFIED** (both PBE and HSE06 from read papers) | 2 | LLZO, LATP |
| **PBE-ONLY** (PBE from paper; HSE06 estimated) | 1 | LGPS |
| **UNVERIFIED** (both values estimated from training knowledge) | 17 | all others |

All unverified entries are labelled `[UNVERIFIED]` in the `reference` column of
`ANCHOR_TABLE`. The correction is mathematically valid and runs correctly; the
LOO-MAE (0.41 eV) reflects uncertainty across the full set, but the true uncertainty
on unverified entries is unknown. Replace them with your own DFT+HSE06 calculations
before production use.

**To replace unverified entries:**

```bash
# Run VASP HSE06 on GGA-relaxed structures, then:
python second_pass.py --fit --anchor-csv my_verified_anchors.csv
```

### Caveats (Part 2)

- Corrects gap **magnitude** only.  Band alignment, effective masses, and k-point
  character are unchanged.
- Apply only to PBE gaps from the same VASP setup as the anchor set (see above).
- On novel chemistries outside the anchor composition space this is extrapolation;
  the LOO-MAE understates the true uncertainty in those cases.
- The correction is calibrated for the inorganic SSE chemical family (OBELiX
  dataset).  Do not apply it to organic electrolytes, polymers, or mixed
  halide/organic frameworks.

### File reference (Part 2)

| File | Contents |
|------|----------|
| `second_pass.py` | Scissor correction module + CLI |
