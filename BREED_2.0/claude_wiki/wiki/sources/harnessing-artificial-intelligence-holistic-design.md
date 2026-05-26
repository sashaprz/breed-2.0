---
title: "Harnessing artificial intelligence to holistic design and identification for solid electrolytes"
year: 2021
doi: 10.1016/j.nanoen.2021.106337
type: primary
material-class: garnet
dataset: materials-project
target: band-gap
relevance: medium
---

# Harnessing artificial intelligence to holistic design and identification for solid electrolytes

## TL;DR
Wang et al. built a two-stage XGBoost pipeline to screen 29,008 garnet structures for solid electrolyte candidates. Stage 1: XGB-C classifies garnets as narrow/wide Eg (AUC=0.885); Stage 2: XGB-R predicts Eg for wide-gap candidates (MAE=0.25 eV, R²=0.866) — ~10⁹× faster than DFT. After ML screening, CI-NEB calculations confirmed ionic conductivities for 29 candidates; 12 new garnet SEs with σᵢ > 10⁻⁴ S/cm and Eg > 4.0 eV were identified. Top performers: Dy₃Ga₂Li₃O₁₂ (σᵢ=3.24 S/cm, Ea=0.12 eV), Ho₃Ga₂Li₃O₁₂ (1.52 S/cm, Ea=0.14 eV). This paper is "Wang et al. 2021, Nano Energy 89, 106337" previously referenced in benchmarks.md.

## ML method
- Architecture: XGBoost classification (XGB-C) + XGBoost regression (XGB-R); σᵢ computed via CI-NEB (DFT), not ML
- Features/inputs: 7 elemental properties per site element (dipole polarizability P, atomic number AN, covalent radius cr, van der Waals radius vr, EN_Ghosh EG, first ionization energy IE, valence electron VE) → 28-dim initial feature vector; XGB F-score selects top 16 features for XGB-C (key: IEB=first ionization of B-site, ANA=atomic number of A-site, PA=polarizability of A-site) and top 15 features for XGB-R (key: PA=polarizability of A-site)
- Target variable: Band gap Eg (eV) as proxy for electronic conductivity σₑ; threshold Eg > 4.0 eV → σₑ < 3.6×10⁻³⁰ S/cm (insulating). σᵢ computed separately via CI-NEB for 29 candidates.
- Training data: 286 garnet structures from Materials Project with DFT-computed (PBE) band gaps; 79 experimentally synthesized, 207 from DFT; 107/286 (~40%) have Eg < 0.5 eV (imbalanced dataset → addressed by XGB-C classification stage)
- Train/test split: 10-fold CV throughout; grid search for hyperparameters
- Performance: XGB-C: average AUC=0.885 across 10 folds; XGB-R: average MAE=0.25 eV, average R²=0.866 across 10 folds
- Best prior ML result on same dataset/target: First garnet Eg screening study at this scale — no comparable baseline

## Materials / chemistry
- Garnet formula {A}₃(B)₂[C]₃<X>₁₂ with space group Ia3̄d
- A-site: 37 elements (rare earths, alkaline earths); B-site: 28 elements (Sc, Ga, Al, etc.); C-site: 14 elements (Li, Si, Ge, Al, etc.); X-site: O or F
- Tolerance factor Tf filtering: 29,008 → 7,067 stable candidates (0.9 < Tf < 1.1)
- 12 confirmed Li-ion superionic conductors: Ea range 0.12–0.35 eV; σᵢ range 4.55×10⁻⁴ to 3.24 S/cm
- Some rare-earth and Sc-containing compositions may be difficult/expensive to synthesize

## Bottleneck / problem identified
1. σᵢ prediction still requires expensive CI-NEB (DFT) — ML only reduces the candidate pool, does not replace physics calculations for final σᵢ estimation.
2. Limited training data (286 structures) constrains model accuracy; no experimental conductivity labels used (DFT-computed Eg only).
3. Tolerance factor Tf is a rough stability screen — 2 of 47 structures failed DFT structural optimization despite passing Tf filter.

## Relevance to BREED
- **This is the Wang et al. 2021 paper previously cited in benchmarks.md** — entry confirmed: XGBoost, 286 garnets from MP, Eg target, 10-fold CV, MAE=0.25 eV, R²=0.866.
- Multi-step workflow (ML screen → DFT validation → σᵢ) is a complementary paradigm to BREED's direct σᵢ prediction. BREED could in principle add Eg as an input feature (band gap available from Materials Project for many compositions).
- Elemental property features (ionization energy, polarizability, atomic number, van der Waals radius) achieve R²=0.866 with only 286 training structures — suggests these features are more data-efficient than purely compositional features.
- The Nernst-Einstein equation used for σᵢ estimation is the same underpinning used throughout SSE computational work.

## Related
- [[materials-space-of-solid-state-electrolytes]]
- [[improving-ionic-conductivity-garnet-gradient-boosting]]
- [[concepts/garnet-electrolytes]]
- [[concepts/tolerance-factor]]
- [[concepts/aimd]]
- [[machine-learning-assisted-property-prediction-solid-state]]
