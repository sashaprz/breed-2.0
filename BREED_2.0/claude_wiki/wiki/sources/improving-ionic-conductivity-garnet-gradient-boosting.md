---
title: "Improving ionic conductivity of garnet solid-state electrolytes using Gradient boosting regression optimized machine learning"
year: 2024
doi: 10.1016/j.jpowsour.2024.234492
type: primary
material-class: garnet
dataset: custom
target: sigma
relevance: high
---

# Improving ionic conductivity of garnet solid-state electrolytes using Gradient boosting regression optimized machine learning

## TL;DR
Ma et al. assembled a 398-entry garnet SSE database from Materials Project, OQMD, AFLOW, and literature, then compared GBR, RF, XGB, and ~10 other models for predicting log₁₀(σᵢ). GBR achieved R²=0.90 on both train and test sets with 10 engineered descriptors focused on the Zr-site (C-site) chemistry and garnet structural parameters. SHAP analysis reveals work temperature (WT) and Zr-site first ionization energy (FIE AVG C) as most predictive features. Five garnet compositions were synthesized to validate model-guided design; predicted vs actual conductivities agreed within roughly a factor of 2.

## ML method
- Architecture: Gradient Boosting Regression (GBR) — best; also compared: Random Forest (RF), XGBoost (XGB), Linear Ridge Regression (LRR), OMP, LAR, SVM, KNN, LASSO, EN, DT, KRR, MEN
- Features/inputs: 10 descriptors selected from 90 candidates via feature importance + Pearson correlation (|r|>0.8 threshold for removal):
  1. total C — element occupancy at Zr site (C point)
  2. WT — work/measurement temperature
  3. Rc Var — variance of ionic radius at C site
  4. FIE AVG C — avg first ionization energy at C site
  5. EAE AVG C — avg electron affinity energy at C site
  6. EAE SD C — std dev of electron affinity at C site
  7. t — garnet tolerance factor (ratio L'/H from ionic radii geometry)
  8. χ_c — avg electronegativity at C site
  9. χ_c Var — variance of electronegativity at C site
  10. ΔS Rc — configurational entropy of Rc
- Target variable: log₁₀(σᵢ) in S/cm; range −8.06 to −1.15, mean −4.13, median −3.91
- Training data: 398 garnet-type SSE compositions from Materials Project + OQMD + AFLOW + literature; all garnet-type (manually classified); bulk/total conductivity not always distinguishable
- Train/test split: 80/20 random split + 10-fold CV; hyperparameters: learning_rate=0.001, max_depth=11, min_samples_leaf=1, min_samples_split=2, n_estimators=300
- Performance: GBR train R²=0.90, test R²=0.90, MSE=1.57×10⁻⁷; RF train R²=0.83, test R²=0.84; XGB train R²=0.89, test R²=0.89; linear models ~R²=0.62–0.63
- Best prior ML result on same dataset/target: Kireeva & Pervov 2017 (SVR on garnets, R²=0.778)

## Materials / chemistry
- All garnet-type Li-SSEs; formula A₃B₂(XO₄)₃; cubic LLZO Li₇La₃Zr₂O₁₂ as primary system
- Diverse Zr-site dopants (Ta, Nb, Fe, Hf, Ti, In, Ce, Al, Ga, etc.); high-entropy garnets (multi-element co-doping) included
- 5 compositions synthesized for experimental validation:
  - HEG (Li₆.₂La₃(Zr₀.₂Hf₀.₂Ti₀.₂Nb₀.₂Ta₀.₂)₂O₁₂): measured 1.42×10⁻⁴, predicted 2.62×10⁻⁴ S/cm
  - LLZInO (Li₇.₄La₃Zr₁.₆In₀.₄O₁₂): measured 2.66×10⁻⁵, predicted 6.70×10⁻⁵ S/cm
  - LLZCeO (Li₇La₃Zr₁.₆Ce₀.₄O₁₂): measured 1.51×10⁻⁴, predicted 2.47×10⁻⁴ S/cm
  - LLZFeO (Li₆.₂La₃Zr₂Fe₀.₂₅O₁₂): measured 1.37×10⁻⁴, predicted 1.08×10⁻⁴ S/cm
  - LLZFeO variant at higher T: measured 4.06×10⁻⁴, predicted 5.07×10⁻⁴ S/cm

## Bottleneck / problem identified
1. Bulk vs total conductivity not consistently separated in literature — only a few papers make the distinction; this adds irreducible noise to the training data.
2. Work temperature (WT) is the most important feature by SHAP, but it is a measurement artifact rather than a material property — models trained this way implicitly learn Arrhenius extrapolation rather than structure-property physics.
3. Outlier detection (Cook's distance > 0.5) required removal of 5 data points with implausibly low or high conductivity — confirms field-wide data quality problem.

## Relevance to BREED
- **Most directly comparable paper to BREED**: same algorithm family (gradient boosting), same target (log₁₀ σᵢ), uses literature experimental data.
- R²=0.90 on garnets is much better than BREED's MAE=1.192 on OBELiX — but garnet-only datasets have lower compositional diversity than OBELiX (spanning all SSE classes), making the problem easier.
- **Work temperature (WT) is the #1 SHAP feature** — BREED should check whether OBELiX includes measurement temperature and, if so, add it as a feature. If temperatures are inconsistent/missing, this explains some of BREED's irreducible error.
- Tolerance factor t is a top-ranked feature — a structural geometry descriptor BREED currently lacks; adding it for garnet-class data could improve garnet predictions.
- Zr-site ionization energy and electronegativity outperform global composition averages — site-specific features are more informative than material-level averages for structured crystal families.

## Related
- [[materials-space-of-solid-state-electrolytes]]
- [[harnessing-artificial-intelligence-holistic-design]]
- [[concepts/garnet-electrolytes]]
- [[concepts/tolerance-factor]]
- [[concepts/ionic-conductivity]]
- [[concepts/activation-energy]]
