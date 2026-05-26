---
title: "A data science approach for advanced solid polymer electrolyte design"
year: 2021
doi: 10.1016/j.commatsci.2020.110108
type: primary
material-class: polymer
dataset: custom
target: sigma
relevance: low
---

# A data science approach for advanced solid polymer electrolyte design

## TL;DR
Liu et al. trained six ML models (linear regression, lasso, ridge, decision tree, random forest, RBF SVM) to predict log(σ) of PEO-LiTFSI solid polymer electrolytes from only two features: temperature and EO/Li ratio (LiTFSI wt%). Random forest on a combined literature+experimental dataset (Dataset 3, ~240 points) achieved RMSE=0.332 log(S/cm). Key finding: adding uncontrolled literature data (Dataset 2) hurt linear model accuracy — heterogeneous reporting conditions are actively harmful. Independently conducted experiments with standardized conditions improved the most complex models.

## ML method
- Architecture: Random Forest (best); compared against: linear regression, lasso, ridge, decision tree, RBF SVM
- Features/inputs: Only 2 features — temperature (°C) and EO/Li ratio (LiTFSI wt%); all other factors (synthesis method, molecular weight, additives) excluded
- Target variable: log(σ) in S/cm for PEO-LiTFSI at 25–65°C
- Training data: Dataset 3 = ~193 literature + 47 independently collected experimental data points; PEO-LiTFSI only, no additives, linear PEO chains; three dataset versions compared (D1: 76 lit, D2: 146 lit, D3: 193 lit + 47 experimental)
- Train/test split: 80/20 random split (fixed random_state=5 across all models); 5-fold CV for hyperparameter optimization
- Performance: Abstract/conclusion RMSE=0.332 log(S/cm); model comparison section reports RF RMSE=0.289, MAE=0.229 on Dataset 3 test set (discrepancy flagged in TODO.md); experimental validation MAE=0.253, RMSE=0.453 log(S/cm) on 3 held-out compositions
- Best prior ML result on same dataset/target: Fujimura et al. (LISICON, different material, MAE ~0.373) — not directly comparable

## Materials / chemistry
- Solid polymer electrolytes (SPEs): PEO host + LiTFSI salt (Li(CF₃SO₂)₂N)
- Linear PEO chains only; no branching, no plasticizers, no additives
- Tested at 25–65°C; 10–60 wt% LiTFSI compositions
- Optimal conductivity: 41–43 wt% LiTFSI (RF prediction, validated experimentally)
- Activation energy (Ea) predicted from Arrhenius fit to RF predictions: ~71.53 kJ/mol at 15 wt% LiTFSI, consistent with literature values (~75–77 kJ/mol)

## Bottleneck / problem identified
1. **Data heterogeneity from literature is actively harmful**: adding 70 more literature data points (D1→D2) made linear models worse. One outlier (Marzantowicz, unusual conditions) skewed high-LiTFSI predictions. Key conclusion: "an apples-to-apples comparison is necessary."
2. **Unreported experimental details**: stirring method, oxygen/humidity level, drying environment, PEO molecular weight are rarely reported in SPE literature → cannot be encoded as features → irreducible noise floor.
3. SPE research is constrained by the same data problem as inorganic SSE: lengthy synthesis/testing makes large datasets impractical.

## Relevance to BREED
- Low direct relevance: BREED targets inorganic SSEs; this paper studies a single polymer electrolyte system with only 2 features.
- **Transferable lesson**: the finding that independently curated experimental data improved models over literature-only data is directly relevant to BREED. OBELiX inherits literature-wide heterogeneity, and BREED's irreducible error floor likely reflects this.
- The extreme simplicity (2 features, high R² on parity plot) shows that for highly constrained systems, even minimal features suffice — the analogy for BREED would be within a single SSE subclass.

## Related
- [[concepts/solid-polymer-electrolytes]]
- [[concepts/ionic-conductivity]]
- [[concepts/activation-energy]]
- [[machine-learning-assisted-property-prediction-solid-state]]
