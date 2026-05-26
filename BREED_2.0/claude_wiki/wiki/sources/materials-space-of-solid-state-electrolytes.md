---
title: "Materials space of solid-state electrolytes: unraveling chemical composition–structure–ionic conductivity relationships in garnet-type metal oxides using cheminformatics virtual screening approaches"
year: 2017
doi: 10.1039/c7cp00518k
type: primary
material-class: garnet
dataset: custom
target: sigma
relevance: medium
---

# Materials space of solid-state electrolytes: unraveling chemical composition–structure–ionic conductivity relationships in garnet-type metal oxides using cheminformatics virtual screening approaches

## TL;DR
First cheminformatics/ML study of Li-ion conductivity in garnet-type oxides. Kireeva & Pervov assembled 98–168 experimental garnet compounds, constructed geometry-based descriptors (ionic radii, octahedral factors, space filling), and built SVR regression models. Achieved R²=0.778, RMSE=0.372 log(S/cm). Also used t-STE for chemical space visualization. A heuristic grid search to infer Al³⁺/Ga³⁺ site preferences improved models to R²=0.81. Several virtual garnet compounds recommended for synthesis.

## ML method
- Architecture: Support vector regression (SVR) with RBF kernel (LIBSVM); t-Stochastic Triplet Embedding (t-STE) for 2D chemical-space visualization
- Features/inputs: Constitutional/compositional bit-string descriptors (per site: tetrahedral, octahedral, dodecahedral); Shannon ionic radii (coordination-aware); octahedral factors and analogues for alternative coordinations; space filling parameter φ = (4π/3 Σ nᵢRᵢ³) / V_cell; optional synthesis information descriptors (sintering temperatures, times); atomic X-ray scattering factor
- Target variable: log₁₀(total ionic conductivity) at ambient temperature
- Training data: 98–168 garnet oxide compounds (multiple subsets for different protocols); cubic and I4̄3d phases; LLZO variants dominate; dataset construction involved deduplication decisions made ad hoc due to multi-order-of-magnitude variance in same-compound data
- Train/test split: 10-fold external cross-validation, averaged over 100 shuffled individual models
- Performance: Dataset 1 (no synthesis info, deduped): R²=0.778, RMSE=0.372, MAE=0.283; optimized model with heuristic Al/Ga site occupancy: R²=0.81, RMSE=0.34, MAE=0.26
- Best prior ML result on same dataset/target: None — first ML study on garnet σ (earlier work used DFT+ML for Li migration barriers in LISICON/olivine, different target)

## Materials / chemistry
- Garnet-type oxides A₃B₂(XO₄)₃; Li₃ through Li₇ phases
- Focus: Al³⁺, Ga³⁺, Fe³⁺-stabilized cubic Li₇La₃Zr₂O₁₂ (LLZO); also Li₅La₃Nb₂O₁₂ and Li₅La₃Ta₂O₁₂ variants
- Key structural detail: Li distributes between tetrahedrally coordinated 24d and octahedrally coordinated 96h sites; ratio depends on Li content; high Li content → more 96h occupation → moisture sensitivity
- Two cubic space groups (Ia3̄d and I4̄3d); tetragonal phase excluded (poor conductivity)

## Bottleneck / problem identified
1. **Intrinsic experimental data variance**: same compound, same synthesis method → conductivity varies by orders of magnitude across studies (see Table 1 in paper: LLNO compounds range 5×10⁻⁷ to 4.4×10⁻⁵ S/cm). This variance is comparable to model RMSE, making the error floor the data itself.
2. **Synthesis information encoding failed**: adding sintering conditions as descriptors did not improve predictions — either the encoding was too simple or the information is too inconsistently reported.
3. **Limited structural data**: Li ionic radius coordination ambiguity reduces model performance; "frozen" tetrahedral coordination assumption for Li desensitizes the model.

## Relevance to BREED
- Establishes R²=0.778 as a garnet-class SVR baseline; BREED targets all material classes (harder problem) so direct comparison is not meaningful, but garnet-specific models outperform mixed-class models due to reduced compositional diversity.
- Geometry-based features (ionic radii, octahedral factors, space filling φ) carry useful signal — these are not in typical composition-only feature sets; BREED could benefit from adding structural geometry features where CIF data is available.
- Confirms that encoding synthesis conditions doesn't help even with explicit parameters — consistent with BREED's approach of not encoding synthesis route.
- Data variance finding is directly relevant: OBELiX inherits literature data variance, and the R²=0.778 ceiling here is partly a data ceiling, not a model limitation.

## Related
- [[improving-ionic-conductivity-garnet-gradient-boosting]]
- [[harnessing-artificial-intelligence-holistic-design]]
- [[concepts/garnet-electrolytes]]
- [[concepts/tolerance-factor]]
- [[concepts/ionic-conductivity]]
