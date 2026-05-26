---
title: "Machine Learning-Assisted Property Prediction of Solid-State Electrolyte"
year: 2024
doi: 10.1002/aenm.202304480
type: review
material-class: mixed
dataset: multi
target: multi
relevance: high
---

# Machine Learning-Assisted Property Prediction of Solid-State Electrolyte

## TL;DR
A comprehensive review of ML applications for predicting SSE properties. Covers the SSE landscape (inorganic, polymer, composite electrolytes), the general ML workflow for materials prediction, and then walks through eight SSE properties targeted by ML: activation energy, band gap, migration barrier, diffusivity, ionic conductivity, modulus, thermal expansion, and reaction energy — with specific case studies for each. Concludes with six future directions. The paper does not introduce a new model; it surveys the field as of early 2024.

## Landscape covered
- Architecture: Survey of many — Linear Regression, Gaussian Process Regression, DNNs, Random Forest, Gradient Boosting, XGBoost, LightGBM, AdaBoost, CrabNet, message-passing neural networks, ML interatomic potentials (moment tensor potentials)
- Features / inputs: Varies by sub-study — composition-based elemental features, crystal structure descriptors (HECS descriptors: composition, structure, conduction pathway, ion distribution, special ions), behavior-based descriptors from early MD trajectories, 2D polymer SMILES, tolerance factor for garnets
- Target variable: Multiple — Ea (activation energy in eV), Eg (band gap in eV), migration barrier (meV), ion diffusivity (log cm²/s), ionic conductivity (log S/cm), shear/bulk modulus (GPa), CTE (thermal expansion coefficient), reaction energy (meV/atom)
- Training data: Various sources; notable curated dataset is Hargreaves et al. (2023, npj Comput. Mater.) — 820 Li-ion conductor entries from 214 refs, ~403 room-temperature compositions
- Train/test split: Mostly random or k-fold cross-validation; no scaffold or leave-one-cluster-out splits reported
- Performance: Key results from cited primary papers (not this review itself):
  - HECS descriptors for Li-argyrodite Ea (Zhao et al. 2021): R²=0.887/train, R²=0.820/test, RMSE=0.02 eV
  - XGB for garnet Eg (Wang et al. 2021): MAE=0.25 eV (10-fold CV), ~10⁹× speedup vs ab initio
  - AdaBoost for anti-perovskite migration barriers (Kim & Siegel 2022): RMSE=71 meV (vacancy, 5 features), 46 meV (interstitial, 3 features)
  - RF + behavior-based descriptors for polymer diffusivity (Khajeh et al. 2023): Li⁺ diffusivity MAE=0.110, TFSI⁻ MAE=0.100, transference number MAE=0.094
  - ML on Na-glass conductivity (Mandal et al. 2023): R²=0.99/train, R²=0.97/test (log₁₀ σ)
  - LightGBM for doped LLZO conductivity classification (Adhyatma et al. 2022): accuracy=0.903 (LOOCV)
  - LightGBM for Li-ion SSE modulus (Choi et al. 2021): R²=0.822/MAE=9.7 GPa (shear); R²=0.892/MAE=14.0 GPa (bulk)
  - Extra Trees for CTE (Kumar et al. 2023): best among RF/XGB/GB (exact metric not extracted, 10-fold CV reported)
- Comparison to baselines: ML consistently shows massive speedup vs DFT/AIMD (10⁹× for garnets, 95 years of computation saved in one study)

## Materials / chemistry
- Material classes: inorganic (oxides, sulfides, phosphates, garnets, argyrodites, NASICON, anti-perovskites), polymer (PEO, PTMC), composite
- Key case studies: Li-argyrodites (activation energy), garnet-type Li₇La₃Zr₂O₁₂-family (band gap, ionic conductivity), anti-perovskite chalco-halides (migration barrier), Li₂S–P₂S₅ glass (migration barrier / "softness"), solid polymer electrolytes (diffusivity, conductivity), sodium aluminophosphate glass (conductivity), Li₁₀GeP₂S₁₂ (diffusion mechanism)
- Ion types: Li⁺ primary focus; Na⁺ also covered (NASICON, glass electrolytes, Na-SSEs)

## Bottlenecks identified
1. **Data scarcity for rare material classes**: small training sets limit model accuracy and generalizability; identified as a critical bottleneck throughout.
2. **Feature engineering subjectivity**: manually curated features introduce bias and are hard to scale; a cited NN model (Lu et al.) reduces this by needing only 3 per-atom inputs.
3. **Multi-property trade-offs**: optimizing ionic conductivity often degrades mechanical stability, and vice versa; no universal solution presented.
4. **Model transferability across crystal families**: ML models trained on one structure type (e.g., anti-perovskites) don't straightforwardly transfer to others; explicitly flagged.
5. **Interface and SEI characterization**: XPS alone can't resolve deeply buried SEI atomic structure; ML+ab initio hybrid needed (Sun et al. AI-ai framework).
6. **Disordered/glassy electrolytes**: limited understanding of how disordered structural attributes control ion transport; ML "softness" metric is a partial solution.

## Relevance to BREED
- Validates GBT/RF as the standard competitive approach for composition-based conductivity prediction.
- HECS descriptors for argyrodites (bottleneck size, anion disorder, Li-Li distance, conduction pathway features) are more informative than composition alone — BREED should consider adding structural/pathway features if CIF data is available.
- ~~The Hargreaves et al. dataset (820 entries, ~403 room-temp σ values) is the closest to a standard benchmark dataset for Li-ion conductors.~~ **SUPERSEDED**: OBELiX (2025, [[obelix]]) is now the standard; it incorporates Hargreaves et al. and adds a leakage-free evaluation protocol. BREED uses OBELiX directly.
- Sendek et al. ML screening result (2.7× better hit rate vs random for fast Li conductors) is the clearest published baseline for composition-only ML.
- ~~Confirms that random k-fold CV is the dominant split methodology and directly comparable to BREED's approach.~~ **SUPERSEDED**: OBELiX (2025) demonstrates that leakage-free splits give substantially worse apparent performance than random CV — random k-fold is now specifically identified as a methodological weakness, not a valid comparison point.
- Multi-property optimization perspective is not currently in BREED's scope but relevant if the project expands.

## Related
- [[obelix]]
- [[data-driven-prediction-ionic-conductivity-llm]]
- [[prediction-3d-potential-landscape-bvse]]
- [[stability-transferability-mlff]]
- [[improving-ionic-conductivity-garnet-gradient-boosting]]
- [[materials-space-of-solid-state-electrolytes]]
- [[harnessing-artificial-intelligence-holistic-design]]
- [[concepts/ionic-conductivity]]
- [[concepts/activation-energy]]
- [[concepts/aimd]]
- [[concepts/bvse]]


