---
title: "OBELiX: A Curated Dataset of Crystal Structures and Experimentally Measured Ionic Conductivities for Lithium Solid-State Electrolytes"
year: 2025
doi: 10.1039/d5dd00441a
type: dataset
material-class: mixed
dataset: obelix
target: sigma
relevance: high
---

# OBELiX: A Curated Dataset of Crystal Structures and Experimentally Measured Ionic Conductivities for Lithium Solid-State Electrolytes

## What it is

OBELiX is a curated, leakage-free benchmark dataset of experimentally measured room-temperature ionic conductivities for lithium solid-state electrolytes. It provides 599 σ entries spanning garnet, sulfide, halide, NASICON, argyrodite, and other structural families, with 321 crystal structure files (CIFs). The dataset's primary contribution beyond raw data collection is a principled leakage-free train/test split and a systematic benchmark of ML models under that split.

## Data sources and construction

OBELiX integrates two prior collections: Hargreaves et al. 2023 (*npj Comput. Mater.* 9, 9) and the Laskowski et al. dataset. Each entry was manually curated from experimental literature; reported values are room-temperature log₁₀(σ / S·cm⁻¹). CIFs were sourced from ICSD and CSD databases where available, yielding 321 structure files out of 599 total entries — roughly 54% CIF coverage.

The leakage-free split is the paper's key methodological contribution. Rather than random k-fold CV, which conflates materials from the same paper or same composition series, OBELiX defines train/test splits via Monte Carlo optimization that groups entries by both composition family and source paper. This yields a 20.2% test set (n=121 for composition-based models; n=67 for the CIF-only subset) that is substantially more independent than random splits. Earlier work using random CV on similar data almost certainly reports inflated performance.

## Experimental reproducibility floor

A critical calibration number: the estimated experimental reproducibility floor, derived from duplicate measurements of the same compound across different studies, is MAD=0.41, RMSD=0.63 log₁₀(S/cm). This is the irreducible noise floor attributable to synthesis variation, measurement conditions, and reporting inconsistencies. No ML model trained on heterogeneous literature data can meaningfully achieve test MAE below this bound. It sets the target: BREED at MAE=1.192 is ~2.9× above the floor; OBELiX's best model (RF at 1.59) is ~3.9× above it.

## Benchmark results

The paper benchmarks RF and MLP on all 599 entries (composition-based), and additionally benchmarks six GNN architectures — PaiNN, SchNet, M3GNet, SO3Net, CGCNN, and disorder-aware variants dis-CGCNN and dis-SO3Net — on the CIF subset only.

| Model | Subset | Test MAE (log₁₀ S/cm) |
|---|---|---|
| RF | Full (n=121) | 1.59 |
| RF | CIF only (n=67) | 1.85 |
| MLP | Full (n=121) | 1.72 |
| MLP | CIF only (n=67) | 2.10 |
| dis-CGCNN (best GNN) | CIF only | 2.71 |
| dis-SO3Net | CIF only | 2.86 |
| PaiNN / SchNet / M3GNet / SO3Net / CGCNN | CIF only | ~2.74–2.89 |

GNNs underperform RF/MLP for two structural reasons: (1) they can only train on the CIF subset (254 training points vs. 478 for composition-based models), and (2) approximately 75% of OBELiX CIF entries have partial site occupancy — crystallographic disorder that standard GNN architectures cannot represent, since they require integer site occupancies. Disorder is precisely the physics that enables Li⁺ conduction in high-performing structures such as argyrodites and Li-stuffed garnets.

Disorder-aware variants (dis-CGCNN, dis-SO3Net) improve marginally over standard GNNs but still fall substantially below RF.

## Key findings

1. RF and MLP substantially outperform all GNNs under a leakage-free evaluation — the "structure matters" intuition reverses when partial occupancy prevents GNNs from accessing the disorder physics that controls conductivity.
2. The experimental reproducibility floor (MAD=0.41) bounds achievable performance. Current best models operate ~3–4× above this bound.
3. Random k-fold CV inflates apparent performance significantly relative to leakage-free evaluation — the magnitude of the inflation is quantified here for the first time on OBELiX.
4. The CIF gap (only 321/599 entries have structures) is a fundamental data quality issue, not just an ML problem.

## Relevance to BREED

Critical. OBELiX is BREED's training dataset. This paper establishes:
- RF MAE=1.59 (full test set, n=121) is the correct published baseline. BREED records "OBELiX RF 1.531" in git; see TODO.md for the discrepancy flag.
- BREED's GBT MAE=1.192 already outperforms all published OBELiX baselines by a meaningful margin.
- The experimental floor (MAD=0.41) is the ultimate target; BREED is ~2.9× above it.
- Adding structure-based features is unlikely to close the remaining gap unless partial occupancy is explicitly handled, since ~75% of CIFs are disordered.

## Related

- [[benchmarks]]
- [[data-driven-prediction-ionic-conductivity-llm]]
- [[prediction-3d-potential-landscape-bvse]]
- [[machine-learning-assisted-property-prediction-solid-state]]
- [[concepts/ionic-conductivity]]
- [[concepts/bvse]]
