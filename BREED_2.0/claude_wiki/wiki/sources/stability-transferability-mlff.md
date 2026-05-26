---
title: "Stability and transferability of machine learning force fields for molecular dynamics simulations of solid electrolytes"
year: 2024
doi: 10.1039/d4dd00140k
type: primary
material-class: mixed
dataset: custom
target: sigma
relevance: medium
---

# Stability and transferability of machine learning force fields for molecular dynamics simulations of solid electrolytes

## TL;DR
Duangdangchote et al. (2024) benchmark multiple MLFF architectures (SchNet, PaiNN, DimeNet, DimeNet++, and others) on MLFF-MD simulations for three sulfide SSEs: LGPS (Li₁₀GeP₂S₁₂), Li₃PS₄, and Li₄GeS₄. Despite comparable energy and force accuracy (R²>0.96 for most models), only DimeNet and DimeNet++ produce physically stable MD trajectories validated by radial distribution function (RDF) comparison to DFT-AIMD references. Other models exhibit atom fusion or lattice mismatch in long-time trajectories. The core finding: energy/force accuracy metrics do not predict MD trajectory stability, and RDF validation is mandatory before reporting σ from MLFF-MD.

## ML method
- **Architectures tested**: SchNet, PaiNN, DimeNet, DimeNet++, and additional variants; all are message-passing GNNs
- **Features / inputs**: Atomic positions and species (standard MLFF inputs; no hand-crafted descriptors)
- **Target variable**: Ion diffusivity D → σ via Nernst-Einstein relation; RDF as stability diagnostic
- **Training data**: Custom DFT-AIMD trajectories for each material (LGPS, Li₃PS₄, Li₄GeS₄); models also tested for transferability (trained on one material, evaluated on others)
- **Train/test split**: Within-material accuracy evaluated on held-out DFT snapshots; cross-material transferability tested explicitly
- **Performance**:
  - DimeNet / DimeNet++: Stable MD trajectories across all three SSEs; full transferability pipeline LGPS → Li₃PS₄ → Li₄GeS₄ passes RDF validation
  - SchNet, PaiNN, others: Comparable energy/force MAE but fail RDF validation — exhibit atom fusion (two atoms merge to unphysical distances) or lattice mismatch (crystal structure drifts from physical geometry)
  - No explicit σ MAE is reported; stability is the primary outcome metric
- **Validation methodology**: RDF computed from 50+ ps MLFF-MD trajectories, compared to reference DFT-AIMD RDF; failure = significant peak shift, missing peaks, or new spurious peaks

## Materials / chemistry
- Sulfide SSEs: LGPS (Li₁₀GeP₂S₁₂), Li₃PS₄, Li₄GeS₄
- MLFF-MD runs at elevated temperatures (800–1200 K) with Arrhenius extrapolation to RT σ
- All three materials are in the sulfide family; transferability to oxide or halide SSEs not tested

## Bottleneck / problem identified
MLFF stability is a systematic underreported failure mode. Most MLFF papers for SSE conductivity report energy/force accuracy on held-out DFT snapshots and infer that MD trajectories are reliable. This paper shows the inference is wrong: high force accuracy does not prevent unphysical behavior in long MD runs. The community lacks a standard stability validation protocol; RDF comparison to reference AIMD should be mandatory.

## Relevance to BREED
BREED does not currently use MLFFs. If MLFF-MD-derived conductivity values are ever added as computational training data or used as a benchmark reference, DimeNet/DimeNet++ is the only architecture validated for sulfide SSEs. However, the immediate practical impact on BREED is low unless a computational data augmentation campaign is planned. The paper is most relevant as a caveat when interpreting published MLFF-MD conductivity predictions in other papers.

## Related
- [[concepts/mlff]]
- [[concepts/aimd]]
- [[concepts/ionic-conductivity]]
- [[obelix]]
- [[machine-learning-assisted-property-prediction-solid-state]]
