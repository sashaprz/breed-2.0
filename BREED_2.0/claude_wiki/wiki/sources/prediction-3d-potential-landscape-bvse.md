---
title: "Prediction of solid-state lithium-ion conductivity using the three-dimensional potential landscape and distribution of lithium ions"
year: 2026
doi: 10.1063/5.0305688
type: primary
material-class: mixed
dataset: custom
target: sigma
relevance: medium
---

# Prediction of solid-state lithium-ion conductivity using the three-dimensional potential landscape and distribution of lithium ions

## TL;DR
Hashizume et al. (2026) introduce R3DVS (Reciprocal 3D Voxelized Structure), a descriptor encoding the BVSE-derived Li⁺ potential energy landscape and Li-ion site density in reciprocal lattice space, making it invariant to unit-cell choice. Fed to a 3D CNN, R3DVS achieves Test MAE=0.810 on a custom multi-material dataset. An ensemble with CrabNet (composition-based transformer, MAE=0.748) reaches MAE=0.711 — and, crucially, correctly ranks polymorphs (cubic vs. tetragonal LLZO, ~1000× σ difference) where composition-only models give identical predictions.

## ML method
- **Architecture**: 3D convolutional neural network (3DCNN) applied to the R3DVS voxel descriptor; ensemble with CrabNet composition-based transformer
- **Features / inputs**: R3DVS descriptor = BVSE energy grid (Morse-type + Coulomb potential for Li⁺ probe) + Li-ion site density, both encoded on a reciprocal lattice grid; invariant to unit-cell choice and orientation
- **Target variable**: Room-temperature log₁₀(σᵢ) in S/cm
- **Training data**: Custom multi-material dataset (exact size not specified in the paper summary; covers garnets, sulfides, and other inorganic SSEs)
- **Train/test split**: Held-out test set; exact split methodology not specified
- **Performance**:
  - 3DCNN (R3DVS): Test MAE=0.810 log₁₀(S/cm)
  - CrabNet (composition only): Test MAE=0.748 log₁₀(S/cm)
  - Ensemble (3DCNN + CrabNet): Test MAE=0.711 log₁₀(S/cm) — best result
- **Best prior ML result on same dataset**: Not stated; dataset is custom, so cross-study comparison is not directly applicable
- BVSE computation is classical (no DFT); requires a CIF as input

## Materials / chemistry
- Mixed: garnet (cubic and tetragonal LLZO), sulfides, and other inorganic SSEs
- Key demonstration: R3DVS + 3DCNN correctly ranks cubic LLZO (σ ≈ 10⁻³ S/cm) above tetragonal LLZO (σ ≈ 10⁻⁶ S/cm) — a ~1000× difference that composition-only models cannot capture because the two polymorphs have identical composition
- BVSE requires CIF input; computation is fast (minutes per structure vs. hours for DFT-NEB)

## Bottleneck / problem identified
Composition-only models are structurally blind — they assign identical predictions to all polymorphs of a composition, regardless of whether σ differs by orders of magnitude. This is a concrete failure mode for BREED and for any composition-based virtual screening workflow, since a stable vs. metastable polymorph or a cubic vs. tetragonal phase will have the same predicted conductivity.

## Relevance to BREED
- BREED currently uses composition-based features. The polymorphism blind spot is a known failure mode: if OBELiX contains multiple entries for different phases of the same composition, BREED assigns them the same prediction.
- Integrating BVSE-derived scalar features (minimum migration barrier height, bottleneck radius) into BREED's feature set is a lower-overhead path than full 3DCNN, and could partially resolve the polymorph issue.
- Full R3DVS integration requires CIFs; only ~54% of OBELiX entries have CIFs, so a structural model cannot be applied to the full training set — same limitation as GNNs on OBELiX.
- CrabNet's MAE=0.748 on this custom dataset is competitive with the best OBELiX results, but dataset mismatch prevents direct comparison.

## Related
- [[obelix]]
- [[data-driven-prediction-ionic-conductivity-llm]]
- [[concepts/bvse]]
- [[concepts/ionic-conductivity]]
- [[machine-learning-assisted-property-prediction-solid-state]]
