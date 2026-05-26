---
title: "Unsupervised machine learning accelerates solid electrolyte discovery"
year: 2021
doi: 10.1016/j.gee.2019.12.003
type: other
material-class: mixed
dataset: custom
target: sigma
relevance: medium
---

# Unsupervised machine learning accelerates solid electrolyte discovery

## TL;DR
A 2-page research highlight (not a primary paper) summarizing Zhang et al. 2019 (Nat. Commun. 10, 5260) from Toyota Research Institute / University of Maryland. The highlighted work applies unsupervised ML clustering to all ICSD Li-containing compounds using modified XRD (mXRD) patterns of the anion lattice as features — no conductivity labels needed. Materials in groups V and VI (moderately distorted anion lattices) correlated with high Li-ion conductivity. AIMD verification found 16 novel SSLCs with σRT > 10⁻⁴ S/cm; three — Li₈N₂Se, Li₆KBiO₆, Li₅P₂N₅ — showed σRT > 10⁻² S/cm.

## Primary paper summarized
Zhang et al. 2019, Nat. Commun. 10, 5260: "Unsupervised discovery of solid-state lithium ion conductors"

## Key insight
Unsupervised clustering (no conductivity labels) bypasses the labeled-data bottleneck. Using only anion-lattice mXRD features, the model is less susceptible to experimental variance in conductivity labels than supervised approaches. The correlation between moderately distorted anion lattices (disordered anion framework) and high conductivity provides a new structural heuristic.

## Limitations noted
- Only anion lattice considered; some grouped compounds did not exhibit fast Li-ion diffusion — more features needed
- σRT checked computationally only (AIMD); other SSE requirements (electrochemical stability, mechanical properties) not validated
- 16 novel SSLCs have structures very different from well-known fast conductors — synthesis feasibility unknown

## Bottleneck / problem identified
Small labeled datasets for SSLCs limit supervised ML. Unsupervised pre-screening with unlabeled structural data is an alternative paradigm. Also: no unified theory explains conductivity across diverse crystal structures.

## Relevance to BREED
- BREED uses supervised GBT which is label-sensitive; if OBELiX label noise is significant, unsupervised pre-screening could complement supervised models.
- The mXRD anion-lattice descriptor carries structural signal not available from composition-only features — supports adding structure-based features.
- The 16 novel SSLCs are structurally unlike well-known fast conductors → composition/structure space coverage matters for dataset design.

## Related
- [[harnessing-artificial-intelligence-holistic-design]]
- [[machine-learning-assisted-property-prediction-solid-state]]
- [[concepts/ionic-conductivity]]
- [[concepts/aimd]]
