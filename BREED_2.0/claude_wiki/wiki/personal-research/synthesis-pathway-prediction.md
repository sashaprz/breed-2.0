---
author: Sasha Przybylski
date: 2026-05-25
type: personal-research
status: surface-level — not peer-reviewed, not published
topics: [synthesis-pathway, multiscale-modeling, polymer-SSE, data-scarcity]
---

# Synthesis Pathway Prediction — Personal Research Notes

Not a published source. My own surface-level thinking, recorded for continuity.

---

## Why predicting full synthesis pathways is hard

- **Non-linear and emergent behavior.** Synthesis outcomes don't compose linearly from individual steps. Small changes in temperature, atmosphere, precursor ratios, or mixing order can produce qualitatively different phases or microstructures. The mapping from synthesis protocol → material is highly non-convex and hard to model analytically.

- **Scale-dependent effects.** Lab-scale synthesis results are often unreliable at production scale. Heat and mass transfer, mixing uniformity, and atmosphere control all behave differently as batch size increases. A protocol that reliably produces a target phase at 1g may fail at 100g.

- **Very little data.** Synthesis parameter spaces are rarely explored systematically in published literature. Papers report a successful protocol, not a grid search over conditions. Negative results (wrong phase, low density, cracks) are almost never published. This makes training any ML model on synthesis data extremely difficult.

- **Models struggle with complexity, time/length scales, and disorder.** MD and DFT operate at time scales (ns–μs) and length scales (nm) far below those relevant to bulk synthesis (minutes–hours, mm–cm). Bridging these scales requires multiscale modeling frameworks that don't yet exist in a general, reliable form for SSEs. Disorder (partial occupancy, grain boundaries, defects) further complicates any atomistic simulation of realistic synthesis intermediates.

## Current approaches for polymer discovery

The dominant paradigm is **guided empirical iteration** — not true prediction, but an intelligent search process:

1. **Candidate design via structure-property relationships and computational screening.** Start from known base polymer systems (e.g., PEO-based, polycarbonate, polysiloxane). Use structure-property intuition or computational screening to identify promising structural motifs or compositional variants worth testing.

2. **High-throughput experimentation over parameter combinations.** From the candidate starting point, test many combinations of synthesis parameters (monomer ratio, crosslink density, plasticizer loading, annealing conditions, etc.). The space is too large to explore exhaustively, so coverage is guided by prior knowledge.

3. **Robotic / automated synthesis platforms.** Automated systems allow fast physical testing of many parameter combinations in parallel, compressing the iteration cycle from weeks to days.

4. **ML or Bayesian optimization as the active learning loop.** The ML model (or a Bayesian optimizer) takes results from tested conditions and recommends what to simulate or physically test next — converging toward a target property region faster than random or grid search. The model isn't predicting the synthesis pathway; it's acting as a surrogate to prioritize which experiments to run.

**Key observation:** this is active learning over synthesis parameter space, not synthesis pathway prediction. It finds a working protocol by converging empirically, but does not explain *why* those parameters work or generalize reliably to new polymer families or production scale.

## Open question

**Can we simulate a polymer SSE across multiple scales?** Polymer behavior is intrinsically multiscale: quantum effects govern bond formation; molecular dynamics captures chain conformation; mesoscale models describe phase separation; continuum models describe bulk processing. No current framework connects all of these for a single material from synthesis conditions → final properties. This may be the hardest version of the synthesis problem.
