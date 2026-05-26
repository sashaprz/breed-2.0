---
title: "Fundamentals of inorganic solid-state electrolytes for batteries"
year: 2019
doi: 10.1038/s41563-019-0431-3
type: review
material-class: mixed
dataset: n/a
target: n/a
relevance: medium
---

# Fundamentals of inorganic solid-state electrolytes for batteries

## TL;DR
Famprikis et al. (2019, *Nature Materials*) survey the fundamental science underpinning inorganic SSEs: multiscale ion transport from atomic hops to device-level behavior, electrochemical stability, mechanical properties, and processing routes. The review identifies three unresolved engineering challenges — utilization of metal anodes, interface stabilization, and physical contact maintenance — as the core blockers to practical solid-state batteries. No ML content; this is a physics-first reference paper.

## Landscape covered
The review organizes around four scales of ion transport:
1. **Atomic scale**: NEB and AIMD reveal vacancy, interstitial, and concerted (paddle-wheel) migration mechanisms. Activation energy Ea correlates with crystal structure and migration pathway geometry.
2. **Micro/mesoscopic scale**: grain boundary resistance (can dominate total conductivity in dense ceramics), pore morphology, blocking grain boundaries vs. conducting grain boundaries.
3. **Macroscopic scale**: effective medium theory for conductivity of composite electrodes; role of tortuosity and percolation.
4. **Device scale**: solid electrolyte thickness requirements; critical current density; dendritic failure.

Materials covered: oxide garnets (LLZO), NASICON-type oxides (LAGP, LATP), sulfide glass-ceramics (Li₂S-P₂S₅), argyrodites (Li₆PS₅X), halide perovskites, hydrides. A brief section covers polymer and composite electrolytes.

The electrochemical stability section analyzes thermodynamic stability windows from DFT and compares to experimental observations, noting that kinetic passivation makes many SSEs appear more stable than thermodynamics predicts. Mechanics section covers fracture, creep, and interface delamination.

## Best results cited

No ML models for conductivity prediction are evaluated. The paper is a physics and materials science review. One passing citation of an ML interatomic potential (for Li₃PO₄ amorphous phase MD, Li et al. 2017) but no conductivity MAE/RMSE metrics reported.

Reference conductivity values (experimental, not ML predictions):
- Best Li-ion SSE: ~10 mS/cm at RT (sulfide superionic conductors, Kato et al.)
- Best Na-ion SSE: ~1 mS/cm (Na₁₁Sn₂PS₁₂)
- Liquid electrolyte benchmark: ~10 mS/cm

## Bottlenecks identified
1. **Li dendrite penetration of SSEs**: critical current density for practical Li metal batteries must reach 3–10 mA/cm²; most SSEs never exceed 0.3 mA/cm² at RT under real interfacial contact; mechanism poorly understood.
2. **Interface instability**: SSEs typically decompose in contact with both Li metal (reductive) and high-voltage cathodes (oxidative); no single SSE is thermodynamically stable against both simultaneously.
3. **Grain boundary resistance**: in polycrystalline SSEs, grain boundaries often block ion transport (blocking grain boundaries) or provide fast paths depending on composition; hard to control during processing.
4. **Reproducibility**: same-compound conductivity varies ~1 order of magnitude across labs (e.g., Na₁₁Sn₂PS₁₂); attributed to synthesis variability, not measurement error.
5. **High conductivity correlates with metastability**: the fastest Li-ion conductors (LGPS, argyrodites) are kinetically trapped metastable phases; synthesis is challenging and ambient stability is poor.
6. **Physical contact maintenance**: electrode cycling causes volume changes; brittle SSEs crack and delaminate; no robust solution at scale.

## Relevance to BREED
- **Measurement quality context**: grain boundary resistance is a significant but inconsistently reported component of total conductivity. OBELiX measurements are drawn from heterogeneous literature sources; some report total (bulk + grain boundary) conductivity, others report bulk-only. This directly contributes to BREED's irreducible error floor.
- **Target definition**: BREED predicts room-temperature log₁₀(σ); this review clarifies that σ_total = f(σ_bulk, σ_GB, microstructure) — predicting bulk σ alone understates the physics complexity but is the standard ML target.
- **SSE physics background**: most useful for understanding which structural and chemical features physically control ion transport, informing which descriptor classes to include in BREED features.

## Related
- [[designing-solid-state-electrolytes-safe-energy]]
- [[lithium-battery-chemistries-enabled-solid-state]]
- [[concepts/ionic-conductivity]]
- [[concepts/activation-energy]]
- [[concepts/aimd]]
- [[concepts/garnet-electrolytes]]
- [[concepts/bvse]]
- [[machine-learning-assisted-property-prediction-solid-state]]
