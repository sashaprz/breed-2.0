---
title: "Recent Advances in Energy Chemistry between Solid-State Electrolyte and Safe Lithium-Metal Anodes"
year: 2019
doi: 10.1016/j.chempr.2018.12.002
type: review
material-class: mixed
dataset: n/a
target: n/a
relevance: low
---

# Recent Advances in Energy Chemistry between Solid-State Electrolyte and Safe Lithium-Metal Anodes

## TL;DR
Cheng et al. (2019, *Chem*) review the challenges specific to pairing Li metal anodes with SSEs: interfacial resistance, dendrite growth, and the large gap between achievable and practical operation current/capacity. The central finding is that even when SSE bulk conductivity matches liquid electrolyte, solid-state Li metal batteries still fall far short of practical performance — the bottleneck is interfacial, not bulk. One prospective mention of ML/AI for future high-throughput screening; no ML performance data.

## Landscape covered
Organized around three current dilemmas: (1) interfacial stability between SSE and Li metal, (2) dendrite growth mechanisms in SSEs, (3) the operation current/capacity gap. A "basic principle" section covers ionic channels in composite electrolytes and space charge layers at SSE/cathode interfaces. The strategies section covers composite electrolytes, interfacial layers (alloy, polymer, liquid wetting), and mixed-conducting 3D frameworks.

Material focus: LLZO garnets (most stable vs. Li metal), LGPS sulfide (highest bulk conductivity but reactive with Li), LATP/LAGP phosphates (react violently with molten Li). Polymer composites (PVDF/LLZTO, PAN/LAGP) are covered as compromise approaches.

## Best results cited

No ML models for conductivity prediction are evaluated.

Quantitative performance of SSE + Li metal cells (experimental, not ML):
- Operation current at RT for solid-state Li metal batteries: 0.01–3 mA/cm² (vs. target 3–10 mA/cm²)
- LLZO + lithiophilic graphite coating: interfacial impedance reduced from thousands of Ω to a few Ω
- 3D Li | polymer-LLZTO-polymer | 3D Li symmetric cell: stable >700 hours
- Stable cycling 400 cycles at 100–200 μA/cm² using n-BuLi wetting

## Bottlenecks identified
1. **Interface resistance dominates even at sufficient bulk conductivity**: when SSE thickness ≤16.7 μm with σ=0.1 mS/cm, interfacial resistance exceeds voltage drop across bulk — bulk conductivity is not the bottleneck.
2. **Dendrite penetration through SSEs with high mechanical modulus**: Monroe–Newman theory (modulus > 2× Li metal suppresses dendrites) is falsified; Li infiltrates both LLZO (shear modulus ~55 GPa) and sulfide SSEs (much lower modulus) — mechanical stiffness alone does not suppress dendrites.
3. **Void formation at Li/SSE interface during stripping**: increases resistance irreversibly; current solutions require external stack pressure.
4. **Space charge layers at SSE/cathode interfaces**: Li ions migrate out of SSE subsurface layer under applied potential, building a depleted, high-resistance zone; suppressed by buffer layers (LiNbO₃) but adds processing complexity.
5. **Grain boundary conductivity temperature dependence**: at RT, grain boundary diffusion is only ~50% of bulk rate in LLZO; raises total resistance significantly.

## Relevance to BREED
Low. The paper is primarily about Li metal anodes and SSE/Li interfaces — outside BREED's scope of predicting bulk ionic conductivity. However, the central quantitative point is relevant context: bulk σ is necessary but not sufficient, and the interfacial resistance gap means that BREED's predictions may not translate to device performance.

## Related
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[designing-solid-state-electrolytes-safe-energy]]
- [[concepts/ionic-conductivity]]
- [[concepts/garnet-electrolytes]]
