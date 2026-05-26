---
title: "Interfaces in Solid-State Lithium Batteries"
year: 2018
doi: 10.1016/j.joule.2018.07.009
type: review
material-class: mixed
dataset: n/a
target: n/a
relevance: medium
---

# Interfaces in Solid-State Lithium Batteries

## TL;DR
Xu et al. (2018, *Joule*) review interface mechanisms across all solid-solid contact types in SSBs: interphase formation, cathode/electrolyte interface, anode/electrolyte interface, polymer/Li metal, and interparticle contacts within composite electrodes. Key mechanistic finding: Li dendrite growth in SSEs is controlled by defect density (voids, grain boundaries, flaws) rather than shear modulus, falsifying the Monroe–Newman model. The space-charge layer at the sulfide SE / oxide cathode interface is identified as the rate-limiting step for high-power operation. Covers advanced in situ characterization methods for buried interfaces.

## Landscape covered
Coverage spans:
- **Interphase formation**: interphase types (SEI = ionically conducting, electronically blocking; MCI = both conducting), Li₂O- and Li₂CO₃-rich SEI from organic liquid analogy, differences in solid-solid interphases
- **Cathode/inorganic SE interface**: space-charge layer mechanism at β-Li₃PS₄/LiCoO₂; LiNbO₃ buffer layer suppression; chemical compatibility at high temperature; oxidation of SE at high voltage; contribution of carbon additives to SE oxidation
- **Anode/inorganic SE interface**: Li metal vs. LGPS (MCI formation), Li metal vs. LLZO (marginally stable, XPS-confirmed reaction at 300–350°C); Li nucleation size inversely proportional to overpotential; dendrite growth defect model
- **Polymer SE / Li metal interface**: PEO/Li interface mechanisms, Li adhesion, void formation during stripping
- **Interparticle contacts**: particle size effects on ionic percolation, role of grain boundaries in composite electrodes
- **Characterization**: in situ TEM, cryo-TEM, XPS, Raman, SFG spectroscopy, on-chip single-nanowire battery

## Best results cited

No ML models for conductivity prediction are evaluated.

Reference conductivities:
- LGPS: 1.2×10⁻² S/cm
- Li₂S–P₂S₅: 1.7×10⁻² S/cm
- Li₉.₅₄Si₁.₇₄P₁.₄₄S₁₁.₇Cl₀.₃: 25 mS/cm

## Bottlenecks identified
1. **Point contact limits ion transport**: solid electrolytes cannot wet electrode surfaces like liquid electrolytes; only a small fraction of electrode particle surface is in ionic contact with the SE.
2. **Space-charge layer at sulfide/oxide interfaces**: at β-Li₃PS₄/LiCoO₂, chemical potential difference drives Li⁺ depletion from SE side → high-resistance zone forms immediately upon contact; LiNbO₃ buffer layer suppresses this but adds processing complexity.
3. **Dendrite growth is defect-controlled, not modulus-controlled**: critical current density for dendrite-free deposition is set by flaw size, grain boundary density, and local electronic conductivity — not bulk shear modulus (falsifies Monroe–Newman model for inorganic SSEs).
4. **Volume change incompatibility**: even small volume changes in intercalation electrodes (~1–4% for LiCoO₂) cause delamination and contact loss in rigid solid-solid interfaces.
5. **In situ characterization of buried solid-solid interfaces** is fundamentally difficult — TEM sample preparation destroys the interface; SFG and on-chip methods provide non-destructive alternatives but are difficult to quantify.

## Relevance to BREED
Low direct relevance. The paper provides mechanistic context for why bulk σ prediction (BREED's target) does not determine device performance. The defect-controlled dendrite model and space-charge layer mechanisms are not encodable as composition-based features.

## Related
- [[understanding-interface-stability-solid-state]]
- [[role-of-interfaces-solid-state-batteries]]
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[concepts/ionic-conductivity]]
- [[concepts/electrochemical-stability-window]]
