---
title: "Designing solid-state electrolytes for safe, energy-dense batteries"
year: 2020
doi: 10.1038/s41578-019-0165-5
type: review
material-class: mixed
dataset: n/a
target: n/a
relevance: medium
---

# Designing solid-state electrolytes for safe, energy-dense batteries

## TL;DR
Zhao, Stalin, Zhao & Archer (2020, *Nature Reviews Materials*) survey SSE design across all ion types and structural families, with emphasis on failure modes, design principles, and high-energy-density battery systems enabled by SSEs. The paper provides the most comprehensive experimental conductivity/activation-energy reference table of the four reviews (Table 1, 25+ SSE materials). Key finding: the Monroe–Newman mechanical model for dendrite suppression does not hold, and DFT stability windows systematically overestimate thermodynamic instability due to kinetic passivation. No ML models for conductivity prediction.

## Landscape covered
Organized by ion-transport physics (mechanisms, pathways, Li vs. Na vs. multivalent ions), failure modes (electrochemical/interfacial instability, dendrite electrodeposition), strategies to improve SSE properties, and high-energy-density battery systems (NCM811, Li–S, Li–air, Na–S, K–S).

Material coverage is the broadest of the four reviews: inorganic SIEs (oxides, sulfides, halides, hydrides, borohydrides), polymer SPEs (PEO, crosslinked networks, composite fillers), thin-film LiPON, and SSEs for multivalent cations (Mg²⁺, Zn²⁺, Ca²⁺, Al³⁺).

Computational content: AIMD for studying concerted migration in LGPS/LLZO/LATP; DFT for stability window calculations; Sendek et al. high-throughput structure screening (>12,000 candidates, no ML conductivity regression model discussed).

## Best results cited

No ML models for conductivity prediction are evaluated.

Reference experimental σ and Ea values (from Table 1 — useful for calibrating BREED target space):
| Material | σ at RT (S/cm) | Ea (eV) |
|---|---|---|
| Li₉.₅₄Si₁.₇₄P₁.₄₄S₁₁.₇Cl₀.₃ | 0.025 | 0.24 |
| Li₇P₃S₁₁ (calc.) | 0.046 | 0.191 |
| Li₆PS₅Br (argyrodite) | 0.011 | 0.10 |
| Li₃.₈₃₃Sn₀.₈₃₃As₀.₁₆₆S₄ | 0.00139 | 0.21 |
| Li₃PS₄ (nanoporous) | 1.6×10⁻⁴ | 0.356 |
| Na₃SbS₄ | 0.001 | 0.22 |
| PEO-LiTFSI (70°C) | ~8×10⁻⁴ | — |

## Bottlenecks identified
1. **Monroe–Newman model failure**: SSE shear modulus > 2× Li metal modulus does not suppress dendrites — Li infiltrates LLZO (oxide) and sulfide SSEs alike. Electronic conductivity of SSEs (enabling internal Li nucleation) is the more important factor than mechanical modulus.
2. **DFT stability windows are too pessimistic**: calculated thermodynamic stability windows are much narrower than experimentally observed windows, because DFT does not account for kinetic inertness and passivating interphase layers.
3. **SSE thickness constraint for energy density**: to achieve 500 Wh/kg, SSE must be ≤5 μm (LLZO), ≤7 μm (LAGP), ≤11 μm (LGPS), ≤20 μm (SPE) — current inorganic SSEs are typically much thicker due to mechanical fragility.
4. **Grain boundary resistance**: 40–50% of total resistance in LLZO pellets; temperature-dependent; must be minimized by sintering optimization but this conflicts with phase stability.
5. **Multivalent cation SSEs**: Mg²⁺, Zn²⁺, Al³⁺ SSEs at ambient temperature are still in early stages; high charge density causes strong electrostatic binding and low mobility.
6. **Interfacial charge transfer**: "a huge challenge" for all-solid-state Li-ion batteries; ALD Al₂O₃ coatings reduce interfacial resistance to ~1 Ω/cm² but add processing steps and cost.

## Relevance to BREED
- **Best single reference table**: Table 1 provides σ and Ea for 25+ SSE materials from argyrodites to hydrides — useful calibration for understanding the spread of BREED's target variable.
- **Grain boundary clarification**: 40–50% of LLZO total resistance comes from grain boundaries; experiments reporting "total" conductivity systematically underreport bulk conductivity, which BREED may or may not be fitting to.
- **Monroe–Newman falsification**: BREED predicts bulk σ; this paper shows that the highest-σ materials (soft sulfides) may fail for other reasons (dendrites, instability), which limits the practical significance of predicting the highest bulk conductivities.
- **High-throughput screening precedent**: Sendek et al. screening of >12,000 candidates via DFT is a precursor to BREED's ML approach; shows that computational screening of conductivity space is established practice.

## Related
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[lithium-battery-chemistries-enabled-solid-state]]
- [[recent-advances-energy-chemistry-solid-state]]
- [[concepts/ionic-conductivity]]
- [[concepts/activation-energy]]
- [[concepts/aimd]]
- [[concepts/garnet-electrolytes]]
- [[concepts/solid-polymer-electrolytes]]
- [[machine-learning-assisted-property-prediction-solid-state]]
