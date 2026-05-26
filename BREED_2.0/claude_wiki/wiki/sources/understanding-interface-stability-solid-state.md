---
title: "Understanding interface stability in solid-state batteries"
year: 2020
doi: 10.1038/s41578-019-0157-5
type: review
material-class: mixed
dataset: n/a
target: n/a
relevance: high
---

# Understanding interface stability in solid-state batteries

## TL;DR
Xiao, Wang, Bo, Kim, Miara & Ceder (2020, *Nature Reviews Materials*) systematically map the electrochemical stability of all major SSE classes using DFT-calculated stability windows and reconcile them with experimental data. The central finding is that conventional cyclic voltammetry (CV) severely overestimates stability windows by 2–3 V — e.g., LGPS is claimed stable 0–5 V by CV but actually oxidizes at ~2.1 V; LLZO is claimed 0–9 V but is stable only to ~3.2 V. The paper establishes that no SSE is simultaneously stable against Li metal (0 V) and high-voltage cathodes (4–5 V), and that passivation layers — not true thermodynamic stability — are what make cells work.

## Landscape covered
Organized by SSE material class:
- **Sulfides** (LGPS, Li₃PS₄ variants, argyrodites Li₆PS₅X): narrow windows ~1.5–2.5 V; form mixed ionic-electronic interphases (MCI) at both ends; argyrodites have slightly wider windows than chain-type sulfides
- **Oxides** — Garnets (LLZO family): 0.05–2.9 V to 0.07–3.2 V; marginally unstable vs. Li metal (20 meV/atom driving force); relatively stable vs. common cathodes; LiPON: reduction below 0.68 V, oxidation above 2.6 V — stability claimed by CV (0–5.5 V) is due to kinetic passivation
- **Perovskites** (LLT, LATP, LAGP): LATP unstable vs. Li metal (Ti⁴⁺ reduction at ~2.17 V); LAGP also unstable; LATP/LAGP oxidation stable to 4.2–4.8 V — suitable for high-voltage cathode side only
- **Antiperovskites** (Li₃OCl): CV claims of 0–8 V described as physically impossible; actual window < 3 V
- **NASICONs**: LiZr₂(PO₄)₃ more stable vs. Li metal than LATP/LAGP (no reducible transition metal)

For each class the review reports: DFT-calculated window, experimental measurement using the carbon-composite cell method, reduction/oxidation products, and driving force for chemical mixing with electrodes.

The review introduces a framework with four stability criteria: (1) electrochemical stability window, (2) topotactic stability (can Li intercalation occur without structure change?), (3) chemical mixing reactivity with neighboring phases, (4) passivation layer ionic/electronic conductivity.

**Why CV overestimates stability:** Standard CV uses a metallic or carbon-containing working electrode. Carbon accelerates kinetics of SE oxidation and creates an electron conduction path, producing an apparent large current at low voltage and masking the true onset. The Ceder group's carbon-composite cell method applies the SE in a composite with carbon, directly testing its electrochemical oxidation with a realistic contact geometry.

## Best results cited

No ML models for conductivity prediction are evaluated.

Selected DFT stability windows (from paper, representative values):
| Material | Reduction onset (V vs Li) | Oxidation limit (V vs Li) |
|---|---|---|
| LGPS | 1.7 | 2.1 |
| Li₃PS₄ (β) | ~0.0 | ~2.5 |
| Li₆PS₅Cl (argyrodite) | ~1.0 | ~2.5 |
| LLZO | 0.05–0.07 | 2.9–3.2 |
| LiPON | ~0.7 | ~2.6 |
| LATP | 2.2 | 4.2 |
| LAGP | 2.7 | 4.3 |

## Bottlenecks identified
1. **No SSE stable against both electrodes**: simultaneous stability at 0 V (Li metal) and 4–5 V (high-energy cathode) is thermodynamically impossible for all known SSEs — the two stability requirements conflict with each other.
2. **CV measurement error is systematic and large**: all early literature claiming wide stability windows (LGPS 0–5 V, LLZO 0–9 V, antiperovskites 0–8 V) used methods that cannot detect SEE decomposition at the relevant rate; carbon-composite methods show true windows are 2–3 V narrower.
3. **Mixed ionic-electronic interphase (MCI)** formation: sulfide SEs against Li metal form a MCI that conducts both Li⁺ and electrons → continuous electrolyte consumption; contrast with solid electrolyte interphase (SEI) that is ionically conductive but electronically blocking.
4. **Cathode coating design**: inorganic coatings (LiNbO₃, LiTaO₃) recommended to prevent mixing at sulfide/oxide cathode interfaces, but coatings themselves have non-trivial mixing enthalpy with sulfide SEs (>100 meV/atom).
5. **Coulombic efficiency is insufficient as a stability metric**: SE decomposition temporarily contributes capacity, masking degradation; short-term CE data cannot detect progressive electrolyte consumption.

## Relevance to BREED
- **Indirect**: BREED predicts bulk σ; this paper addresses what limits the *usefulness* of high-σ materials. The materials with highest σ (sulfides) have the worst stability windows.
- **Data quality analogy**: the systematic CV overestimation problem (claims of 0–5V stability for LGPS widely cited in literature) is structurally identical to the conductivity measurement reproducibility problem — experimental data reported in literature is systematically unreliable for quantities measured outside equilibrium. If any BREED feature relies on reported stability window data, those features may be systematically biased.
- **Concept**: electrochemical stability window as a complementary property to σ; a Pareto frontier across σ and stability is the actual design target.

## Related
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[designing-solid-state-electrolytes-safe-energy]]
- [[role-of-interfaces-solid-state-batteries]]
- [[concepts/electrochemical-stability-window]]
- [[concepts/garnet-electrolytes]]
- [[concepts/argyrodite-electrolytes]]
- [[concepts/nasicon-electrolytes]]
- [[machine-learning-assisted-property-prediction-solid-state]]
