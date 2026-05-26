# Electrochemical Stability Window

## One-line definition
The voltage range over which a solid electrolyte does not electrochemically decompose, bounded below by the Li metal reduction potential and above by the cathode oxidation potential.

## Why it matters for ionic conductivity
The ESW determines which electrode combinations a given SSE can survive without forming degradation products. High-σ SSEs tend to have narrow ESWs, creating a fundamental conductivity–stability trade-off. This is the primary reason bulk σ alone cannot determine device viability.

## Detailed explanation
The ESW is bounded by two reactions:
- **Reduction limit**: the voltage at which electrons from the anode can reduce the SE (e.g., Li metal at 0 V vs. Li⁺/Li reduces sulfides at ~1.5–2.0 V)
- **Oxidation limit**: the voltage at which the cathode drives oxidation of the SE (e.g., high-V cathodes at 4–5 V oxidize most known SSEs)

No SSE is thermodynamically stable against both Li metal (0 V) and a high-energy cathode (4–5 V) simultaneously. All practical cells rely on **passivation layers** (analogous to the SEI in liquid electrolytes) that are kinetically stable even though the SSE is thermodynamically unstable.

### Why CV measurements overestimate the ESW by 2–3 V
Standard cyclic voltammetry uses a metallic or carbon-containing working electrode. Carbon provides an electron percolation path that accelerates SE oxidation kinetics, causing current at low overpotentials and masking the true decomposition onset. This is the origin of widely-cited (but unphysical) claims such as LGPS stable 0–5 V, LLZO stable 0–9 V, and antiperovskites stable 0–8 V. The Ceder group's **carbon-composite cell method** (SE powder mixed with carbon, tested electrochemically) gives accurate kinetically-relevant windows 2–3 V narrower than CV.

### DFT-calculated stability windows (Xiao et al. 2020, Nature Reviews Materials)
| Material class | Reduction (V vs. Li) | Oxidation (V vs. Li) |
|---|---|---|
| LGPS | 1.7 | 2.1 |
| Li₃PS₄ (β, sulfide) | ~0.0 | ~2.5 |
| Li₆PS₅Cl (argyrodite) | ~1.0 | ~2.5 |
| LLZO (garnet) | 0.05–0.07 | 2.9–3.2 |
| LiPON | ~0.7 | ~2.6 |
| LATP (NASICON) | 2.2 | 4.2 |
| LAGP (NASICON) | 2.7 | 4.3 |

### Interphase types
- **SEI (solid electrolyte interphase)**: ionically conductive, electronically blocking — favored for anode-side stability; passivates and halts further decomposition
- **MCI (mixed conducting interphase)**: both ionically and electronically conductive — forms at sulfide/Li metal interfaces; does not self-passivate, causing continuous SE consumption

### Design implication: passivation vs. thermodynamic stability
For SSEs where the ESW is narrower than the operating voltage window, the practical question is not "is the window wide enough?" but "do the decomposition products form a stable passivation layer?" LiPON (kinetically stable but thermodynamically narrow) and LiF-rich interphases are the best current examples of functional passivation.

## Prerequisites
- [[ionic-conductivity]]
- [[activation-energy]]

## Sources discussing this
- [[understanding-interface-stability-solid-state]] — primary source; DFT windows and carbon-composite method
- [[interfaces-in-solid-state-lithium-batteries]] — space-charge layer mechanism at sulfide/oxide interfaces
- [[role-of-interfaces-solid-state-batteries]] — grain boundary and composite electrode context
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[designing-solid-state-electrolytes-safe-energy]]

## Related
- [[garnet-electrolytes]]
- [[nasicon-electrolytes]]
- [[argyrodite-electrolytes]]
- [[ionic-conductivity]]
