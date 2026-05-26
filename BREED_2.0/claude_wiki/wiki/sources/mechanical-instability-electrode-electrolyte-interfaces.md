---
title: "Mechanical instability of electrode-electrolyte interfaces in solid-state batteries"
year: 2018
doi: 10.1103/PhysRevMaterials.2.105407
type: primary
material-class: mixed
dataset: n/a
target: n/a
relevance: low
---

# Mechanical instability of electrode-electrolyte interfaces in solid-state batteries

## TL;DR
Bucci et al. (2018, *Physical Review Materials*) develop an analytical fracture mechanics model for electrode/SE interface delamination in composite solid-state electrodes. The key result: delamination nucleates at just 7.5% volumetric change (≈2.5% radius change), which is exceeded by most intercalation compounds during normal cycling. Compliant electrolytes (E < 25 GPa) delay but cannot prevent delamination. Nanostructuring electrodes does not prevent crack nucleation — only promotes more stable (ductile) crack propagation after it occurs. Once contiguous delamination occurs, the benefit of high SE conductivity is negated.

## ML method
N/A — this is an analytical mechanics study using cohesive zone fracture theory (Del Piero framework). No data-driven or machine learning components.

## Materials / chemistry
- Generic composite electrode geometry: spherical intercalation active material particle embedded in solid electrolyte matrix
- Model applies to all inorganic SSE classes; no material-specific chemistry
- Electrolyte Young's modulus range explored: 20–200 GPa (covering sulfide to oxide SSEs)
- Interfacial fracture energy range: 1–10 J/m²
- Volume change range: 0–40% (covers graphite to Si anodes and intercalation cathodes)

## Bottleneck / problem identified
1. **Mechanical delamination is near-universal**: for standard material properties, 7.5% volumetric change is sufficient to nucleate delamination — this encompasses essentially all practical intercalation electrode materials.
2. **Nanostructuring doesn't help crack nucleation**: reducing particle size does not prevent crack initiation; it only changes the crack growth regime from unstable (brittle) to stable (ductile). Crack initiation threshold is size-independent.
3. **Design conflict**: preventing crack nucleation requires compliant SE (low E, high Fc) while promoting stable crack growth after nucleation requires stiff SE (high E, low Fc) — these are opposing requirements that cannot be simultaneously satisfied.
4. **High SE conductivity amplifies delamination impact**: once delamination occurs, impedance rise is larger when SE conductivity is high (since the interfacial kinetics become rate-limiting relative to bulk transport). A 2× increase in charge time from delamination occurs when SE Li mobility is 10× higher than electrode mobility.
5. **No chemical reactions modeled**: electrochemical decomposition, new phase formation, and space-charge layers are outside scope — the paper demonstrates that mechanical failure alone (no chemistry) is sufficient to cause battery failure.

## Relevance to BREED
Low. This paper addresses electrode-level mechanical failure — a failure mode beyond bulk σ prediction. However, it reinforces the "bulk σ ≠ device performance" open problem: even a material with perfectly predicted high conductivity may fail due to mechanical delamination at the electrode interface. Relevant primarily as context for why BREED's predictions cannot alone determine device viability.

## Related
- [[understanding-interface-stability-solid-state]]
- [[role-of-interfaces-solid-state-batteries]]
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[concepts/ionic-conductivity]]
