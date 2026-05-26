---
title: "Lithium battery chemistries enabled by solid-state electrolytes"
year: 2017
doi: 10.1038/natrevmats.2016.103
type: review
material-class: mixed
dataset: n/a
target: n/a
relevance: low
---

# Lithium battery chemistries enabled by solid-state electrolytes

## TL;DR
Manthiram et al. (2017, *Nature Reviews Materials*) survey Li battery chemistries made possible by SSEs: all-solid-state Li-ion, Li–air, Li–S, Li–Br, and mediator-ion aqueous systems. The organizing principle is what battery chemistries become accessible when you replace liquid electrolyte with solid. Contains useful reference tables of conductivity ranges across SSE classes and device performance metrics. No ML content; the core bottleneck is solid/solid interfacial resistance and poor physical contact — not bulk conductivity.

## Landscape covered
The review covers three SSE material families (inorganic, polymer/composite, thin-film) and maps each to the battery chemistries it enables. Inorganic SSEs are categorized by structure type: perovskite-type (Li₃ₓLa₂/₃₋ₓTiO₃, LLTO), NASICON-type (LAGP, LATP), garnet-type (LLZO variants), and sulfide glass-ceramics. Polymer SSEs center on PEO-LiTFSI. Thin-film SSEs (LiPON) enable microbatteries with >10,000 cycles but at low areal capacity.

The battery systems section covers: all-solid-state Li-ion (oxide and sulfide SSEs), Li–air (LATP/LAGP as humidity-selective separator), Li–S (ceramic separator enabling polysulfide blocking), Li–Br, and a mediator-ion aqueous concept using LATP as Li-selective membrane. Each system section identifies device-level performance metrics and failure modes.

## Best results cited

No ML models for conductivity prediction are evaluated.

Representative experimental σ values (from Table 1, not ML predictions):
- Li₃ₓLa₂/₃₋ₓTiO₃ (perovskite): >10⁻³ S/cm (bulk; grain boundary conductivity ~3 orders lower)
- Li₆.₅La₃Zr₁.₇₅Te₀.₂₅O₁₂ (garnet): 1.02×10⁻³ S/cm at RT
- Highest sulfide glass-ceramic: 6.9×10⁻⁴ S/cm (Li₂S–SiS₂ doped with Li₃PO₄)
- LiPON thin film: ~10⁻⁶ S/cm
- PEO-LiTFSI: ~10⁻⁴ S/cm at 65–78°C

## Bottlenecks identified
1. **Interfacial charge-transfer resistance is the primary limiting factor** for all-solid-state batteries; it is more critical than bulk conductivity and currently unsolved at scale.
2. **Volume change compatibility**: electrode materials expand/contract during cycling; solid electrolytes are brittle and cannot accommodate this mechanically → capacity fade, contact loss.
3. **PEO instability above ~4 V**: cannot pair PEO with high-energy-density cathodes (LiCoO₂ at 4.2 V, LiNi₀.₅Mn₁.₅O₄ at 4.7 V).
4. **LATP chemical incompatibility with Li–S**: Ti⁴⁺ reduction at ~2.4 V vs. Li⁺/Li during Li–S discharge destroys the NASICON separator.
5. **Mediator-ion limitation**: LATP-type SSEs are selective for monovalent ions (Li⁺, Na⁺); no practical divalent SSE at ambient temperature exists.

## Relevance to BREED
Low direct relevance. The paper establishes that bulk σ prediction (what BREED does) is necessary but not sufficient — interfacial resistance dominates practical device performance. BREED's predictions inform which materials are worth synthesizing but cannot tell you whether the material will perform well in a full cell.

## Related
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[designing-solid-state-electrolytes-safe-energy]]
- [[concepts/ionic-conductivity]]
- [[concepts/garnet-electrolytes]]
- [[concepts/solid-polymer-electrolytes]]
