# NASICON Electrolytes

## One-line definition
A family of solid electrolytes based on the NASICON crystal structure (Na Super Ionic CONductor, general formula M₂(XO₄)₃ with fast-ion channels along the c-axis), adapted for Li-ion conduction in materials such as LAGP and LATP.

## Why it matters for ionic conductivity
NASICON-type Li-ion conductors (especially LAGP: Li₁₊ₓAlₓGe₂₋ₓ(PO₄)₃ and LATP: Li₁₊ₓAlₓTi₂₋ₓ(PO₄)₃) achieve σ ~ 10⁻⁴–10⁻³ S/cm at RT in the bulk, with the advantage of oxide-class chemical stability and humidity tolerance. They appear in OBELiX and are a material class relevant to BREED.

## Detailed explanation
The NASICON framework consists of corner-sharing MO₆ octahedra and XO₄ tetrahedra forming a 3D skeleton. Interstitial channels along the c-axis host mobile cations. In Na-NASICON (NaZr₂(PO₄)₃), Na⁺ hops between M1 (6b) and M2 (18e) sites. Li-adapted NASICONs substitute Zr → Ti or Ge on the M-site and Al → on the M-site to increase carrier concentration:

Li₁₊ₓAlₓTi₂₋ₓ(PO₄)₃ (LATP): x ≈ 0.3, σ_bulk ~ 10⁻³ S/cm, Ea ~ 0.35 eV  
Li₁₊ₓAlₓGe₂₋ₓ(PO₄)₃ (LAGP): x ≈ 0.5, σ_bulk ~ 10⁻⁴ S/cm, Ea ~ 0.30 eV

Key issues:
- **LATP instability with Li metal**: Ti⁴⁺ is reduced to Ti³⁺ at ~2.4 V vs. Li⁺/Li, making LATP incompatible with Li metal anodes and Li–S systems.
- **LAGP reactivity**: reacts violently with molten Li metal, releasing O₂ and causing flash fires — unsafe for Li metal batteries.
- **Grain boundary resistance**: NASICON ceramics have high grain boundary resistance that often dominates total conductivity; requires careful densification (hot pressing, SPS).
- **Application**: primarily used as membranes in Li–air and Li–S batteries that use liquid catholytes — the SSE is a separator rather than a single-ion conductor throughout the cell.

NASICON-type materials appeared early in computational high-throughput screening (Sendek et al. >12,000 candidates) and in the OBELiX dataset. Their 3D open framework provides clear ion migration channels amenable to both BVSE analysis and composition-based ML.

## Prerequisites
- [[ionic-conductivity]]
- [[activation-energy]]

## Sources discussing this
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[lithium-battery-chemistries-enabled-solid-state]]
- [[designing-solid-state-electrolytes-safe-energy]]
- [[machine-learning-assisted-property-prediction-solid-state]]
- [[understanding-interface-stability-solid-state]] — DFT ESW: LATP reduction at 2.2 V (Ti⁴⁺ reduction), oxidation stable to 4.2 V; LAGP reduction at 2.7 V, stable to 4.3 V; both unsuitable for Li metal anode but stable vs. high-V cathodes
- [[role-of-interfaces-solid-state-batteries]] — NASICON grain boundary conductivity is 1 order of magnitude below bulk (total ~10⁻⁴ vs bulk ~10⁻³ S/cm); sintering quality critically controls total conductivity

## Related
- [[ionic-conductivity]]
- [[garnet-electrolytes]]
- [[activation-energy]]
- [[electrochemical-stability-window]]
