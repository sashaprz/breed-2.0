# Garnet Electrolytes

## One-line definition
Oxide-based solid electrolytes with the garnet crystal structure (general formula A₃B₂[XO₄]₃), characterized by a 3D Li-ion migration network, high chemical stability, and compatibility with Li metal anodes.

## Why it matters for ionic conductivity
Garnets (especially LLZO: Li₇La₃Zr₂O₁₂) are among the most studied SSE candidates, achieving σᵢ ~ 10⁻⁴–10⁻³ S/cm at RT, stability against Li metal, and a wide electrochemical window (~6 V). They are the subject of multiple ML studies for conductivity prediction.

## Detailed explanation
The garnet structure A₃B₂(XO₄)₃ provides a 3D network of corner- and edge-sharing polyhedra. In Li-stuffed garnets (Li₅–Li₇), Li occupies two sites: tetrahedrally coordinated 24d sites and octahedrally coordinated 96h sites. Higher Li content shifts occupation toward 96h sites, which are connected to form the fast-ion conduction pathway. The cubic phase (space group Ia3̄d) is most conductive; the tetragonal phase (I4₁/acd) has ordered Li and poor conductivity.

Al³⁺, Ga³⁺, or Ta⁵⁺ doping on the Zr-site or Li-site stabilizes the cubic phase and tunes Li⁺ carrier concentration. High-entropy garnets (multiple simultaneous dopants at the Zr site) are an emerging approach for further tuning.

The tolerance factor Tf (a geometric ratio of ionic radii) is a primary stability predictor: values in 0.9–1.1 indicate stable garnet structures. ML models trained on garnets consistently find Tf among the top features.

Key performance targets: Ea < 0.3 eV, σᵢ > 10⁻⁴ S/cm for bulk; grain boundary resistance typically limits total conductivity.

## Prerequisites
- [[ionic-conductivity]]
- [[tolerance-factor]]
- [[activation-energy]]

## Sources discussing this
- [[machine-learning-assisted-property-prediction-solid-state]]
- [[materials-space-of-solid-state-electrolytes]]
- [[improving-ionic-conductivity-garnet-gradient-boosting]]
- [[harnessing-artificial-intelligence-holistic-design]]
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[designing-solid-state-electrolytes-safe-energy]]
- [[understanding-interface-stability-solid-state]] — DFT ESW for LLZO (0.05–0.07 V to 2.9–3.2 V); marginally unstable vs. Li metal (20 meV/atom); relatively stable vs. common cathodes
- [[role-of-interfaces-solid-state-batteries]] — LLZO grain boundary conductivity equals bulk (~10⁻⁴ S/cm); exceptional compared to LLTO (2-order GB penalty) and NASICON (1-order penalty)

## Related
- [[ionic-conductivity]]
- [[tolerance-factor]]
- [[activation-energy]]
- [[electrochemical-stability-window]]
