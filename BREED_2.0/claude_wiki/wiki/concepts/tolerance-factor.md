# Tolerance Factor

## One-line definition
A dimensionless geometric parameter computed from ionic radii of site-occupying elements that quantifies the structural stability of garnet (and perovskite) crystal structures.

## Why it matters for ionic conductivity
A tolerance factor near 1 indicates a stable garnet structure; values outside 0.9–1.1 for garnets suggest structural instability. It appears as a top-ranked ML feature for garnet SSE conductivity prediction and is used as a fast pre-filter to reduce large virtual compound libraries to synthesizable candidates.

## Detailed explanation
For garnet-type oxides, the tolerance factor Tf is defined geometrically from the ionic radii of the A, B, and C site elements. In the formulation used by Wang et al. 2021 and Ma et al. 2024:

Tf = L' / H

where L' and H are characteristic lengths derived from the lattice geometry (the ratio of the average nearest-neighbor distance that would be expected if the sites were perfectly packed). The exact formula varies by author; the conceptual content is identical: Tf measures how well the ionic radii at each site match the ideal geometry.

Physically, Tf < 0.9 means the A-site ions are too small for the dodecahedral cavities (buckled lattice); Tf > 1.1 means the octahedral B-site ions are too large (strained bonds). Both cases produce distorted structures with worse Li-ion pathways.

In practice:
- Wang et al. 2021 used 0.9 < Tf < 1.1 to reduce 29,008 virtual garnets to 7,067 stable candidates before ML screening
- Ma et al. 2024 include Tf as one of 10 final features; it ranks ~3rd by SHAP importance for log₁₀(σᵢ) prediction

The tolerance factor concept originates from perovskites (Goldschmidt 1926) and has been adapted for garnets, NASICONs, and other framework oxides.

## Prerequisites
- [[garnet-electrolytes]]

## Sources discussing this
- [[materials-space-of-solid-state-electrolytes]]
- [[improving-ionic-conductivity-garnet-gradient-boosting]]
- [[harnessing-artificial-intelligence-holistic-design]]

## Related
- [[garnet-electrolytes]]
- [[ionic-conductivity]]
