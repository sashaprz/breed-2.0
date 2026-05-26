# Argyrodite Electrolytes

## One-line definition
Sulfide-based SSEs with the argyrodite crystal structure (general formula Li₆PS₅X, X = Cl, Br, I), notable for achieving among the highest room-temperature Li-ion conductivities of any inorganic SSE class.

## Why it matters for ionic conductivity
Argyrodites reach σ ~ 10⁻³–10⁻² S/cm at RT — approaching or matching liquid electrolytes — making them prime candidates for high-power solid-state batteries. They are a major material class in OBELiX and appear prominently in ML benchmarks. Their high conductivity arises from Li site disorder (partially occupied 48h sites) and soft, polarizable S²⁻/halide lattice, which are also the features that make them challenging for structure-based ML.

## Detailed explanation
The argyrodite structure (space group F4̄3m) places Li on a disordered set of 48h Wyckoff positions distributed around the tetrahedral voids of the thiophosphate (PS₄³⁻) framework. The partial occupancy — Li ions statistically distributed over ~12 of the 24 available 48h sites per unit cell — is central to conductivity: it provides a continuous Li percolation network throughout the 3D crystal.

Key structural parameters controlling σ:
- **Anion disorder (S/X distribution)**: the fraction of halide X on the 4c vs. 4a cage sites changes the local Li environment. Cl-rich compositions favor different site occupancies than Br-rich ones; optimal disorder maximizes the connected 48h pathway.
- **Bottleneck size**: the critical jump between 48h sites passes through a triangular window of sulfide/halide anions. Larger bottleneck → lower Ea.
- **Li-Li repulsion distance**: closer Li neighbors correlate with correlated hops and lower apparent Ea.

Representative conductivities:
- Li₆PS₅Cl: ~1–3 mS/cm at RT, Ea ≈ 0.3 eV (cold-pressed) to 0.1 eV (hot-pressed)
- Li₆PS₅Br: ~11 mS/cm at RT, Ea ≈ 0.10 eV (Zhao et al. 2020 compilation)
- Li₆PS₅I: ~10⁻⁴ S/cm at RT (much lower — I on 4a site blocks pathway)

The HECS descriptor study (Zhao et al. 2021, cited in machine-learning-assisted) found bottleneck size and anion disorder to be the #1 and #2 SHAP features for argyrodite Ea prediction — structural features inaccessible from composition alone, and only partially accessible from CIF when disorder occupancies are partial (as they are in ~75% of OBELiX CIF entries).

## Prerequisites
- [[ionic-conductivity]]
- [[activation-energy]]

## Sources discussing this
- [[machine-learning-assisted-property-prediction-solid-state]]
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[designing-solid-state-electrolytes-safe-energy]]
- [[obelix]]
- [[understanding-interface-stability-solid-state]] — argyrodite (Li₆PS₅Cl) ESW: reduction at ~1.0 V, oxidation at ~2.5 V; narrow window requires passivation or buffer coatings for practical use

## Related
- [[ionic-conductivity]]
- [[activation-energy]]
- [[bvse]]
- [[garnet-electrolytes]]
- [[electrochemical-stability-window]]
