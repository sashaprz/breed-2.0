# Activation Energy

## One-line definition
The minimum energy barrier an ion must overcome to hop from one site to an adjacent site in the crystal lattice, controlling how steeply conductivity depends on temperature.

## Why it matters for ionic conductivity
Activation energy Ea directly sets the temperature dependence of ionic conductivity via the Arrhenius relation: σ = σ₀ exp(−Ea / k_BT). Lower Ea means higher conductivity at room temperature and weaker temperature sensitivity. For battery applications, Ea < 0.3 eV is considered excellent; most SSEs are in the 0.2–0.6 eV range.

## Detailed explanation
Ea is the saddle-point energy along the minimum energy path (MEP) between two adjacent ion sites in the crystal. It is distinct from but related to the migration barrier: for simple vacancy hopping, they are the same; for more complex correlated/concerted diffusion, the apparent Ea extracted from Arrhenius fitting reflects contributions from multiple hop types.

Ea is computed by:
- **NEB (Nudged Elastic Band)**: DFT-based, supercell with one vacancy, finds MEP. Accurate but expensive.
- **BVSE (Bond Valence Site Energy)**: Fast empirical method using bond valence parameters. Correlates well with NEB for simple hops.
- **AIMD + Arrhenius fitting**: Run MD at high temperatures, extract diffusivity at each T, fit Arrhenius. Captures collective effects but requires long simulations.

Key structural factors controlling Ea (from HECS descriptor work on Li-argyrodites):
- Bottleneck size (the narrowest point along the Li migration pathway)
- Anion site disorder (increasing disorder lowers Ea)
- Li-Li distance (affects correlated diffusion)
- Lattice parameter / unit cell volume

For BREED: Ea is not directly predicted by BREED's current models, but it is strongly correlated with room-temperature log₁₀(σ) and could serve as an intermediate target or auxiliary feature.

## Prerequisites
- [[ionic-conductivity]]
- [[migration-barrier]]

## Sources discussing this
- [[machine-learning-assisted-property-prediction-solid-state]]
- [[improving-ionic-conductivity-garnet-gradient-boosting]]
- [[harnessing-artificial-intelligence-holistic-design]]
- [[data-science-approach-advanced-solid-polymer]]
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[designing-solid-state-electrolytes-safe-energy]]

## Related
- [[migration-barrier]]
- [[ionic-conductivity]]
- [[bvse]]
- [[aimd]]
