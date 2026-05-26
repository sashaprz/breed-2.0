# Migration Barrier

## One-line definition
The energy barrier for a single ion hop event between two specific lattice sites, computed as the maximum energy along the migration path minus the energy at the starting site.

## Why it matters for ionic conductivity
The migration barrier directly limits how fast ions can move. High barriers (> ~0.5 eV) produce poor ionic conductors; low barriers (< ~0.2 eV) enable superionic behavior. It is a microscopic quantity that, when averaged over all inequivalent hops in a crystal, predicts macroscopic conductivity.

## Detailed explanation
Migration barrier is typically computed by NEB calculations in DFT. For vacancy migration, a single vacancy is placed in a supercell and the barrier to hop to an adjacent vacancy site is found. For interstitial migration, an extra ion is placed in an interstitial site and the barrier for it to displace a lattice ion (interstitialcy mechanism) is computed.

The distinction from activation energy: Ea is an experimentally or statistically extracted quantity (from Arrhenius fits to conductivity vs temperature); the migration barrier is a single-hop DFT number. In simple materials they are nearly equal; in complex materials with correlated diffusion, Ea < individual hop barriers because ions cooperate.

Key factors controlling migration barrier (from Kim & Siegel 2022 study on anti-perovskites):
- Vacancy migration: ~70% determined by lattice characteristics (hopping distance, channel width)
- Interstitial (dumbbell) migration: ~50% determined by lattice characteristics; anion polarizability also important
- Best ML model (AdaBoost+ERTR, 5 features): RMSE = 71 meV for vacancy, 46 meV for interstitial

## Prerequisites
- [[ionic-conductivity]]

## Sources discussing this
- [[machine-learning-assisted-property-prediction-solid-state]]
- [[harnessing-artificial-intelligence-holistic-design]]

## Related
- [[activation-energy]]
- [[ionic-conductivity]]
- [[bvse]]
- [[aimd]]
