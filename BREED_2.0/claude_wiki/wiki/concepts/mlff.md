# MLFF (Machine Learning Force Field)

## One-line definition
An ML model trained on DFT energies and forces that approximates the potential energy surface of a material, enabling fast molecular dynamics simulations to compute ion diffusivity and conductivity.

## Why it matters for ionic conductivity
DFT-AIMD costs 10²–10³ CPU-hours per simulation point and is limited to 10–100 ps trajectories — often too short to sample rare Li⁺ hopping events. MLFFs achieve DFT-level energy and force accuracy at ~100–1000× lower cost, making nanosecond-scale MD feasible. Longer simulations improve diffusivity statistics and lower the temperature needed for Arrhenius extrapolation to RT σ.

## Detailed explanation
MLFFs fit the Born-Oppenheimer potential energy surface E({**r**_i}) from DFT-computed energies, forces, and (sometimes) stress tensors on a training set of configurations. At inference time, given atomic positions, the MLFF predicts forces on all atoms in microseconds rather than hours.

Common architectures used in SSE research:

- **SchNet, PaiNN**: Message-passing GNNs with continuous convolutions; fast but rotationally equivariant only via data augmentation
- **DimeNet, DimeNet++**: Include angular information (bond angles); more expensive but more stable in long MD trajectories for sulfide SSEs (Duangdangchote 2024)
- **NequIP, MACE**: Strictly equivariant architectures; often most data-efficient for extrapolation
- **CHGNet**: Pre-trained MLFF on ~1.5 million Materials Project DFT calculations; used by Kim et al. 2026 as a structure relaxation engine to generate 152 new crystal structures from stoichiometric predictions
- **M3GNet**: Universal MLFF trained on ~180,000 Materials Project structures; benchmarked as a GNN model for σ prediction in OBELiX

In MLFF-MD, conductivity is computed via Nernst-Einstein from the mean-squared displacement of Li⁺: σ = (n z² e² D) / (k_B T). Simulations run at elevated temperatures (typically 800–1200 K) with Arrhenius extrapolation to RT, inheriting the same extrapolation risk as AIMD.

**Critical stability caveat**: Good energy/force accuracy (R²>0.96 on held-out DFT snapshots) does not guarantee physically stable MD trajectories. Duangdangchote et al. 2024 demonstrated that several popular MLFFs (SchNet, PaiNN) exhibit atom fusion and lattice mismatch during long sulfide SSE simulations, even when their force errors are numerically comparable to passing models (DimeNet, DimeNet++). RDF comparison to reference DFT-AIMD is mandatory validation before reporting σ from MLFF-MD.

## Typical performance / cost
- Energy MAE: ~1–10 meV/atom vs. DFT
- Force MAE: ~10–100 meV/Å vs. DFT
- MD speed: 10²–10⁴× faster than DFT-AIMD per step
- Trajectory stability: only a subset of architectures produce physically valid long-time trajectories; DimeNet/DimeNet++ are best-validated for sulfide SSEs; universal force fields (CHGNet, M3GNet) trade some stability for broad coverage

## When to use
Use MLFFs when ion diffusivity or conductivity is needed from MD and DFT-AIMD is too expensive. Must validate trajectory stability via RDF before reporting σ. Treat universal pre-trained MLFFs (CHGNet, M3GNet) as fast hypothesis-generators, not final answers. Do not report MLFF-MD σ without a stability check.

## Sources using or evaluating this
- [[stability-transferability-mlff]]
- [[obelix]] (M3GNet benchmarked as GNN-based σ predictor)
- [[data-driven-prediction-ionic-conductivity-llm]] (CHGNet used for structure generation)

## Related methods
- [[aimd]]
- [[bvse]]
