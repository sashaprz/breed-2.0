# AIMD (Ab Initio Molecular Dynamics)

## One-line definition
A simulation method that propagates atomic trajectories using forces from quantum-mechanical DFT calculations, used to compute ion diffusivity and ionic conductivity at finite temperature.

## Why it matters for ionic conductivity
AIMD is the primary high-accuracy computational method for predicting σᵢ without experiment. It naturally captures collective and correlated diffusion effects that static NEB calculations miss, and is used to generate training data for ML models and to validate ML-screened candidates.

## Detailed explanation
In AIMD, the quantum-mechanical forces on each atom are computed at every timestep (typically from DFT with GGA functionals, e.g., PBE) and Newton's equations of motion are integrated. Simulations run at elevated temperatures (600–1500 K) to achieve sufficient ion diffusion within computationally tractable time windows (10–100 ps).

Diffusivity D is extracted from the mean-squared displacement (MSD) of the mobile ion:

D = lim_{t→∞} ⟨|r(t) − r(0)|²⟩ / (6t)

Conductivity is then obtained via the Nernst-Einstein equation: σ = (n z² e² D) / (k_B T).

For solid-state electrolytes, AIMD at 900–1500 K followed by Arrhenius extrapolation to RT is standard. The main limitations are:
- Expensive: AIMD for a ~200-atom supercell typically requires 10²–10³ CPU-hours per simulation point
- Short timescales: correlated motion and rare-event hops may be missed at lower temperatures
- DFT errors: GGA systematically underestimates barriers; HSE hybrid functionals improve accuracy but are ~10× more expensive

AIMD serves two roles in ML for SSE: (1) generating computational labels for supervised ML training when experimental data is unavailable; (2) validating ML-screened candidates before experimental synthesis.

## Prerequisites
- [[ionic-conductivity]]
- [[migration-barrier]]

## Sources discussing this
- [[machine-learning-assisted-property-prediction-solid-state]]
- [[harnessing-artificial-intelligence-holistic-design]]
- [[unsupervised-machine-learning-accelerates-solid]]
- [[stability-transferability-mlff]]
- [[data-driven-prediction-ionic-conductivity-llm]]
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[designing-solid-state-electrolytes-safe-energy]]

## Related
- [[migration-barrier]]
- [[activation-energy]]
- [[bvse]]
