# Ionic Conductivity

## One-line definition
The rate at which ions move through a material under an applied electric field, measured in S/cm (siemens per centimeter).

## Why it matters for ionic conductivity
It is the primary target property — both for experimental SSE design and for BREED's ML prediction task. Higher ionic conductivity means faster ion transport between electrodes, better power output, and lower internal resistance.

## Detailed explanation
Ionic conductivity σ is related to diffusivity D via the Nernst-Einstein equation:

σ = (n z² e² D) / (k_B T)

where n is carrier concentration, z is charge, e is elementary charge, k_B is Boltzmann's constant, T is temperature.

In SSEs, σ depends critically on: (1) crystal structure — whether fast-ion pathways exist (e.g., 3D Li-sublattice in garnets, 1D channels in NASICON); (2) defect concentration — vacancies and interstitial sites enable ion hopping; (3) dopant substitution — changes carrier concentration and bottleneck geometry; (4) temperature — Arrhenius behavior, σ = σ₀ exp(−Ea/k_BT), where Ea is activation energy.

Room-temperature σ is typically reported as log₁₀(σ / S·cm⁻¹). Values above −3 (i.e., >10⁻³ S/cm) are considered good for solid-state battery application. Best inorganic SSEs (e.g., Li₆PS₅Cl argyrodite, Li₁₀GeP₂S₁₂) reach 10⁻² to 10⁻² S/cm. Polymer electrolytes are typically 3–5 orders of magnitude lower at room temperature.

For ML models, log₁₀(σ) is the standard target. The distribution is roughly log-normal, spanning ~10 orders of magnitude across all inorganic SSEs.

## Prerequisites
- [[activation-energy]]
- [[migration-barrier]]

## Sources discussing this
- [[machine-learning-assisted-property-prediction-solid-state]]
- [[materials-space-of-solid-state-electrolytes]]
- [[improving-ionic-conductivity-garnet-gradient-boosting]]
- [[data-science-approach-advanced-solid-polymer]]
- [[harnessing-artificial-intelligence-holistic-design]]
- [[unsupervised-machine-learning-accelerates-solid]]
- [[obelix]]
- [[data-driven-prediction-ionic-conductivity-llm]]
- [[prediction-3d-potential-landscape-bvse]]
- [[stability-transferability-mlff]]
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[lithium-battery-chemistries-enabled-solid-state]]
- [[recent-advances-energy-chemistry-solid-state]]
- [[designing-solid-state-electrolytes-safe-energy]]
- [[understanding-interface-stability-solid-state]]
- [[interfaces-in-solid-state-lithium-batteries]]
- [[role-of-interfaces-solid-state-batteries]]
- [[mechanical-instability-electrode-electrolyte-interfaces]]

## Related
- [[activation-energy]]
- [[migration-barrier]]
- [[garnet-electrolytes]]
- [[solid-polymer-electrolytes]]
- [[electrochemical-stability-window]]
