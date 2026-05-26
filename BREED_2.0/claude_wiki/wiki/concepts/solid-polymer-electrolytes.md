# Solid Polymer Electrolytes (SPEs)

## One-line definition
Electrolytes based on a polymer matrix (e.g., polyethylene oxide, PEO) with dissolved lithium salt, offering flexibility and non-flammability but typically much lower room-temperature ionic conductivity than inorganic SSEs.

## Why it matters for ionic conductivity
SPEs are a major class of SSE candidate for flexible and wearable batteries. Room-temperature σ ~ 10⁻⁸–10⁻⁵ S/cm (typically 3–5 orders of magnitude lower than best inorganic SSEs). Their ML prediction is structurally distinct from inorganic SSE prediction: composition space is molecular (SMILES/polymer backbone), and temperature and salt concentration are the primary tunable parameters.

## Detailed explanation
The most-studied SPE system is PEO (polyethylene oxide) + LiTFSI (lithium bis(trifluoromethanesulfonyl)imide). Li⁺ transport occurs via coupled motion with the ether oxygen segments of PEO chains; above the glass transition temperature Tg, chain segmental motion enables Li⁺ hopping. Below Tg, conductivity drops sharply.

Key parameters controlling conductivity in PEO-LiTFSI:
- EO/Li ratio (salt concentration): optimal is ~40–43 wt% LiTFSI; too much salt reduces chain mobility
- Temperature: Arrhenius behavior with Ea ~ 70–80 kJ/mol (much higher than inorganic SSEs in absolute terms)
- PEO molecular weight and chain architecture: rarely reported in literature → source of dataset heterogeneity

ML models for SPEs are simpler than for inorganics — Liu et al. 2021 achieved competitive RMSE with just 2 features (temperature and EO/Li ratio) — but this simplicity is possible only because the system is so chemically constrained. Extending to diverse polymer chemistries requires SMILES-based or GNN-based representations.

## Prerequisites
- [[ionic-conductivity]]

## Sources discussing this
- [[data-science-approach-advanced-solid-polymer]]
- [[machine-learning-assisted-property-prediction-solid-state]]
- [[lithium-battery-chemistries-enabled-solid-state]]
- [[designing-solid-state-electrolytes-safe-energy]]

## Related
- [[ionic-conductivity]]
- [[activation-energy]]
