# BVSE (Bond Valence Sum Energy)

## One-line definition
A classical force-field approximation to the Li⁺ 3D migration energy landscape in a crystal structure, computed from tabulated atomic radii and valence parameters without DFT.

## Why it matters for ionic conductivity
BVSE maps where a probe Li⁺ ion is energetically comfortable and where it faces barriers as it moves through the unit cell. From this 3D energy landscape one can extract migration pathways, barrier heights, and bottleneck geometry — the key physics of ion transport — at a computational cost orders of magnitude below NEB or AIMD, making it practical for screening thousands of structures.

## Detailed explanation
The BVSE potential energy for a probe Li⁺ at position **r** is typically computed as:

E_BVSE(**r**) = E_repulsive(**r**) + E_Coulomb(**r**)

where E_repulsive uses Morse-type empirical potential parameters calibrated against known crystal structures (SoftBV or similar databases), and E_Coulomb is the Coulomb interaction between the probe ion and all framework atoms treated as point charges.

The result is a 3D energy grid sampled over the unit cell. From this grid one extracts:
- **Migration barrier Eb**: the energy along the minimum-energy path between adjacent Li sites; the primary screening output
- **Network connectivity**: whether migration is 1D (channel), 2D (layer), or 3D (isotropic) — 3D connectivity is strongly associated with high σ
- **Bottleneck radius**: the minimum void radius along the migration path, related to the geometric bottleneck for Li⁺ hopping

BVSE barriers are systematically lower than DFT-NEB barriers because the classical potential underbinds the probe ion, but the relative ranking of materials is usually preserved. This makes BVSE useful for large-scale virtual screening even when absolute values are inaccurate.

Two primary roles in the ML pipeline for SSE conductivity:

1. **Screening feature (scalar)**: Eb and bottleneck radius can be computed for thousands of structures and used as scalar inputs to composition/structure ML models. He et al. (cited in OBELiX) computed 12,000 BVSE-derived Eb values for a large materials database, providing a quantitative migration barrier feature at scale.

2. **3D structural descriptor (grid)**: The full BVSE energy grid can be voxelized and fed to a 3D CNN. Hashizume et al. 2026 encode both the BVSE energy grid and Li-ion site density on a reciprocal lattice grid (the R3DVS descriptor), making the representation invariant to unit-cell choice. This 3DCNN approach correctly ranks polymorphs (cubic vs. tetragonal LLZO) where composition-only models give identical predictions.

Compared to AIMD: AIMD gives physically correct diffusivities at finite temperature, capturing collective and correlated motion. BVSE is a static 0 K approximation — it misses thermal fluctuations, anharmonicity, and correlated hops, so it underestimates entropy contributions to σ. BVSE is best used for fast pre-filtering and feature generation, with DFT or AIMD reserved for final validation.

## Prerequisites
- [[ionic-conductivity]]
- [[migration-barrier]]

## Sources discussing this
- [[prediction-3d-potential-landscape-bvse]]
- [[obelix]]
- [[machine-learning-assisted-property-prediction-solid-state]]

## Related
- [[aimd]]
- [[migration-barrier]]
- [[activation-energy]]
- [[mlff]]
