# 01 — Composition-Only Baseline (Tier 1 Features)

16 features from chemical formula only (no CIF needed). `anion_type` is one-hot encoded into 5 columns:

| Feature | Physical motivation |
|---|---|
| `li_count` | Carrier density proxy — absolute Li count in formula unit |
| `mean_electronegativity` | Composition-weighted Pauling EN — lower EN → weaker Li⁺ binding |
| `anion_electronegativity` | EN of most electronegative anion (O: 3.44, S: 2.58) |
| `cation_anion_en_diff` | Anion EN − weighted cation EN — bond ionicity proxy |
| `mean_ionic_radius` | Composition-weighted ionic radius — channel size proxy |
| `anion_ionic_radius` | Radius of dominant anion (S²⁻: 1.84 Å vs O²⁻: 1.40 Å) |
| `radius_std` | Std dev of per-atom radii — size mismatch / structural diversity |
| `anion_polarizability` | Weighted anion polarizability (Å³) — soft anions lower migration barriers |
| `mean_atomic_mass` | Composition-weighted atomic mass — heavier → softer lattice (phonon proxy) |
| `n_elements` | Number of distinct elements — more → disorder, shallower barriers |
| `mean_atomic_number` | Composition-weighted atomic number |
| `anion_halide` | One-hot: halide framework (F, Cl, Br, I) |
| `anion_mixed` | One-hot: mixed anion framework |
| `anion_other` | One-hot: other anion type |
| `anion_oxide` | One-hot: oxide framework |
| `anion_sulfide` | One-hot: sulfide framework |

**Results on full 829-pt dataset:**
- GBT default: Test MAE **1.144**, R² 0.521
- OBELiX RF baseline (their features, 478 pts): 1.531 ([arxiv:2502.14234](https://arxiv.org/abs/2502.14234))
- Our GBT on OBELiX-only (478 pts): 1.440

The 22% improvement over the [published baseline](https://arxiv.org/abs/2502.14234) comes from more data, not better features. The model pattern-matches chemistry groups (sulfides conduct, oxides don't) — it will miss unusual good conductors and can't rank within a material family.
