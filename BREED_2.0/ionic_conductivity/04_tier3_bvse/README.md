# 04 — Tier 3: Bond Valence Site Energy (BVSE)

Maps the energy a Li⁺ ion experiences at every grid point in the crystal based on distances to all neighboring ions. Finds the lowest-energy pathway between Li sites and extracts the migration barrier (saddle point energy), which approximates activation energy Eₐ → σ ∝ exp(−Eₐ/kT). Implemented with bvlain.

**The catch:** only works on ordered structures (no partial occupancy). Partial occupancy = sites that are probabilistically occupied, common in fast conductors where Li disorder creates the transport pathway. Forcing full occupancy gives physically garbage results.

**Results on ordered-only subset (146 entries):**
- `barrier_3d` Spearman ρ = **−0.455**, p = 7.9×10⁻⁹ — strong independent signal
- Force-ordered (221 entries): drops to −0.220 — garbage structures dilute signal
- Composition-only on this same 146-entry subset: MAE ~1.4 (vs 1.144 on full 829)

**Why BVSE lost to composition-only:**

Two compounding problems. First, requiring ordered structures cuts 60% of the dataset — and even a strong feature can't overcome training on 146 vs 829 samples. Second, and more importantly, there's a selection bias: the fast conductors (garnets like LLZO, argyrodites like Li₆PS₅Cl) almost all have partially occupied Li sites — the disorder *is* the conduction mechanism. Li hopping fast means it's statistically smeared across multiple nearby sites, which shows up as partial occupancy in the CIF. By requiring ordered structures, you're biasing the training set toward materials where Li sits in well-defined sites, which tends to be the slower conductors. So BVSE has a real physical signal (ρ = −0.455) but can't be used at scale on this dataset.

Used in the final ensemble for the 235 OBELiX entries where real CIF BVSE was available — it adds signal there without the dataset restriction problem, since the ensemble blends it with the full-dataset composition model.
