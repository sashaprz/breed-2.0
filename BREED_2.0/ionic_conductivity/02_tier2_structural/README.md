# 02 — Tier 2 Structural Features

Added local geometry around Li sites from CIFs: coordination number, Li–anion bond distance (mean + variance), polyhedral volume and distortion, tetrahedral vs octahedral fraction, min Li–Li distance. Computed with pymatgen CrystalNN/VoronoiNN. Partial occupancy required manual species parsing ('Li:0.875' format).

**Results on CIF subset (~339 train):**
- Tier 1 only: MAE 2.068
- Tier 1 + Tier 2: MAE 1.953
- XGBoost default: MAE **1.535**

Modest improvement (~6% from adding structural geometry). Restricting to the CIF subset — losing ~370 samples — dominates the gain. Tier 2 captures where atoms are, not what Li does.
