# 03 — Tier 3: Zeo++ Geometric Features

**What Zeo++ is:** C++ tool for void-space analysis in crystals. Builds a Voronoi decomposition then computes geometric channel properties — bottleneck diameter (largest sphere that can travel through the whole crystal), largest included sphere, and probe-occupiable volume (POAV).

Compiled from source in WSL. Default CCDC van-der-Waals radii caused 127/408 failures on dense electrolytes (atoms too close). Fixed by switching to ionic radii → 404/408 success.

**POAV attempt:** Used a 0.76 Å Li-sized probe to measure accessible void volume. Only 22/404 structures had any connected channels — because Li in solid electrolytes *hops* between sites, it doesn't flow through open channels like in a zeolite. POAV is near-zero for the other 382, making it useless as a discriminating feature.

**Results (bottleneck + density features):**
- Tier 1 + Zeo: MAE **1.550** vs Tier 1 alone: 1.550 — net zero improvement
- Structural features (+0.193 MAE gain) exactly cancelled by losing 114 more samples (−0.208 MAE loss)

Zeo++ is the right tool for porous materials (MOFs, zeolites) but wrong for dense ionic conductors.
