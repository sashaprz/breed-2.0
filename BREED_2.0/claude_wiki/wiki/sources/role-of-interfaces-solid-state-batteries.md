---
title: "Role of Interfaces in Solid-State Batteries"
year: 2023
doi: 10.1002/adma.202206402
type: review
material-class: mixed
dataset: n/a
target: n/a
relevance: medium
---

# Role of Interfaces in Solid-State Batteries

## TL;DR
Miao, Guan, Ma, Li & Nan (2023, *Advanced Materials*) review all interface types in solid-state batteries, including grain boundaries within SEs, composite electrode interfaces, and planar electrode/SE separator interfaces. The paper's most important quantitative contribution is systematic grain boundary resistance data across oxide SE classes: LLTO total conductivity is 2 orders of magnitude below bulk (10⁻⁵ vs 10⁻³ S/cm); NASICON is 1 order below bulk; garnet LLZO is exceptional — grain boundary and bulk conductivities are both ~10⁻⁴ S/cm. This difference explains LLZO's preferred position in oxide SSE research despite not having the highest bulk conductivity.

## Landscape covered
Six interface categories:
1. **Grain boundaries in inorganic SEs**: structural reconstruction at GBs in perovskite (LLTO) and NASICON creates ion-blocking layers; LLZO avoids this due to higher cubic symmetry — a key open mechanistic question. GB thickness ~1–2 unit cells.
2. **Composite solid electrolyte interfaces**: polymer/inorganic filler interfaces in CSEs; activation energy reduction (PEO from ~1.0 eV to ~0.5 eV with oxide fillers); RT conductivity up to ~1 mS/cm in optimized CSEs.
3. **Composite electrode interfaces**: physical contact and chemical/electrochemical stability; cold vs. warm pressing effects; elastomeric binder approaches; delamination during cycling.
4. **Cathode/SE separator interfaces**: cathode coatings (LiNbO₃, LLTO, Al₂O₃ ALD); spatial-confined degradation; halide SSE compatibility advantages.
5. **Anode/SE separator interfaces**: chemical/electrochemical stability; Li dendrite growth mechanisms (electro-chemo-mechanical coupling, void formation during stripping); Li-alloy anodes (Li–In) as workaround; anode-free designs.
6. **Current collector/electrode interfaces**: halide SE reaction with Al current collector; carbon barrier layers.

## Best results cited

No ML models for conductivity prediction.

Key quantitative grain boundary data (most valuable for BREED context):
| SE material | Bulk σ (S/cm) | Total σ (S/cm) | GB penalty |
|---|---|---|---|
| LLTO (perovskite) | ~10⁻³ | ~10⁻⁵ | ×100 |
| NASICON (LATP/LAGP) | ~10⁻³ | ~10⁻⁴ | ×10 |
| LLZO (garnet) | ~10⁻⁴ | ~10⁻⁴ | ×1 (exceptional) |

SSB cycling performance with Li₆PS₅Cl SE (Table 1 highlight):
- With composite SE: up to 20,000 cycles at 71% capacity retention (1.61 C rate)
- Best standard cells: 850 cycles at 91.5% retention (1/3 C)

## Bottlenecks identified
1. **Grain boundary resistance is material-class dependent and large**: LLTO and NASICON lose 1–2 orders of magnitude to grain boundaries; explains why literature conductivity data for these classes shows >1 order of magnitude scatter — sample-to-sample variation in sintering quality directly controls total conductivity.
2. **Mechanistic reason LLZO avoids GB blocking**: hypothesis is higher cubic symmetry allows better accommodation of grain orientation differences; this remains an open question, and predicting GB-free materials from composition alone is unsolved.
3. **Electron-beam-sensitive SEs (sulfides, halides)** cannot be studied at atomic resolution — GB characterization at the unit-cell level is limited to oxide systems.
4. **Li-stripping void formation** at Li/SE interface increases local current density → accelerates dendrite formation; external stack pressure mitigates but doesn't solve.
5. **Composite SE energy/conductivity/processability triangle**: optimizing one property tends to degrade others.

## Relevance to BREED
- **Critical context for training label quality**: grain boundary resistance data quantifies how much total conductivity (what literature reports, what BREED trains on) deviates from bulk conductivity (what crystal structure controls). For LLTO/NASICON-class materials, training labels may be 1–2 orders of magnitude below the intrinsic material value — this is a systematic bias in BREED's training data, not random noise.
- **LLZO exception**: LLZO's grain boundaries don't penalize conductivity, making LLZO experimental data more reliable as training labels.
- **Explains within-class scatter**: the scatter in BREED predictions for perovskite/NASICON classes may be partly explained by uncontrolled GB resistance variation, not composition-conductivity physics.

## Related
- [[understanding-interface-stability-solid-state]]
- [[fundamentals-of-inorganic-solid-state-electrolytes]]
- [[designing-solid-state-electrolytes-safe-energy]]
- [[concepts/ionic-conductivity]]
- [[concepts/garnet-electrolytes]]
- [[concepts/nasicon-electrolytes]]
- [[concepts/electrochemical-stability-window]]
