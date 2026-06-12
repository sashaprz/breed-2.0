# bulk_modulus

Bulk modulus predictor for inorganic solid-state electrolytes (SSEs).

Bulk modulus matters for SSEs because it governs mechanical compatibility with the electrodes during cycling — too stiff and the electrolyte cracks under the electrodes' volume changes, too soft and it can't resist lithium dendrite penetration.

---

## How it works (plain-language summary)

1. **Relax the structure** to find the equilibrium volume V0 — the volume
   at which the crystal sits at its lowest energy (and zero internal pressure).
2. **Scan volumes around V0.** Generate ~9 new structures by scaling the whole
   unit cell up and down by a few percent (e.g. ±5%). This scaling is
   *isotropic* — it stretches/shrinks the cell uniformly so its shape stays the
   same, only its size changes.
3. **Relax atomic positions at each fixed volume.** The cell size is locked at
   each scan point, but the atoms inside are allowed to settle into their
   lowest-energy arrangement for that volume. Then record the total energy.
4. **Fit E(V).** Plot energy vs. volume for all scan points. Near the minimum
   this traces out a smooth bowl-shaped curve, well-described by the
   Birch-Murnaghan equation of state.
5. **Read K off the curvature of that curve at its minimum** (V0).

### What bulk modulus actually is

Bulk modulus K measures resistance to uniform (all-sides) compression: how
much pressure P you need to apply to squeeze a material by a given fractional
volume change:

```
K = -V (dP/dV)
```

Large K = stiff, hard to compress (diamond). Small K = soft, squishes easily
under modest pressure (a soft halide). For a quick intuitive primer on what
bulk modulus means physically, see
[this video I made](https://youtube.com/shorts/SG4xx-lQ898?si=PF7e_ONP4euBpi2k).

### Why curvature gives the bulk modulus

Pressure itself is the volume-derivative of energy (a standard thermodynamic
identity):

```
P = -dE/dV
```

Substituting into the definition of K:

```
dP/dV = -d^2E/dV^2
K = -V * (dP/dV) = -V * (-d^2E/dV^2) = V * (d^2E/dV^2)
```

Evaluated at the equilibrium volume V0 — where dE/dV = 0, i.e. zero pressure,
the minimum of the curve — this becomes:

```
K = V0 * (d^2E/dV^2) |_V0
```

That second derivative is exactly the **curvature of the E(V) curve at its
minimum**. Here's the deeper reason that's true, not just an algebraic
coincidence:

Near the minimum, E(V) is smooth and its *first* derivative is zero by
definition of V0 — so the leading behaviour as you move away from V0 is purely
quadratic (a Taylor expansion has no linear term left). Write the fractional
volume change as strain `eps = (V - V0) / V0` and the energy *density*
`u = E / V0` (energy per unit volume — intensive, so it doesn't depend on how
big a simulation cell you happened to use). Then:

```
u(eps) ~= u0 + (1/2) K * eps^2,   where   K = d^2u/deps^2 |_eps=0
```

This is *exactly* the form of Hooke's law for a spring, `U(x) = (1/2) k x^2`,
where the spring constant `k = U''(0)` is the curvature of the potential well.
Here, volumetric strain `eps` plays the role of the spring's displacement, and
K is the "spring constant" of the crystal under uniform compression/expansion
— except expressed in energy-per-volume units, which is precisely what
pressure (and hence K) is.

So: a stiff material's energy density rises sharply for a small volumetric
strain — a narrow, steep well, high curvature, high K. A soft, compressible
material has a shallow, wide well — low curvature, low K. The volume-scan +
relax-at-each-point procedure above is a numerical way of mapping out that
well so its curvature at the bottom can be measured. The Birch-Murnaghan fit
(Phase 4) is a convenient functional form for E(V) that captures this
curvature and returns K (and its pressure derivative K0') directly as fit
parameters.

### Why bulk modulus matters for SSEs

- **Cycling-induced volume changes.** Electrodes expand and contract as Li
  inserts/extracts during charge/discharge. The SSE has to keep intimate
  contact at those interfaces without cracking — a function of its
  stiffness and how it deforms under stress.
- **Stack pressure.** Solid-state cells are typically assembled under applied
  external pressure to maintain interfacial contact. How much a candidate
  compresses under that pressure — and how much pressure is needed to keep it
  in contact — is set directly by K.
- **Dendrite suppression (mechanical route).** Separately from the electronic
  route (band gap — see `BREED_2.0/bandgap/README.md`), the Monroe-Newman
  mechanical criterion shows that a sufficiently stiff solid electrolyte
  (elastic/shear modulus large relative to Li metal's) can suppress dendrite
  penetration by deforming elastically rather than cracking. K is a
  first-pass, cheap-to-compute proxy for this stiffness before any full
  elastic-tensor calculation.
- **Processability trade-off.** Very compliant ceramics are easier to densify
  (lower sintering temperatures) but may deform too much under stack pressure;
  very stiff ones resist deformation but can be brittle and prone to
  interfacial fracture. K is one axis of that trade-off used during candidate
  screening.

---

## Final model: `physics_bulk_modulus.py`

**Method:** MLIP + Birch-Murnaghan equation of state (3rd-order BM fit).

A machine-learned interatomic potential (MACE-MP, SevenNet, CHGNet, or M3GNet)
samples the E(V) curve by relaxing internal coordinates at a fixed cell volume
over nine scan points spanning ±5% of the equilibrium volume. The bulk modulus
K is extracted from the curvature of the BM fit at the minimum.

**Why this approach wins over composition/structure ML regressors:**
- Operates on first-principles energetics, not learned property correlations.
- Gives a physically-shaped E(V) curve rather than an unconstrained prediction.
- Naturally produces K0' (pressure derivative) and a stress cross-check as built-in
  diagnostics — you know whether to trust the result before using it.

**MLIP priority (auto-selected):** MACE-MP (float64) > SevenNet > CHGNet > M3GNet.
Use MACE-MP in float64 for production; the second derivative amplifies noise and
float32 (CHGNet) degrades K reproducibility.

### Quick start

```bash
pip install mace-torch            # preferred backend

# Static K at 0 K from an MP entry:
python physics_bulk_modulus.py --mp-id mp-1234

# From a CIF file:
python physics_bulk_modulus.py --cif path/to/structure.cif

# Run the Phase-5 validation gate (known SSEs from MP):
python physics_bulk_modulus.py --validate

# Finite-temperature K(T) via quasi-harmonic approximation:
python physics_bulk_modulus.py --mp-id mp-1234 --qha
```

### Pipeline phases

| Phase | Function | What it does |
|-------|----------|-------------|
| 0 | `load_calculator()` | Lock one MLIP + float64 for all subsequent calls |
| 1 | `acquire_structure()` / `order_structure()` | Pull DFT-relaxed structure; order disorder via Ewald before MLIP sees it |
| 2 | `relax_structure()` | Full cell + positions relaxation; re-symmetrize; volume sanity check vs MP |
| 3 | `volume_scan()` | Scale V0 ±5% (9 pts); relax internals at fixed cell each point |
| 4 | `fit_birch_murnaghan()` | 3rd-order BM fit; K0' alarm (physical range 3.5–4.5); stress cross-check |
| 5 | `validate()` | Phase-5 gate: known SSEs vs MP experimental K values |
| 3b | `volume_scan_qha()` | Per-volume phonons (phonopy + MLIP forces) → F_vib(V,T) |
| 4b | `run_qha()` / `compute_qha()` | BM fit per temperature → K(T), V(T), thermal expansion, Gruneisen |

---

## Accuracy (benchmark run 2026-06-03, MACE-medium/float64)

Validated on 11 materials spanning the full stiffness range (soft halides ~24 GPa to diamond 435 GPa), with reference values from Materials Project DFT VRH.

**Aggregate:**

| Metric | Value |
|--------|-------|
| n | 11 |
| MAE | **5.67 GPa** |
| MAPE | **5.75%** |
| RMSE | 7.14 GPa |
| Bias | +0.51 GPa (near-zero) |
| Median % error | 3.1% |
| R² | 0.996 |
| Pearson r | 0.998 |
| Spearman r | 0.991 |
| Qualitative rank correct | yes (oxides > halides/sulfides) |

**Per-material:**

| Material | Family | Ref (GPa) | Pred (GPa) | Error % |
|----------|--------|-----------|-----------|---------|
| Si | other | 88.9 | 74.6 | -16.1% |
| TiO2 | oxide | 208.5 | 209.4 | +0.4% |
| Li2O | oxide | 118.8 | 130.9 | +10.2% |
| Li3PO4 | oxide | 148.8 | 153.7 | +3.2% |
| Li2S | sulfide | 40.5 | 34.5 | -14.7% |
| LiCl | halide | 32.2 | 33.2 | +3.1% |
| C (diamond) | other | 435.2 | 445.2 | +2.3% |
| MgO | oxide | 151.5 | 146.9 | -3.0% |
| Al2O3 | oxide | 231.0 | 227.5 | -1.5% |
| NaCl | halide | 23.8 | 25.4 | +7.0% |
| SiC | other | 212.7 | 216.2 | +1.7% |

9 of 11 within 10%; 6 of 11 within 4%. The two larger errors (Si and Li2S) are consistent with known MACE-MP behaviour: Si is a covalent semiconductor with a different electronic character from the oxide/sulfide training distribution, and Li2S is at the soft edge where second-derivative noise is largest.

To reproduce: run `python benchmark_accuracy.py --set standard` from `BREED/env/bulk_modulus/`.

---

## Other models (in `BREED/env/bulk_modulus/`)

Two earlier ML regression models were developed before the physics approach and live in the dev environment:

| File | Approach | Notes |
|------|----------|-------|
| `composition_bulk_modulus_predictor.py` | Random forest on composition features | Can't distinguish polymorphs; fast, no CIF needed |
| `bulk_modulus_structure.py` | Random forest on structure + composition features | Requires CIF; trained on MP elastic data |

The physics model (`physics_bulk_modulus.py`) supersedes both — it requires no labeled K training data and extrapolates more reliably to novel SSE compositions outside the MP training distribution.

---

## Test suite

| File | What it tests |
|------|--------------|
| `test_tier0_eos.py` | Phase 0–4 on a simple elemental EOS (sanity baseline) |
| `test_tier12_si_tio2.py` | Phase 1–2 structure acquisition and relaxation on Si/TiO2 |
| `test_tier3_disorder.py` | Phase 1 disorder handling (Ewald ordering, occupancy snapping) |
| `test_tier4_sse.py` | Phase 3–4 on SSE candidates from the OBELiX chemical family |
| `test_tier5_phonons_qha.py` | Phase 3b/4b phonon gate and QHA finite-T path |
| `test_volume_check_fix.py` | Regression test for the MLIP V0 vs MP V0 sanity check |

Run in order of ascending cost: tier 0 first, tier 5 last (requires phonopy and is slow).

---

## Dependencies

```
mace-torch    # or sevenn / chgnet / matgl
phonopy       # for --qha / --validate-phonons only
pymatgen      # structure I/O and MP API
ase           # ASE >= 3.23 for FrechetCellFilter
```
