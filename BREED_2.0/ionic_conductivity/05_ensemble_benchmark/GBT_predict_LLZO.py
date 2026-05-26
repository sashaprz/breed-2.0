# GBT_predict_LLZO.py
#
# Uses the composition-only GBT baseline model (04_baseline_model) extended
# with structural BVSE features to predict the ionic conductivity of LLZO.
#
# BVSE features (computed live for LLZO; from bvse_features_combined.csv for
# training samples with CIFs; zero-filled when unavailable):
#   bvse_energy              — 3D activation energy Ea (eV)
#   topology_score           — dimensionality / 3.0  (0→none, 0.33→1D, 0.67→2D, 1.0→3D)
#   bvse_anisotropy          — E_1D − E_3D (eV)
#   bvse_bottleneck          — min free-sphere radius for 3-D void percolation (Å)
#   bvse_accessible          — fraction of grid points with E < E_min + 1 eV
#   bvse_li_count            — total Li atoms per unit cell
#   bvse_available           — 1 if computed from a CIF (real or proxy), 0 if unavailable

import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from pymatgen.core import Composition, Element
from pymatgen.io.cif import CifWriter, CifParser
from mp_api.client import MPRester
from bvlain import Lain

# ── Materials Project API key ─────────────────────────────────────────────────
MP_API_KEY = os.environ.get("MP_API_KEY")
if MP_API_KEY is None:
    raise EnvironmentError(
        "Set the MP_API_KEY environment variable before running this script.\n"
        "Get your key at https://materialsproject.org/api"
    )

# ── 1. Fetch LLZO from Materials Project ─────────────────────────────────────
FORMULA = "Li7La3Zr2O12"
print(f"Fetching {FORMULA} from Materials Project...")

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.search(
        formula=FORMULA,
        fields=["material_id", "structure", "entries"],
    )

if not docs:
    raise RuntimeError(f"No entries found for '{FORMULA}' on Materials Project.")

def _e_above_hull(doc):
    try:
        return doc.entries["GGA+U"].energy_above_hull
    except (KeyError, AttributeError, TypeError):
        try:
            return next(iter(doc.entries.values())).energy_above_hull
        except Exception:
            return float("inf")

docs.sort(key=_e_above_hull)
best = docs[0]
llzo_formula = best.structure.composition.reduced_formula
e_hull = _e_above_hull(best)
print(f"  material_id  : {best.material_id}")
if e_hull < float("inf"):
    print(f"  E above hull : {e_hull:.4f} eV/atom")
print(f"  Formula used : {llzo_formula}")

cif_path = "LLZO_mp.cif"
CifWriter(best.structure).write_file(cif_path)
print(f"  CIF written  : {cif_path}")

# ── 2. BVSE + void distribution for LLZO via bvlain ──────────────────────────
print("\nRunning BVSE for Li1+ on LLZO CIF ...")
calc = Lain(verbose=True)
calc.read_file(cif_path)

# Energy landscape → 1D/2D/3D percolation barriers
calc.bvse_distribution(mobile_ion="Li1+", r_cut=10.0, resolution=0.2)
barriers = calc.percolation_barriers()
llzo_barrier_1d = barriers["E_1D"]
llzo_barrier_2d = barriers["E_2D"]
llzo_barrier_3d = barriers["E_3D"]

# Dimensionality: highest finite percolation dimension
if llzo_barrier_3d < np.inf:
    llzo_dimensionality = 3
elif llzo_barrier_2d < np.inf:
    llzo_dimensionality = 2
elif llzo_barrier_1d < np.inf:
    llzo_dimensionality = 1
else:
    llzo_dimensionality = 0

llzo_topology_score = llzo_dimensionality / 3.0

# Anisotropy: spread across finite percolation barriers
_finite_b = [b for b in [llzo_barrier_1d, llzo_barrier_2d, llzo_barrier_3d] if b < np.inf]
llzo_anisotropy = max(_finite_b) - min(_finite_b) if len(_finite_b) > 1 else 0.0

# Accessible volume fraction: fraction of grid points with E < E_min + 1 eV
E_min_llzo = float(calc.data.min())
llzo_accessible_fraction = float(np.mean(calc.data < (E_min_llzo + 1.0)))

# Void/geometric landscape → bottleneck radius
calc.void_distribution(mobile_ion="Li1+", r_cut=10.0, resolution=0.2)
void_radii = calc.percolation_radii()
llzo_bottleneck_radius = void_radii["r_3D"] if void_radii["r_3D"] > 0 else void_radii["r_2D"]

# Li site count from the fetched structure
llzo_li_site_count = float(best.structure.composition[Element('Li')])

print(f"\n  BVSE results for LLZO:")
print(f"    barrier_1d          = {llzo_barrier_1d:.4f} eV")
print(f"    barrier_2d          = {llzo_barrier_2d:.4f} eV")
print(f"    barrier_3d (Ea)     = {llzo_barrier_3d:.4f} eV")
print(f"    dimensionality      = {llzo_dimensionality}")
print(f"    bottleneck_radius   = {llzo_bottleneck_radius:.4f} Å")
print(f"    accessible_fraction = {llzo_accessible_fraction:.4f}")
print(f"    li_site_count       = {llzo_li_site_count:.0f}")

# ── 3. Tier-1 composition feature extraction (mirrors 03_tier1_features.ipynb) ─

ANION_POLARIZABILITY = {
    'O':  1.2,  'S':  5.5,  'Se': 7.0,  'Te': 9.9,
    'F':  1.0,  'Cl': 3.0,  'Br': 4.2,  'I':  6.4,
    'N':  1.5,
}

def identify_anions(comp):
    candidates = set(ANION_POLARIZABILITY.keys())
    anions = [el for el in comp.elements if el.symbol in candidates]
    return anions if anions else [max(comp.elements, key=lambda e: e.X)]

def identify_cations(comp, anions):
    anion_symbols = {a.symbol for a in anions}
    return [el for el in comp.elements if el.symbol not in anion_symbols]

def get_exclusive_anion_type(comp):
    """Return the dominant anion class (oxide/sulfide/halide) by atom count.
    If multiple types present, the one with the highest total count wins;
    ties break in halide > sulfide > oxide order."""
    halogens = {'F', 'Cl', 'Br', 'I'}
    counts = {
        'halide':  sum(comp[Element(e)] for e in halogens if Element(e) in comp),
        'sulfide': comp[Element('S')] if Element('S') in comp else 0.0,
        'oxide':   comp[Element('O')] if Element('O') in comp else 0.0,
    }
    present = {k: v for k, v in counts.items() if v > 0}
    if not present:
        return 'other'
    return max(present, key=lambda k: present[k])

def get_ionic_radius(element, default=1.0):
    try:
        for charge in element.common_oxidation_states:
            if charge in element.ionic_radii:
                return element.ionic_radii[charge]
        if element.ionic_radii:
            return list(element.ionic_radii.values())[0]
    except Exception:
        pass
    return default

def compute_defect_proxies(comp):
    """Return li_stoich_deviation, charge_compensation_proxy, dopant_presence."""
    li_actual = comp[Element('Li')] if Element('Li') in comp.elements else 0.0
    li_stoich_deviation        = 0.0
    charge_compensation_proxy  = 0.0
    dopant_presence            = 0.0
    try:
        oxi = comp.add_charges_from_oxi_state_guesses()
        non_li_charge = sum(
            oxi[sp] * sp.oxi_state for sp in oxi if sp.element.symbol != 'Li'
        )
        li_ideal = -non_li_charge
        li_stoich_deviation = abs(li_actual - li_ideal)
        for sp in oxi:
            expected = sp.element.common_oxidation_states[0] if sp.element.common_oxidation_states else 0
            charge_compensation_proxy += oxi[sp] * abs(sp.oxi_state - expected)
    except Exception:
        pass
    try:
        reduced = comp.reduced_composition
        dopant_presence = float(sum(1 for v in reduced.values() if abs(v - round(v)) > 0.05))
    except Exception:
        pass
    return {
        'li_stoich_deviation':       li_stoich_deviation,
        'charge_compensation_proxy': charge_compensation_proxy,
        'dopant_presence':           dopant_presence,
    }


def extract_composition_features(formula_string):
    """Return the 19 tier-1 + defect-proxy feature dict for a composition string."""
    try:
        comp = Composition(formula_string)
    except Exception:
        return None

    total_atoms = comp.num_atoms
    elements = comp.elements
    anions = identify_anions(comp)
    cations = identify_cations(comp, anions)

    li_count = comp[Element('Li')] if Element('Li') in elements else 0
    en_weighted = sum(comp[el] * el.X for el in elements) / total_atoms
    anion_en = max(a.X for a in anions) if anions else 0
    cation_total = sum(comp[el] for el in cations)
    cation_en_w = sum(comp[el] * el.X for el in cations) / cation_total if cation_total else 0

    radii = {el: get_ionic_radius(el) for el in elements}
    mean_radius = sum(comp[el] * radii[el] for el in elements) / total_atoms
    anion_radius = max(radii[a] for a in anions) if anions else 0
    radius_std = np.std([radii[el] for el in elements for _ in range(int(comp[el]))])

    anion_total = sum(comp[a] for a in anions)
    anion_pol = (
        sum(comp[a] * ANION_POLARIZABILITY.get(a.symbol, 1.0) for a in anions) / anion_total
        if anion_total else 0
    )

    return {
        'li_count':                li_count,
        'mean_electronegativity':  en_weighted,
        'anion_electronegativity': anion_en,
        'cation_anion_en_diff':    anion_en - cation_en_w,
        'mean_ionic_radius':       mean_radius,
        'anion_ionic_radius':      anion_radius,
        'radius_std':              radius_std,
        'anion_polarizability':    anion_pol,
        'mean_atomic_mass':        float(sum(comp[el] * el.atomic_mass for el in elements) / total_atoms),
        'n_elements':              len(elements),
        'mean_atomic_number':      sum(comp[el] * el.Z for el in elements) / total_atoms,
        'anion_type':              get_exclusive_anion_type(comp),
        **compute_defect_proxies(comp),
    }

# ── 4. Load training data + merge BVSE features ───────────────────────────────
# Training set: 478 OBELiX entries only.
# BVSE source hierarchy:
#   real      — original dataset CIF (not in cif_metadata)   → confidence = 1.0
#   real      — exact formula/chemsys match from MP          → confidence = 0.85
#   proxy     — relaxed parent structure (minor dopants dropped) →
#               confidence = clip(exp(-3 * lattice_score), 0.05, 0.75)
#   guessed   — no CIF, median-imputed                       → confidence = 0.0
BASE = os.path.dirname(__file__)
print(f"\nLoading training data (OBELiX-only, 478 entries) ...")

comp_train = pd.read_csv(os.path.join(BASE, 'comp_train.csv'))
obelix_ids = set(comp_train.loc[comp_train['source'] == 'obelix', 'id'])

train   = pd.read_csv(os.path.join(BASE, 'comp_train_features.csv'))
train   = train[train['id'].isin(obelix_ids)].reset_index(drop=True)
bvse_df = pd.read_csv(os.path.join(BASE, 'bvse_features_combined.csv'))

META_COLS = ['id', 'composition', 'log_conductivity']
comp_feature_cols = [c for c in train.columns if c not in META_COLS]

# Recompute exclusive anion one-hot encoding (oxide/sulfide/halide) from composition.
# Overwrites any anion_* columns from the CSV; drops anion_mixed / anion_other.
_ANION_COLS = ['anion_oxide', 'anion_sulfide', 'anion_halide']
_anion_types = train['composition'].apply(
    lambda f: get_exclusive_anion_type(Composition(f)) if isinstance(f, str) else 'other'
)
for col in _ANION_COLS:
    train[col] = (_anion_types == col.replace('anion_', '')).astype(float)
comp_feature_cols = [c for c in comp_feature_cols if not c.startswith('anion_')] + _ANION_COLS

nan_mask = train[comp_feature_cols].isna().any(axis=1)
train = train[~nan_mask].reset_index(drop=True)
print(f"  {len(train)} OBELiX training samples")

# Merge BVSE columns (new columns absent in old CSV rows → NaN, imputed below)
bvse_merge_cols = ['cif_id', 'barrier_1d', 'barrier_2d', 'barrier_3d',
                   'bottleneck_radius', 'accessible_fraction']
bvse_merge_cols = [c for c in bvse_merge_cols if c in bvse_df.columns]
train = train.merge(
    bvse_df[bvse_merge_cols].rename(columns={'cif_id': 'id'}),
    on='id', how='left'
)
for col in ['bottleneck_radius', 'accessible_fraction']:
    if col not in train.columns:
        train[col] = np.nan

# Load CIF metadata produced by generate_obelix_cifs.py
meta_path = os.path.join(BASE, 'cif_metadata.csv')
if os.path.exists(meta_path):
    meta_df = pd.read_csv(meta_path)
    train = train.merge(
        meta_df[['cif_id', 'source_type', 'lattice_score']].rename(columns={'cif_id': 'id'}),
        on='id', how='left'
    )
else:
    train['source_type']  = np.nan
    train['lattice_score'] = np.nan

# Dimensionality: derived directly from barriers for all entries that have them
def _dimensionality(row):
    if pd.isna(row.get('barrier_3d')):
        return np.nan
    if row['barrier_3d'] < 9.99: return 3.0
    if row['barrier_2d'] < 9.99: return 2.0
    if row['barrier_1d'] < 9.99: return 1.0
    return 0.0

train['dimensionality'] = train.apply(_dimensionality, axis=1)

# Li site count: read from CIF files for all OBELiX entries that have a CIF
cif_dir = os.path.join(BASE, 'cifs')
li_counts = {}
for cif_path in glob.glob(os.path.join(cif_dir, '*.cif')):
    cif_id = os.path.splitext(os.path.basename(cif_path))[0]
    try:
        struct = CifParser(cif_path).get_structures(primitive=False)[0]
        li_counts[cif_id] = float(struct.composition[Element('Li')])
    except Exception:
        pass
train['li_site_count'] = train['id'].map(li_counts)
print(f"  li_site_count available for {train['li_site_count'].notna().sum()} entries")

def _bvse_source_confidence(row):
    if pd.isna(row['barrier_3d']):
        return 'guessed', 0.0
    src = row['source_type']
    if pd.isna(src):
        # In bvse_features_combined.csv but no cif_metadata entry → original dataset CIF
        return 'real', 1.0
    if src in ('exact_formula', 'exact_chemsys'):
        return 'real', 0.85
    if src == 'proxy_parent':
        ls   = float(row['lattice_score']) if pd.notna(row['lattice_score']) else 0.5
        conf = float(np.clip(np.exp(-3.0 * ls), 0.05, 0.75))
        return 'proxy', conf
    return 'guessed', 0.0

sc = train.apply(_bvse_source_confidence, axis=1, result_type='expand')
train['bvse_source']    = sc[0]
train['bvse_confidence'] = sc[1].astype(float)
train['bvse_source_real']  = (train['bvse_source'] == 'real').astype(float)
train['bvse_source_proxy'] = (train['bvse_source'] == 'proxy').astype(float)

n_real   = (train['bvse_source'] == 'real').sum()
n_proxy  = (train['bvse_source'] == 'proxy').sum()
n_guess  = (train['bvse_source'] == 'guessed').sum()
print(f"  BVSE coverage — real: {n_real}  proxy: {n_proxy}  guessed: {n_guess}")

# ── Defect proxies from composition (for all training samples) ────────────────
defect_cols = ['li_stoich_deviation', 'charge_compensation_proxy', 'dopant_presence', 'defect_strength']
defect_vals = train['composition'].apply(
    lambda f: pd.Series(compute_defect_proxies(Composition(f))
                        if f else {'li_stoich_deviation': np.nan, 'charge_compensation_proxy': np.nan, 'dopant_presence': np.nan})
)
for col in ['li_stoich_deviation', 'charge_compensation_proxy', 'dopant_presence']:
    train[col] = defect_vals[col].fillna(0.0)
train['defect_strength'] = train['li_stoich_deviation'] + train['dopant_presence']

_defect_zero = (
    (train['li_stoich_deviation'] == 0) &
    (train['dopant_presence'] == 0) &
    (train['charge_compensation_proxy'] == 0)
)
n_dropped = _defect_zero.sum()
train = train[~_defect_zero].reset_index(drop=True)
print(f"  Dropped {n_dropped} entries with all defect proxies == 0")

# ── BVSE feature construction ─────────────────────────────────────────────────
# All BVSE features kept in raw eV — no log/exp mixing.
train['bvse_energy']    = train['barrier_3d']
train['topology_score'] = train['dimensionality'] / 3.0
train['bvse_anisotropy'] = (train[['barrier_1d', 'barrier_2d', 'barrier_3d']].max(axis=1)
                            - train[['barrier_1d', 'barrier_2d', 'barrier_3d']].min(axis=1))

_bvse_value_cols = {
    'bvse_energy':    'bvse_energy',
    'topology_score': 'topology_score',
    'bvse_anisotropy': 'bvse_anisotropy',
    'bvse_bottleneck': 'bottleneck_radius',
    'bvse_accessible': 'accessible_fraction',
    'bvse_li_count':   'li_site_count',
}

bvse_value_cols = []
for feat, src in _bvse_value_cols.items():
    raw = train[feat] if feat in train.columns else train[src]
    train[feat] = raw.fillna(0.0)
    bvse_value_cols.append(feat)

# Single binary flag: 1 = BVSE from a real or proxy CIF, 0 = unavailable (zeros above)
train['bvse_available'] = (train['bvse_source'] != 'guessed').astype(float)

# Derived BVSE features — one encoding per physical quantity
train['li_mobility_capacity'] = train['bvse_accessible'] * train['bvse_li_count']

_derived_cols = ['li_mobility_capacity']
bvse_cols = bvse_value_cols + ['bvse_available'] + _derived_cols
feature_cols = comp_feature_cols + defect_cols + bvse_cols
X_train = train[feature_cols].values
y_train = train['log_conductivity'].values
w_train = train['bvse_source'].map({'real': 1.0, 'proxy': 0.5, 'guessed': 0.1}).values

# ── 5. Train model ────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42,
)
n_feat = len(feature_cols)
print(f"Training GBT model ({n_feat} features: comp + defect + BVSE) ...")
model.fit(X_train, y_train, sample_weight=w_train)

cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                            scoring='neg_mean_absolute_error')
print(f"  5-fold CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 6. Compute tier-1 features for LLZO and assemble prediction row ───────────
print(f"\nComputing tier-1 features for {llzo_formula} ...")
raw = extract_composition_features(llzo_formula)
if raw is None:
    raise RuntimeError(f"Feature extraction failed for '{llzo_formula}'.")

anion_type = raw.pop('anion_type')
feat_df = pd.DataFrame([raw])
feat_df[f'anion_{anion_type}'] = 1

for col in comp_feature_cols:
    if col not in feat_df.columns:
        feat_df[col] = 0

# Defect proxies for LLZO
llzo_defect = compute_defect_proxies(Composition(llzo_formula))
for k, v in llzo_defect.items():
    feat_df[k] = v
feat_df['defect_strength'] = llzo_defect['li_stoich_deviation'] + llzo_defect['dopant_presence']

# BVSE values — all raw eV, no log/exp mixing
feat_df['bvse_energy']        = llzo_barrier_3d
feat_df['topology_score']     = llzo_dimensionality / 3.0
feat_df['bvse_anisotropy']    = llzo_anisotropy
feat_df['bvse_bottleneck']    = llzo_bottleneck_radius
feat_df['bvse_accessible']    = llzo_accessible_fraction
feat_df['bvse_li_count']      = llzo_li_site_count
feat_df['bvse_available']     = 1.0

# Derived features — one encoding per physical quantity
feat_df['li_mobility_capacity'] = llzo_accessible_fraction * llzo_li_site_count

X_llzo = feat_df[feature_cols].values

print("\n  Feature values for LLZO:")
for col, val in zip(feature_cols, X_llzo[0]):
    print(f"    {col:<28} {val:.4f}")

# ── 7. Predict and report ─────────────────────────────────────────────────────
log_sigma  = float(model.predict(X_llzo)[0])
sigma      = 10 ** log_sigma
train_mae  = mean_absolute_error(y_train, model.predict(X_train))
sigma_low  = 10 ** (log_sigma - train_mae)
sigma_high = 10 ** (log_sigma + train_mae)

print(f"\n{'='*62}")
print(f"  GBT + BVSE Prediction: ionic conductivity of {FORMULA}")
print(f"{'='*62}")
print(f"  bvse_energy (Ea)       = {llzo_barrier_3d:.4f} eV")
print(f"  topology_score         = {llzo_dimensionality/3.0:.3f}  (dimensionality/3)")
print(f"  bvse_anisotropy        = {llzo_anisotropy:.4f} eV")
print(f"  bvse_bottleneck        = {llzo_bottleneck_radius:.4f} Å")
print(f"  bvse_accessible        = {llzo_accessible_fraction:.4f}")
print(f"  bvse_li_count          = {llzo_li_site_count:.0f}")
print(f"  li_mobility_capacity   = {llzo_accessible_fraction * llzo_li_site_count:.4f}")
print(f"  li_stoich_deviation    = {llzo_defect['li_stoich_deviation']:.3f}")
print(f"  charge_compensation    = {llzo_defect['charge_compensation_proxy']:.3f}")
print(f"  dopant_presence        = {llzo_defect['dopant_presence']:.0f}")
print(f"  defect_strength        = {llzo_defect['li_stoich_deviation'] + llzo_defect['dopant_presence']:.3f}")
print(f"  bvse_available         = 1  (BVSE computed from real MP CIF)")
print(f"  log₁₀(σ) predicted     = {log_sigma:.3f}")
print(f"  σ (300 K)              = {sigma:.4e} S/cm")
print(f"  ± 1 train-MAE range    = [{sigma_low:.4e}, {sigma_high:.4e}] S/cm")
print(f"\n  Experimental cubic LLZO: ~10⁻⁴ – 10⁻³ S/cm")
print(f"{'='*62}")
