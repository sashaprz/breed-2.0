# XGB_ensemble_LLZO.py
#
# Ensemble of two XGBoost models for LLZO ionic conductivity prediction:
#
#   Model 1 — BVSE + composition (high-fidelity, OBELiX-only, ~478 samples)
#             Identical feature set to XGB_predict_LLZO.py.
#             BVSE barriers computed live from the MP CIF.
#             Params cached in xgb_best_params.json.
#
#   Model 2 — Composition-only (full training set, OBELiX + Liverpool, ~826 samples)
#             No structure-derived features; composition + defect proxies only.
#             Params cached in xgb_comp_only_best_params.json.
#
#   Ensemble — weighted average of log₁₀(σ) predictions.
#              Weights are determined by inverse 5-fold CV MAE (better model
#              gets higher weight automatically).

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from pymatgen.core import Composition, Element
from pymatgen.io.cif import CifWriter, CifParser
from mp_api.client import MPRester
from bvlain import Lain

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Materials Project API key ─────────────────────────────────────────────────
MP_API_KEY = os.environ.get("MP_API_KEY")
if MP_API_KEY is None:
    raise EnvironmentError(
        "Set the MP_API_KEY environment variable before running this script.\n"
        "Get your key at https://materialsproject.org/api"
    )

EARLY_STOPPING_ROUNDS = 30
N_OPTUNA_TRIALS       = 200

parser = argparse.ArgumentParser(description="Predict ionic conductivity for any solid electrolyte formula.")
parser.add_argument("formula", nargs="?", help="Chemical formula, e.g. Li10GeP2S12")
args = parser.parse_args()

if args.formula:
    FORMULA = args.formula.strip()
else:
    FORMULA = input("Enter chemical formula (e.g. Li10GeP2S12): ").strip()

if not FORMULA:
    sys.exit("Error: no formula provided.")

# Known oxidation states for common solid-electrolyte elements.
# Used to manually decorate the MP structure before BVSE so that
# bvlain can resolve the correct BV parameters for each species.
OXIDATION_STATES = {
    'Li': +1, 'Na': +1, 'K':  +1,
    'Mg': +2, 'Ca': +2, 'Ba': +2, 'Zn': +2, 'Cd': +2,
    'Al': +3, 'Ga': +3, 'In': +3, 'B':  +3, 'La': +3, 'Y': +3,
    'Si': +4, 'Ge': +4, 'Sn': +4, 'Ti': +4, 'Zr': +4, 'Hf': +4,
    'P':  +5, 'As': +5, 'Nb': +5, 'Ta': +5,
    'O':  -2, 'S':  -2, 'Se': -2, 'Te': -2,
    'F':  -1, 'Cl': -1, 'Br': -1, 'I':  -1,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

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

def get_anion_type(anions):
    symbols = {a.symbol for a in anions}
    halogens = {'F', 'Cl', 'Br', 'I'}
    types = []
    if 'O' in symbols:     types.append('oxide')
    if 'S' in symbols:     types.append('sulfide')
    if symbols & halogens: types.append('halide')
    if not types:          return 'other'
    if len(types) == 1:    return types[0]
    return 'mixed'

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
    li_actual = comp[Element('Li')] if Element('Li') in comp.elements else 0.0
    li_stoich_deviation       = 0.0
    charge_compensation_proxy = 0.0
    dopant_presence           = 0.0
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
    try:
        comp = Composition(formula_string)
    except Exception:
        return None

    total_atoms = comp.num_atoms
    elements    = comp.elements
    anions      = identify_anions(comp)
    cations     = identify_cations(comp, anions)

    li_count      = comp[Element('Li')] if Element('Li') in elements else 0
    en_weighted   = sum(comp[el] * el.X for el in elements) / total_atoms
    anion_en      = max(a.X for a in anions) if anions else 0
    cation_total  = sum(comp[el] for el in cations)
    cation_en_w   = sum(comp[el] * el.X for el in cations) / cation_total if cation_total else 0

    radii        = {el: get_ionic_radius(el) for el in elements}
    mean_radius  = sum(comp[el] * radii[el] for el in elements) / total_atoms
    anion_radius = max(radii[a] for a in anions) if anions else 0
    radius_std   = np.std([radii[el] for el in elements for _ in range(int(comp[el]))])

    anion_total = sum(comp[a] for a in anions)
    anion_pol   = (
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
        'anion_type':              get_anion_type(anions),
        **compute_defect_proxies(comp),
    }

def _cv_mae_xgb(X, y, w, params, return_folds=False):
    """5-fold CV MAE with inner 85/15 early-stopping split."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes = []
    for tr_idx, val_idx in kf.split(X):
        X_ot, X_ov = X[tr_idx], X[val_idx]
        y_ot, y_ov = y[tr_idx], y[val_idx]
        w_ot       = w[tr_idx] if w is not None else None

        split_kw = dict(test_size=0.15, random_state=42)
        if w_ot is not None:
            X_it, X_iv, y_it, y_iv, w_it, _ = train_test_split(X_ot, y_ot, w_ot, **split_kw)
        else:
            X_it, X_iv, y_it, y_iv = train_test_split(X_ot, y_ot, **split_kw)
            w_it = None

        m = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbosity=0)
        m.fit(X_it, y_it, sample_weight=w_it, eval_set=[(X_iv, y_iv)], verbose=False)
        fold_maes.append(mean_absolute_error(y_ov, m.predict(X_ov)))

    return fold_maes if return_folds else float(np.mean(fold_maes))

def _load_or_tune(params_path, X, y, w, label):
    """Return best XGBoost params, loading from cache or running Optuna."""
    if os.path.exists(params_path):
        with open(params_path) as f:
            saved = json.load(f)
        best_params = {k: v for k, v in saved.items() if k != 'cv_mae'}
        print(f"\nLoaded cached params for {label} from {params_path}")
        if 'cv_mae' in saved:
            print(f"  Cached CV MAE: {saved['cv_mae']:.4f}")
        return best_params

    print(f"\nRunning Optuna for {label} ({N_OPTUNA_TRIALS} trials, 5-fold CV) ...")

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),
            'max_depth':        trial.suggest_int('max_depth', 2, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma':            trial.suggest_float('gamma', 0, 5.0),
            'random_state':     42,
        }
        return _cv_mae_xgb(X, y, w, params)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    best_params = {**study.best_params, 'random_state': 42}
    print(f"  Best 5-fold CV MAE: {study.best_value:.4f}")
    with open(params_path, 'w') as f:
        json.dump({**best_params, 'cv_mae': study.best_value}, f, indent=2)
    print(f"  Params saved to {params_path}")
    return best_params

def _train_final(X, y, w, params):
    """Train final model with 85/15 early-stopping split."""
    split_kw = dict(test_size=0.15, random_state=42)
    if w is not None:
        X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(X, y, w, **split_kw)
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, **split_kw)
        w_tr = None
    model = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbosity=0)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Fetch LLZO from Materials Project + compute BVSE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"Fetching {FORMULA} from Materials Project...")
with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.search(formula=FORMULA, fields=["material_id", "structure", "entries"])

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
best         = docs[0]
lpscl_formula = best.structure.composition.reduced_formula
e_hull       = _e_above_hull(best)
print(f"  material_id  : {best.material_id}")
if e_hull < float("inf"):
    print(f"  E above hull : {e_hull:.4f} eV/atom")
print(f"  Formula used : {lpscl_formula}")

# Assign oxidation states before writing CIF so bvlain resolves BV parameters correctly.
structure = best.structure.copy()
comp      = structure.composition

oxi_dict = {}
missing  = []
for el in comp.elements:
    if el.symbol in OXIDATION_STATES:
        oxi_dict[el.symbol] = OXIDATION_STATES[el.symbol]
    else:
        missing.append(el.symbol)

if missing:
    print(f"  WARNING: no oxidation state defined for {missing} — attempting pymatgen guess")
    try:
        guessed = comp.add_charges_from_oxi_state_guesses()
        for sym in missing:
            matches = [sp for sp in guessed if sp.element.symbol == sym]
            if matches:
                oxi_dict[sym] = int(matches[0].oxi_state)
    except Exception:
        pass

charge_sum = sum(comp[Element(sym)] * oxi for sym, oxi in oxi_dict.items())
print(f"  Oxidation states : {oxi_dict}")
print(f"  Charge sum check : {charge_sum:+.2f}  (target: 0)")
if abs(charge_sum) > 0.1:
    print(f"  WARNING: structure is not charge-neutral — BVSE lookup may be unreliable")

structure.add_oxidation_state_by_element(oxi_dict)

cif_path = os.path.join(BASE, 'data', 'cifs', 'test_candidates', f"{FORMULA}_mp.cif")
CifWriter(structure).write_file(cif_path)
print(f"  CIF written  : {cif_path}")

print(f"\nRunning BVSE for Li1+ on {FORMULA} ...")
calc = Lain(verbose=True)
# Pass the pre-decorated structure directly and disable oxi_check so
# BVAnalyzer does not overwrite our manually assigned oxidation states.
calc.read_structure(structure, oxi_check=False)
calc.bvse_distribution(mobile_ion="Li1+", r_cut=10.0, resolution=0.2)
barriers             = calc.percolation_barriers()
lpscl_barrier_1d      = barriers["E_1D"]
lpscl_barrier_2d      = barriers["E_2D"]
lpscl_barrier_3d      = barriers["E_3D"]

if   lpscl_barrier_3d < np.inf: lpscl_dimensionality = 3
elif lpscl_barrier_2d < np.inf: lpscl_dimensionality = 2
elif lpscl_barrier_1d < np.inf: lpscl_dimensionality = 1
else:                           lpscl_dimensionality = 0

_finite_b       = [b for b in [lpscl_barrier_1d, lpscl_barrier_2d, lpscl_barrier_3d] if b < np.inf]
lpscl_anisotropy = max(_finite_b) - min(_finite_b) if len(_finite_b) > 1 else 0.0

E_min_lpscl              = float(calc.data.min())
lpscl_accessible_fraction = float(np.mean(calc.data < (E_min_lpscl + 1.0)))

calc.void_distribution(mobile_ion="Li1+", r_cut=10.0, resolution=0.2)
void_radii            = calc.percolation_radii()
lpscl_bottleneck_radius = void_radii["r_3D"] if void_radii["r_3D"] > 0 else void_radii["r_2D"]
lpscl_li_site_count     = float(best.structure.composition[Element('Li')])

print(f"  barrier_3d (Ea)     = {lpscl_barrier_3d:.4f} eV")
print(f"  dimensionality      = {lpscl_dimensionality}")
print(f"  bottleneck_radius   = {lpscl_bottleneck_radius:.4f} Å")
print(f"  accessible_fraction = {lpscl_accessible_fraction:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Model 1: BVSE + composition, OBELiX-only (~478 samples)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*62}")
print("Model 1: BVSE + composition  (OBELiX-only, high-fidelity)")
print(f"{'─'*62}")

comp_train_raw = pd.read_csv(os.path.join(BASE, 'data', 'splits', 'comp_train.csv'))
obelix_ids     = set(comp_train_raw.loc[comp_train_raw['source'] == 'obelix', 'id'])

train1   = pd.read_csv(os.path.join(BASE, 'data', 'features', 'comp_train_features.csv'))
train1   = train1[train1['id'].isin(obelix_ids)].reset_index(drop=True)
bvse_df  = pd.read_csv(os.path.join(BASE, 'data', 'features', 'bvse_features_combined.csv'))

META_COLS       = ['id', 'composition', 'log_conductivity']
comp_feat_cols  = [c for c in train1.columns if c not in META_COLS]

nan_mask = train1[comp_feat_cols].isna().any(axis=1)
train1   = train1[~nan_mask].reset_index(drop=True)
print(f"  OBELiX samples after NaN filter: {len(train1)}")

bvse_merge_cols = [c for c in ['cif_id', 'barrier_1d', 'barrier_2d', 'barrier_3d',
                                'bottleneck_radius', 'accessible_fraction'] if c in bvse_df.columns]
train1 = train1.merge(
    bvse_df[bvse_merge_cols].rename(columns={'cif_id': 'id'}),
    on='id', how='left'
)
for col in ['bottleneck_radius', 'accessible_fraction']:
    if col not in train1.columns:
        train1[col] = np.nan

meta_path = os.path.join(BASE, 'data', 'features', 'cif_metadata.csv')
if os.path.exists(meta_path):
    meta_df = pd.read_csv(meta_path)
    train1 = train1.merge(
        meta_df[['cif_id', 'source_type', 'lattice_score']].rename(columns={'cif_id': 'id'}),
        on='id', how='left'
    )
else:
    train1['source_type']   = np.nan
    train1['lattice_score'] = np.nan

def _dimensionality(row):
    if pd.isna(row.get('barrier_3d')): return np.nan
    if row['barrier_3d'] < 9.99: return 3.0
    if row['barrier_2d'] < 9.99: return 2.0
    if row['barrier_1d'] < 9.99: return 1.0
    return 0.0

train1['dimensionality'] = train1.apply(_dimensionality, axis=1)

cif_dir   = os.path.join(BASE, 'data', 'cifs', 'training')
li_counts = {}
for cif_p in glob.glob(os.path.join(cif_dir, '*.cif')):
    cif_id = os.path.splitext(os.path.basename(cif_p))[0]
    try:
        struct = CifParser(cif_p).get_structures(primitive=False)[0]
        li_counts[cif_id] = float(struct.composition[Element('Li')])
    except Exception:
        pass
train1['li_site_count'] = train1['id'].map(li_counts)

def _bvse_source_confidence(row):
    if pd.isna(row['barrier_3d']):
        return 'guessed', 0.0
    src = row['source_type']
    if pd.isna(src):
        return 'real', 1.0
    if src in ('exact_formula', 'exact_chemsys'):
        return 'real', 0.85
    if src == 'proxy_parent':
        ls   = float(row['lattice_score']) if pd.notna(row['lattice_score']) else 0.5
        conf = float(np.clip(np.exp(-3.0 * ls), 0.05, 0.75))
        return 'proxy', conf
    return 'guessed', 0.0

sc = train1.apply(_bvse_source_confidence, axis=1, result_type='expand')
train1['bvse_source']     = sc[0]
train1['bvse_confidence'] = sc[1].astype(float)

n_real  = (train1['bvse_source'] == 'real').sum()
n_proxy = (train1['bvse_source'] == 'proxy').sum()
n_guess = (train1['bvse_source'] == 'guessed').sum()
print(f"  BVSE coverage — real: {n_real}  proxy: {n_proxy}  guessed: {n_guess}")

defect_vals1 = train1['composition'].apply(
    lambda f: pd.Series(compute_defect_proxies(Composition(f)) if f else
              {'li_stoich_deviation': np.nan, 'charge_compensation_proxy': np.nan, 'dopant_presence': np.nan})
)
for col in ['li_stoich_deviation', 'charge_compensation_proxy', 'dopant_presence']:
    train1[col] = defect_vals1[col].fillna(0.0)
train1['defect_strength'] = train1['li_stoich_deviation'] + train1['dopant_presence']

train1['bvse_energy']     = train1['barrier_3d']
train1['topology_score']  = train1['dimensionality'] / 3.0
train1['bvse_anisotropy'] = (train1[['barrier_1d', 'barrier_2d', 'barrier_3d']].max(axis=1)
                             - train1[['barrier_1d', 'barrier_2d', 'barrier_3d']].min(axis=1))

_bvse_value_map = {
    'bvse_energy':    'bvse_energy',
    'topology_score': 'topology_score',
    'bvse_anisotropy': 'bvse_anisotropy',
    'bvse_bottleneck': 'bottleneck_radius',
    'bvse_accessible': 'accessible_fraction',
    'bvse_li_count':   'li_site_count',
}
bvse_value_cols = []
for feat, src in _bvse_value_map.items():
    raw = train1[feat] if feat in train1.columns else train1[src]
    train1[feat] = raw.fillna(0.0)
    bvse_value_cols.append(feat)

train1['bvse_available']      = (train1['bvse_source'] != 'guessed').astype(float)
train1['bvse_energy_inv']     = 1.0 / (train1['bvse_energy'] + 1e-6)
train1['li_mobility_capacity'] = train1['bvse_accessible'] * train1['bvse_li_count']

defect_cols1  = ['li_stoich_deviation', 'dopant_presence', 'defect_strength']
_derived_bvse = ['bvse_energy_inv', 'li_mobility_capacity']
bvse_cols     = bvse_value_cols + ['bvse_available'] + _derived_bvse
feat_cols1    = comp_feat_cols + defect_cols1 + bvse_cols

X1 = train1[feat_cols1].values
y1 = train1['log_conductivity'].values
w1 = train1['bvse_source'].map({'real': 1.0, 'proxy': 0.5, 'guessed': 0.1}).values

best_params1 = _load_or_tune(
    os.path.join(BASE, 'xgb_best_params.json'), X1, y1, w1, "Model 1 (BVSE+comp)"
)
model1       = _train_final(X1, y1, w1, best_params1)
fold_maes1   = _cv_mae_xgb(X1, y1, w1, best_params1, return_folds=True)
cv_mae1      = float(np.mean(fold_maes1))
print(f"  5-fold CV MAE: {cv_mae1:.3f} ± {np.std(fold_maes1):.3f}  (n={len(train1)})")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Model 2: composition-only, full training set (OBELiX + Liverpool)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*62}")
print("Model 2: composition-only  (OBELiX + Liverpool, full dataset)")
print(f"{'─'*62}")

train2 = pd.read_csv(os.path.join(BASE, 'data', 'features', 'comp_train_features.csv'))
nan_mask2 = train2[comp_feat_cols].isna().any(axis=1)
train2    = train2[~nan_mask2].reset_index(drop=True)

src2 = train2['id'].isin(obelix_ids).map({True: 'obelix', False: 'liverpool'})
n_ob2  = (src2 == 'obelix').sum()
n_liv2 = (src2 == 'liverpool').sum()
print(f"  Samples: {len(train2)} total  ({n_ob2} OBELiX + {n_liv2} Liverpool)")

defect_vals2 = train2['composition'].apply(
    lambda f: pd.Series(compute_defect_proxies(Composition(f)) if f else
              {'li_stoich_deviation': np.nan, 'charge_compensation_proxy': np.nan, 'dopant_presence': np.nan})
)
for col in ['li_stoich_deviation', 'charge_compensation_proxy', 'dopant_presence']:
    train2[col] = defect_vals2[col].fillna(0.0)
train2['defect_strength'] = train2['li_stoich_deviation'] + train2['dopant_presence']

defect_cols2 = ['li_stoich_deviation', 'dopant_presence', 'defect_strength']
feat_cols2   = comp_feat_cols + defect_cols2

X2 = train2[feat_cols2].values
y2 = train2['log_conductivity'].values
w2 = None  # uniform weights — no BVSE confidence for comp-only

best_params2 = _load_or_tune(
    os.path.join(BASE, 'xgb_comp_only_best_params.json'), X2, y2, w2, "Model 2 (comp-only)"
)
model2     = _train_final(X2, y2, w2, best_params2)
fold_maes2 = _cv_mae_xgb(X2, y2, w2, best_params2, return_folds=True)
cv_mae2    = float(np.mean(fold_maes2))
print(f"  5-fold CV MAE: {cv_mae2:.3f} ± {np.std(fold_maes2):.3f}  (n={len(train2)})")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Build LLZO feature vectors for each model
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nComputing features for {lpscl_formula} ...")
raw = extract_composition_features(lpscl_formula)
if raw is None:
    raise RuntimeError(f"Feature extraction failed for '{lpscl_formula}'.")

anion_type = raw.pop('anion_type')
base_df    = pd.DataFrame([raw])
base_df[f'anion_{anion_type}'] = 1
for col in comp_feat_cols:
    if col not in base_df.columns:
        base_df[col] = 0

lpscl_defect = compute_defect_proxies(Composition(lpscl_formula))
for k, v in lpscl_defect.items():
    base_df[k] = v
base_df['defect_strength'] = lpscl_defect['li_stoich_deviation'] + lpscl_defect['dopant_presence']

# Model 1 extra: BVSE features
feat1_df = base_df.copy()
feat1_df['bvse_energy']         = lpscl_barrier_3d
feat1_df['topology_score']      = lpscl_dimensionality / 3.0
feat1_df['bvse_anisotropy']     = lpscl_anisotropy
feat1_df['bvse_bottleneck']     = lpscl_bottleneck_radius
feat1_df['bvse_accessible']     = lpscl_accessible_fraction
feat1_df['bvse_li_count']       = lpscl_li_site_count
feat1_df['bvse_available']      = 1.0
feat1_df['bvse_energy_inv']     = 1.0 / (lpscl_barrier_3d + 1e-6)
feat1_df['li_mobility_capacity'] = lpscl_accessible_fraction * lpscl_li_site_count

X_llzo1 = feat1_df[feat_cols1].values
X_llzo2 = base_df[feat_cols2].values

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — Ensemble: inverse-MAE weighted average
# ═══════════════════════════════════════════════════════════════════════════════
log_sigma1 = float(model1.predict(X_llzo1)[0])
log_sigma2 = float(model2.predict(X_llzo2)[0])

# Inverse-MAE weights: better model (lower MAE) gets higher weight
inv1   = 1.0 / cv_mae1
inv2   = 1.0 / cv_mae2
w_ens1 = inv1 / (inv1 + inv2)
w_ens2 = inv2 / (inv1 + inv2)

log_sigma_ens = w_ens1 * log_sigma1 + w_ens2 * log_sigma2

# Uncertainty: propagate CV MAE through ensemble weights
ens_mae_lower = w_ens1 * cv_mae1 + w_ens2 * cv_mae2  # weighted average MAE

sigma1    = 10 ** log_sigma1
sigma2    = 10 ** log_sigma2
sigma_ens = 10 ** log_sigma_ens
sigma_low_ens  = 10 ** (log_sigma_ens - ens_mae_lower)
sigma_high_ens = 10 ** (log_sigma_ens + ens_mae_lower)

# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"  Ensemble Prediction: ionic conductivity of {FORMULA}")
print(f"{'='*62}")
print(f"  Model 1 (BVSE+comp, {len(train1)} samples)")
print(f"    CV MAE        = {cv_mae1:.3f}  →  weight = {w_ens1:.3f}")
print(f"    log₁₀(σ)      = {log_sigma1:.3f}  →  σ = {sigma1:.4e} S/cm")
print(f"  Model 2 (comp-only, {len(train2)} samples)")
print(f"    CV MAE        = {cv_mae2:.3f}  →  weight = {w_ens2:.3f}")
print(f"    log₁₀(σ)      = {log_sigma2:.3f}  →  σ = {sigma2:.4e} S/cm")
print(f"  {'─'*58}")
print(f"  Ensemble log₁₀(σ) = {log_sigma_ens:.3f}")
print(f"  σ (300 K)          = {sigma_ens:.4e} S/cm")
print(f"  ± 1 MAE range      = [{sigma_low_ens:.4e}, {sigma_high_ens:.4e}] S/cm")
print(f"\n  (check literature for experimental reference for {FORMULA})")
print(f"{'='*62}")
