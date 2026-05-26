# XGB_analysis.py
#
# Post-training analysis of the XGBoost + BVSE model.
# Requires xgb_best_params.json (produced by XGB_predict_LLZO.py).
# Falls back to reasonable defaults if the file is missing.
#
# Analyses:
#   1. SHAP decomposition — % contribution by feature group
#   2. OOF residual vs bvse_energy — physics-gap vs noise check
#   3. Abs-error by anion type — chemistry-failure clustering
#   4. Charge-imbalance feature ablation — add 1 column → MAE delta
#   5. BVSE ablation — remove structural features → MAE jump

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from pymatgen.core import Composition, Element
from pymatgen.io.cif import CifParser

warnings.filterwarnings('ignore')

BASE                  = os.path.dirname(__file__)
EARLY_STOPPING_ROUNDS = 30
PARAMS_PATH           = os.path.join(BASE, 'xgb_best_params.json')
FALLBACK_PARAMS       = {
    'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
    'reg_alpha': 0.01, 'reg_lambda': 1.0, 'gamma': 0.1, 'random_state': 42,
}

# ── Composition helpers ───────────────────────────────────────────────────────

ANION_POLARIZABILITY = {
    'O': 1.2, 'S': 5.5, 'Se': 7.0, 'Te': 9.9,
    'F': 1.0, 'Cl': 3.0, 'Br': 4.2, 'I': 6.4, 'N': 1.5,
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
    li_stoich_deviation, charge_compensation_proxy, dopant_presence = 0.0, 0.0, 0.0
    try:
        oxi = comp.add_charges_from_oxi_state_guesses()
        non_li_charge = sum(oxi[sp] * sp.oxi_state for sp in oxi if sp.element.symbol != 'Li')
        li_stoich_deviation = abs(li_actual - (-non_li_charge))
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

def compute_charge_imbalance(comp):
    """Absolute total formal charge using each element's most common oxidation state.
    Non-zero when a composition cannot be charge-balanced with standard oxidation states."""
    total = 0.0
    for el in comp.elements:
        if el.common_oxidation_states:
            total += comp[el] * el.common_oxidation_states[0]
    return abs(total)

# ── Data loading (mirrors XGB_predict_LLZO.py sections 3-4) ──────────────────

print("Loading training data ...")
comp_train_df = pd.read_csv(os.path.join(BASE, 'comp_train.csv'))
obelix_ids    = set(comp_train_df.loc[comp_train_df['source'] == 'obelix', 'id'])

train   = pd.read_csv(os.path.join(BASE, 'comp_train_features.csv'))
train   = train[train['id'].isin(obelix_ids)].reset_index(drop=True)
bvse_df = pd.read_csv(os.path.join(BASE, 'bvse_features_combined.csv'))

META_COLS = ['id', 'composition', 'log_conductivity']
comp_feature_cols = [c for c in train.columns if c not in META_COLS]

train = train[~train[comp_feature_cols].isna().any(axis=1)].reset_index(drop=True)
print(f"  {len(train)} OBELiX training samples, {len(comp_feature_cols)} comp features")

bvse_merge_cols = [c for c in ['cif_id', 'barrier_1d', 'barrier_2d', 'barrier_3d',
                                'bottleneck_radius', 'accessible_fraction'] if c in bvse_df.columns]
train = train.merge(bvse_df[bvse_merge_cols].rename(columns={'cif_id': 'id'}), on='id', how='left')
for col in ['bottleneck_radius', 'accessible_fraction']:
    if col not in train.columns:
        train[col] = np.nan

meta_path = os.path.join(BASE, 'cif_metadata.csv')
if os.path.exists(meta_path):
    meta_df = pd.read_csv(meta_path)
    train = train.merge(
        meta_df[['cif_id', 'source_type', 'lattice_score']].rename(columns={'cif_id': 'id'}),
        on='id', how='left'
    )
else:
    train['source_type'] = np.nan
    train['lattice_score'] = np.nan

def _dimensionality(row):
    if pd.isna(row.get('barrier_3d')): return np.nan
    if row['barrier_3d'] < 9.99: return 3.0
    if row['barrier_2d'] < 9.99: return 2.0
    if row['barrier_1d'] < 9.99: return 1.0
    return 0.0

train['dimensionality'] = train.apply(_dimensionality, axis=1)

cif_dir  = os.path.join(BASE, 'cifs')
li_counts = {}
for cif_p in glob.glob(os.path.join(cif_dir, '*.cif')):
    cid = os.path.splitext(os.path.basename(cif_p))[0]
    try:
        struct = CifParser(cif_p).get_structures(primitive=False)[0]
        li_counts[cid] = float(struct.composition[Element('Li')])
    except Exception:
        pass
train['li_site_count'] = train['id'].map(li_counts)

def _bvse_source(row):
    if pd.isna(row['barrier_3d']): return 'guessed', 0.0
    src = row['source_type']
    if pd.isna(src): return 'real', 1.0
    if src in ('exact_formula', 'exact_chemsys'): return 'real', 0.85
    if src == 'proxy_parent':
        ls   = float(row['lattice_score']) if pd.notna(row['lattice_score']) else 0.5
        conf = float(np.clip(np.exp(-3.0 * ls), 0.05, 0.75))
        return 'proxy', conf
    return 'guessed', 0.0

sc = train.apply(_bvse_source, axis=1, result_type='expand')
train['bvse_source']    = sc[0]
train['bvse_confidence'] = sc[1].astype(float)

defect_vals = train['composition'].apply(
    lambda f: pd.Series(compute_defect_proxies(Composition(f)) if f
              else {'li_stoich_deviation': np.nan, 'charge_compensation_proxy': np.nan, 'dopant_presence': np.nan})
)
for col in ['li_stoich_deviation', 'charge_compensation_proxy', 'dopant_presence']:
    train[col] = defect_vals[col].fillna(0.0)
train['defect_strength'] = train['li_stoich_deviation'] + train['dopant_presence']
defect_cols = ['li_stoich_deviation', 'dopant_presence', 'defect_strength']

train['bvse_energy']     = train['barrier_3d']
train['topology_score']  = train['dimensionality'] / 3.0
train['bvse_anisotropy'] = (train[['barrier_1d', 'barrier_2d', 'barrier_3d']].max(axis=1)
                            - train[['barrier_1d', 'barrier_2d', 'barrier_3d']].min(axis=1))

for feat, src in {'bvse_energy': 'bvse_energy', 'topology_score': 'topology_score',
                  'bvse_anisotropy': 'bvse_anisotropy', 'bvse_bottleneck': 'bottleneck_radius',
                  'bvse_accessible': 'accessible_fraction', 'bvse_li_count': 'li_site_count'}.items():
    raw = train[feat] if feat in train.columns else train[src]
    train[feat] = raw.fillna(0.0)

train['bvse_available']       = (train['bvse_source'] != 'guessed').astype(float)
train['bvse_energy_inv']      = 1.0 / (train['bvse_energy'] + 1e-6)
train['li_mobility_capacity'] = train['bvse_accessible'] * train['bvse_li_count']

bvse_value_cols = ['bvse_energy', 'topology_score', 'bvse_anisotropy',
                   'bvse_bottleneck', 'bvse_accessible', 'bvse_li_count']
bvse_cols       = bvse_value_cols + ['bvse_available', 'bvse_energy_inv', 'li_mobility_capacity']
feature_cols    = comp_feature_cols + defect_cols + bvse_cols

X_train = train[feature_cols].values.astype(float)
y_train = train['log_conductivity'].values
w_train = train['bvse_source'].map({'real': 1.0, 'proxy': 0.5, 'guessed': 0.1}).values

# Anion type label per sample (from one-hot columns for plots)
anion_onehot_cols = [c for c in comp_feature_cols if c.startswith('anion_')]
def _anion_label(row):
    for col in anion_onehot_cols:
        if row.get(col, 0) == 1:
            return col.replace('anion_', '')
    return 'other'
train['anion_label'] = train.apply(_anion_label, axis=1)

# ── Load model params ─────────────────────────────────────────────────────────

if os.path.exists(PARAMS_PATH):
    with open(PARAMS_PATH) as f:
        saved = json.load(f)
    best_params = {k: v for k, v in saved.items() if k != 'cv_mae'}
    print(f"Loaded params from {PARAMS_PATH}  (cached CV MAE: {saved.get('cv_mae', 'n/a')})")
else:
    best_params = FALLBACK_PARAMS
    print(f"WARNING: {PARAMS_PATH} not found — using fallback params. Run XGB_predict_LLZO.py first.")

# ── CV helpers ────────────────────────────────────────────────────────────────

def _cv_mae(params, X, y, w, return_folds=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes = []
    for tr_idx, val_idx in kf.split(X):
        X_out_tr, X_out_val = X[tr_idx], X[val_idx]
        y_out_tr, y_out_val = y[tr_idx], y[val_idx]
        w_out_tr = w[tr_idx]
        X_in_tr, X_in_val, y_in_tr, y_in_val, w_in_tr, _ = train_test_split(
            X_out_tr, y_out_tr, w_out_tr, test_size=0.15, random_state=42
        )
        m = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbosity=0)
        m.fit(X_in_tr, y_in_tr, sample_weight=w_in_tr,
              eval_set=[(X_in_val, y_in_val)], verbose=False)
        fold_maes.append(mean_absolute_error(y_out_val, m.predict(X_out_val)))
    return fold_maes if return_folds else float(np.mean(fold_maes))

def _oof_predictions(params, X, y, w):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr_idx, val_idx in kf.split(X):
        X_out_tr, X_out_val = X[tr_idx], X[val_idx]
        y_out_tr, y_out_val = y[tr_idx], y[val_idx]
        w_out_tr = w[tr_idx]
        X_in_tr, X_in_val, y_in_tr, y_in_val, w_in_tr, _ = train_test_split(
            X_out_tr, y_out_tr, w_out_tr, test_size=0.15, random_state=42
        )
        m = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbosity=0)
        m.fit(X_in_tr, y_in_tr, sample_weight=w_in_tr,
              eval_set=[(X_in_val, y_in_val)], verbose=False)
        oof[val_idx] = m.predict(X_out_val)
    return oof

# ── Train final model ─────────────────────────────────────────────────────────

print(f"\nTraining final model on {len(X_train)} samples ({len(feature_cols)} features) ...")
X_fin_tr, X_fin_val, y_fin_tr, y_fin_val, w_fin_tr, _ = train_test_split(
    X_train, y_train, w_train, test_size=0.15, random_state=42
)
model = xgb.XGBRegressor(**best_params, early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbosity=0)
model.fit(X_fin_tr, y_fin_tr, sample_weight=w_fin_tr,
          eval_set=[(X_fin_val, y_fin_val)], verbose=False)
print(f"  Best iteration: {model.best_iteration}")

print("\nComputing OOF predictions for residual analyses ...")
oof_pred = _oof_predictions(best_params, X_train, y_train, w_train)
oof_residual = y_train - oof_pred
oof_mae = mean_absolute_error(y_train, oof_pred)
print(f"  OOF MAE: {oof_mae:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 1: SHAP decomposition
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── Analysis 1: SHAP decomposition ───")

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)          # (n_samples, n_features)
mean_abs    = np.abs(shap_values).mean(axis=0)         # mean |SHAP| per feature

feat_idx    = {f: i for i, f in enumerate(feature_cols)}
comp_shap   = mean_abs[[feat_idx[f] for f in comp_feature_cols]].sum()
defect_shap = mean_abs[[feat_idx[f] for f in defect_cols]].sum()
bvse_shap   = mean_abs[[feat_idx[f] for f in bvse_cols]].sum()
total_shap  = comp_shap + defect_shap + bvse_shap

bvse_energy_pct  = mean_abs[feat_idx['bvse_energy']]  / total_shap * 100
topology_pct     = mean_abs[feat_idx['topology_score']]/ total_shap * 100

print(f"\n  Feature-group SHAP contributions:")
print(f"    Composition  : {comp_shap/total_shap*100:5.1f}%  ({comp_shap:.4f})")
print(f"    Defect proxy : {defect_shap/total_shap*100:5.1f}%  ({defect_shap:.4f})")
print(f"    BVSE total   : {bvse_shap/total_shap*100:5.1f}%  ({bvse_shap:.4f})")
print(f"      bvse_energy alone : {bvse_energy_pct:.1f}%")
print(f"      topology_score    : {topology_pct:.1f}%")

# Top-15 individual features
sorted_idx = np.argsort(mean_abs)[::-1]
print(f"\n  Top-15 features by mean |SHAP|:")
for rank, i in enumerate(sorted_idx[:15], 1):
    group = ('BVSE' if feature_cols[i] in bvse_cols
             else 'defect' if feature_cols[i] in defect_cols else 'comp')
    print(f"    {rank:2d}. {feature_cols[i]:<28} {mean_abs[i]:.4f}  [{group}]")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: feature-group pie / bar
groups  = ['Composition', 'Defect proxy', 'BVSE']
g_vals  = [comp_shap, defect_shap, bvse_shap]
colors  = ['#4C72B0', '#DD8452', '#55A868']
bars    = axes[0].barh(groups, [v / total_shap * 100 for v in g_vals], color=colors)
axes[0].set_xlabel('% of total mean |SHAP|')
axes[0].set_title('Feature-group SHAP contributions')
for bar, val in zip(bars, g_vals):
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{val/total_shap*100:.1f}%', va='center')
axes[0].set_xlim(0, max(v / total_shap * 100 for v in g_vals) * 1.15)

# Right: top-15 individual features
top15_names = [feature_cols[i] for i in sorted_idx[:15]]
top15_vals  = [mean_abs[i] for i in sorted_idx[:15]]
top15_colors = ['#55A868' if n in bvse_cols else '#DD8452' if n in defect_cols
                else '#4C72B0' for n in top15_names]
axes[1].barh(range(15), top15_vals[::-1], color=top15_colors[::-1])
axes[1].set_yticks(range(15))
axes[1].set_yticklabels(top15_names[::-1], fontsize=8)
axes[1].set_xlabel('Mean |SHAP value|')
axes[1].set_title('Top-15 features (green=BVSE, orange=defect, blue=comp)')

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'analysis_shap_groups.png'), dpi=150)
plt.close()
print("  Saved: analysis_shap_groups.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 2: OOF residual vs bvse_energy
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── Analysis 2: OOF residual vs bvse_energy ───")

bvse_e = train['bvse_energy'].values
available_mask = train['bvse_available'].values == 1.0

print(f"  Samples with real/proxy BVSE: {available_mask.sum()}")
corr_all  = np.corrcoef(bvse_e, oof_residual)[0, 1]
corr_avail = np.corrcoef(bvse_e[available_mask], oof_residual[available_mask])[0, 1]
print(f"  Pearson r(Ea, residual) — all:       {corr_all:+.3f}")
print(f"  Pearson r(Ea, residual) — BVSE avail:{corr_avail:+.3f}")

anion_colors = {'oxide': '#4C72B0', 'sulfide': '#DD8452',
                'halide': '#55A868', 'mixed': '#8172B2', 'other': '#937860'}
labels = train['anion_label'].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for anion in sorted(set(labels)):
    mask = labels == anion
    axes[0].scatter(bvse_e[mask], oof_residual[mask], alpha=0.5, s=20,
                    color=anion_colors.get(anion, 'grey'), label=anion)
axes[0].axhline(0, color='k', linewidth=0.8, linestyle='--')
axes[0].set_xlabel('bvse_energy / Ea (eV)')
axes[0].set_ylabel('OOF residual (actual − predicted)')
axes[0].set_title(f'Residual vs bvse_energy  (r={corr_all:+.2f})')
axes[0].legend(fontsize=8)

# Right: residual histogram split by BVSE availability
axes[1].hist(oof_residual[available_mask],  bins=25, alpha=0.6,
             label=f'BVSE available (n={available_mask.sum()})', color='#55A868')
axes[1].hist(oof_residual[~available_mask], bins=25, alpha=0.6,
             label=f'BVSE guessed (n={(~available_mask).sum()})', color='#DD8452')
axes[1].axvline(0, color='k', linestyle='--', linewidth=0.8)
axes[1].set_xlabel('OOF residual (actual − predicted)')
axes[1].set_ylabel('Count')
axes[1].set_title('Residual distribution by BVSE availability')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'analysis_residual_vs_bvse.png'), dpi=150)
plt.close()
print("  Saved: analysis_residual_vs_bvse.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 3: Abs-error by anion type
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── Analysis 3: Error clustering by anion type ───")

abs_err = np.abs(oof_residual)
results = []
for anion in sorted(set(labels)):
    mask = labels == anion
    errs = abs_err[mask]
    results.append({'anion': anion, 'n': mask.sum(), 'mean_mae': errs.mean(),
                    'median': np.median(errs), 'p75': np.percentile(errs, 75)})
    print(f"  {anion:8s}: n={mask.sum():3d}  mean_MAE={errs.mean():.3f}  "
          f"median={np.median(errs):.3f}  p75={np.percentile(errs, 75):.3f}")

anion_order = [r['anion'] for r in sorted(results, key=lambda x: -x['mean_mae'])]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
box_data = [abs_err[labels == a] for a in anion_order]
bp = axes[0].boxplot(box_data, labels=anion_order, patch_artist=True, notch=False)
for patch, anion in zip(bp['boxes'], anion_order):
    patch.set_facecolor(anion_colors.get(anion, 'grey'))
    patch.set_alpha(0.7)
for i, (anion, data) in enumerate(zip(anion_order, box_data)):
    jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(data))
    axes[0].scatter(np.full(len(data), i + 1) + jitter, data, alpha=0.3, s=8,
                    color=anion_colors.get(anion, 'grey'))
axes[0].set_ylabel('|OOF residual|')
axes[0].set_title('Abs error by anion type (sorted by mean)')
axes[0].axhline(oof_mae, color='k', linestyle='--', linewidth=0.8, label=f'overall MAE={oof_mae:.2f}')
axes[0].legend(fontsize=8)

# Bar: mean MAE with counts
means = [r['mean_mae'] for r in sorted(results, key=lambda x: -x['mean_mae'])]
ns    = [r['n']        for r in sorted(results, key=lambda x: -x['mean_mae'])]
bar_colors = [anion_colors.get(a, 'grey') for a in anion_order]
bars = axes[1].bar(anion_order, means, color=bar_colors, alpha=0.8)
axes[1].axhline(oof_mae, color='k', linestyle='--', linewidth=0.8, label=f'overall={oof_mae:.2f}')
for bar, n in zip(bars, ns):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'n={n}', ha='center', va='bottom', fontsize=8)
axes[1].set_ylabel('Mean |OOF residual|')
axes[1].set_title('Mean abs error by anion type')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'analysis_error_by_anion.png'), dpi=150)
plt.close()
print("  Saved: analysis_error_by_anion.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 4: Charge-imbalance feature ablation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── Analysis 4: Charge-imbalance feature ablation ───")

print("  Computing charge_imbalance for all training samples ...")
ci_vals = train['composition'].apply(
    lambda f: compute_charge_imbalance(Composition(f)) if f else 0.0
).values
X_with_ci = np.column_stack([X_train, ci_vals])
feature_cols_ci = feature_cols + ['charge_imbalance']

mae_base = _cv_mae(best_params, X_train, y_train, w_train)
mae_ci   = _cv_mae(best_params, X_with_ci, y_train, w_train)
delta    = mae_base - mae_ci

print(f"  CV MAE  without charge_imbalance : {mae_base:.4f}")
print(f"  CV MAE  with    charge_imbalance : {mae_ci:.4f}")
print(f"  Delta (positive = improvement)   : {delta:+.4f}")
if delta > 0.005:
    print("  >> Feature is helpful — worth including.")
elif delta < -0.005:
    print("  >> Feature hurts — discard.")
else:
    print("  >> Feature is neutral — no meaningful impact.")

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 5: BVSE ablation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── Analysis 5: BVSE ablation ───")

comp_defect_cols = comp_feature_cols + defect_cols
X_no_bvse = train[comp_defect_cols].values.astype(float)

mae_full    = _cv_mae(best_params, X_train, y_train, w_train)
mae_no_bvse = _cv_mae(best_params, X_no_bvse, y_train, w_train)
bvse_gain   = mae_no_bvse - mae_full

print(f"  CV MAE  full model  (comp+defect+BVSE)  : {mae_full:.4f}")
print(f"  CV MAE  comp+defect only (BVSE removed) : {mae_no_bvse:.4f}")
print(f"  BVSE information gain (MAE reduction)   : {bvse_gain:+.4f}")
bvse_pct = bvse_gain / mae_no_bvse * 100
print(f"  Relative improvement from BVSE          : {bvse_pct:.1f}%")

# ── Ablation summary figure ───────────────────────────────────────────────────
conditions = ['comp+defect\n(no BVSE)', 'full model\n(+BVSE)', f'full +\ncharge_imbalance']
mae_vals   = [mae_no_bvse, mae_full, mae_ci]
bar_cols   = ['#DD8452', '#4C72B0', '#55A868']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(conditions, mae_vals, color=bar_cols, alpha=0.85, width=0.5)
for bar, val in zip(bars, mae_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('5-fold CV MAE (log₁₀ S/cm)')
ax.set_title('Feature ablation — CV MAE comparison')
ax.set_ylim(0, max(mae_vals) * 1.15)

# Annotation: BVSE gain arrow
ax.annotate('',
    xy=(1, mae_full + 0.02), xytext=(0, mae_no_bvse - 0.02),
    arrowprops=dict(arrowstyle='->', color='k', lw=1.5))
ax.text(0.5, (mae_full + mae_no_bvse) / 2, f'−{bvse_gain:.3f}',
        ha='center', va='center', fontsize=9, color='k',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='grey', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'analysis_ablation.png'), dpi=150)
plt.close()
print("  Saved: analysis_ablation.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"""
{'='*62}
  XGBoost Analysis Summary
{'='*62}
  OOF MAE (full model)               : {oof_mae:.4f}
  SHAP: BVSE group contribution       : {bvse_shap/total_shap*100:.1f}%
  SHAP: bvse_energy alone             : {bvse_energy_pct:.1f}%
  SHAP: topology_score alone          : {topology_pct:.1f}%
  Residual–Ea correlation             : {corr_all:+.3f}
  Charge imbalance feature delta      : {delta:+.4f}
  BVSE info gain (MAE reduction)      : {bvse_gain:+.4f} ({bvse_pct:.1f}%)
  Figures: analysis_*.png
{'='*62}""")
