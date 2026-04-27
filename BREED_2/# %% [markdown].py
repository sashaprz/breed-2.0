# %% [markdown]
# # 02 — CIF Collection & Dataset Organization
# 
# **Goal**: Collect CIFs for as many entries as possible, then create four clean datasets:
# 
# | Dataset | Description |
# |---------|-------------|
# | `comp_train.csv` | All training entries (for composition-only model) |
# | `comp_test.csv` | All test entries (for composition-only model) |
# | `struct_train.csv` | Only entries with CIFs (for structure-based model) |
# | `struct_test.csv` | Only test entries with CIFs (for structure-based model) |
# 
# All CIF files go in `cifs/` folder, named `{id}.cif`.
# 
# **Prerequisites**: 
# - Run `01_data_loading.ipynb` first → need `clean_train.csv` and `clean_test_DO_NOT_TOUCH.csv`
# - Extract `train_cifs.tar.gz` and `test_cifs.tar.gz` into this directory (use 7-Zip)
# - `pip install pymatgen`

# %%
import pandas as pd
import numpy as np
import os
import shutil
import time
import requests
from pymatgen.core import Composition, Structure

# %% [markdown]
# ## Step 1: Organize existing OBELiX CIFs

# %%
os.makedirs('cifs', exist_ok=True)

# Copy from OBELiX archives if cifs/ is empty
cif_count = len([f for f in os.listdir('cifs') if f.endswith('.cif')])
if cif_count == 0:
    obelix_cif_dirs = [
        'raw_data/train_cifs/train_randomized_cifs',
        'raw_data/test_cifs/test_randomized_cifs'
    ]
    for d in obelix_cif_dirs:
        if not os.path.exists(d):
            print(f"WARNING: {d} not found")
            continue
        for f in os.listdir(d):
            if f.endswith('.cif'):
                shutil.copy2(os.path.join(d, f), os.path.join('cifs', f))
    cif_count = len([f for f in os.listdir('cifs') if f.endswith('.cif')])

print(f"CIFs in cifs/: {cif_count}")

# %% [markdown]
# ## Step 2: Load datasets and find all entries missing CIFs

# %%
train = pd.read_csv('clean_train.csv')
test = pd.read_csv('clean_test_DO_NOT_TOUCH.csv')

cif_files = set(f.replace('.cif', '') for f in os.listdir('cifs') if f.endswith('.cif'))

train['has_cif'] = train['id'].isin(cif_files)
test['has_cif'] = test['id'].isin(cif_files)

print(f"=== BEFORE MP QUERIES ===")
print(f"Train: {len(train)} total, {train['has_cif'].sum()} with CIF, {(~train['has_cif']).sum()} missing")
print(f"  OBELiX:    {(train['source']=='obelix').sum()} total, {((train['source']=='obelix') & train['has_cif']).sum()} with CIF")
print(f"  Liverpool: {(train['source']=='liverpool').sum()} total, {((train['source']=='liverpool') & train['has_cif']).sum()} with CIF")
print(f"Test:  {len(test)} total, {test['has_cif'].sum()} with CIF, {(~test['has_cif']).sum()} missing")

# Combine all missing entries for MP querying
all_missing = pd.concat([
    train[~train['has_cif']], 
    test[~test['has_cif']]
], ignore_index=True)
print(f"\nTotal entries missing CIFs: {len(all_missing)}")

# %% [markdown]
# ## Step 3: Query Materials Project for missing CIFs
# 
# Uses the MP REST API directly (no mp-api package needed).
# Searches by reduced formula, picks the most thermodynamically stable structure.

# %%
MP_API_KEY = "Ocv14brcANyWZe6exk5zpgxnhuNTIgAq"  # Replace if you regenerated it

def get_reduced_formula(comp_string):
    try:
        return Composition(comp_string).reduced_formula
    except Exception:
        return None

all_missing['reduced_formula'] = all_missing['composition'].apply(get_reduced_formula)
failed_parse = all_missing['reduced_formula'].isna().sum()
if failed_parse > 0:
    print(f"WARNING: {failed_parse} compositions couldn't be parsed")

unique_formulas = all_missing['reduced_formula'].dropna().unique()
print(f"Unique formulas to query: {len(unique_formulas)}")

#very little cifs match because 

# %%
def query_mp_for_structure(formula):
    """Query MP REST API directly for a formula. Return best structure or None."""
    try:
        resp = requests.get(
            "https://api.materialsproject.org/materials/summary/",
            params={
                "formula": formula,
                "_fields": "material_id,formula_pretty,energy_above_hull,structure",
                "_limit": 10
            },
            headers={"X-API-KEY": MP_API_KEY},
            timeout=30
        )
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data:
            return None
        best = min(data, key=lambda r: r.get("energy_above_hull") or 999)
        return best
    except Exception as e:
        print(f"  Error querying {formula}: {e}")
        return None

# Query MP for all missing formulas
mp_results = {}
for i, formula in enumerate(unique_formulas):
    if i % 50 == 0:
        print(f"  Querying {i}/{len(unique_formulas)}...")
    mp_results[formula] = query_mp_for_structure(formula)
    time.sleep(0.3)

matched = sum(1 for v in mp_results.values() if v is not None)
missed = sum(1 for v in mp_results.values() if v is None)
print(f"\nDone! Matched: {matched}, No match: {missed}")

# %% [markdown]
# ## Step 4: Save matched CIF files

# %%
saved = 0
for formula, result in mp_results.items():
    if result is None or result.get("structure") is None:
        continue
    
    struct = Structure.from_dict(result["structure"])
    
    mask = all_missing['reduced_formula'] == formula
    ids = all_missing.loc[mask, 'id'].tolist()
    
    for entry_id in ids:
        cif_path = os.path.join('cifs', f'{entry_id}.cif')
        struct.to(filename=cif_path)
        saved += 1

print(f"Saved {saved} new CIF files from Materials Project")
print(f"Total CIFs in cifs/: {len([f for f in os.listdir('cifs') if f.endswith('.cif')])}")

# %% [markdown]
# ## Step 5: Create the four datasets
# 
# Two pairs (composition-only and structure-based), each with train/test split.
# 
# - **Composition datasets** (`comp_train.csv`, `comp_test.csv`): ALL entries, no filtering
# - **Structure datasets** (`struct_train.csv`, `struct_test.csv`): Only entries with a CIF in `cifs/`
# 
# The test set is always from OBELiX's original test split, keeping results comparable to their paper.

# %%
# Refresh CIF file list after MP downloads
cif_files = set(f.replace('.cif', '') for f in os.listdir('cifs') if f.endswith('.cif'))

train['has_cif'] = train['id'].isin(cif_files)
test['has_cif'] = test['id'].isin(cif_files)

# ============================================
# COMPOSITION-ONLY DATASETS (keep everything)
# ============================================
comp_train = train.copy()
comp_test = test.copy()

# ============================================
# STRUCTURE-BASED DATASETS (only entries with CIFs)
# ============================================
struct_train = train[train['has_cif']].copy()
struct_test = test[test['has_cif']].copy()

# ============================================
# Summary
# ============================================
print("=" * 55)
print("FINAL DATASET SUMMARY")
print("=" * 55)
print(f"\n--- Composition-only model ---")
print(f"  comp_train.csv:  {len(comp_train)} entries")
print(f"    OBELiX:    {(comp_train['source']=='obelix').sum()}")
print(f"    Liverpool: {(comp_train['source']=='liverpool').sum()}")
print(f"  comp_test.csv:   {len(comp_test)} entries")

print(f"\n--- Structure-based model ---")
print(f"  struct_train.csv: {len(struct_train)} entries")
print(f"    OBELiX:    {(struct_train['source']=='obelix').sum()}")
print(f"    Liverpool: {(struct_train['source']=='liverpool').sum()}")
print(f"  struct_test.csv:  {len(struct_test)} entries")

print(f"\n--- Dropped (no CIF, excluded from structure model) ---")
print(f"  Train: {len(comp_train) - len(struct_train)} entries")
print(f"  Test:  {len(comp_test) - len(struct_test)} entries")
print(f"\n--- CIF files ---")
print(f"  Total in cifs/: {len(cif_files)}")

# %% [markdown]
# ## Step 6: Verify test set integrity

# %%
# Sanity checks
# 1. No test IDs leaked into train
test_ids = set(comp_test['id'])
train_ids = set(comp_train['id'])
overlap = test_ids & train_ids
assert len(overlap) == 0, f"DATA LEAK: {len(overlap)} IDs appear in both train and test!"
print("✓ No ID overlap between train and test")

# 2. struct datasets are proper subsets of comp datasets
assert set(struct_train['id']).issubset(set(comp_train['id'])), "struct_train has IDs not in comp_train!"
assert set(struct_test['id']).issubset(set(comp_test['id'])), "struct_test has IDs not in comp_test!"
print("✓ Structure datasets are subsets of composition datasets")

# 3. Every entry in struct datasets has a CIF file
for entry_id in struct_train['id']:
    assert os.path.exists(f'cifs/{entry_id}.cif'), f"Missing CIF: {entry_id}"
for entry_id in struct_test['id']:
    assert os.path.exists(f'cifs/{entry_id}.cif'), f"Missing CIF: {entry_id}"
print("✓ Every structure dataset entry has a CIF file on disk")

# 4. Test set is OBELiX only
assert (comp_test['source'] == 'obelix').all(), "Test set contains non-OBELiX entries!"
print("✓ Test set is OBELiX only (comparable to paper baseline)")

print("\nAll checks passed!")

# %%
# Save all four datasets
comp_train.to_csv('comp_train.csv', index=False)
comp_test.to_csv('comp_test.csv', index=False)
struct_train.to_csv('struct_train.csv', index=False)
struct_test.to_csv('struct_test.csv', index=False)

# Save list of what was dropped for reference
dropped = train[~train['has_cif']][['id', 'composition', 'source', 'log_conductivity']]
dropped.to_csv('dropped_no_cif.csv', index=False)

print("Saved:")
print(f"  comp_train.csv     ({len(comp_train)} entries)")
print(f"  comp_test.csv      ({len(comp_test)} entries)")
print(f"  struct_train.csv   ({len(struct_train)} entries)")
print(f"  struct_test.csv    ({len(struct_test)} entries)")
print(f"  dropped_no_cif.csv ({len(dropped)} entries)")
print("\n✓ Done! Ready for feature extraction.")

# %%
import shutil, os

os.makedirs('cifs', exist_ok=True)

cif_sources = ['raw_data/cifs', 'raw_data/train_cifs', 'raw_data/test_cifs']

copied = 0
for d in cif_sources:
    if not os.path.exists(d):
        continue
    for f in os.listdir(d):
        if f.endswith('.cif'):
            shutil.copy2(os.path.join(d, f), os.path.join('cifs', f))
            copied += 1

print(f"Copied {copied} CIF files to cifs/")
print(f"Unique CIFs: {len([f for f in os.listdir('cifs') if f.endswith('.cif')])}")

# %%
import shutil, os

nested = 'raw_data/train_cifs/train_randomized_cifs'
count = 0
for f in os.listdir(nested):
    if f.endswith('.cif'):
        shutil.copy2(os.path.join(nested, f), os.path.join('cifs', f))
        count += 1
print(f"Copied {count} new CIFs")

# Check for same nesting in test
test_nested = 'raw_data/test_cifs'
for root, dirs, files in os.walk(test_nested):
    cifs = [f for f in files if f.endswith('.cif')]
    if cifs:
        print(f"Found {len(cifs)} CIFs in {root}")
        for f in cifs:
            shutil.copy2(os.path.join(root, f), os.path.join('cifs', f))

cif_files = set(f.replace('.cif', '') for f in os.listdir('cifs') if f.endswith('.cif'))
train['has_cif'] = train['id'].isin(cif_files)
test['has_cif'] = test['id'].isin(cif_files)
print(f"\nCIFs on disk: {len(cif_files)}")
print(f"struct_train: {train['has_cif'].sum()}")
print(f"struct_test:  {test['has_cif'].sum()}")

# %%
train_check = pd.read_csv('clean_train.csv')
test_check = pd.read_csv('clean_test_DO_NOT_TOUCH.csv')
print(f"clean_train: {len(train_check)} rows")
print(f"clean_test: {len(test_check)} rows")
print(train_check.head())


