import pandas as pd
from mp_api.client import MPRester
from collections import defaultdict
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = r"C:\Users\Sasha\repos\genetic_algo\new_ionic_cond_predictor\BREED_2.0\raw_data\obelix_train.csv"
API_KEY = "YOUR_API_KEY_HERE"  # or use environment variable

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# try to detect formula column
possible_cols = ["formula", "composition", "Formula", "comp"]
formula_col = next((c for c in possible_cols if c in df.columns), None)

if formula_col is None:
    raise ValueError(f"No formula column found. Columns are: {df.columns}")

formulas = df[formula_col].dropna().unique()
print(f"Total unique formulas: {len(formulas)}")

# -----------------------------
# QUERY MATERIALS PROJECT
# -----------------------------
with MPRester(API_KEY) as mpr:

    has_structure = {}
    missing = []

    for f in tqdm(formulas):
        try:
            # summary endpoint includes structure if available
            docs = mpr.summary.search(
                formula=f,
                fields=["material_id", "formula_pretty", "structure"]
            )

            if len(docs) == 0:
                has_structure[f] = False
                continue

            # check if any returned entry has structure
            structured = any(getattr(doc, "structure", None) is not None for doc in docs)

            has_structure[f] = structured

        except Exception as e:
            print(f"[ERROR] {f}: {e}")
            has_structure[f] = False
            missing.append(f)

# -----------------------------
# ANALYSIS
# -----------------------------
total = len(has_structure)
available = sum(has_structure.values())
missing_count = total - available

print("\n================ RESULTS ================")
print(f"Total formulas checked : {total}")
print(f"Has structure (CIF)    : {available}")
print(f"Missing structure      : {missing_count}")
print(f"Coverage              : {available/total:.2%}")

# optional breakdown
no_struct = [k for k, v in has_structure.items() if not v]

print("\nExample missing structures:")
print(no_struct[:20])