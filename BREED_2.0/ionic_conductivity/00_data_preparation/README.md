# 00 — Data Preparation

Combined OBELiX (~599 entries reported in paper, 478 after filtering) and the Liverpool/NPJ database (800+ raw entries, 348 after filtering). Extracted held-out test set *first* before any feature work to prevent leakage.

**Why Liverpool dropped from 800+ to 348:**
- Non-room-temperature measurements removed (largest cut — the database mixes many temperatures)
- Duplicates with OBELiX removed
- 32 glassy/amorphous entries dropped — pymatgen can't parse them and they follow different physics

**Output:**
- Composition set: 826 total → 709 train / 120 test (approximate; source labels in `comp_train.csv`)
- CIF subset: ~408 entries that had known CIF files (from OBELiX's included structures or matched via Materials Project) → ~339 train / 68 test
