# Improved CGCNN PBE band-gap training

A self-contained pipeline to **retrain CGCNN to predict Materials Project PBE band
gaps** with eight improvements over the stock `txie-93/cgcnn` model. Designed to be
copied to a GPU machine, where it fetches its own MP data and trains.

## What's improved (vs the vendored `band-gap.pth.tar`)

| # | Improvement | Where |
|---|-------------|-------|
| 1 | Retrain on **current** MP data (not the ~2018 snapshot) | `fetch_mp_training_data.py` |
| 2 | **Composition-grouped** train/val/test split (no polymorph leakage) | `make_splits.py` |
| 3 | Separate **metal/non-metal classifier** | `train_classifier.py` |
| 4 | Regressor trained **on non-metals only** | `train_regressor.py` |
| 5 | **Optimized** metal decision threshold (val-tuned, not fixed 0.5) | `tune_threshold.py` |
| 6 | **Band-gap range weighting** in the loss (wide-gap up-weighted) | `train_regressor.py` |
| 7 | **Per-range MAE** tracked (0–1 / 1–3 / 3–6 / >6 eV) | `train_regressor.py` |
| 8 | **Ensemble** of 3–5 seeds; prediction std = uncertainty | `train_ensemble.py`, `predict.py` |

## This is the whole bundle

Copy **this entire `cgcnn_train/` folder** to the GPU machine. It vendors everything
it needs (`cgcnn/model.py`, `cgcnn/data.py`, `atom_init.json`, and the old
`band-gap.pth.tar` for optional warm-start). Nothing else from the repo is required.

```
cgcnn_train/
├── cgcnn/{model.py,data.py,__init__.py}   # vendored graph builder + model
├── atom_init.json                          # 92-dim atom features (required)
├── band-gap.pth.tar                        # optional warm-start weights
├── config.py                               # all defaults (paths, device, bins, hparams)
├── common.py  loaders.py                   # shared utils + dataset/loader
├── fetch_mp_training_data.py               # step 1
├── make_splits.py                          # step 2
├── cache_graphs.py                         # step 3 (optional speedup)
├── train_classifier.py  tune_threshold.py  # steps 4, 6
├── train_regressor.py  train_ensemble.py   # step 5
├── predict.py                              # inference
├── requirements.txt
└── README_TRAINING.md
```

Artifacts created at runtime (all gitignored): `data/` (CIFs + `labels.csv` + graph
cache), `splits/`, `models/` (checkpoints, threshold, histories).

## Setup (on the GPU machine)

```bash
cd cgcnn_train

# 1. PyTorch with CUDA (pick your toolkit), then the rest:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Materials Project API key (gitignored file). Free key: materialsproject.org/api
echo "YOUR_MP_API_KEY" > .mp_api_key
```

> **Security:** the key is read from `--api-key` → `$MP_API_KEY` → `.mp_api_key` and
> is never written into source. If you shared a key in plaintext anywhere, rotate it.

GPU is auto-detected (`config.get_device()`): CUDA → MPS → CPU, with mixed precision
on CUDA. CPU-only works but is slow at full scale.

## Run order

```bash
# 1. Fetch ALL suitable non-deprecated MP materials (resumable; several GB).
python fetch_mp_training_data.py
#    optional: --max-ehull 0.1 (stability filter) | --max-n N (cap) | --smoke

# 2. Composition-grouped split (asserts no group spans two splits).
python make_splits.py

# 3. (Recommended) precompute graphs once so the 5 seeds don't rebuild them.
python cache_graphs.py

# 4. Train classifier + ensemble of regressors.
python train_ensemble.py --n-models 5        # add --warm-start to seed from band-gap.pth.tar

# 5. Tune the metal decision threshold on validation.
python tune_threshold.py

# 6. Predict (gap ± uncertainty).
python predict.py --test --out test_predictions.csv   # also prints per-range test MAE
python predict.py --cif some_candidate.cif
python predict.py --cif-dir ./my_candidates --out preds.csv
```

## Quick end-to-end smoke test (minutes)

Validate the whole chain before the full run:

```bash
python fetch_mp_training_data.py --smoke            # ~40 materials
python make_splits.py --val 0.2 --test 0.2
python train_ensemble.py --n-models 2 --epochs 3    # tiny
python tune_threshold.py
python predict.py --test
```

## Tuning knobs (all CLI flags)

- **Dataset size**: default = everything. `--max-n` (cap), `--max-ehull` (stability).
- **Ensemble size**: `train_ensemble.py --n-models {3..5}`.
- **Gap weighting (#6)**: `train_regressor.py --weight-scheme {config,inverse,none}`.
  `config` uses `WEIGHT_BY_BIN` in `config.py`; `inverse` derives weights from this
  dataset's bin counts.
- **Epochs / batch / lr**: flags on every train script; defaults in `config.py`.
- **Architecture** (`ATOM_FEA_LEN`, `N_CONV`, …) lives in `config.py` and matches
  `band-gap.pth.tar` so `--warm-start` works.

## Outputs

- `models/classifier.pth.tar`, `models/regressor_seed{0..N}.pth.tar`
- `models/threshold.json` — tuned metal threshold + metric
- `models/regressor_seed*_history.json` — per-epoch val MAE **by gap range** (#7)
- `predict.py` returns `{material_id, predicted_band_gap_eV, uncertainty_eV,
  p_metal, classified_metal}` — drop-in for downstream BREED screening.

## Notes

- **Time/disk**: fetching the full MP set is the long pole (downloads structures for
  ~150k materials — several GB, can take a while; it's resumable, so interrupt
  freely). `cache_graphs.py` front-loads the CPU graph build once. Training time
  scales with `--n-models` × `--epochs`.
- The original `BREED/env/bandgap/` code is untouched; this folder is additive.
