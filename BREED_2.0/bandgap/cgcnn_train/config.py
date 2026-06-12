"""
Shared configuration for the improved CGCNN PBE band-gap training pipeline.

Everything here is a *default*; almost all values are overridable by CLI flags on
the individual scripts. Paths are resolved relative to this file so the whole
`cgcnn_train/` folder can be copied anywhere and still work.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths (all relative to this package so the bundle is portable)
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent

ATOM_INIT_FILE = PKG_DIR / "atom_init.json"          # 92-dim atom features (required)
WARM_START_WEIGHTS = PKG_DIR / "band-gap.pth.tar"    # official CGCNN, optional warm start

DATA_DIR = PKG_DIR / "data"
CIF_DIR = DATA_DIR / "cifs"                           # <material_id>.cif
GRAPH_CACHE_DIR = DATA_DIR / "graphs"                 # <material_id>.pt (optional cache)
LABELS_CSV = DATA_DIR / "labels.csv"                  # id, formula, band_gap, is_metal, ...

SPLITS_DIR = PKG_DIR / "splits"                       # train/val/test id lists
TRAIN_IDS = SPLITS_DIR / "train_ids.txt"
VAL_IDS = SPLITS_DIR / "val_ids.txt"
TEST_IDS = SPLITS_DIR / "test_ids.txt"
SPLIT_META = SPLITS_DIR / "split_meta.json"

MODELS_DIR = PKG_DIR / "models"                       # trained checkpoints land here
CLASSIFIER_CKPT = MODELS_DIR / "classifier.pth.tar"
THRESHOLD_JSON = MODELS_DIR / "threshold.json"


def regressor_ckpt(seed: int) -> Path:
    return MODELS_DIR / f"regressor_seed{seed}.pth.tar"


def regressor_history(seed: int) -> Path:
    return MODELS_DIR / f"regressor_seed{seed}_history.json"


# ---------------------------------------------------------------------------
# Device (lights up the GPU automatically on the powerful machine)
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon fallback if present, else CPU.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
USE_AMP = DEVICE.type == "cuda"   # mixed precision only helps on CUDA
# autocast() needs a device_type it recognizes; when AMP is off we pass "cpu"
# (a no-op since enabled=False) so an MPS/other device can't trip its validation.
AMP_DEVICE_TYPE = "cuda" if USE_AMP else "cpu"


# ---------------------------------------------------------------------------
# Band-gap bins
# ---------------------------------------------------------------------------
# Reporting / loss-weighting bins (the four ranges you asked to track).
# (lo, hi) in eV; the last bin is open-ended (hi = inf).
METRIC_BINS = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, float("inf"))]
METRIC_BIN_LABELS = ["0-1 eV", "1-3 eV", "3-6 eV", ">6 eV"]

# Finer bins used only for stratified *fetching* so wide-gap insulators (rare in
# MP) are pulled in adequate numbers.
FETCH_BINS = [
    (0.0, 0.001),   # metals (exactly 0)
    (0.001, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
    (5.0, 6.0),
    (6.0, 8.0),
    (8.0, 20.0),
]

# A nonmetal is anything MP does not flag as a metal. Used to route the
# regressor (nonmetals only) vs the classifier (all materials).
METAL_GAP_EPS = 1e-6


# ---------------------------------------------------------------------------
# CGCNN architecture (matches the vendored band-gap.pth.tar so warm-start works)
# ---------------------------------------------------------------------------
ATOM_FEA_LEN = 64
N_CONV = 3
H_FEA_LEN = 128
N_H = 1
MAX_NUM_NBR = 12
RADIUS = 8.0
GAUSS_DMIN = 0.0
GAUSS_STEP = 0.2


# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
EPOCHS = 60
BATCH_SIZE = 256
LR = 0.01
OPTIM = "Adam"
WEIGHT_DECAY = 0.0
LR_MILESTONES = [40, 55]
LR_GAMMA = 0.1
MOMENTUM = 0.9            # only used when OPTIM == "SGD"
WORKERS = 0              # graph build is in-process; raise on Linux GPU boxes
PRINT_FREQ = 20

# Wide-gap loss weighting (improvement #6). Per-sample weight = WEIGHT_BY_BIN for
# the bin the *target* falls into. Rare wide-gap materials are up-weighted so they
# contribute more strongly to the loss. Tune via --weight-scheme on the regressor.
# Index aligns with METRIC_BINS.
WEIGHT_BY_BIN = [1.0, 1.0, 2.0, 4.0]

# Split fractions (composition-grouped).
VAL_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 42

# Ensemble
N_MODELS = 5


def ensure_dirs() -> None:
    for d in (DATA_DIR, CIF_DIR, SPLITS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
