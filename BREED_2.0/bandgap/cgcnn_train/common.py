"""
Shared training utilities: label/split loading, model construction, the target
Normalizer, device helpers, and per-gap-range MAE.

`Normalizer` and `AverageMeter` are lifted from cgcnn_pretrained/main.py so the
checkpoints stay compatible with the existing CGCNN tooling.
"""
from __future__ import annotations

import csv

import numpy as np
import torch

from cgcnn.model import CrystalGraphConvNet
import config


# ---------------------------------------------------------------------------
# Labels + splits
# ---------------------------------------------------------------------------
def load_labels() -> dict[str, dict]:
    """material_id -> {band_gap: float, is_metal: int, reduced_formula, ...}."""
    out: dict[str, dict] = {}
    with open(config.LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            out[row["material_id"]] = {
                "band_gap": float(row["band_gap"]),
                "is_metal": int(row["is_metal"]),
                "reduced_formula": row.get("reduced_formula", ""),
            }
    return out


def load_split(path) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def classifier_pairs(ids, labels):
    """(id, is_metal) for ALL ids -- the classifier sees metals and nonmetals."""
    return [(i, labels[i]["is_metal"]) for i in ids if i in labels]


def regressor_pairs(ids, labels):
    """(id, band_gap) for NONMETALS only (improvement #4)."""
    return [(i, labels[i]["band_gap"]) for i in ids
            if i in labels and not labels[i]["is_metal"]]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(orig_atom_fea_len, nbr_fea_len, classification=False):
    model = CrystalGraphConvNet(
        orig_atom_fea_len, nbr_fea_len,
        atom_fea_len=config.ATOM_FEA_LEN,
        n_conv=config.N_CONV,
        h_fea_len=config.H_FEA_LEN,
        n_h=config.N_H,
        classification=classification,
    )
    return model.to(config.DEVICE)


def move_input(inp, device=None):
    """Move a collate_pool input tuple (atom_fea, nbr_fea, nbr_fea_idx,
    crystal_atom_idx) onto the device."""
    device = device or config.DEVICE
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inp
    return (
        atom_fea.to(device, non_blocking=True),
        nbr_fea.to(device, non_blocking=True),
        nbr_fea_idx.to(device, non_blocking=True),
        [idx.to(device, non_blocking=True) for idx in crystal_atom_idx],
    )


# ---------------------------------------------------------------------------
# Normalizer + AverageMeter (from main.py, unchanged behaviour)
# ---------------------------------------------------------------------------
class Normalizer:
    """Normalize a Tensor and restore it later."""
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------------------------
# Per-gap-range MAE (improvement #7)
# ---------------------------------------------------------------------------
def mae_by_range(targets, preds):
    """Dict label -> {n, mae} over config.METRIC_BINS."""
    t = np.asarray(targets, dtype=float)
    p = np.asarray(preds, dtype=float)
    err = np.abs(p - t)
    out = {}
    for (lo, hi), lab in zip(config.METRIC_BINS, config.METRIC_BIN_LABELS):
        m = (t >= lo) & (t < hi)
        out[lab] = {"n": int(m.sum()),
                    "mae": float(err[m].mean()) if m.any() else None}
    return out


def gap_bin_index(value):
    """Index into config.METRIC_BINS for a gap value."""
    for k, (lo, hi) in enumerate(config.METRIC_BINS):
        if lo <= value < hi:
            return k
    return len(config.METRIC_BINS) - 1
