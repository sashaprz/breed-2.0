#!/usr/bin/env python3
"""
first_pass.py
==============
Stage-1 band-gap triage for GA-generated SSE candidates.

Thin, GA-facing wrapper around the CGCNN classifier + 5-regressor ensemble in
`cgcnn_train/` (see `cgcnn_train/predict.py` for the training-side inference
script this is derived from). Two things this module adds over predict.py:

1. `BandGapPredictor` loads the classifier + regressor ensemble ONCE and keeps
   them in memory, so a GA driver can call `.predict_cif_dir(...)` once per
   generation without re-loading checkpoints every time.
2. `screen_population()` applies the Stage-1 -> Stage-2 triage gate: a
   candidate proceeds to DFT relaxation + VASP PBE (Stage 2, `second_pass.py`)
   only if its predicted band gap could plausibly clear the electronic-
   insulation bar once its own uncertainty is taken into account.

Screening rule
--------------
    cutoff(candidate) = INSULATOR_THRESHOLD_EV - uncertainty_eV(candidate)
    passes_first_pass  = predicted_band_gap_eV >= cutoff

Equivalently: pass if `predicted + uncertainty >= INSULATOR_THRESHOLD_EV`,
i.e. the top of the ensemble's spread reaches the insulation bar.
INSULATOR_THRESHOLD_EV defaults to 3.0 eV (a defensible cut for "this is an
electronic insulator"). Subtracting the per-candidate uncertainty makes the
gate more permissive for predictions the ensemble itself is unsure about,
so borderline candidates aren't dropped on Stage-1 noise alone -- Stage 2
(real VASP PBE + scissor correction) is the actual decision point.

Metals (`classified_metal=True`) are pinned to `predicted_band_gap_eV=0.0`,
`uncertainty_eV=0.0`, so their cutoff is exactly INSULATOR_THRESHOLD_EV and
they are correctly rejected.

Usage (CLI)
-----------
    # Single structure
    python first_pass.py --cif candidate.cif

    # Directory of GA candidates, with Stage-1 screening
    python first_pass.py --cif-dir ./generation_042 --out gen042_screen.csv

    # Predict only, no pass/fail column
    python first_pass.py --cif-dir ./generation_042 --no-screen --out gen042_preds.csv

    # Custom insulation threshold
    python first_pass.py --cif-dir ./generation_042 --threshold 2.5

Usage (Python API)
------------------
    from first_pass import BandGapPredictor, screen_population

    predictor = BandGapPredictor()                    # loads weights once
    records = predictor.predict_cif_dir("./gen_042")  # call again per generation
    screen_population(records)                        # adds passes_first_pass
    survivors = [r for r in records if r["passes_first_pass"]]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# cgcnn_train/ is a self-contained, portable package whose modules
# (config, common, loaders, cgcnn.*) use absolute imports assuming the
# package directory itself is on sys.path -- add it so this module works
# regardless of the caller's cwd.
_CGCNN_TRAIN_DIR = Path(__file__).resolve().parent / "cgcnn_train"
if str(_CGCNN_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_CGCNN_TRAIN_DIR))

import config
import common
from loaders import GraphDataset
from cgcnn.data import collate_pool


# --------------------------------------------------------------------------- #
#  Stage-1 -> Stage-2 screening                                                #
# --------------------------------------------------------------------------- #

INSULATOR_THRESHOLD_EV = 3.0


def screen_population(
    records: list[dict],
    threshold_ev: float = INSULATOR_THRESHOLD_EV,
    verbose: bool = True,
) -> list[dict]:
    """Annotate each record with the Stage-1 triage decision (in-place).

    Adds ``first_pass_cutoff_eV`` (= ``threshold_ev - uncertainty_eV``) and
    ``passes_first_pass`` (= ``predicted_band_gap_eV >= first_pass_cutoff_eV``).
    Records with ``passes_first_pass=True`` are the survivors that should go
    on to DFT relaxation + VASP PBE (Stage 2).
    """
    for rec in records:
        cutoff = threshold_ev - rec["uncertainty_eV"]
        rec["first_pass_cutoff_eV"] = round(cutoff, 4)
        rec["passes_first_pass"] = bool(rec["predicted_band_gap_eV"] >= cutoff)

    if verbose:
        n_pass = sum(r["passes_first_pass"] for r in records)
        print(f"\n  Stage-1 screen: {n_pass}/{len(records)} candidates pass "
              f"(insulator threshold = {threshold_ev:.2f} eV minus "
              f"per-candidate uncertainty)")

    return records


# --------------------------------------------------------------------------- #
#  Predictor: load once, predict many times                                   #
# --------------------------------------------------------------------------- #

class BandGapPredictor:
    """Classifier + 5-regressor CGCNN ensemble, loaded once and reused.

    Same prediction logic as ``cgcnn_train/predict.py``:
    classifier -> P(metal); if P(metal) >= tuned threshold, gap = 0 eV;
    else gap = mean of regressor ensemble, uncertainty = std of ensemble.
    """

    def __init__(self, batch_size: int | None = None):
        self.batch_size = batch_size or config.BATCH_SIZE
        self.threshold = self._load_threshold()
        self.classifier = self._load_classifier()
        self.regressors = self._load_regressors()
        print(f"=> BandGapPredictor: classifier + {len(self.regressors)} "
              f"regressors loaded on {config.DEVICE}, "
              f"metal threshold P>={self.threshold:.3f}")

    @staticmethod
    def _load_threshold() -> float:
        if config.THRESHOLD_JSON.exists():
            return float(json.loads(config.THRESHOLD_JSON.read_text())["threshold"])
        print("[warn] no threshold.json (run tune_threshold.py); defaulting to 0.5")
        return 0.5

    @staticmethod
    def _load_classifier():
        ckpt = torch.load(config.CLASSIFIER_CKPT, map_location=config.DEVICE)
        dims = ckpt["graph_dims"]
        model = common.build_model(dims[0], dims[1], classification=True)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    @staticmethod
    def _load_regressors():
        members = []
        for ckpt_path in sorted(config.MODELS_DIR.glob("regressor_seed*.pth.tar")):
            ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
            dims = ckpt["graph_dims"]
            model = common.build_model(dims[0], dims[1], classification=False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            norm = common.Normalizer(torch.zeros(1))
            norm.load_state_dict(ckpt["normalizer"])
            members.append((model, norm))
        if not members:
            raise SystemExit("No regressor checkpoints found in "
                             "cgcnn_train/models/ -- run train_ensemble.py first.")
        return members

    @torch.no_grad()
    def _predict_ids(self, ids: list[str], cif_dir: Path) -> list[dict]:
        dataset = GraphDataset([(i, 0.0) for i in ids], cif_dir=cif_dir, use_cache=False)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                            collate_fn=collate_pool,
                            pin_memory=(config.DEVICE.type == "cuda"))

        results = []
        for inp, _, batch_ids in loader:
            inp = common.move_input(inp)
            with torch.autocast(device_type=config.AMP_DEVICE_TYPE, enabled=config.USE_AMP):
                clf_out = self.classifier(*inp)
                p_metal = torch.exp(clf_out.float())[:, 1].cpu().numpy()

                member_preds = []
                for model, norm in self.regressors:
                    out = model(*inp).detach().float()
                    member_preds.append(norm.denorm(out).view(-1).cpu().numpy())
            member_preds = np.stack(member_preds, axis=0)  # (n_members, B)

            for k, mid in enumerate(batch_ids):
                is_metal = bool(p_metal[k] >= self.threshold)
                if is_metal:
                    gap, unc = 0.0, 0.0
                else:
                    gap = float(member_preds[:, k].mean())
                    gap = max(0.0, gap)
                    unc = float(member_preds[:, k].std())
                results.append({
                    "material_id": mid,
                    "predicted_band_gap_eV": round(gap, 4),
                    "uncertainty_eV": round(unc, 4),
                    "p_metal": round(float(p_metal[k]), 4),
                    "classified_metal": is_metal,
                })
        return results

    def predict_cif(self, cif_path: str | Path) -> dict:
        """Predict for a single .cif file."""
        cif_path = Path(cif_path)
        return self._predict_ids([cif_path.stem], cif_path.parent)[0]

    def predict_cif_dir(self, cif_dir: str | Path) -> list[dict]:
        """Predict for every .cif file in a directory."""
        cif_dir = Path(cif_dir)
        ids = sorted(c.stem for c in cif_dir.glob("*.cif"))
        return self._predict_ids(ids, cif_dir)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cif", help="single .cif file")
    g.add_argument("--cif-dir", help="directory of .cif files (GA population)")
    ap.add_argument("--threshold", type=float, default=INSULATOR_THRESHOLD_EV,
                    help=f"insulator band-gap threshold in eV "
                         f"(default: {INSULATOR_THRESHOLD_EV})")
    ap.add_argument("--no-screen", action="store_true",
                    help="predict only; skip the pass/fail Stage-1 gate")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None, help="write results to this CSV")
    args = ap.parse_args()

    predictor = BandGapPredictor(batch_size=args.batch_size)

    if args.cif:
        records = [predictor.predict_cif(args.cif)]
    else:
        records = predictor.predict_cif_dir(args.cif_dir)

    if not args.no_screen:
        screen_population(records, threshold_ev=args.threshold)

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        print(f"=> wrote {args.out}")
    else:
        for r in records[:50]:
            line = (f"  {r['material_id']:<16} gap={r['predicted_band_gap_eV']:>7.3f} "
                    f"+/- {r['uncertainty_eV']:.3f} eV  "
                    f"{'METAL' if r['classified_metal'] else ''}")
            if "passes_first_pass" in r:
                line += "  PASS" if r["passes_first_pass"] else "  reject"
            print(line)
        if len(records) > 50:
            print(f"  ... ({len(records)} total; use --out to write all)")


if __name__ == "__main__":
    main()
