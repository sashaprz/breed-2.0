#!/usr/bin/env python3
"""
Train ONE band-gap regressor (one ensemble member).

Improvements implemented here:
  #4  trains only on NON-METALS (gap > 0), so the regressor can focus on
      predicting finite gaps accurately instead of also learning the metal spike.
  #6  band-gap RANGE WEIGHTING: each sample's squared error is multiplied by a
      per-bin weight (rare wide-gap materials up-weighted) so they contribute more
      strongly to the loss. Weights come from config.WEIGHT_BY_BIN, or set
      --weight-scheme inverse to derive them from this training set's bin counts.
  #7  per-range MAE tracked every epoch for [0,1), [1,3), [3,6), [>6) eV and saved
      to a history JSON, exposing failure modes hidden by a single aggregate MAE.

The model is the stock CrystalGraphConvNet (regression head). Targets are
normalized with the same Normalizer as the original CGCNN so checkpoints stay
compatible. `--seed` makes this one ensemble member; train_ensemble.py calls it
across seeds (improvement #8).

Usage
-----
    python train_regressor.py --seed 0
    python train_regressor.py --seed 1 --weight-scheme inverse --epochs 60
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import config
import common
from loaders import make_loader


# ---------------------------------------------------------------------------
# Per-sample loss weights by gap bin (improvement #6)
# ---------------------------------------------------------------------------
def resolve_bin_weights(train_pairs, scheme):
    """Return a list of weights aligned with config.METRIC_BINS."""
    if scheme == "config":
        return list(config.WEIGHT_BY_BIN)
    if scheme == "inverse":
        counts = np.zeros(len(config.METRIC_BINS))
        for _, gap in train_pairs:
            counts[common.gap_bin_index(gap)] += 1
        counts = np.maximum(counts, 1.0)
        inv = counts.sum() / (len(counts) * counts)   # inverse-frequency, mean ~1
        return inv.tolist()
    if scheme == "none":
        return [1.0] * len(config.METRIC_BINS)
    raise ValueError(scheme)


def sample_weights(targets, bin_weights):
    """Map a target tensor -> per-sample weight tensor on device."""
    w = torch.empty_like(targets)
    flat = targets.view(-1)
    for i, v in enumerate(flat.tolist()):
        w.view(-1)[i] = bin_weights[common.gap_bin_index(v)]
    return w.to(config.DEVICE)


def weighted_mse(pred_normed, target_normed, weights):
    """Sum(w * (pred - target)^2) / Sum(w)  -- a weighted mean squared error."""
    se = (pred_normed - target_normed) ** 2
    return (weights * se).sum() / weights.sum().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------
def train_epoch(model, loader, normalizer, optimizer, scaler, bin_weights):
    model.train()
    losses = common.AverageMeter()
    for inp, target, _ in loader:
        inp = common.move_input(inp)
        target_normed = normalizer.norm(target).to(config.DEVICE)
        w = sample_weights(target, bin_weights)
        with torch.autocast(device_type=config.AMP_DEVICE_TYPE, enabled=config.USE_AMP):
            out = model(*inp)
            loss = weighted_mse(out, target_normed, w)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.update(float(loss.detach()), target.size(0))
    return losses.avg


@torch.no_grad()
def evaluate(model, loader, normalizer):
    model.eval()
    preds, targets = [], []
    for inp, target, _ in loader:
        inp = common.move_input(inp)
        with torch.autocast(device_type=config.AMP_DEVICE_TYPE, enabled=config.USE_AMP):
            out = model(*inp)
        pred = normalizer.denorm(out.detach().float().cpu())
        preds.extend(pred.view(-1).tolist())
        targets.extend(target.view(-1).tolist())
    mae = float(np.mean(np.abs(np.array(preds) - np.array(targets)))) if preds else float("nan")
    return mae, targets, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--weight-scheme", choices=["config", "inverse", "none"],
                    default="config", help="gap-range loss weighting (#6)")
    ap.add_argument("--warm-start", action="store_true",
                    help="initialize from the vendored band-gap.pth.tar where shapes match")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    labels = common.load_labels()
    train_pairs = common.regressor_pairs(common.load_split(config.TRAIN_IDS), labels)
    val_pairs = common.regressor_pairs(common.load_split(config.VAL_IDS), labels)
    print(f"regressor seed={args.seed}: train={len(train_pairs)} (nonmetals) "
          f"val={len(val_pairs)} on {config.DEVICE}")

    bin_weights = resolve_bin_weights(train_pairs, args.weight_scheme)
    print(f"  gap-bin weights ({args.weight_scheme}): "
          + ", ".join(f"{l}={w:.2f}" for l, w in zip(config.METRIC_BIN_LABELS, bin_weights)))

    use_cache = not args.no_cache
    train_ds, train_loader = make_loader(train_pairs, args.batch_size, shuffle=True,
                                         use_cache=use_cache)
    _, val_loader = make_loader(val_pairs, args.batch_size, shuffle=False,
                                use_cache=use_cache)

    orig_atom_fea_len, nbr_fea_len = train_ds.sample_graph_dims()
    model = common.build_model(orig_atom_fea_len, nbr_fea_len, classification=False)

    if args.warm_start and config.WARM_START_WEIGHTS.exists():
        _maybe_warm_start(model)

    # Normalizer fit on the nonmetal training targets.
    norm_targets = torch.tensor([g for _, g in train_pairs], dtype=torch.float)
    normalizer = common.Normalizer(norm_targets)

    if config.OPTIM == "Adam":
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=config.MOMENTUM,
                              weight_decay=config.WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=config.LR_MILESTONES, gamma=config.LR_GAMMA)
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

    history, best_mae, best_state = [], float("inf"), None
    for epoch in range(args.epochs):
        tr_loss = train_epoch(model, train_loader, normalizer, optimizer, scaler, bin_weights)
        va_mae, va_t, va_p = evaluate(model, val_loader, normalizer)
        by_range = common.mae_by_range(va_t, va_p)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_mae": va_mae, "val_mae_by_range": by_range})
        rng = "  ".join(f"{l}:{(by_range[l]['mae'] if by_range[l]['mae'] is not None else float('nan')):.3f}"
                        for l in config.METRIC_BIN_LABELS)
        print(f"[reg s{args.seed}] epoch {epoch:3d}  train_loss={tr_loss:.4f}  "
              f"val_MAE={va_mae:.4f}  [{rng}]")
        if va_mae < best_mae:
            best_mae = va_mae
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save({
        "state_dict": best_state,
        "normalizer": normalizer.state_dict(),
        "graph_dims": [orig_atom_fea_len, nbr_fea_len],
        "best_val_mae": best_mae,
        "weight_scheme": args.weight_scheme,
        "seed": args.seed,
        "args": vars(args),
    }, config.regressor_ckpt(args.seed))
    with open(config.regressor_history(args.seed), "w") as f:
        json.dump(history, f, indent=2)

    print(f"=> saved regressor seed {args.seed} (best val MAE {best_mae:.4f}) "
          f"-> {config.regressor_ckpt(args.seed)}")


def _maybe_warm_start(model):
    """Load overlapping tensors from the vendored CGCNN weights (best-effort)."""
    ckpt = torch.load(config.WARM_START_WEIGHTS, map_location="cpu")
    src = ckpt.get("state_dict", ckpt)
    own = model.state_dict()
    loaded = 0
    for k, v in src.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
    model.load_state_dict(own)
    print(f"  warm-started {loaded}/{len(own)} tensors from band-gap.pth.tar")


if __name__ == "__main__":
    main()
