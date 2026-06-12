#!/usr/bin/env python3
"""
Train the metal / non-metal CGCNN classifier.

IMPROVEMENT #3: a dedicated binary classifier instead of forcing one regressor to
straddle metals (gap = 0) and insulators. At inference, anything classified a metal
is pinned to exactly 0 eV (killing the ~0.4 eV prediction floor the old single
regressor had), and only non-metals go to the band-gap regressor.

Target = MP `is_metal`. The model is the SAME CrystalGraphConvNet with
`classification=True` (2-logit LogSoftmax head + NLLLoss). Class imbalance is
handled with a class-weighted NLLLoss. The validation P(metal) values are saved so
tune_threshold.py can pick the decision threshold (improvement #5).

Usage
-----
    python train_classifier.py                      # uses config defaults
    python train_classifier.py --epochs 40 --batch-size 256 --seed 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import config
import common
from loaders import make_loader


def class_weights(pairs):
    """Inverse-frequency weights for [nonmetal(0), metal(1)] -> tensor on device."""
    y = np.array([t for _, t in pairs])
    n0 = max(1, int((y == 0).sum()))
    n1 = max(1, int((y == 1).sum()))
    total = n0 + n1
    w = torch.tensor([total / (2.0 * n0), total / (2.0 * n1)], dtype=torch.float)
    return w.to(config.DEVICE)


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    losses, correct, total = common.AverageMeter(), 0, 0
    all_p_metal, all_true = [], []

    for inp, target, _ in loader:
        inp = common.move_input(inp)
        y = target.view(-1).long().to(config.DEVICE)  # 0 nonmetal / 1 metal

        with torch.set_grad_enabled(train), \
             torch.autocast(device_type=config.AMP_DEVICE_TYPE, enabled=config.USE_AMP):
            out = model(*inp)                 # log-softmax logits, shape (B, 2)
            loss = criterion(out, y)

        if train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        prob = torch.exp(out.detach().float())            # (B, 2) probabilities
        p_metal = prob[:, 1]
        pred = (p_metal >= 0.5).long()
        correct += int((pred == y).sum())
        total += y.numel()
        losses.update(float(loss.detach()), y.numel())
        all_p_metal.extend(p_metal.cpu().tolist())
        all_true.extend(y.cpu().tolist())

    acc = correct / max(1, total)
    return losses.avg, acc, all_p_metal, all_true


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", help="ignore data/graphs cache")
    args = ap.parse_args()

    config.ensure_dirs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    labels = common.load_labels()
    train_pairs = common.classifier_pairs(common.load_split(config.TRAIN_IDS), labels)
    val_pairs = common.classifier_pairs(common.load_split(config.VAL_IDS), labels)
    print(f"classifier: train={len(train_pairs)} val={len(val_pairs)} on {config.DEVICE}")

    use_cache = not args.no_cache
    train_ds, train_loader = make_loader(train_pairs, args.batch_size, shuffle=True,
                                         use_cache=use_cache)
    _, val_loader = make_loader(val_pairs, args.batch_size, shuffle=False,
                                use_cache=use_cache)

    orig_atom_fea_len, nbr_fea_len = train_ds.sample_graph_dims()
    model = common.build_model(orig_atom_fea_len, nbr_fea_len, classification=True)

    criterion = nn.NLLLoss(weight=class_weights(train_pairs))
    if config.OPTIM == "Adam":
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=config.MOMENTUM,
                              weight_decay=config.WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=config.LR_MILESTONES, gamma=config.LR_GAMMA)
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

    best_acc, best_state = -1.0, None
    best_val_p, best_val_true = None, None
    for epoch in range(args.epochs):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler)
        with torch.no_grad():
            va_loss, va_acc, va_p, va_true = run_epoch(model, val_loader, criterion)
        scheduler.step()
        print(f"[clf] epoch {epoch:3d}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_val_p, best_val_true = va_p, va_true

    torch.save({
        "state_dict": best_state,
        "graph_dims": [orig_atom_fea_len, nbr_fea_len],
        "best_val_acc": best_acc,
        "args": vars(args),
    }, config.CLASSIFIER_CKPT)

    # Save val P(metal) for threshold tuning (improvement #5).
    with open(config.MODELS_DIR / "classifier_val_probs.json", "w") as f:
        json.dump({"p_metal": best_val_p, "is_metal": best_val_true}, f)

    print(f"=> saved classifier (val acc {best_acc:.3f}) -> {config.CLASSIFIER_CKPT}")


if __name__ == "__main__":
    main()
