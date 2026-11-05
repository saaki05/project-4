"""Supervised training and evaluation of change-detection networks."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from terravision_models import metrics as M
from terravision_models.models import ChangeNet


@dataclass(frozen=True)
class FitConfig:
    epochs: int = 30
    batch: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 8
    dice_weight: float = 1.0
    amp: bool = True  # mixed precision, used only on CUDA


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bce_dice_loss(logits: Tensor, target: Tensor, dice_weight: float = 1.0,
                  pos_weight: float | None = None) -> Tensor:
    """Binary cross-entropy plus soft Dice; Dice counters the heavy class imbalance."""
    t = target.float()
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, t, pos_weight=pw)
    p = torch.sigmoid(logits)
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
    return bce + dice_weight * dice


@torch.no_grad()
def predict_proba(model: ChangeNet, before: np.ndarray, after: np.ndarray, batch: int = 32,
                  device: torch.device | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(before), batch):
        b = torch.from_numpy(before[i:i + batch]).to(device)
        a = torch.from_numpy(after[i:i + batch]).to(device)
        out.append(torch.sigmoid(model(b, a)).cpu().numpy())
    return np.concatenate(out)


def fit(model: ChangeNet, train: tuple, val: tuple, cfg: FitConfig, seed: int = 0,
        device: torch.device | None = None) -> dict:
    """Train with early stopping on validation F1; restores the best weights."""
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    tb, ta, tm = (torch.from_numpy(x) for x in train)
    vb, va, vm = val
    best, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(tb))
        total = 0.0
        for i in range(0, len(order), cfg.batch):
            idx = torch.from_numpy(order[i:i + cfg.batch])
            b, a, m = tb[idx].to(device), ta[idx].to(device), tm[idx].to(device)
            with torch.autocast(device.type, enabled=use_amp):
                loss = bce_dice_loss(model(b, a).float(), m, cfg.dice_weight)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(idx)
        prob = predict_proba(model, vb, va, device=device)
        f1 = M.scores(prob >= 0.5, vm)["f1"]
        history.append({"epoch": epoch, "train_loss": total / len(tb), "val_f1": f1})
        if f1 > best + 1e-4:
            best, best_state, stale = f1, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.cpu()
    return {"best_val_f1": best, "history": history}


def evaluate(model: ChangeNet, val: tuple, test: tuple, device: torch.device | None = None) -> dict:
    """Test scores with the threshold chosen on the validation split."""
    pv = predict_proba(model, val[0], val[1], device=device)
    thr = M.best_threshold(pv, val[2])
    pt = predict_proba(model, test[0], test[1], device=device)
    return {**M.scores(pt >= thr, test[2]), "threshold": thr}
