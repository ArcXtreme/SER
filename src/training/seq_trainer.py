"""
E3/E4 - deterministic training loop for variable-length sequence models.

Same rules as src/training/trainer.py (E2):
  - AdamW, class-weighted cross-entropy (weights from TRAINING labels only)
  - after every epoch: macro-F1 on the VALIDATION recordings; best epoch kept (ties -> earliest);
    stop after `patience` epochs without improvement; never receives test rows
  - the seed is set BEFORE the model factory is called (seed controls initial weights)
Additions for sequence models:
  - length-bucketed, seeded mini-batches (SequenceData.train_batches)
  - gradient-norm clipping at `grad_clip` = 1.0 (fixed a priori, shared by E3 and E4); the
    pre-clipping norm is logged per epoch as a training-stability diagnostic.
    PILOT FINDING (results/E3/pilot_*.json): mean gradient norms are 2-7 on CREMA-D, so the clip
    is active on 96-100 % of CREMA-D steps (26-85 % on RAVDESS). It therefore acts as per-step
    gradient normalisation inside AdamW, not as a rarely-triggered safety net. No divergence was
    observed (max norm 31, monotone loss). Kept unchanged so that E3 and E4 share one optimiser;
    E2 (src/training/trainer.py) uses no clipping - an optimiser difference between E2 and E3.
  - a non-finite loss or gradient raises an error (never silently skipped)
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.training.sequence_data import SequenceData
from src.training.trainer import class_weights, set_determinism


@dataclass(frozen=True)
class SeqTrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10
    class_weighted: bool = True
    seed: int = 0
    grad_clip: float = 1.0
    bucket: int = 20
    eval_batch_size: int = 64

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def predict_rows(model: nn.Module, data: SequenceData, rows: np.ndarray, batch_size: int = 64,
                 return_logits: bool = False):
    """Predictions for `rows`, returned in the ORDER OF `rows` (batches are length-sorted internally)."""
    model.eval()
    rows = np.asarray(rows)
    pos = {int(r): i for i, r in enumerate(rows)}
    logits = np.zeros((len(rows), 6), dtype=np.float32)
    for b in data.eval_batches(rows, batch_size):
        out = model(data.batch(b)).float().cpu().numpy()
        logits[[pos[int(r)] for r in b]] = out
    pred = logits.argmax(1)
    return (pred, logits) if return_logits else pred


def _macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=list(range(6)), zero_division=0))


def train_sequence_classifier(model_fn, data: SequenceData, train: tuple, val: tuple, cfg: SeqTrainConfig) -> dict:
    """train / val = (rows, labels) for TRAIN / VALIDATION recordings of the fold."""
    tr_rows, ytr = np.asarray(train[0]), np.asarray(train[1])
    va_rows, yva = np.asarray(val[0]), np.asarray(val[1])
    if set(tr_rows.tolist()) != data.fitted_on:
        raise RuntimeError("standardisers were not fitted on exactly these training rows")
    set_determinism(cfg.seed)
    model = model_fn()
    dev = next(model.parameters()).device
    y_lookup = torch.full((data.n,), -1, dtype=torch.long, device=dev)
    y_lookup[torch.as_tensor(tr_rows, device=dev)] = torch.as_tensor(ytr, device=dev)
    w = class_weights(ytr) if cfg.class_weighted else np.ones(6)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    best = {"f1": -1.0, "epoch": -1, "state": None}
    history = []
    t0 = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        te = time.perf_counter()
        model.train()
        tot, nb, gnorms = 0.0, 0, []
        for b in data.train_batches(tr_rows, cfg.batch_size, gen, cfg.bucket):
            batch = data.batch(b)
            y = y_lookup[batch["rows"]]
            loss = loss_fn(model(batch), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            opt.step()
            tot += float(loss.detach()); nb += 1; gnorms.append(float(g))
        f1 = _macro_f1(yva, predict_rows(model, data, va_rows, cfg.eval_batch_size))
        history.append({"epoch": epoch, "train_loss": tot / nb, "val_macro_f1": f1,
                        "grad_norm_mean": float(np.mean(gnorms)), "grad_norm_max": float(np.max(gnorms)),
                        "frac_clipped": float(np.mean(np.array(gnorms) > cfg.grad_clip)),
                        "seconds": time.perf_counter() - te})
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
        elif epoch - best["epoch"] >= cfg.patience:
            break
    model.load_state_dict(best["state"])
    return {"model": model, "state_dict": best["state"], "best_epoch": best["epoch"], "best_val_macro_f1": best["f1"],
            "history": history, "train_seconds": time.perf_counter() - t0, "epochs_run": len(history),
            "class_weights": w.tolist()}
