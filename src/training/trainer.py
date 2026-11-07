"""
Stage 5 - generic, deterministic training loop with validation-based early stopping.

  train_classifier(model, (X_train, y_train), (X_val, y_val), cfg)
      - mini-batch AdamW on class-weighted (or unweighted) cross-entropy
      - after every epoch: macro-F1 on the VALIDATION set; the best epoch's weights are kept
        (ties -> earliest epoch); stop after `patience` epochs without improvement
      - receives no test data (model selection and early stopping use validation only)
  predict(model, X) - batched inference, no gradients

Determinism: seeds for python / numpy / torch, a seeded CPU generator for shuffling,
torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG (required by cuBLAS for
deterministic matmuls). The same seed therefore reproduces the same weights (tested).

Class weights (training labels only): w_c = n_train / (C * n_train,c)  (sklearn "balanced").
torch's CrossEntropyLoss(weight=w, reduction="mean") computes
    L = - sum_i w_{y_i} log p_i(y_i) / sum_i w_{y_i}.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch import nn  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10
    class_weighted: bool = True
    seed: int = 0
    # G-study (report section 8): optional global-norm gradient clipping for the pooled head. The default None keeps
    # the E2 code path unchanged (no clipping call at all), so every E2 run is reproduced exactly (gates R1-R5).
    grad_clip: float | None = None

    def to_dict(self):
        return asdict(self)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def class_weights(y_train: np.ndarray, n_classes: int = 6) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"class missing from training labels: {counts}")
    return len(y_train) / (n_classes * counts)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = [model(X[i:i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).cpu().numpy()


def train_classifier(model_fn, train: tuple, val: tuple, cfg: TrainConfig) -> dict:
    """
    model_fn: zero-argument factory returning the model ON ITS DEVICE. It is called AFTER the
              seed is set, so the seed controls the weight initialisation too (a model built
              before seeding would start from different weights in every run).
    train / val: (X tensor on the target device, y numpy int array).
    Returns {"model" (best-epoch weights loaded), "state_dict" (best epoch, CPU), "best_epoch",
             "best_val_macro_f1", "history", "train_seconds", "epochs_run", "class_weights"}.
    """
    set_determinism(cfg.seed)
    model = model_fn()
    dev = next(model.parameters()).device
    Xtr, ytr = train
    Xva, yva = val
    yt = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    w = class_weights(ytr) if cfg.class_weighted else np.ones(6)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    best = {"f1": -1.0, "epoch": -1, "state": None}
    history = []
    t0 = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(yt), generator=gen).to(dev)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            loss = loss_fn(model(Xtr[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            opt.step()
            tot += float(loss.detach()); nb += 1
        f1 = float(f1_score(yva, predict(model, Xva), average="macro", labels=list(range(6)), zero_division=0))
        history.append({"epoch": epoch, "train_loss": tot / nb, "val_macro_f1": f1})
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
        elif epoch - best["epoch"] >= cfg.patience:
            break
    model.load_state_dict(best["state"])
    return {"model": model, "state_dict": best["state"], "best_epoch": best["epoch"], "best_val_macro_f1": best["f1"],
            "history": history, "train_seconds": time.perf_counter() - t0, "epochs_run": len(history),
            "class_weights": w.tolist()}
