"""
Stage 5 - identical evaluation metrics for E1-E4 (and the cross-corpus experiments).

Definitions (C = 6 classes, n_c = number of test recordings of class c):
  accuracy (WA, weighted accuracy) = correct / all                     (favours frequent classes)
  UA (unweighted accuracy)         = (1/C) sum_c recall_c              (= balanced accuracy;
                                     identical to macro recall by definition - both reported)
  macro precision / recall / F1    = unweighted mean over classes of the per-class values
  F1_c = 2 P_c R_c / (P_c + R_c);   a class never predicted gets P_c = 0 (zero_division=0)
Primary metrics (agreed): macro-F1 and UA. RAVDESS is imbalanced (neutral = half of the
other classes), so accuracy alone would reward ignoring neutral.

Aggregation across runs: per fold first (average seeds), then mean and sample std (ddof=1)
over folds. We never report the best fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.preprocessing.dataset_index import CLASSES

METRICS = ("accuracy", "UA", "macro_precision", "macro_recall", "macro_f1")


def compute_metrics(y_true, y_pred, n_classes: int = len(CLASSES)) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "UA": float(r.mean()),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "per_class": {CLASSES[c]: {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]),
                                   "support": int(s[c])} for c in labels},
        "confusion": cm.tolist(),          # rows = true class, columns = predicted (order CLASSES)
    }


def summarise(per_fold: list[dict], keys=METRICS) -> dict:
    """mean / std (ddof=1) / min / max over folds of each metric."""
    out = {}
    for k in keys:
        v = np.array([d[k] for d in per_fold], dtype=np.float64)
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max()), "n_folds": int(len(v))}
    return out
