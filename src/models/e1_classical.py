"""
E1 - Traditional machine-learning baseline: utterance-level MFCC statistics + SVM / Random Forest.

WHY THIS BASELINE?
  * It is the classical SER pipeline (hand-crafted spectral features + a kernel / ensemble
    classifier), so it tells us how much a large pretrained model actually adds.
  * It has few hyper-parameters, trains in seconds and is deterministic (SVM) -> a stable
    reference point and a check that the data, splits and metrics pipeline is correct.
  * SVM (RBF): maximum-margin classifier in an implicit feature space; strong on small
    datasets (RAVDESS: 704 training recordings) with standardised dense features.
  * Random Forest: bagged decision trees on random feature subsets; scale-invariant,
    non-linear, little tuning, and gives a different inductive bias from the SVM.

FROM SEQUENCES TO VECTORS (aggregation)
  A recording is a (T, 39) MFCC matrix with variable T. SVM / RF need one fixed-length vector,
  so each of the 39 dimensions is summarised over time ("statistical functionals"):
      mean         ->  39 dims   average spectral envelope
      mean_std     ->  78 dims   + its variability over time (dynamics)
      functionals  -> 195 dims   + 10th / 50th / 90th percentiles (distribution shape,
                                   robust to single outlier frames, unlike min / max)
  Each statistic uses only the frames of that one recording (no fitted parameters), so
  aggregation cannot leak information between recordings or partitions.

TRAIN / VALIDATION / TEST PROTOCOL (per fold, per seed)
  select_on_validation(): every (aggregation, hyper-parameter) candidate is fitted on the
      TRAIN speakers (StandardScaler + classifier in one Pipeline, so the scaler is fitted on
      train only) and scored on the VALIDATION speakers by macro-F1. It never receives test
      data - the function has no test arguments.
  The selected candidate is the one with the highest validation macro-F1 (ties -> first in
  grid order). Only afterwards is the test partition predicted, once.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

AGGREGATIONS = ("mean", "mean_std", "functionals")
PERCENTILES = (10, 50, 90)


def aggregate(frames: np.ndarray, kind: str) -> np.ndarray:
    """(T, D) frame matrix -> fixed-length vector. Statistics over time, per dimension."""
    x = np.asarray(frames, dtype=np.float64)
    parts = [x.mean(0)]
    if kind in ("mean_std", "functionals"):
        parts.append(x.std(0))
    if kind == "functionals":
        parts.extend(np.percentile(x, PERCENTILES, axis=0))
    if kind not in AGGREGATIONS:
        raise ValueError(kind)
    return np.concatenate(parts).astype(np.float32)


def aggregate_store(store, kinds=AGGREGATIONS) -> dict[str, np.ndarray]:
    """Aggregate every recording of a FeatureStore: kind -> (n_recordings, d)."""
    out = {k: [] for k in kinds}
    for i in range(len(store.manifest)):
        f = store.frames("mfcc", i)
        for k in kinds:
            out[k].append(aggregate(f, k))
    return {k: np.stack(v) for k, v in out.items()}


@dataclass(frozen=True)
class E1Config:
    aggregations: tuple = AGGREGATIONS
    svm_C: tuple = (0.1, 1.0, 10.0, 100.0)
    svm_gamma_mult: tuple = (0.1, 1.0, 10.0)        # gamma = mult / d (d = input dimension)
    rf_n_estimators: int = 300
    rf_max_features: tuple = ("sqrt", 0.2)
    rf_min_samples_leaf: tuple = (1, 5)
    rf_seeds: tuple = (0, 1, 2)
    class_weight: str = "balanced"                  # same rule as class-weighted CE in E2-E4
    selection_metric: str = "macro_f1"              # on VALIDATION speakers only
    scaler: str = "StandardScaler (fit on train fold)"

    def to_dict(self):
        return asdict(self)


def candidates(model: str, cfg: E1Config) -> list[dict]:
    if model == "svm":
        return [{"C": C, "gamma_mult": g} for C, g in product(cfg.svm_C, cfg.svm_gamma_mult)]
    if model == "rf":
        return [{"max_features": mf, "min_samples_leaf": ml}
                for mf, ml in product(cfg.rf_max_features, cfg.rf_min_samples_leaf)]
    raise ValueError(model)


def build(model: str, hp: dict, d: int, cfg: E1Config, seed: int) -> Pipeline:
    if model == "svm":
        clf = SVC(kernel="rbf", C=hp["C"], gamma=hp["gamma_mult"] / d, class_weight=cfg.class_weight)
    else:
        clf = RandomForestClassifier(n_estimators=cfg.rf_n_estimators, max_features=hp["max_features"],
                                     min_samples_leaf=hp["min_samples_leaf"], class_weight=cfg.class_weight,
                                     random_state=seed, n_jobs=-1)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def model_size(pipe: Pipeline, model: str) -> dict:
    """'Capacity' of non-parametric models, reported instead of a parameter count."""
    clf = pipe.named_steps["clf"]
    if model == "svm":
        n_sv = int(clf.support_vectors_.shape[0])
        return {"n_support_vectors": n_sv,
                "stored_numbers": int(clf.support_vectors_.size + clf.dual_coef_.size + clf.intercept_.size)}
    nodes = [e.tree_.node_count for e in clf.estimators_]
    return {"n_trees": len(nodes), "total_nodes": int(sum(nodes)), "mean_depth": float(np.mean([e.tree_.max_depth for e in clf.estimators_]))}


def _fit_score(model, hp, agg, Xtr, ytr, Xva, yva, cfg, seed):
    pipe = build(model, hp, Xtr.shape[1], cfg, seed)
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    pred = pipe.predict(Xva)
    return {"aggregation": agg, "hp": hp, "val_macro_f1": float(f1_score(yva, pred, average="macro")),
            "fit_seconds": fit_s}


def select_on_validation(model: str, X: dict, y_train, y_val, rows_train, rows_val, cfg: E1Config,
                         seed: int, n_jobs: int = 1) -> dict:
    """
    X: aggregation -> (n_recordings, d) feature matrix of the WHOLE corpus (unlabelled
       functionals; only the train/val rows are indexed here).
    Returns the full candidate table and the selected candidate. No test argument exists.
    """
    from joblib import Parallel, delayed
    jobs = [(agg, hp) for agg in cfg.aggregations for hp in candidates(model, cfg)]
    table = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(model, hp, agg, X[agg][rows_train], y_train, X[agg][rows_val], y_val, cfg, seed)
        for agg, hp in jobs)
    best = max(range(len(table)), key=lambda j: (table[j]["val_macro_f1"], -j))   # ties -> first
    per_agg = {}
    for agg in cfg.aggregations:
        idx = [j for j, r in enumerate(table) if r["aggregation"] == agg]
        per_agg[agg] = table[max(idx, key=lambda j: (table[j]["val_macro_f1"], -j))]
    return {"table": table, "selected": table[best], "selected_per_aggregation": per_agg}
