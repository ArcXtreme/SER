"""
E1 driver: MFCC statistics + SVM / Random Forest, speaker-independent folds (Stage 3 files).

Per (dataset, fold, model, seed) run:
  1. rows of train / val / test speakers from the saved split (fold_rows -> load_fold checks)
  2. select_on_validation(): all aggregation x hyper-parameter candidates, fit on train,
     macro-F1 on val. Written to <run>.selection.json BEFORE any test row is touched.
  3. the selected candidate (and, for the descriptive aggregation comparison, the val-best
     candidate of each aggregation) is refitted alone on train to measure training time;
     predictions on val are checked to equal those of the selection phase (determinism).
  4. test partition predicted once; metrics + timings + model size written to <run>.json;
     fitted pipeline saved with joblib under C:\\SER_data\\checkpoints\\E1\\.
Resumable: finished runs (same config hash) are skipped.

Run:     python scripts/stage5_e1_run.py --datasets RAVDESS CREMA-D
Summary: python scripts/stage5_e1_run.py --datasets RAVDESS CREMA-D --summarise
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.metrics import METRICS, compute_metrics, summarise  # noqa: E402
from src.features.store import FeatureStore  # noqa: E402
from src.models.e1_classical import (E1Config, aggregate_store, build, model_size,  # noqa: E402
                                     select_on_validation)
from src.preprocessing.dataset_index import CLASSES  # noqa: E402
from src.preprocessing.splits import DEFAULT_FOLDS, split_path  # noqa: E402
from src.training.data import fold_rows  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402

CFG = E1Config()


def run_hash(ds: str, store) -> str:
    split_sha = hashlib.sha256(split_path(ds).read_bytes()).hexdigest()
    blob = json.dumps({"e1": CFG.to_dict(), "cache": store.meta["cache_id"], "split": split_sha}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_aggregates(ds: str, store) -> dict:
    """Utterance-level MFCC functionals, cached once per feature cache (label-free)."""
    path = PATHS.cache / "stage5" / f"E1_mfcc_agg_{ds}_{store.meta['cache_id']}.npz"
    if path.exists():
        z = np.load(path)
        X = {k: z[k] for k in CFG.aggregations}
        if all(len(v) == len(store.manifest) for v in X.values()):
            return X
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    X = aggregate_store(store, CFG.aggregations)
    np.savez(path, **X)
    print(f"[E1] {ds}: aggregated {len(store.manifest)} recordings in {time.perf_counter() - t0:.1f}s -> {path}")
    return X


def one_run(ds, store, X, fr, model, seed, rhash, jobs):
    tag = f"fold{fr.fold}_{model}_seed{seed}"
    out = PATHS.results / "E1" / ds / f"{tag}.json"
    if out.exists() and json.loads(out.read_text()).get("run_hash") == rhash:
        print(f"  {tag}: done (skip)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    ytr, yva, yte = fr.labels["train"], fr.labels["val"], fr.labels["test"]
    rtr, rva, rte = fr.rows["train"], fr.rows["val"], fr.rows["test"]

    # ---- 1) selection on validation speakers (no test data involved) ----
    t0 = time.perf_counter()
    sel = select_on_validation(model, X, ytr, yva, rtr, rva, CFG, seed, n_jobs=jobs if model == "svm" else 1)
    sel_seconds = time.perf_counter() - t0
    sel_record = {"run_hash": rhash, "dataset": ds, "fold": fr.fold, "model": model, "seed": seed,
                  "selection_seconds": sel_seconds, **sel}
    out.with_suffix(".selection.json").write_text(json.dumps(sel_record, indent=1))

    # ---- 2) refit selected candidates alone (timing + determinism check), then test once ----
    def final(cand):
        agg, hp = cand["aggregation"], cand["hp"]
        pipe = build(model, hp, X[agg].shape[1], CFG, seed)
        t = time.perf_counter(); pipe.fit(X[agg][rtr], ytr); fit_s = time.perf_counter() - t
        pv = pipe.predict(X[agg][rva])
        val = compute_metrics(yva, pv)
        if abs(val["macro_f1"] - cand["val_macro_f1"]) > 1e-12:
            raise RuntimeError(f"{tag}: refit is not deterministic ({val['macro_f1']} vs {cand['val_macro_f1']})")
        t = time.perf_counter(); pt = pipe.predict(X[agg][rte]); inf_s = time.perf_counter() - t
        return pipe, {"aggregation": agg, "hp": hp, "input_dim": int(X[agg].shape[1]),
                      "train_seconds": fit_s, "test_inference_ms_per_recording": 1000 * inf_s / len(rte),
                      "model_size": model_size(pipe, model), "val": val, "test": compute_metrics(yte, pt),
                      "test_pred": pt.tolist()}

    pipe, head = final(sel["selected"])
    per_agg = {}
    for agg, cand in sel["selected_per_aggregation"].items():
        per_agg[agg] = head if cand is sel["selected"] or cand == sel["selected"] else final(cand)[1]
    ck = PATHS.checkpoints / "E1" / ds / f"{tag}.joblib"
    ck.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ck)
    rec = {"run_hash": rhash, "experiment": "E1", "dataset": ds, "fold": fr.fold, "model": model, "seed": seed,
           "config": CFG.to_dict(), "feature_cache": store.meta["cache_id"],
           "n": {p: int(len(fr.rows[p])) for p in fr.rows}, "speakers": fr.speakers,
           "selection_seconds": sel_seconds, "headline": head, "per_aggregation": per_agg,
           "test_rows": rte.tolist(), "checkpoint": str(ck),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.write_text(json.dumps(rec, indent=1))
    h = head["test"]
    print(f"  {tag}: sel={head['aggregation']} {head['hp']}  val F1={head['val']['macro_f1']:.3f}  "
          f"test F1={h['macro_f1']:.3f} UA={h['UA']:.3f} acc={h['accuracy']:.3f}  "
          f"(selection {sel_seconds:.0f}s, fit {head['train_seconds']:.2f}s)")


def summarise_dataset(ds: str):
    runs = [json.loads(p.read_text()) for p in sorted((PATHS.results / "E1" / ds).glob("fold*_seed*.json"))
            if not p.name.endswith(".selection.json")]
    rows = []
    for r in runs:
        for variant, d in [("headline", r["headline"])] + [(f"agg={a}", v) for a, v in r["per_aggregation"].items()]:
            rows.append({"model": r["model"], "fold": r["fold"], "seed": r["seed"], "variant": variant,
                         "aggregation": d["aggregation"], "hp": json.dumps(d["hp"]),
                         "val_macro_f1": d["val"]["macro_f1"], **{f"test_{k}": d["test"][k] for k in METRICS},
                         "train_seconds": d["train_seconds"], "infer_ms": d["test_inference_ms_per_recording"],
                         **{f"size_{k}": v for k, v in d["model_size"].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(PATHS.results / "E1" / ds / "runs.csv", index=False)
    summary = {"dataset": ds, "n_runs": len(runs), "protocol": "per fold: seeds averaged first; mean/std over folds"}
    for (model, variant), g in df.groupby(["model", "variant"]):
        per_fold = g.groupby("fold")[[f"test_{k}" for k in METRICS] + ["val_macro_f1", "train_seconds", "infer_ms"]].mean()
        seed_std = g.groupby("fold")["test_macro_f1"].std(ddof=1).mean() if g.seed.nunique() > 1 else 0.0
        summary.setdefault(model, {})[variant] = {
            "test": summarise([{k: row[f"test_{k}"] for k in METRICS} for _, row in per_fold.iterrows()]),
            "val_macro_f1_mean": float(per_fold.val_macro_f1.mean()),
            "mean_within_fold_seed_std_macro_f1": float(seed_std),
            "train_seconds_mean": float(per_fold.train_seconds.mean()),
            "infer_ms_per_recording_mean": float(per_fold.infer_ms.mean()),
            "selected_aggregations": g.aggregation.value_counts().to_dict() if variant == "headline" else None,
            "selected_hp": g.hp.value_counts().to_dict(),
        }
    # pooled confusion + per-class metrics (headline), summed over folds and seeds
    for model in df.model.unique():
        cm = np.zeros((6, 6), dtype=np.int64)
        per_class = []
        for r in runs:
            if r["model"] == model:
                cm += np.array(r["headline"]["test"]["confusion"])
                per_class.append(r["headline"]["test"]["per_class"])
        summary[model]["headline"]["confusion_summed"] = cm.tolist()
        summary[model]["headline"]["per_class_f1_mean"] = {c: float(np.mean([p[c]["f1"] for p in per_class])) for c in CLASSES}
        summary[model]["headline"]["per_class_recall_mean"] = {c: float(np.mean([p[c]["recall"] for p in per_class])) for c in CLASSES}
    (PATHS.results / "E1" / ds / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["RAVDESS"])
    ap.add_argument("--models", nargs="+", default=["svm", "rf"])
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    cfg_path = PATHS.configs / "e1_mfcc_classical.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(CFG.to_dict(), indent=1))
    for ds in args.datasets:
        if args.summarise:
            s, _ = summarise_dataset(ds)
            for model in ("svm", "rf"):
                if model in s:
                    t = s[model]["headline"]["test"]
                    print(ds, model, {k: f"{t[k]['mean']:.3f}±{t[k]['std']:.3f}" for k in METRICS})
            continue
        store = FeatureStore.open(ds, index)
        X = load_aggregates(ds, store)
        rhash = run_hash(ds, store)
        print(f"[E1] {ds}: run hash {rhash}, aggregations {{{', '.join(f'{k}: {v.shape[1]}' for k, v in X.items())}}}")
        for k in (args.folds if args.folds is not None else range(DEFAULT_FOLDS[ds])):
            fr = fold_rows(store, index, ds, k)
            for model in args.models:
                for seed in (CFG.rf_seeds if model == "rf" else (0,)):
                    one_run(ds, store, X, fr, model, seed, rhash, args.jobs)


if __name__ == "__main__":
    main()
