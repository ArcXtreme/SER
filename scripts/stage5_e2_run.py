"""
E2 driver: frozen HuBERT (pooled hidden states) + MLP head, speaker-independent folds.

Per (dataset, fold) - all decisions on VALIDATION macro-F1, staged search:
  A  representation x pooling   15 representations (13 layers, concat(1,9), weighted(all))
                                x {mean, mean_std}, default head  -> 30 runs (= layer-wise probe)
  B  head hyper-parameters      hidden {256,512} x dropout {0.2,0.4} x lr {1e-3,3e-4} -> 8 runs
  C  loss weighting             selected config with UNWEIGHTED CE (controlled: nothing else
                                changes) -> 1 run; the better of weighted / unweighted is kept
  -- selection.json is written here; phases A-C never see test rows --
  D  final                      selected config, seeds 0,1,2: train, save checkpoint, reload it,
                                predict the test speakers once per seed
Resumable: every run writes its own JSON; finished runs (same run hash) are skipped.

Run:      python scripts/stage5_e2_run.py --datasets RAVDESS CREMA-D
Summary:  python scripts/stage5_e2_run.py --datasets RAVDESS CREMA-D --summarise
"""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluation.metrics import METRICS, compute_metrics, summarise  # noqa: E402
from src.features.store import FeatureStore  # noqa: E402
from src.models.e2_hubert import POOLINGS, REPRESENTATIONS, HubertHead, n_params, pooled_input  # noqa: E402
from src.preprocessing.dataset_index import CLASSES  # noqa: E402
from src.preprocessing.splits import DEFAULT_FOLDS, split_path  # noqa: E402
from src.training.data import fold_rows, without_test  # noqa: E402
from src.training.preprocessing import FrameStandardizer  # noqa: E402
from src.training.trainer import TrainConfig, device, predict, train_classifier  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402
from src.utils.resources import gpu_peak_mib, process_ram_mib  # noqa: E402


@dataclass(frozen=True)
class E2Config:
    representations: tuple = REPRESENTATIONS
    poolings: tuple = POOLINGS
    default_hidden: int = 256
    default_dropout: float = 0.2
    default_lr: float = 1e-3
    grid_hidden: tuple = (256, 512)
    grid_dropout: tuple = (0.2, 0.4)
    grid_lr: tuple = (1e-3, 3e-4)
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10
    selection_seed: int = 0
    final_seeds: tuple = (0, 1, 2)
    selection_metric: str = "validation macro-F1 (ties -> earlier candidate)"


CFG = E2Config()


def run_hash(ds, store):
    split_sha = hashlib.sha256(split_path(ds).read_bytes()).hexdigest()
    blob = json.dumps({"e2": asdict(CFG), "cache": store.meta["cache_id"], "split": split_sha}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- data + model
def make_inputs(pooled, rows_fit, rows_apply_list, rep, pooling):
    """Standardiser fitted on rows_fit (TRAIN only), applied to each array in rows_apply_list."""
    Xfit = pooled_input(pooled, rows_fit, rep, pooling)
    shape = Xfit.shape[1:]
    std = FrameStandardizer().fit([Xfit.reshape(len(Xfit), -1)])
    dev = device()
    out = []
    for rows in rows_apply_list:
        X = pooled_input(pooled, rows, rep, pooling)
        out.append(torch.from_numpy(std.transform(X.reshape(len(X), -1)).reshape(len(X), *shape)).to(dev))
    return out, std, shape


def build_model(shape, rep, hidden, dropout):
    d_in = shape[-1]
    return HubertHead(d_in, hidden, dropout, weighted_layers=(rep == "weighted_all")).to(device())


def train_cfg(hp, seed):
    return TrainConfig(lr=hp["lr"], weight_decay=CFG.weight_decay, batch_size=CFG.batch_size,
                       max_epochs=CFG.max_epochs, patience=CFG.patience, class_weighted=hp["class_weighted"], seed=seed)


def run_candidate(pooled, fr, cand: dict, seed: int) -> dict:
    """Train on TRAIN speakers, early-stop / score on VALIDATION speakers. No test argument."""
    (Xtr, Xva), _, shape = make_inputs(pooled, fr.rows["train"], [fr.rows["train"], fr.rows["val"]],
                                       cand["rep"], cand["pooling"])
    res = train_classifier(lambda: build_model(shape, cand["rep"], cand["hidden"], cand["dropout"]),
                           (Xtr, fr.labels["train"]), (Xva, fr.labels["val"]), train_cfg(cand, seed))
    model = res["model"]
    val = compute_metrics(fr.labels["val"], predict(model, Xva))
    out = {"candidate": cand, "seed": seed, "val_macro_f1": res["best_val_macro_f1"], "val": val,
           "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], "train_seconds": res["train_seconds"],
           "n_params": n_params(model), "input_shape": list(shape), "history": res["history"]}
    if cand["rep"] == "weighted_all":
        out["layer_weights"] = torch.softmax(model.layer_weighting.logits.detach().cpu(), 0).tolist()
    return out


def cached(path: Path, rhash: str, fn):
    if path.exists():
        r = json.loads(path.read_text())
        if r.get("run_hash") == rhash:
            return r
    r = fn()
    r["run_hash"] = rhash
    write_atomic(path, r)
    return r


def write_atomic(path: Path, obj, indent=None):
    """Write to a temp file, then rename: an interruption never leaves a truncated result file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=indent))
    tmp.replace(path)


def best_of(results: list[dict]) -> dict:
    return results[max(range(len(results)), key=lambda j: (results[j]["val_macro_f1"], -j))]


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, std, cand, shape, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "model_args": {"shape": list(shape), "rep": cand["rep"], "hidden": cand["hidden"], "dropout": cand["dropout"]},
                "candidate": cand, "standardizer": {"mean": std.mean_, "std": std.std_}, **meta}, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location=device(), weights_only=False)
    a = ck["model_args"]
    model = build_model(tuple(a["shape"]), a["rep"], a["hidden"], a["dropout"])
    model.load_state_dict(ck["state_dict"])
    std = FrameStandardizer(); std.mean_, std.std_, std._fitted = ck["standardizer"]["mean"], ck["standardizer"]["std"], True
    return model, std, ck


def final_and_test(ds, store, pooled, fr, cand, seed, rhash, selection_val_f1):
    """Phase D: train selected config (train speakers), checkpoint, reload, predict TEST once."""
    tag = f"fold{fr.fold}_seed{seed}"
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    (Xtr, Xva), std, shape = make_inputs(pooled, fr.rows["train"], [fr.rows["train"], fr.rows["val"]], cand["rep"], cand["pooling"])
    res = train_classifier(lambda: build_model(shape, cand["rep"], cand["hidden"], cand["dropout"]),
                           (Xtr, fr.labels["train"]), (Xva, fr.labels["val"]), train_cfg(cand, seed))
    model = res["model"]
    if seed == CFG.selection_seed and abs(res["best_val_macro_f1"] - selection_val_f1) > 1e-12:
        raise RuntimeError(f"{tag}: retraining did not reproduce the selection run "
                           f"({res['best_val_macro_f1']} vs {selection_val_f1})")
    ck = PATHS.checkpoints / "E2" / ds / f"{tag}.pt"
    save_checkpoint(ck, model, std, cand, shape, {"dataset": ds, "fold": fr.fold, "seed": seed, "run_hash": rhash,
                                                  "feature_cache": store.meta["cache_id"],
                                                  "best_epoch": res["best_epoch"], "train_cfg": train_cfg(cand, seed).to_dict()})
    # ---- test: from the reloaded checkpoint, exactly once ----
    model2, std2, _ = load_checkpoint(ck)
    Xva2 = torch.from_numpy(std2.transform(pooled_input(pooled, fr.rows["val"], cand["rep"], cand["pooling"]).reshape(len(fr.rows["val"]), -1)).reshape(len(fr.rows["val"]), *shape)).to(device())
    if not np.array_equal(predict(model2, Xva2), predict(model, Xva)):
        raise RuntimeError(f"{tag}: reloaded checkpoint predicts differently")
    t = time.perf_counter()
    Xte = pooled_input(pooled, fr.rows["test"], cand["rep"], cand["pooling"])
    Xte = torch.from_numpy(std2.transform(Xte.reshape(len(Xte), -1)).reshape(len(Xte), *shape)).to(device())
    pt = predict(model2, Xte)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    infer = time.perf_counter() - t
    return {"dataset": ds, "fold": fr.fold, "seed": seed, "candidate": cand, "n_params": n_params(model),
            "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], "train_seconds": res["train_seconds"],
            "test_inference_ms_per_recording": 1000 * infer / len(fr.rows["test"]),
            "val": compute_metrics(fr.labels["val"], predict(model, Xva)), "test": compute_metrics(fr.labels["test"], pt),
            "test_pred": pt.tolist(), "test_rows": fr.rows["test"].tolist(), "checkpoint": str(ck),
            "gpu_peak_mib": gpu_peak_mib(), "process_ram_mib": process_ram_mib(), "history": res["history"]}


# --------------------------------------------------------------------------- per fold
def select_on_validation(pooled, fr_sel, d: Path, rhash: str):
    """Phases A-C. fr_sel has NO test partition (without_test), so test rows cannot be used."""
    base = {"hidden": CFG.default_hidden, "dropout": CFG.default_dropout, "lr": CFG.default_lr, "class_weighted": True}
    A = [cached(d / f"A_{r}_{p}.json", rhash, lambda r=r, p=p: run_candidate(pooled, fr_sel, {"rep": r, "pooling": p, **base},
                                                                                CFG.selection_seed))
         for r in CFG.representations for p in CFG.poolings]
    a = best_of(A)["candidate"]
    B = [cached(d / f"B_h{h}_d{dr}_lr{lr}.json", rhash,
                lambda h=h, dr=dr, lr=lr: run_candidate(pooled, fr_sel, {**a, "hidden": h, "dropout": dr, "lr": lr}, CFG.selection_seed))
         for h, dr, lr in product(CFG.grid_hidden, CFG.grid_dropout, CFG.grid_lr)]
    b = best_of(B)
    C = cached(d / "C_unweighted.json", rhash,
               lambda: run_candidate(pooled, fr_sel, {**b["candidate"], "class_weighted": False}, CFG.selection_seed))
    return A, a, B, b, C


def run_fold(ds, store, pooled, fr, rhash):
    d = PATHS.results / "E2" / ds / f"fold{fr.fold}"
    t0 = time.perf_counter()
    A, a, B, b, C = select_on_validation(pooled, without_test(fr), d, rhash)
    final = C if C["val_macro_f1"] > b["val_macro_f1"] else b          # tie -> keep weighted (default)
    sel = {"run_hash": rhash, "dataset": ds, "fold": fr.fold, "selected": final["candidate"],
           "selected_val_macro_f1": final["val_macro_f1"],
           "A": [{"rep": r["candidate"]["rep"], "pooling": r["candidate"]["pooling"], "val_macro_f1": r["val_macro_f1"],
                  "best_epoch": r["best_epoch"]} for r in A],
           "A_selected": a, "B": [{**r["candidate"], "val_macro_f1": r["val_macro_f1"]} for r in B],
           "B_selected": b["candidate"], "C_weighted_val_macro_f1": b["val_macro_f1"],
           "C_unweighted_val_macro_f1": C["val_macro_f1"], "selection_seconds": time.perf_counter() - t0}
    prev = json.loads((d / "selection.json").read_text()) if (d / "selection.json").exists() else None
    if not (prev and prev["run_hash"] == rhash and prev["selected"] == sel["selected"]):
        write_atomic(d / "selection.json", sel, indent=1)              # written BEFORE any test prediction
    # (an identical existing selection is kept, so resuming never rewrites it after its D files)
    for seed in CFG.final_seeds:
        out = d / f"D_seed{seed}.json"
        if out.exists() and json.loads(out.read_text()).get("run_hash") == rhash:
            continue
        r = final_and_test(ds, store, pooled, fr, final["candidate"], seed, rhash, final["val_macro_f1"])
        r["run_hash"] = rhash
        write_atomic(out, r)
    finals = [json.loads((d / f"D_seed{s}.json").read_text()) for s in CFG.final_seeds]
    f1 = [f["test"]["macro_f1"] for f in finals]
    print(f"  fold{fr.fold}: sel {final['candidate']}  val F1 {final['val_macro_f1']:.3f}  "
          f"test F1 seeds {np.round(f1, 3).tolist()}  (selection {sel['selection_seconds']:.0f}s)")


# --------------------------------------------------------------------------- summary
def summarise_dataset(ds):
    root = PATHS.results / "E2" / ds
    folds = sorted(int(p.name[4:]) for p in root.glob("fold*") if p.is_dir())
    S = {"dataset": ds, "config": asdict(CFG), "folds": {}}
    per_fold_test, rowsA, sel_rows, finals_all = [], [], [], []
    for k in folds:
        sel = json.loads((root / f"fold{k}" / "selection.json").read_text())
        finals = [json.loads((root / f"fold{k}" / f"D_seed{s}.json").read_text()) for s in CFG.final_seeds]
        finals_all += finals
        m = {key: float(np.mean([f["test"][key] for f in finals])) for key in METRICS}
        per_fold_test.append(m)
        S["folds"][k] = {"selected": sel["selected"], "val_macro_f1_selection": sel["selected_val_macro_f1"],
                         "test_mean_over_seeds": m,
                         "test_macro_f1_per_seed": [f["test"]["macro_f1"] for f in finals],
                         "val_macro_f1_per_seed": [f["val"]["macro_f1"] for f in finals],
                         "n_params": finals[0]["n_params"], "train_seconds_mean": float(np.mean([f["train_seconds"] for f in finals])),
                         "infer_ms_mean": float(np.mean([f["test_inference_ms_per_recording"] for f in finals])),
                         "gpu_peak_mib_max": max(f["gpu_peak_mib"] or 0 for f in finals),
                         "ram_peak_mib_max": max(f["process_ram_mib"]["peak"] or 0 for f in finals),
                         "weighted_vs_unweighted_val": [sel["C_weighted_val_macro_f1"], sel["C_unweighted_val_macro_f1"]]}
        rowsA += [{"fold": k, **a} for a in sel["A"]]
        sel_rows.append({"fold": k, **sel["selected"]})
    S["test"] = summarise(per_fold_test)
    S["seed_std_macro_f1_mean_within_fold"] = float(np.mean([np.std(v["test_macro_f1_per_seed"], ddof=1) for v in S["folds"].values()]))
    A = pd.DataFrame(rowsA)
    S["phaseA_val_macro_f1"] = {f"{r}|{p}": {"mean": float(g.val_macro_f1.mean()), "std": float(g.val_macro_f1.std(ddof=1))}
                                for (r, p), g in A.groupby(["rep", "pooling"])}
    piv = A.pivot_table(index=["fold", "rep"], columns="pooling", values="val_macro_f1")
    dlt = (piv["mean_std"] - piv["mean"]).groupby("rep")
    S["pooling_val_diff_mean_std_minus_mean"] = {r: {"mean": float(g.mean()), "n_better": int((g > 0).sum()), "n": int(len(g))}
                                                 for r, g in dlt}
    S["selection_counts"] = {c: pd.DataFrame(sel_rows)[c].astype(str).value_counts().to_dict()
                             for c in ("rep", "pooling", "hidden", "dropout", "lr", "class_weighted")}
    cm = np.sum([np.array(f["test"]["confusion"]) for f in finals_all], axis=0)
    S["confusion_summed_over_folds_and_seeds"] = cm.tolist()
    S["per_class_f1_mean"] = {c: float(np.mean([f["test"]["per_class"][c]["f1"] for f in finals_all])) for c in CLASSES}
    S["per_class_precision_mean"] = {c: float(np.mean([f["test"]["per_class"][c]["precision"] for f in finals_all])) for c in CLASSES}
    S["per_class_recall_mean"] = {c: float(np.mean([f["test"]["per_class"][c]["recall"] for f in finals_all])) for c in CLASSES}
    # E1 comparison on the same folds (test vs test only)
    e1 = pd.read_csv(PATHS.results / "E1" / ds / "runs.csv")
    e1 = e1[e1.variant == "headline"].groupby(["model", "fold"]).test_macro_f1.mean()
    e2f = pd.Series([p["macro_f1"] for p in per_fold_test], index=folds)
    S["paired_vs_E1"] = {m: {"diff_per_fold": (e2f - e1[m]).round(4).tolist(), "mean": float((e2f - e1[m]).mean()),
                             "std": float((e2f - e1[m]).std(ddof=1)), "n_E2_better": int(((e2f - e1[m]) > 0).sum())}
                         for m in ("svm", "rf")}
    (root / "summary.json").write_text(json.dumps(S, indent=1))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["RAVDESS"])
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    (PATHS.configs / "e2_hubert_head.json").write_text(json.dumps(asdict(CFG), indent=1))
    for ds in args.datasets:
        if args.summarise:
            S = summarise_dataset(ds)
            t = S["test"]
            print(ds, {k: f"{t[k]['mean']:.3f}±{t[k]['std']:.3f}" for k in METRICS})
            continue
        store = FeatureStore.open(ds, index)
        pooled = np.asarray(store.array("hubert_pooled"))              # (n, 13, 1536) fp32, in RAM
        rhash = run_hash(ds, store)
        print(f"[E2] {ds}: run hash {rhash}, pooled {pooled.shape}, device {device()}, RAM {process_ram_mib()}")
        for k in (args.folds if args.folds is not None else range(DEFAULT_FOLDS[ds])):
            run_fold(ds, store, pooled, fold_rows(store, index, ds, k), rhash)
        print(f"[E2] {ds}: done. GPU peak {gpu_peak_mib()} MiB, RAM {process_ram_mib()}")


if __name__ == "__main__":
    main()
