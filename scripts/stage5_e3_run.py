"""
E3 driver: frame-level concatenation fusion (HuBERT + MFCC + prosody), speaker-independent folds.

Per (dataset, fold):
  L  HuBERT layer   = best single hidden state of THIS fold's E2 phase-A VALIDATION results
                      (max over the two poolings; ties -> lower layer). Never averaged over folds:
                      fold k's test speakers are validation speakers of other folds.
  S  search (validation macro-F1, seed 0), each tuned model gets the same budget:
       concat_HMP  dropout {0.2, 0.4} x lr {1e-3, 3e-4}   4 runs   (main E3 model)
       sa_HMP      dropout {0.2, 0.4} x lr {1e-3, 3e-4}   4 runs   (parameter-matched self-attention control)
  -- selection.json is written here; phase S never sees test rows (without_test) --
  D  final, seeds 0, 1, 2, checkpoint -> reload -> predict TEST once:
       concat_HMP, sa_HMP                    (own selected hyper-parameters)
       stream ablations concat_{H, M, P, HM, HP, MP}       (hyper-parameters of concat_HMP)
       DisVoice ablations concat_HMP_dv / concat_HMP_dvind (+103 DisVoice features / + 103 indicators)
Resumable: each run writes its own JSON (atomic); finished runs with the same run hash are skipped.

  python scripts/stage5_e3_run.py --datasets RAVDESS --folds 0 --pilot   # selection only + checks
  python scripts/stage5_e3_run.py --datasets RAVDESS CREMA-D             # full experiment
  python scripts/stage5_e3_run.py --datasets RAVDESS CREMA-D --summarise
"""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluation.metrics import METRICS, compute_metrics, summarise  # noqa: E402
from src.features.store import FeatureStore  # noqa: E402
from src.models.e3_fusion import (SUBSETS, E3Fusion, e3_fusion_params, e4_reference_fusion_params,  # noqa: E402
                                  matched_sa_blocks, n_params)
from src.preprocessing.dataset_index import CLASSES  # noqa: E402
from src.preprocessing.splits import DEFAULT_FOLDS, split_path  # noqa: E402
from src.training.data import fold_rows, without_test  # noqa: E402
from src.training.seq_trainer import SeqTrainConfig, predict_rows, train_sequence_classifier  # noqa: E402
from src.training.sequence_data import SequenceData, load_hubert_layer  # noqa: E402
from src.training.trainer import device  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402
from src.utils.resources import gpu_peak_mib, process_ram_mib  # noqa: E402

TUNED = ("concat_HMP", "sa_HMP")
ABLATIONS = ("concat_H", "concat_HM", "concat_HP", "concat_MP", "concat_M", "concat_P",
             "concat_HMP_dv", "concat_HMP_dvind")


@dataclass(frozen=True)
class E3Config:
    d_model: int = 128
    n_heads: int = 4
    d_ffn: int = 256
    head_hidden: int = 256
    n_sa_blocks: int = field(default_factory=lambda: matched_sa_blocks(128, 4, 256))
    grid_dropout: tuple = (0.2, 0.4)
    grid_lr: tuple = (1e-3, 3e-4)
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10
    grad_clip: float = 1.0
    bucket: int = 20
    class_weighted: bool = True
    selection_seed: int = 0
    final_seeds: tuple = (0, 1, 2)
    tuned: tuple = TUNED
    ablations: tuple = ABLATIONS
    layer_rule: str = ("best single hidden state of the fold's own E2 phase-A validation macro-F1 "
                       "(max over poolings; ties -> lower layer)")
    selection_metric: str = "validation macro-F1 (ties -> earlier candidate)"


CFG = E3Config()


# --------------------------------------------------------------------------- variants
def parse_variant(name: str) -> dict:
    """'concat_HMP' / 'sa_HMP' / 'concat_HMP_dv' / 'concat_HMP_dvind' -> model spec."""
    parts = name.split("_")
    kind, subset = parts[0], parts[1]
    dv = parts[2] if len(parts) > 2 else None
    if kind not in ("concat", "sa") or subset not in SUBSETS or dv not in (None, "dv", "dvind"):
        raise ValueError(name)
    return {"name": name, "streams": SUBSETS[subset], "n_sa_blocks": CFG.n_sa_blocks if kind == "sa" else 0,
            "disvoice_indicators": None if dv is None else (dv == "dvind")}


def hp_source(name: str) -> str:
    return "sa_HMP" if name.startswith("sa_") else "concat_HMP"


def build_model(spec: dict, hp: dict) -> E3Fusion:
    dvd = 0 if spec["disvoice_indicators"] is None else (206 if spec["disvoice_indicators"] else 103)
    return E3Fusion(spec["streams"], CFG.d_model, hp["dropout"], CFG.head_hidden, spec["n_sa_blocks"],
                    CFG.n_heads, CFG.d_ffn, disvoice_dim=dvd).to(device())


def train_cfg(hp: dict, seed: int) -> SeqTrainConfig:
    return SeqTrainConfig(lr=hp["lr"], weight_decay=CFG.weight_decay, batch_size=CFG.batch_size,
                          max_epochs=CFG.max_epochs, patience=CFG.patience, class_weighted=CFG.class_weighted,
                          seed=seed, grad_clip=CFG.grad_clip, bucket=CFG.bucket)


# --------------------------------------------------------------------------- layer selection
def select_layer(phase_a: list[dict]) -> dict:
    """phase_a: E2 selection.json 'A' entries (validation only). Single layers only."""
    best = {}
    for a in phase_a:
        if a["rep"].startswith("layer"):
            l = int(a["rep"][5:])
            best[l] = max(best.get(l, -1.0), a["val_macro_f1"])
    if sorted(best) != list(range(13)):
        raise ValueError(f"E2 phase A incomplete: layers {sorted(best)}")
    layer = max(best, key=lambda l: (best[l], -l))
    return {"layer": layer, "val_macro_f1_per_layer": {str(l): best[l] for l in range(13)}}


def fold_layer(ds: str, fold: int) -> dict:
    p = PATHS.results / "E2" / ds / f"fold{fold}" / "selection.json"
    info = select_layer(json.loads(p.read_text())["A"])
    info["source"] = p.relative_to(PATHS.project).as_posix()
    info["source_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return info


# --------------------------------------------------------------------------- runs
def run_hash(ds, store, layer):
    split_sha = hashlib.sha256(split_path(ds).read_bytes()).hexdigest()
    blob = json.dumps({"e3": asdict(CFG), "cache": store.meta["cache_id"], "split": split_sha, "layer": layer},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def run_candidate(data: SequenceData, fr, variant: str, hp: dict, seed: int) -> dict:
    """Train on TRAIN speakers, early-stop / score on VALIDATION speakers. No test argument."""
    spec = parse_variant(variant)
    data.fit(fr.rows["train"], spec["disvoice_indicators"])
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    res = train_sequence_classifier(lambda: build_model(spec, hp), data, (fr.rows["train"], fr.labels["train"]),
                                    (fr.rows["val"], fr.labels["val"]), train_cfg(hp, seed))
    model = res["model"]
    val = compute_metrics(fr.labels["val"], predict_rows(model, data, fr.rows["val"]))
    return {"variant": variant, "hp": hp, "seed": seed, "layer": data.layer, "val_macro_f1": res["best_val_macro_f1"],
            "val": val, "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
            "train_seconds": res["train_seconds"], "n_params": n_params(model),
            "n_params_fusion": e3_fusion_params(model), "gpu_peak_mib": gpu_peak_mib(),
            "process_ram_mib": process_ram_mib(), "history": res["history"]}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=indent))
    tmp.replace(path)


def best_of(results: list[dict]) -> dict:
    return results[max(range(len(results)), key=lambda j: (results[j]["val_macro_f1"], -j))]


def search(data, fr_sel, variant, d: Path, rhash) -> list[dict]:
    return [cached(d / f"S_{variant}_d{dr}_lr{lr}.json", rhash,
                   lambda dr=dr, lr=lr: run_candidate(data, fr_sel, variant, {"dropout": dr, "lr": lr}, CFG.selection_seed))
            for dr, lr in product(CFG.grid_dropout, CFG.grid_lr)]


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, data, variant, hp, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    dv = None
    if data.dv is not None:
        p = data.dv_pre
        dv = {"fill": p.fill_, "log_cols": p.log_cols_, "mean": p.mean_, "std": p.std_, "indicators": p.add_indicators}
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "variant": variant, "hp": hp,
                "spec": parse_variant(variant), "e3_config": asdict(CFG), "layer": data.layer,
                "standardizers": {s: {"mean": z.mean_, "std": z.std_, "passthrough": z.passthrough}
                                  for s, z in data.std.items()}, "disvoice_preprocessor": dv, **meta}, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location=device(), weights_only=False)
    model = build_model(ck["spec"], ck["hp"])
    model.load_state_dict(ck["state_dict"])
    return model, ck


def final_and_test(ds, store, data, fr, variant, hp, seed, rhash, selection_val_f1, layer_info):
    """Phase D: train (train speakers), checkpoint, reload, predict TEST once."""
    tag = f"fold{fr.fold}_{variant}_seed{seed}"
    spec = parse_variant(variant)
    data.fit(fr.rows["train"], spec["disvoice_indicators"])
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    res = train_sequence_classifier(lambda: build_model(spec, hp), data, (fr.rows["train"], fr.labels["train"]),
                                    (fr.rows["val"], fr.labels["val"]), train_cfg(hp, seed))
    model = res["model"]
    if selection_val_f1 is not None and seed == CFG.selection_seed and abs(res["best_val_macro_f1"] - selection_val_f1) > 1e-12:
        raise RuntimeError(f"{tag}: retraining did not reproduce the selection run "
                           f"({res['best_val_macro_f1']} vs {selection_val_f1})")
    ck = PATHS.checkpoints / "E3" / ds / f"{tag}.pt"
    save_checkpoint(ck, model, data, variant, hp, {"dataset": ds, "fold": fr.fold, "seed": seed, "run_hash": rhash,
                                                   "feature_cache": store.meta["cache_id"], "best_epoch": res["best_epoch"],
                                                   "train_cfg": train_cfg(hp, seed).to_dict()})
    model2, ck2 = load_checkpoint(ck)
    for s, z in data.std.items():                       # checkpoint statistics == the ones used now
        if not (np.array_equal(ck2["standardizers"][s]["mean"], z.mean_) and np.array_equal(ck2["standardizers"][s]["std"], z.std_)):
            raise RuntimeError(f"{tag}: checkpoint standardiser differs ({s})")
    pv = predict_rows(model, data, fr.rows["val"])
    if not np.array_equal(predict_rows(model2, data, fr.rows["val"]), pv):
        raise RuntimeError(f"{tag}: reloaded checkpoint predicts differently")
    # ---- test: from the reloaded checkpoint, exactly once ----
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t = time.perf_counter()
    pt = predict_rows(model2, data, fr.rows["test"])
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    infer = time.perf_counter() - t
    return {"dataset": ds, "fold": fr.fold, "seed": seed, "variant": variant, "hp": hp, "layer": data.layer,
            "layer_info": layer_info, "n_params": n_params(model), "n_params_fusion": e3_fusion_params(model),
            "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], "train_seconds": res["train_seconds"],
            "class_weights": res["class_weights"],
            "test_inference_ms_per_recording": 1000 * infer / len(fr.rows["test"]),
            "val": compute_metrics(fr.labels["val"], pv), "test": compute_metrics(fr.labels["test"], pt),
            "test_pred": pt.tolist(), "test_rows": fr.rows["test"].tolist(), "checkpoint": str(ck),
            "gpu_peak_mib": gpu_peak_mib(), "process_ram_mib": process_ram_mib(), "history": res["history"]}


# --------------------------------------------------------------------------- per fold
class DataCache:
    """Keeps the SequenceData of the last HuBERT layer (consecutive folds often share a layer)."""

    def __init__(self, store):
        self.store, self.data, self.load_seconds = store, None, {}

    def get(self, layer):
        if self.data is None or self.data.layer != layer:
            self.data = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            t = time.perf_counter()
            H = load_hubert_layer(self.store, layer)
            self.data = SequenceData(self.store, layer, device(), hubert=H, need_disvoice=True)
            self.load_seconds[layer] = time.perf_counter() - t
            print(f"    loaded HuBERT layer {layer}: {H.shape} fp16 in {self.load_seconds[layer]:.0f}s, RAM {process_ram_mib()}")
        return self.data


def select_fold(ds, store, dc, fr, d: Path):
    """Phases L + S and selection.json. Returns (selection dict, data, rhash)."""
    layer_info = fold_layer(ds, fr.fold)
    rhash = run_hash(ds, store, layer_info["layer"])
    data = dc.get(layer_info["layer"])
    t0 = time.perf_counter()
    fr_sel = without_test(fr)
    S = {v: search(data, fr_sel, v, d, rhash) for v in CFG.tuned}
    chosen = {v: best_of(S[v]) for v in CFG.tuned}
    sel = {"run_hash": rhash, "dataset": ds, "fold": fr.fold, "layer": layer_info["layer"], "layer_info": layer_info,
           "selected": {v: chosen[v]["hp"] for v in CFG.tuned},
           "selected_val_macro_f1": {v: chosen[v]["val_macro_f1"] for v in CFG.tuned},
           "search": {v: [{**r["hp"], "val_macro_f1": r["val_macro_f1"], "best_epoch": r["best_epoch"],
                           "train_seconds": r["train_seconds"]} for r in S[v]] for v in CFG.tuned},
           "n_params": {v: chosen[v]["n_params"] for v in CFG.tuned},
           "n_params_fusion": {v: chosen[v]["n_params_fusion"] for v in CFG.tuned},
           "e4_reference_fusion_params": e4_reference_fusion_params(CFG.d_model, CFG.n_heads, CFG.d_ffn),
           "selection_seconds": time.perf_counter() - t0}
    prev = json.loads((d / "selection.json").read_text()) if (d / "selection.json").exists() else None
    if not (prev and prev["run_hash"] == rhash and prev["selected"] == sel["selected"]):
        write_atomic(d / "selection.json", sel, indent=1)        # written BEFORE any test prediction
    return (prev if prev and prev["run_hash"] == rhash and prev["selected"] == sel["selected"] else sel), data, rhash, S


def run_fold(ds, store, dc, fr):
    d = PATHS.results / "E3" / ds / f"fold{fr.fold}"
    sel, data, rhash, _ = select_fold(ds, store, dc, fr, d)
    for variant in CFG.tuned + CFG.ablations:
        hp = sel["selected"][hp_source(variant)]
        sel_f1 = sel["selected_val_macro_f1"][variant] if variant in CFG.tuned else None
        for seed in CFG.final_seeds:
            out = d / f"D_{variant}_seed{seed}.json"
            if out.exists() and json.loads(out.read_text()).get("run_hash") == rhash:
                continue
            r = final_and_test(ds, store, data, fr, variant, hp, seed, rhash, sel_f1, sel["layer_info"])
            r["run_hash"] = rhash
            write_atomic(out, r)
        f1 = [json.loads((d / f"D_{variant}_seed{s}.json").read_text())["test"]["macro_f1"] for s in CFG.final_seeds]
        print(f"  fold{fr.fold} L{sel['layer']} {variant:17s} hp {hp}  test F1 seeds {np.round(f1, 3).tolist()}", flush=True)


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided paired sign-flip (permutation) p-value of the mean fold difference.
    With k folds the smallest attainable p is 2 / 2^k (RAVDESS 0.031, CREMA-D 0.0625): descriptive only."""
    d = np.asarray(d, float)
    k = len(d)
    signs = np.array(list(product((-1, 1), repeat=k)))
    stats = np.abs((signs * d).mean(1))
    return float((stats >= abs(d.mean()) - 1e-12).mean())


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    return {"diff_per_fold": d.round(4).tolist(), "mean": float(d.mean()), "std": float(d.std(ddof=1)),
            "n_better": int((d > 0).sum()), "n": int(len(d)), "sign_flip_p": sign_flip_p(d.values)}


def shapley(v: dict, base: float) -> dict:
    """Shapley value of each stream for the value function v(subset) (v(empty) = base)."""
    from itertools import combinations
    from math import factorial
    players = ("H", "M", "P")
    val = lambda S: base if not S else v["".join(p for p in players if p in S)]
    out = {}
    for i in players:
        others = [p for p in players if p != i]
        phi = 0.0
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                w = factorial(len(S)) * factorial(len(players) - len(S) - 1) / factorial(len(players))
                phi += w * (val(set(S) | {i}) - val(set(S)))
        out[i] = phi
    return out


def summarise_dataset(ds):
    root = PATHS.results / "E3" / ds
    folds = sorted(int(p.name[4:]) for p in root.glob("fold*") if p.is_dir())
    variants = CFG.tuned + CFG.ablations
    S = {"dataset": ds, "config": asdict(CFG), "folds": {}, "variants": {}}
    fold_means = {v: {} for v in variants}                      # variant -> fold -> metric dict (mean over seeds)
    val_means = {v: {} for v in variants}
    finals_all = {v: [] for v in variants}
    for k in folds:
        sel = json.loads((root / f"fold{k}" / "selection.json").read_text())
        S["folds"][k] = {"layer": sel["layer"], "selected": sel["selected"], "selected_val_macro_f1": sel["selected_val_macro_f1"],
                         "search": sel["search"], "variants": {}}
        for v in variants:
            fin = [json.loads((root / f"fold{k}" / f"D_{v}_seed{s}.json").read_text()) for s in CFG.final_seeds]
            finals_all[v] += fin
            fold_means[v][k] = {m: float(np.mean([f["test"][m] for f in fin])) for m in METRICS}
            val_means[v][k] = float(np.mean([f["val"]["macro_f1"] for f in fin]))
            S["folds"][k]["variants"][v] = {
                "test_mean_over_seeds": fold_means[v][k], "test_macro_f1_per_seed": [f["test"]["macro_f1"] for f in fin],
                "val_macro_f1_per_seed": [f["val"]["macro_f1"] for f in fin], "best_epoch": [f["best_epoch"] for f in fin],
                "epochs_run": [f["epochs_run"] for f in fin], "train_seconds": [f["train_seconds"] for f in fin],
                "infer_ms": [f["test_inference_ms_per_recording"] for f in fin], "n_params": fin[0]["n_params"],
                "gpu_peak_mib": max(f["gpu_peak_mib"] or 0 for f in fin),
                "ram_peak_mib": max(f["process_ram_mib"]["peak"] or 0 for f in fin)}
    for v in variants:
        fin = finals_all[v]
        per_seed = {s: summarise([{m: f["test"][m] for m in METRICS} for f in fin if f["seed"] == s]) for s in CFG.final_seeds}
        cm = np.sum([np.array(f["test"]["confusion"]) for f in fin], axis=0)
        S["variants"][v] = {
            "test": summarise([fold_means[v][k] for k in folds]),
            "val_macro_f1": {"mean": float(np.mean(list(val_means[v].values()))), "std": float(np.std(list(val_means[v].values()), ddof=1))},
            "test_per_seed": {str(s): {m: per_seed[s][m] for m in ("macro_f1", "UA", "accuracy", "macro_precision")} for s in CFG.final_seeds},
            "seed_std_macro_f1_mean_within_fold": float(np.mean([np.std(S["folds"][k]["variants"][v]["test_macro_f1_per_seed"], ddof=1) for k in folds])),
            "per_class_f1_mean": {c: float(np.mean([f["test"]["per_class"][c]["f1"] for f in fin])) for c in CLASSES},
            "per_class_recall_mean": {c: float(np.mean([f["test"]["per_class"][c]["recall"] for f in fin])) for c in CLASSES},
            "per_class_precision_mean": {c: float(np.mean([f["test"]["per_class"][c]["precision"] for f in fin])) for c in CLASSES},
            "confusion_summed_over_folds_and_seeds": cm.tolist(),
            "n_params": int(fin[0]["n_params"]), "n_params_fusion": int(fin[0]["n_params_fusion"]),
            "best_epoch_mean": float(np.mean([f["best_epoch"] for f in fin])), "epochs_run_max": int(max(f["epochs_run"] for f in fin)),
            "train_seconds_mean": float(np.mean([f["train_seconds"] for f in fin])),
            "infer_ms_mean": float(np.mean([f["test_inference_ms_per_recording"] for f in fin])),
            "gpu_peak_mib_max": float(max(f["gpu_peak_mib"] or 0 for f in fin)),
            "ram_peak_mib_max": float(max(f["process_ram_mib"]["peak"] or 0 for f in fin))}
    F = {v: pd.Series({k: fold_means[v][k]["macro_f1"] for k in folds}) for v in variants}
    U = {v: pd.Series({k: fold_means[v][k]["UA"] for k in folds}) for v in variants}
    e2s = json.loads((PATHS.results / "E2" / ds / "summary.json").read_text())
    F["E2"] = pd.Series({k: e2s["folds"][str(k)]["test_mean_over_seeds"]["macro_f1"] for k in folds})
    U["E2"] = pd.Series({k: e2s["folds"][str(k)]["test_mean_over_seeds"]["UA"] for k in folds})
    e1 = pd.read_csv(PATHS.results / "E1" / ds / "runs.csv")
    e1 = e1[e1.variant == "headline"].groupby(["model", "fold"])[["test_macro_f1", "test_UA"]].mean()
    F["E1-SVM"] = e1.loc["svm"].test_macro_f1.reindex(folds)
    U["E1-SVM"] = e1.loc["svm"].test_UA.reindex(folds)
    comps = {
        # main comparisons
        "concat_HMP - E2": ("concat_HMP", "E2"), "concat_HMP - E1-SVM": ("concat_HMP", "E1-SVM"),
        "sa_HMP - concat_HMP": ("sa_HMP", "concat_HMP"), "sa_HMP - E2": ("sa_HMP", "E2"),
        # leave-one-stream-out (contribution of a stream given the other two)
        "HuBERT | M+P   (HMP - MP)": ("concat_HMP", "concat_MP"),
        "MFCC | H+P     (HMP - HP)": ("concat_HMP", "concat_HP"),
        "prosody | H+M  (HMP - HM)": ("concat_HMP", "concat_HM"),
        # add-one-stream to HuBERT
        "MFCC | H       (HM - H)": ("concat_HM", "concat_H"),
        "prosody | H    (HP - H)": ("concat_HP", "concat_H"),
        "M+P | H        (HMP - H)": ("concat_HMP", "concat_H"),
        # sequence model vs pooled MLP on HuBERT alone
        "concat_H - E2": ("concat_H", "E2"),
        # without HuBERT
        "prosody | M    (MP - M)": ("concat_MP", "concat_M"), "MFCC | P (MP - P)": ("concat_MP", "concat_P"),
        "concat_M - E1-SVM": ("concat_M", "E1-SVM"),
        # DisVoice
        "DisVoice | HMP (dv - HMP)": ("concat_HMP_dv", "concat_HMP"),
        "indicators | dv (dvind - dv)": ("concat_HMP_dvind", "concat_HMP_dv"),
    }
    S["paired_test_macro_f1"] = {name: paired(F[a], F[b]) for name, (a, b) in comps.items()}
    S["paired_test_UA"] = {name: paired(U[a], U[b]) for name, (a, b) in comps.items()}
    subset_mean = {v.split("_")[1]: S["variants"][v]["test"]["macro_f1"]["mean"] for v in variants
                   if v.startswith("concat_") and len(v.split("_")) == 2}
    S["shapley_test_macro_f1"] = {"base_v_empty": 1 / 6, "values": shapley(subset_mean, 1 / 6),
                                  "per_fold": {k: shapley({s: F[f"concat_{s}"][k] for s in SUBSETS}, 1 / 6) for k in folds},
                                  "note": "v(S) = test macro-F1 of the concat model on stream subset S (mean over folds); "
                                          "v(empty) = 1/6 (chance); values sum to v(HMP) - 1/6"}
    S["layer_per_fold"] = {k: S["folds"][k]["layer"] for k in folds}
    (root / "summary.json").write_text(json.dumps(S, indent=1))
    rows = [{"variant": v, "fold": k, "seed": f["seed"], **{f"test_{m}": f["test"][m] for m in METRICS},
             "val_macro_f1": f["val"]["macro_f1"], "best_epoch": f["best_epoch"], "train_seconds": f["train_seconds"],
             "n_params": f["n_params"], "layer": f["layer"], "dropout": f["hp"]["dropout"], "lr": f["hp"]["lr"]}
            for v in variants for k in folds
            for f in [json.loads((root / f"fold{k}" / f"D_{v}_seed{s}.json").read_text()) for s in CFG.final_seeds]]
    pd.DataFrame(rows).to_csv(root / "runs.csv", index=False)
    for v in variants:
        t = S["variants"][v]["test"]
        print(f"{ds} {v:17s} " + "  ".join(f"{m} {t[m]['mean']:.3f}±{t[m]['std']:.3f}" for m in ("macro_f1", "UA", "accuracy", "macro_precision")))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["RAVDESS"])
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--pilot", action="store_true", help="fold selection only (no test) + checks")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    (PATHS.configs / "e3_concat_fusion.json").write_text(json.dumps(asdict(CFG), indent=1))
    for ds in args.datasets:
        if args.summarise:
            summarise_dataset(ds)
            continue
        store = FeatureStore.open(ds, index)
        dc = DataCache(store)
        print(f"[E3] {ds}: device {device()}, n_sa_blocks {CFG.n_sa_blocks}, RAM {process_ram_mib()}", flush=True)
        for k in (args.folds if args.folds is not None else range(DEFAULT_FOLDS[ds])):
            fr = fold_rows(store, index, ds, k)
            if args.pilot:
                pilot(ds, store, dc, fr)
            else:
                run_fold(ds, store, dc, fr)
        print(f"[E3] {ds}: done. GPU peak {gpu_peak_mib()} MiB, RAM {process_ram_mib()}", flush=True)


# --------------------------------------------------------------------------- pilot
def alignment_check(store, data, fr, n=24, seed=0) -> dict:
    """Real-data check: batched, padded, standardised frames == the cache frames of the same recording."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(np.concatenate([fr.rows["train"], fr.rows["val"]]), n, replace=False)
    b = data.batch(rows)
    worst = {}
    for j, r in enumerate(rows):
        T = int(store.manifest.n_frames[r])
        ref = {"hubert": np.asarray(store.frames("hubert", r)[:, data.layer, :], np.float32),
               "mfcc": np.asarray(store.frames("mfcc", r), np.float32),
               "prosody": np.asarray(store.frames("prosody_frames", r), np.float32)}
        assert int(b["mask"][j].sum()) == T and bool(b["mask"][j, :T].all())
        for s, x in ref.items():
            want = data.std[s].transform(x)
            got = b[s][j, :T].cpu().numpy()
            worst[s] = max(worst.get(s, 0.0), float(np.abs(got - want).max()))
            assert float(b[s][j, T:].abs().sum()) == 0.0, "padding not zero"
    # train-only statistics: recompute the MFCC mean from the training frames directly
    tr = np.concatenate([np.asarray(store.frames("mfcc", r), np.float64) for r in fr.rows["train"]])
    mfcc_mean_diff = float(np.abs(tr.mean(0) - data.std["mfcc"].mean_).max())
    return {"rows_checked": int(n), "max_abs_diff_vs_cache": worst, "padding_zero": True,
            "mfcc_mean_vs_train_frames_max_abs_diff": mfcc_mean_diff,
            "prosody_passthrough": {"mean": data.std["prosody"].mean_[[1, 2]].tolist(),
                                    "std": data.std["prosody"].std_[[1, 2]].tolist()}}


def pilot(ds, store, dc, fr):
    """Fold-level pilot WITHOUT test predictions: alignment, memory, stability, timing."""
    d = PATHS.results / "E3" / ds / f"fold{fr.fold}"
    t0 = time.perf_counter()
    sel, data, rhash, S = select_fold(ds, store, dc, fr, d)
    data.fit(fr.rows["train"], None)
    al = alignment_check(store, data, fr)
    runs = [r for v in CFG.tuned for r in S[v]]
    stab = [{"variant": r["variant"], "hp": r["hp"], "val_macro_f1": r["val_macro_f1"], "best_epoch": r["best_epoch"],
             "epochs_run": r["epochs_run"], "train_seconds": r["train_seconds"],
             "sec_per_epoch": r["train_seconds"] / r["epochs_run"],
             "loss_first_last": [r["history"][0]["train_loss"], r["history"][-1]["train_loss"]],
             "grad_norm_max": max(h["grad_norm_max"] for h in r["history"]),
             "frac_steps_clipped": float(np.mean([h["frac_clipped"] for h in r["history"]])),
             "gpu_peak_mib": r["gpu_peak_mib"], "ram_peak_mib": r["process_ram_mib"]["peak"]} for r in runs]
    t_concat = np.mean([s["train_seconds"] for s in stab if s["variant"] == "concat_HMP"])
    t_sa = np.mean([s["train_seconds"] for s in stab if s["variant"] == "sa_HMP"])
    n_concat_d = 3 * (1 + len(CFG.ablations))
    per_fold = 4 * t_concat + 4 * t_sa + 3 * t_sa + n_concat_d * t_concat
    out = {"dataset": ds, "fold": fr.fold, "run_hash": rhash, "layer_info": sel["layer_info"],
           "selected": sel["selected"], "selected_val_macro_f1": sel["selected_val_macro_f1"],
           "n_params": sel["n_params"], "n_params_fusion": sel["n_params_fusion"],
           "e4_reference_fusion_params": sel["e4_reference_fusion_params"], "alignment": al, "runs": stab,
           "hubert_layer_load_seconds": dc.load_seconds, "pilot_seconds": time.perf_counter() - t0,
           "estimate": {"train_seconds_per_run": {"concat_HMP": t_concat, "sa_HMP": t_sa},
                        "runs_per_fold": 8 + 3 * 2 + 3 * len(CFG.ablations),
                        "est_minutes_per_fold_upper": per_fold / 60,
                        "est_hours_all_folds": per_fold * DEFAULT_FOLDS[ds] / 3600,
                        "note": "ablations without HuBERT are cheaper than concat_HMP: upper bound"},
           "test_predictions_made": False}
    write_atomic(PATHS.results / "E3" / f"pilot_{ds}.json", out, indent=1)
    print(json.dumps({k: out[k] for k in ("layer_info", "selected", "selected_val_macro_f1", "n_params", "alignment",
                                          "estimate", "hubert_layer_load_seconds")}, indent=1, default=str))
    for s in stab:
        print(s)


if __name__ == "__main__":
    main()
