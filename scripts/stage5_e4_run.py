"""
E4 driver: bidirectional cross-attention fusion (HuBERT + MFCC + prosody) and its comparators,
speaker-independent folds. Revised protocol approved 2026-11-06 (reports/stage5_models.md section 6).

Per (dataset, fold):
  L  HuBERT layer: identical rule and source files as E3 (fold's own E2 phase-A VALIDATION results).
  S  search, validation macro-F1, seed 0, fold WITHOUT its test partition. Every tuned model gets
     the SAME 8 candidates: dropout {0.2, 0.4} x lr {1e-3, 3e-4, 1e-4, 3e-5}, max 150 epochs
     (patience 10). Grid order: dropout outer, lr 1e-3 -> 3e-5 inner (ties -> earlier candidate,
     which keeps E3's tie-breaking among the E3 cells).
       xattn_HMP   E4 cross-attention fusion                 (model under test)
       sa_HMP      parameter-matched self-attention control  (E3 code, 4 blocks)
       concat_HMP  E3 concatenation fusion, re-searched      (primary comparator)
       concat_H    E3 code, HuBERT stream only               (HuBERT-only frame model)
  -- selection.json is written here, before any test prediction --
  G  guard before any test prediction: the E3-comparable cells of THIS fold must reproduce E3's
     saved S runs exactly; otherwise the run stops (no test prediction, no restart).
  D  final, seeds 0, 1, 2, checkpoint -> reload -> predict TEST once:
       the 4 tuned models (own selection) and the ablations xattn_{HM, HP, H_MPcat} (xattn_HMP's hp)
Resumable: every run writes its own JSON atomically; files with the same run hash are reused, so
the fold-0 S runs of the pilot are NOT retrained (E4Config must not change, it defines the hash).

E3 reproduction: there is no LR scheduler and no E3 S run reached its 50-epoch cap (max 43), so
concat_HMP / sa_HMP at lr {1e-3, 3e-4} must reproduce E3's S runs exactly (validation F1, best
epoch, epochs run, per-epoch history). The pilot compares them with results/E3 (read-only).

  python scripts/stage5_e4_run.py --datasets RAVDESS CREMA-D --folds 0 --pilot   # validation only
  python scripts/stage5_e4_run.py --datasets RAVDESS                             # full (one dataset per process)
  python scripts/stage5_e4_run.py --datasets RAVDESS CREMA-D --summarise
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluation.metrics import METRICS, compute_metrics, summarise  # noqa: E402
from src.preprocessing.dataset_index import CLASSES  # noqa: E402
from src.features.store import FeatureStore  # noqa: E402
from src.models.e3_fusion import (SUBSETS, E3Fusion, e3_fusion_params, e4_reference_fusion_params,  # noqa: E402
                                  matched_sa_blocks, n_params)
from src.models.e4_cross_attention import STRUCTURES, E4CrossAttention, e4_fusion_params  # noqa: E402
from src.preprocessing.splits import DEFAULT_FOLDS, split_path  # noqa: E402
from src.training.data import fold_rows, without_test  # noqa: E402
from src.training.seq_trainer import SeqTrainConfig, predict_rows, train_sequence_classifier  # noqa: E402
from src.training.sequence_data import SequenceData, load_hubert_layer  # noqa: E402
from src.training.trainer import device  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402
from src.utils.resources import gpu_peak_mib, process_ram_mib  # noqa: E402

# pure helpers shared with E3 (layer rule, atomic writes, tie-breaking, alignment check): imported, not copied
_spec = importlib.util.spec_from_file_location("e3drv", ROOT / "scripts" / "stage5_e3_run.py")
E3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E3)
fold_layer, write_atomic, best_of, alignment_check = E3.fold_layer, E3.write_atomic, E3.best_of, E3.alignment_check
sign_flip_p, paired = E3.sign_flip_p, E3.paired

TUNED = ("xattn_HMP", "sa_HMP", "concat_HMP", "concat_H")
ABLATIONS = ("xattn_HM", "xattn_HP", "xattn_H_MPcat")
E3_COMPARABLE = ("concat_HMP", "sa_HMP")          # tuned in E3 on lr {1e-3, 3e-4}


@dataclass(frozen=True)
class E4Config:
    d_model: int = 128
    n_heads: int = 4
    d_ffn: int = 256
    head_hidden: int = 256
    n_sa_blocks: int = field(default_factory=lambda: matched_sa_blocks(128, 4, 256))
    grid_dropout: tuple = (0.2, 0.4)
    grid_lr: tuple = (1e-3, 3e-4, 1e-4, 3e-5)
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 150
    patience: int = 10
    grad_clip: float = 1.0
    bucket: int = 20
    class_weighted: bool = True
    selection_seed: int = 0
    final_seeds: tuple = (0, 1, 2)
    tuned: tuple = TUNED
    ablations: tuple = ABLATIONS
    e3_reproduction_lrs: tuple = (1e-3, 3e-4)
    layer_rule: str = ("best single hidden state of the fold's own E2 phase-A validation macro-F1 "
                       "(max over poolings; ties -> lower layer)  [identical to E3]")
    selection_metric: str = "validation macro-F1 (ties -> earlier candidate; dropout outer, lr descending inner)"
    positional_encoding: str = ("none (as in the E3 control): cross-attention is content-based and does not encode "
                                "the shared frame positions; the model is invariant to joint time permutation")


CFG = E4Config()


# --------------------------------------------------------------------------- variants
def parse_variant(name: str) -> dict:
    kind, rest = name.split("_", 1)
    if kind == "xattn" and rest in STRUCTURES:
        return {"name": name, "kind": "xattn", "structure": rest, "streams": STRUCTURES[rest]}
    if name in ("sa_HMP", "concat_HMP", "concat_H"):
        return {"name": name, "kind": kind, "streams": SUBSETS[rest],
                "n_sa_blocks": CFG.n_sa_blocks if kind == "sa" else 0}
    raise ValueError(name)


def hp_source(name: str) -> str:
    return "xattn_HMP" if name.startswith("xattn_") else name


def build_model(spec: dict, hp: dict):
    if spec["kind"] == "xattn":
        return E4CrossAttention(spec["structure"], CFG.d_model, hp["dropout"], CFG.head_hidden, CFG.n_heads,
                                CFG.d_ffn).to(device())
    # exactly the E3 constructor call (stage5_e3_run.build_model), so seeds give identical initial weights
    return E3Fusion(spec["streams"], CFG.d_model, hp["dropout"], CFG.head_hidden, spec["n_sa_blocks"],
                    CFG.n_heads, CFG.d_ffn, disvoice_dim=0).to(device())


def fusion_params(model) -> int:
    return e4_fusion_params(model) if isinstance(model, E4CrossAttention) else e3_fusion_params(model)


def train_cfg(hp: dict, seed: int) -> SeqTrainConfig:
    return SeqTrainConfig(lr=hp["lr"], weight_decay=CFG.weight_decay, batch_size=CFG.batch_size,
                          max_epochs=CFG.max_epochs, patience=CFG.patience, class_weighted=CFG.class_weighted,
                          seed=seed, grad_clip=CFG.grad_clip, bucket=CFG.bucket)


def candidates() -> list[dict]:
    return [{"dropout": dr, "lr": lr} for dr, lr in product(CFG.grid_dropout, CFG.grid_lr)]


def s_name(variant: str, hp: dict) -> str:
    return f"S_{variant}_d{hp['dropout']}_lr{hp['lr']}.json"


def run_hash(ds, store, layer):
    split_sha = hashlib.sha256(split_path(ds).read_bytes()).hexdigest()
    blob = json.dumps({"e4": asdict(CFG), "cache": store.meta["cache_id"], "split": split_sha, "layer": layer},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- phase S
def run_candidate(data: SequenceData, fr, variant: str, hp: dict, seed: int) -> dict:
    """Train on TRAIN speakers, early-stop / score on VALIDATION speakers. No test argument."""
    spec = parse_variant(variant)
    data.fit(fr.rows["train"], None)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    res = train_sequence_classifier(lambda: build_model(spec, hp), data, (fr.rows["train"], fr.labels["train"]),
                                    (fr.rows["val"], fr.labels["val"]), train_cfg(hp, seed))
    model = res["model"]
    val = compute_metrics(fr.labels["val"], predict_rows(model, data, fr.rows["val"]))
    return {"variant": variant, "hp": hp, "seed": seed, "layer": data.layer, "val_macro_f1": res["best_val_macro_f1"],
            "val": val, "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
            "hit_max_epochs": res["epochs_run"] >= CFG.max_epochs,
            "train_seconds": res["train_seconds"], "n_params": n_params(model), "n_params_fusion": fusion_params(model),
            "gpu_peak_mib": gpu_peak_mib(), "process_ram_mib": process_ram_mib(), "history": res["history"]}


def search_one(data, fr_sel, variant, hp, d: Path, rhash) -> dict:
    path = d / s_name(variant, hp)
    if path.exists():
        r = json.loads(path.read_text())
        if r.get("run_hash") == rhash:
            return r
    r = run_candidate(data, fr_sel, variant, hp, CFG.selection_seed)
    r["run_hash"] = rhash
    write_atomic(path, r)
    print(f"    S {variant:10s} d{hp['dropout']} lr{hp['lr']:<7g} val F1 {r['val_macro_f1']:.4f}  best ep {r['best_epoch']:3d} "
          f"run {r['epochs_run']:3d}{' CAP' if r['hit_max_epochs'] else ''}  {r['train_seconds']:6.1f}s  "
          f"GPU {r['gpu_peak_mib']:.0f} MiB  RAM {r['process_ram_mib']}", flush=True)
    return r


class DataCache:
    def __init__(self, store):
        self.store, self.data, self.load_seconds = store, None, {}

    def get(self, layer):
        if self.data is None or self.data.layer != layer:
            self.data = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"  [stage] loading HuBERT layer {layer}", flush=True)     # read by the RAM guard (1 s sampling)
            t = time.perf_counter()
            H = load_hubert_layer(self.store, layer)
            self.data = SequenceData(self.store, layer, device(), hubert=H, need_disvoice=False)
            self.load_seconds[layer] = time.perf_counter() - t
            print(f"    loaded HuBERT layer {layer}: {H.shape} fp16 in {self.load_seconds[layer]:.0f}s, "
                  f"RAM {process_ram_mib()}", flush=True)
        return self.data


def search_order():
    """E3-comparable cells first (reproduction is known within minutes), then the rest of the grid.
    The order only affects run time: every candidate is an independent seeded run."""
    first = [(v, hp) for v in E3_COMPARABLE for hp in candidates() if hp["lr"] in CFG.e3_reproduction_lrs]
    rest = [(v, hp) for v in CFG.tuned for hp in candidates() if (v, hp) not in first]
    return first + rest


def select_fold(ds, store, dc, fr, d: Path):
    layer_info = fold_layer(ds, fr.fold)
    rhash = run_hash(ds, store, layer_info["layer"])
    data = dc.get(layer_info["layer"])
    t0 = time.perf_counter()
    fr_sel = without_test(fr)
    done = {}
    print(f"  [stage] {ds} fold {fr.fold}: phase S (search; saved runs with run hash {rhash} are reused)", flush=True)
    for v, hp in search_order():
        done[(v, hp["dropout"], hp["lr"])] = search_one(data, fr_sel, v, hp, d, rhash)
    S = {v: [done[(v, hp["dropout"], hp["lr"])] for hp in candidates()] for v in CFG.tuned}   # canonical grid order
    chosen = {v: best_of(S[v]) for v in CFG.tuned}
    sel = {"run_hash": rhash, "dataset": ds, "fold": fr.fold, "layer": layer_info["layer"], "layer_info": layer_info,
           "selected": {v: chosen[v]["hp"] for v in CFG.tuned},
           "selected_val_macro_f1": {v: chosen[v]["val_macro_f1"] for v in CFG.tuned},
           "search": {v: [{**r["hp"], "val_macro_f1": r["val_macro_f1"], "best_epoch": r["best_epoch"],
                           "epochs_run": r["epochs_run"], "hit_max_epochs": r["hit_max_epochs"],
                           "train_seconds": r["train_seconds"]} for r in S[v]] for v in CFG.tuned},
           "n_params": {v: chosen[v]["n_params"] for v in CFG.tuned},
           "n_params_fusion": {v: chosen[v]["n_params_fusion"] for v in CFG.tuned},
           "e4_reference_fusion_params": e4_reference_fusion_params(CFG.d_model, CFG.n_heads, CFG.d_ffn),
           "selection_seconds": time.perf_counter() - t0}
    prev = json.loads((d / "selection.json").read_text()) if (d / "selection.json").exists() else None
    if not (prev and prev["run_hash"] == rhash and prev["selected"] == sel["selected"]):
        write_atomic(d / "selection.json", sel, indent=1)        # written BEFORE any test prediction
    return sel, data, rhash, S


# --------------------------------------------------------------------------- phase D
def save_checkpoint(path, model, data, variant, hp, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "variant": variant, "hp": hp,
                "spec": parse_variant(variant), "e4_config": asdict(CFG), "layer": data.layer,
                "standardizers": {s: {"mean": z.mean_, "std": z.std_, "passthrough": z.passthrough}
                                  for s, z in data.std.items()}, **meta}, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location=device(), weights_only=False)
    model = build_model(ck["spec"], ck["hp"])
    model.load_state_dict(ck["state_dict"])
    return model, ck


def final_and_test(ds, store, data, fr, variant, hp, seed, rhash, selection_val_f1, layer_info):
    """Phase D: train (train speakers), checkpoint, reload, predict TEST once (from the reloaded checkpoint)."""
    tag = f"fold{fr.fold}_{variant}_seed{seed}"
    spec = parse_variant(variant)
    data.fit(fr.rows["train"], None)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    res = train_sequence_classifier(lambda: build_model(spec, hp), data, (fr.rows["train"], fr.labels["train"]),
                                    (fr.rows["val"], fr.labels["val"]), train_cfg(hp, seed))
    model = res["model"]
    if selection_val_f1 is not None and seed == CFG.selection_seed and abs(res["best_val_macro_f1"] - selection_val_f1) > 1e-12:
        raise RuntimeError(f"{tag}: retraining did not reproduce the selection run "
                           f"({res['best_val_macro_f1']} vs {selection_val_f1})")
    ck = PATHS.checkpoints / "E4" / ds / f"{tag}.pt"
    save_checkpoint(ck, model, data, variant, hp, {"dataset": ds, "fold": fr.fold, "seed": seed, "run_hash": rhash,
                                                   "feature_cache": store.meta["cache_id"] if hasattr(store, "meta") else None,
                                                   "best_epoch": res["best_epoch"],
                                                   "train_cfg": train_cfg(hp, seed).to_dict()})
    model2, ck2 = load_checkpoint(ck)
    for s, z in data.std.items():
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
            "layer_info": layer_info, "n_params": n_params(model), "n_params_fusion": fusion_params(model),
            "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
            "hit_max_epochs": res["epochs_run"] >= CFG.max_epochs, "train_seconds": res["train_seconds"],
            "class_weights": res["class_weights"],
            "test_inference_ms_per_recording": 1000 * infer / len(fr.rows["test"]),
            "val": compute_metrics(fr.labels["val"], pv), "test": compute_metrics(fr.labels["test"], pt),
            "test_pred": pt.tolist(), "test_rows": fr.rows["test"].tolist(), "checkpoint": str(ck),
            "gpu_peak_mib": gpu_peak_mib(), "process_ram_mib": process_ram_mib(), "history": res["history"]}


def guard_e3_reproduction(ds, fold, S):
    """Stop BEFORE any test prediction if the E3-comparable cells do not reproduce E3 exactly."""
    rep = e3_reproduction(ds, fold, S)
    bad = [f"{c['variant']} {c['hp']}" for c in rep if not c["exact"]]
    if len(rep) != 2 * len(CFG.grid_dropout) * len(CFG.e3_reproduction_lrs) or bad:
        raise RuntimeError(f"{ds} fold {fold}: E3 reproduction discrepancy {bad} (n={len(rep)}) - stopped before phase D")
    return rep


def run_fold(ds, store, dc, fr, check_e3: bool = True):
    d = PATHS.results / "E4" / ds / f"fold{fr.fold}"
    sel, data, rhash, S = select_fold(ds, store, dc, fr, d)
    saved = json.loads((d / "selection.json").read_text())
    if saved["run_hash"] != rhash or saved["selected"] != sel["selected"]:
        raise RuntimeError(f"{ds} fold {fr.fold}: selection.json does not match the search - stopped")
    if check_e3:
        print(f"  [stage] {ds} fold {fr.fold}: guard - E3 reproduction check", flush=True)
        guard_e3_reproduction(ds, fr.fold, S)
    for variant in CFG.tuned + CFG.ablations:
        hp = saved["selected"][hp_source(variant)]
        sel_f1 = saved["selected_val_macro_f1"][variant] if variant in CFG.tuned else None
        for seed in CFG.final_seeds:
            out = d / f"D_{variant}_seed{seed}.json"
            if out.exists() and json.loads(out.read_text()).get("run_hash") == rhash:
                continue
            print(f"  [stage] {ds} fold {fr.fold}: phase D {variant} seed {seed} (training)", flush=True)
            r = final_and_test(ds, store, data, fr, variant, hp, seed, rhash, sel_f1, saved["layer_info"])
            r["run_hash"] = rhash
            write_atomic(out, r)
            print(f"    D {variant:14s} seed {seed} hp {hp}  best ep {r['best_epoch']:3d} run {r['epochs_run']:3d}  "
                  f"{r['train_seconds']:6.1f}s  GPU {r['gpu_peak_mib']:.0f} MiB  RAM {r['process_ram_mib']}", flush=True)
        f1 = [json.loads((d / f"D_{variant}_seed{s}.json").read_text())["test"]["macro_f1"] for s in CFG.final_seeds]
        print(f"  fold{fr.fold} L{saved['layer']} {variant:14s} hp {hp}  test F1 seeds {np.round(f1, 3).tolist()}", flush=True)


# --------------------------------------------------------------------------- swap diagnostic (supplementary)
def derangement(n: int, seed: int) -> np.ndarray:
    """Permutation of range(n) without fixed points (seeded; rejection sampling)."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not np.any(p == np.arange(n)):
            return p


def swapped_batch(data: SequenceData, rows, donors, streams=("mfcc", "prosody")) -> dict:
    """Batch of `rows` whose `streams` are replaced by the frames of `donors` (same position in the
    batch), time-stretched to the recipient's length by nearest-index mapping t -> floor(t*L_d/L_r).
    HuBERT and the mask stay those of the recipient. donors == rows reproduces data.batch(rows)."""
    b = data.batch(rows)
    db = data.batch(donors)
    L = data.length[torch.as_tensor(np.asarray(rows), dtype=torch.long, device=data.dev)]
    Ld = data.length[torch.as_tensor(np.asarray(donors), dtype=torch.long, device=data.dev)]
    T = b["mask"].shape[1]
    t = torch.arange(T, device=data.dev)[None, :]
    src = torch.minimum((t * Ld[:, None]) // L[:, None], Ld[:, None] - 1)            # (B, T) donor frame index
    for s in streams:
        x = torch.gather(db[s], 1, src[..., None].expand(-1, -1, db[s].shape[-1]))
        b[s] = torch.where(b["mask"][..., None], x, torch.zeros((), device=data.dev))
    return b


@torch.no_grad()
def predict_swapped(model, data, rows, donors, batch_size=64) -> np.ndarray:
    model.eval()
    rows, donors = np.asarray(rows), np.asarray(donors)
    order = np.argsort(data.lengths_np[rows], kind="stable")
    pred = np.zeros(len(rows), dtype=np.int64)
    for j in range(0, len(rows), batch_size):
        idx = order[j:j + batch_size]
        pred[idx] = model(swapped_batch(data, rows[idx], donors[idx])).argmax(1).cpu().numpy()
    return pred


# --------------------------------------------------------------------------- summary
def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


KEY_COMPARISONS = {"E4 - concat (primary)": ("xattn_HMP", "concat_HMP"),
                   "E4 - control": ("xattn_HMP", "sa_HMP"),
                   "E4 - HuBERT-only": ("xattn_HMP", "concat_H")}
SECONDARY = {
    "control - concat": ("sa_HMP", "concat_HMP"),
    "concat - HuBERT-only (M+P given H, fair search)": ("concat_HMP", "concat_H"),
    "control - HuBERT-only": ("sa_HMP", "concat_H"),
    "concat (E4 search) - concat (E3)": ("concat_HMP", "E3:concat_HMP"),
    "control (E4 search) - control (E3)": ("sa_HMP", "E3:sa_HMP"),
    "HuBERT-only (E4 search) - H (E3, concat hp)": ("concat_H", "E3:concat_H"),
    "E4 - E2": ("xattn_HMP", "E2"), "E4 - E1-SVM": ("xattn_HMP", "E1-SVM"),
    "concat - E2": ("concat_HMP", "E2"), "HuBERT-only - E2": ("concat_H", "E2"),
    "prosody | E4 (HMP - HM)": ("xattn_HMP", "xattn_HM"),
    "MFCC | E4 (HMP - HP)": ("xattn_HMP", "xattn_HP"),
    "stage-1 MFCC<->prosody attention (HMP - H_MPcat)": ("xattn_HMP", "xattn_H_MPcat"),
    "cross-attention vs concat, same inputs (H_MPcat - concat)": ("xattn_H_MPcat", "concat_HMP"),
    "H<->M cross-attention - HuBERT-only": ("xattn_HM", "concat_H"),
    "H<->P cross-attention - HuBERT-only": ("xattn_HP", "concat_H"),
}


def summarise_dataset(ds):
    root = PATHS.results / "E4" / ds
    folds = sorted(int(p.name[4:]) for p in root.glob("fold*") if p.is_dir())
    variants = CFG.tuned + CFG.ablations
    lo, hi = min(CFG.grid_lr), max(CFG.grid_lr)
    S = {"dataset": ds, "config": asdict(CFG), "folds": {}, "variants": {}}
    fold_means = {v: {} for v in variants}
    val_means = {v: {} for v in variants}
    finals_all = {v: [] for v in variants}
    s_cap = []
    for k in folds:
        sel = json.loads((root / f"fold{k}" / "selection.json").read_text())
        s_cap += [f"fold{k} {v} {c}" for v in CFG.tuned for c in sel["search"][v] if c["hit_max_epochs"]]
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
            "confusion_summed_over_folds_and_seeds": cm.tolist(),
            "n_params": int(fin[0]["n_params"]), "n_params_fusion": int(fin[0]["n_params_fusion"]),
            "best_epoch_mean": float(np.mean([f["best_epoch"] for f in fin])), "epochs_run_max": int(max(f["epochs_run"] for f in fin)),
            "train_seconds_mean": float(np.mean([f["train_seconds"] for f in fin])),
            "train_seconds_median": float(np.median([f["train_seconds"] for f in fin])),
            "sec_per_epoch_median": float(np.median([f["train_seconds"] / f["epochs_run"] for f in fin])),
            "infer_ms_mean": float(np.mean([f["test_inference_ms_per_recording"] for f in fin])),
            "gpu_peak_mib_max": float(max(f["gpu_peak_mib"] or 0 for f in fin)),
            "ram_peak_mib_max": float(max(f["process_ram_mib"]["peak"] or 0 for f in fin))}
    S["selection_edges"] = {v: {"lower_edge_folds": sum(S["folds"][k]["selected"][v]["lr"] == lo for k in folds),
                                "upper_edge_folds": sum(S["folds"][k]["selected"][v]["lr"] == hi for k in folds),
                                "n_folds": len(folds),
                                "selected_lr_per_fold": [S["folds"][k]["selected"][v]["lr"] for k in folds]} for v in CFG.tuned}
    S["cap_hits"] = {"S": s_cap, "D": [f"fold{f['fold']} {f['variant']} seed{f['seed']}" for v in variants
                                       for f in finals_all[v] if f["hit_max_epochs"]]}
    F = {v: pd.Series({k: fold_means[v][k]["macro_f1"] for k in folds}) for v in variants}
    U = {v: pd.Series({k: fold_means[v][k]["UA"] for k in folds}) for v in variants}
    e2s = json.loads((PATHS.results / "E2" / ds / "summary.json").read_text())
    F["E2"] = pd.Series({k: e2s["folds"][str(k)]["test_mean_over_seeds"]["macro_f1"] for k in folds})
    U["E2"] = pd.Series({k: e2s["folds"][str(k)]["test_mean_over_seeds"]["UA"] for k in folds})
    e1 = pd.read_csv(PATHS.results / "E1" / ds / "runs.csv")
    e1 = e1[e1.variant == "headline"].groupby(["model", "fold"])[["test_macro_f1", "test_UA"]].mean()
    F["E1-SVM"] = e1.loc["svm"].test_macro_f1.reindex(folds)
    U["E1-SVM"] = e1.loc["svm"].test_UA.reindex(folds)
    e3s = json.loads((PATHS.results / "E3" / ds / "summary.json").read_text())
    for v in ("concat_HMP", "sa_HMP", "concat_H"):
        F[f"E3:{v}"] = pd.Series({k: e3s["folds"][str(k)]["variants"][v]["test_mean_over_seeds"]["macro_f1"] for k in folds})
        U[f"E3:{v}"] = pd.Series({k: e3s["folds"][str(k)]["variants"][v]["test_mean_over_seeds"]["UA"] for k in folds})
    comps = {**KEY_COMPARISONS, **SECONDARY}
    S["paired_test_macro_f1"] = {n: paired(F[a], F[b]) for n, (a, b) in comps.items()}
    S["paired_test_UA"] = {n: paired(U[a], U[b]) for n, (a, b) in comps.items()}
    S["holm_key_comparisons_macro_f1"] = holm({n: S["paired_test_macro_f1"][n]["sign_flip_p"] for n in KEY_COMPARISONS})
    S["layer_per_fold"] = {k: S["folds"][k]["layer"] for k in folds}
    (root / "summary.json").write_text(json.dumps(S, indent=1))
    rows = [{"variant": v, "fold": k, "seed": f["seed"], **{f"test_{m}": f["test"][m] for m in METRICS},
             "val_macro_f1": f["val"]["macro_f1"], "best_epoch": f["best_epoch"], "epochs_run": f["epochs_run"],
             "train_seconds": f["train_seconds"], "n_params": f["n_params"], "layer": f["layer"],
             "dropout": f["hp"]["dropout"], "lr": f["hp"]["lr"]}
            for v in variants for k in folds
            for f in [json.loads((root / f"fold{k}" / f"D_{v}_seed{s}.json").read_text()) for s in CFG.final_seeds]]
    pd.DataFrame(rows).to_csv(root / "runs.csv", index=False)
    for v in variants:
        t = S["variants"][v]["test"]
        print(f"{ds} {v:14s} " + "  ".join(f"{m} {t[m]['mean']:.3f}±{t[m]['std']:.3f}" for m in ("macro_f1", "UA", "accuracy", "macro_precision")))
    return S


DECISION_MIN_BETTER = {"RAVDESS": 5, "CREMA-D": 4}


def decision(summaries: dict) -> dict:
    """Pre-registered rule (report section 6.6): 'E4 improves on X' only if mean delta > 0 on BOTH datasets,
    delta > 0 in >= 5/6 RAVDESS and >= 4/5 CREMA-D folds, and mean delta > E4's mean within-fold seed std."""
    out = {}
    for name in KEY_COMPARISONS:
        per = {}
        for ds, S in summaries.items():
            p = S["paired_test_macro_f1"][name]
            seed_sd = S["variants"]["xattn_HMP"]["seed_std_macro_f1_mean_within_fold"]
            per[ds] = {"mean_delta": p["mean"], "n_better": p["n_better"], "n": p["n"], "p": p["sign_flip_p"],
                       "p_holm": S["holm_key_comparisons_macro_f1"][name], "e4_seed_std": seed_sd,
                       "criteria": {"mean_positive": p["mean"] > 0,
                                    "enough_folds_better": p["n_better"] >= DECISION_MIN_BETTER[ds],
                                    "exceeds_seed_std": p["mean"] > seed_sd}}
        ok = len(per) == 2 and all(all(v["criteria"].values()) for v in per.values())
        out[name] = {"per_dataset": per, "verdict": "E4 improves" if ok else "no consistent improvement"}
    return out


# --------------------------------------------------------------------------- pilot
def e3_reproduction(ds, fold, S) -> list[dict]:
    """Compare the E3-comparable cells with the saved E3 S runs (read-only). Exact equality required."""
    out = []
    for v in E3_COMPARABLE:
        for r in S[v]:
            if r["hp"]["lr"] not in CFG.e3_reproduction_lrs:
                continue
            p = PATHS.results / "E3" / ds / f"fold{fold}" / s_name(v, r["hp"])
            e3 = json.loads(p.read_text())
            chk = {"variant": v, "hp": r["hp"], "e3_file": p.relative_to(PATHS.project).as_posix(),
                   "val_macro_f1": [e3["val_macro_f1"], r["val_macro_f1"]],
                   "best_epoch": [e3["best_epoch"], r["best_epoch"]],
                   "epochs_run": [e3["epochs_run"], r["epochs_run"]],
                   "history_val_f1_identical": [h["val_macro_f1"] for h in e3["history"]] == [h["val_macro_f1"] for h in r["history"]],
                   "history_train_loss_identical": [h["train_loss"] for h in e3["history"]] == [h["train_loss"] for h in r["history"]],
                   "val_confusion_identical": e3["val"]["confusion"] == r["val"]["confusion"]}
            chk["exact"] = bool(chk["val_macro_f1"][0] == chk["val_macro_f1"][1] and chk["best_epoch"][0] == chk["best_epoch"][1]
                                and chk["epochs_run"][0] == chk["epochs_run"][1] and chk["history_val_f1_identical"]
                                and chk["history_train_loss_identical"] and chk["val_confusion_identical"])
            out.append(chk)
    return out


def pilot(ds, store, dc, fr):
    """Fold-level pilot WITHOUT test predictions: reproduction, epochs, time, memory, alignment."""
    d = PATHS.results / "E4" / ds / f"fold{fr.fold}"
    t0 = time.perf_counter()
    sel, data, rhash, S = select_fold(ds, store, dc, fr, d)
    data.fit(fr.rows["train"], None)
    al = alignment_check(store, data, fr)
    runs = [{"variant": r["variant"], "hp": r["hp"], "val_macro_f1": r["val_macro_f1"], "best_epoch": r["best_epoch"],
             "epochs_run": r["epochs_run"], "hit_max_epochs": r["hit_max_epochs"], "train_seconds": r["train_seconds"],
             "sec_per_epoch": r["train_seconds"] / r["epochs_run"],
             "loss_first_last": [r["history"][0]["train_loss"], r["history"][-1]["train_loss"]],
             "grad_norm_max": max(h["grad_norm_max"] for h in r["history"]),
             "frac_steps_clipped": float(np.mean([h["frac_clipped"] for h in r["history"]])),
             "gpu_peak_mib": r["gpu_peak_mib"], "ram_peak_mib": r["process_ram_mib"]["peak"]}
            for v in CFG.tuned for r in S[v]]
    rep = e3_reproduction(ds, fr.fold, S)
    lo, hi = min(CFG.grid_lr), max(CFG.grid_lr)
    t_sel = {v: next(r["train_seconds"] for r in S[v] if r["hp"] == sel["selected"][v]) for v in CFG.tuned}
    s_fold = sum(r["train_seconds"] for v in CFG.tuned for r in S[v])
    d_fold = 3 * sum(t_sel.values()) + 3 * len(CFG.ablations) * 0.6 * t_sel["xattn_HMP"]
    out = {"dataset": ds, "fold": fr.fold, "run_hash": rhash, "layer_info": sel["layer_info"],
           "selected": sel["selected"], "selected_val_macro_f1": sel["selected_val_macro_f1"],
           "selected_at_lower_edge": {v: sel["selected"][v]["lr"] == lo for v in CFG.tuned},
           "selected_at_upper_edge": {v: sel["selected"][v]["lr"] == hi for v in CFG.tuned},
           "n_params": sel["n_params"], "n_params_fusion": sel["n_params_fusion"],
           "e4_reference_fusion_params": sel["e4_reference_fusion_params"],
           "e3_reproduction": rep, "e3_reproduction_all_exact": all(c["exact"] for c in rep),
           "runs_hitting_max_epochs": [f"{r['variant']} {r['hp']}" for r in runs if r["hit_max_epochs"]],
           "alignment": al, "runs": runs, "hubert_layer_load_seconds": dc.load_seconds,
           "pilot_seconds": time.perf_counter() - t0,
           "estimate_from_fold0": {"S_seconds_fold": s_fold, "S_hours_all_folds": s_fold * DEFAULT_FOLDS[ds] / 3600,
                                   "D_seconds_fold": d_fold, "D_hours_all_folds": d_fold * DEFAULT_FOLDS[ds] / 3600,
                                   "note": "D = 3 seeds x selected-config time of each tuned model + 3 ablations x 3 seeds "
                                           "x 0.6 x E4 time (0.57-0.62 from the synthetic benchmark); folds differ"},
           "test_predictions_made": False}
    write_atomic(PATHS.results / "E4" / f"pilot_{ds}.json", out, indent=1)
    print(json.dumps({k: out[k] for k in ("layer_info", "selected", "selected_val_macro_f1", "selected_at_lower_edge",
                                          "n_params", "e3_reproduction_all_exact", "runs_hitting_max_epochs",
                                          "estimate_from_fold0", "hubert_layer_load_seconds")}, indent=1, default=str),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["RAVDESS"])
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--pilot", action="store_true", help="validation-only fold selection + checks (no test)")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    if args.summarise:
        sums = {ds: summarise_dataset(ds) for ds in args.datasets}
        if set(sums) == {"RAVDESS", "CREMA-D"}:
            dec = decision(sums)
            write_atomic(PATHS.results / "E4" / "decision.json", dec, indent=1)
            for n, v in dec.items():
                print(f"DECISION {n}: {v['verdict']}  " + "; ".join(
                    f"{ds} {x['mean_delta']:+.4f} ({x['n_better']}/{x['n']}, p {x['p']:.3f}, Holm {x['p_holm']:.3f})"
                    for ds, x in v["per_dataset"].items()))
        return
    cfg_path = PATHS.configs / "e4_cross_attention.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != json.loads(json.dumps(asdict(CFG))):
        raise SystemExit("configs/e4_cross_attention.json differs from E4Config: protocol changed? stopped.")
    cfg_path.write_text(json.dumps(asdict(CFG), indent=1))
    index = pd.read_csv(PATHS.metadata / "index.csv")
    for ds in args.datasets:
        print(f"  [stage] {ds}: opening feature store", flush=True)
        store = FeatureStore.open(ds, index)
        dc = DataCache(store)
        print(f"[E4] {ds}: {'pilot' if args.pilot else 'full run'}; device {device()}, n_sa_blocks {CFG.n_sa_blocks}, "
              f"RAM {process_ram_mib()}", flush=True)
        for k in (args.folds if args.folds is not None else range(DEFAULT_FOLDS[ds])):
            fr = fold_rows(store, index, ds, k)
            if args.pilot:
                pilot(ds, store, dc, fr)
            else:
                t0 = time.perf_counter()
                run_fold(ds, store, dc, fr)
                print(f"  fold{k} done in {(time.perf_counter() - t0) / 60:.1f} min, RAM {process_ram_mib()}", flush=True)
        dc.data = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"[E4] {ds}: done. GPU peak {gpu_peak_mib()} MiB, RAM {process_ram_mib()}", flush=True)


if __name__ == "__main__":
    main()
