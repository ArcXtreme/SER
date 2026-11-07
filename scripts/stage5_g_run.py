"""
G-study driver (report section 8): why does pooled HuBERT (E2) beat frame-level HuBERT models?
PILOT MODE ONLY (team approval 2026-11-07): phase S (validation-only search) on fold 0. Phase D (test) is not
implemented here and the driver refuses to run without --pilot.

Groups (one process each, run sequentially under the RAM guard, scripts/run_g_pilot.ps1):
  pooled   P0, P0c, P0W, P0UM, P0B32         E2 code path (E2's make_inputs / build_model / train_classifier)
  frameL   B0, Z, Znoclip, ZnoLN, Z768, Z768noLN, ZB32   frame pipeline on HuBERT layer L (E3/E4 rule)
  frameUM  ZUM                               frame pipeline on the uniform 13-layer average (derived file)
  zw       ZW (CREMA-D only)                 all 13 layers streamed from the cache, learned weighting
Every search run gets without_test(fold): the test partition is not accessible.

Implementation gates (report section 8.6; any failure stops the process with an error, nothing is relaxed):
  R1 P0 (h256,w,p0.2,lr1e-3) == E2 A_layer{L}_mean_std      R2 P0W same cell == E2 A_weighted_all_mean_std
  R3 P0 h256 cells == E2 B_h256_* (folds where E2 selected layer L + mean_std)
  R4 P0B32 h512 cells == E2 B_h512_*  (same folds)          R5 P0B32 unweighted cell == E2 C_unweighted
  R6 Z G8 cells == E4 S_concat_H_*                           (fold 0 in the pilot)
  equality of: val macro-F1, best epoch, epochs run, per-epoch val-F1 and train-loss history, val confusion.
  DETERMINISM PROBE: one P0 cell and one Z cell are re-run in the same process; they must be bit-identical
  (distinguishes GPU non-determinism from genuine implementation mismatches).
  B0 INPUT CHECK: B0's standardised pooled inputs == E2's standardised pooled inputs (layer L) within 1e-3.

  python scripts/stage5_g_run.py --pilot --dataset RAVDESS --group pooled
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.features.store import FeatureStore  # noqa: E402
from src.models.e2_hubert import n_params  # noqa: E402
from src.models.g_frame import LayerWeightedFrameModel, PooledFrameHead, build_frame_model  # noqa: E402
from src.preprocessing.splits import split_path  # noqa: E402
from src.training.data import fold_rows, without_test  # noqa: E402
from src.training.preprocessing import FrameStandardizer  # noqa: E402
from src.training.seq_trainer import SeqTrainConfig, predict_rows, train_sequence_classifier  # noqa: E402
from src.training.sequence_data import SequenceData, load_hubert_layer  # noqa: E402
from src.training.streamed_layers import StreamedLayers  # noqa: E402
from src.training.trainer import TrainConfig, device, predict, train_classifier  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402
from src.utils.resources import gpu_peak_mib, process_ram_mib  # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


E2 = _load("e2drv", "stage5_e2_run.py")          # E2's own input preparation and model construction (unchanged)
E3 = _load("e3drv", "stage5_e3_run.py")          # layer rule, atomic writes


@dataclass(frozen=True)
class GConfig:
    grid_dropout: tuple = (0.2, 0.4)
    grid_lr: tuple = (1e-3, 3e-4, 1e-4, 3e-5)
    head_hidden: int = 256
    b32_hidden: tuple = (256, 512)
    b32_class_weighted: tuple = (True, False)
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 150
    patience: int = 10
    grad_clip: float = 1.0
    bucket: int = 20
    selection_seed: int = 0
    final_seeds: tuple = (0, 1, 2)
    pooling: str = "mean_std"
    zw_datasets: tuple = ("CREMA-D",)
    b0_input_tolerance: float = 1e-3


CFG = GConfig()
NO_CLIP = float("inf")      # frame trainer: clip_grad_norm_ with max_norm=inf scales by exactly 1.0 (tested)

# arm -> spec. "key" = model identity used in S-file names (B32 arms share the G8 cells of their base arm).
ARMS = {
    "P0":       {"pipe": "pooled", "key": "P0", "rep": "L", "clip": False, "grid": "G8"},
    "P0c":      {"pipe": "pooled", "key": "P0c", "rep": "L", "clip": True, "grid": "G8"},
    "P0W":      {"pipe": "pooled", "key": "P0W", "rep": "weighted_all", "clip": False, "grid": "G8"},
    "P0UM":     {"pipe": "pooled", "key": "P0UM", "rep": "UM", "clip": False, "grid": "G8"},
    "P0B32":    {"pipe": "pooled", "key": "P0", "rep": "L", "clip": False, "grid": "B32"},
    "B0":       {"pipe": "frame", "key": "B0", "model": "B0", "clip": False, "grid": "G8", "stream": "L"},
    "Z":        {"pipe": "frame", "key": "Z", "model": "Z", "d": 128, "ln": True, "clip": True, "grid": "G8", "stream": "L"},
    "Znoclip":  {"pipe": "frame", "key": "Znoclip", "model": "Z", "d": 128, "ln": True, "clip": False, "grid": "G8", "stream": "L"},
    "ZnoLN":    {"pipe": "frame", "key": "ZnoLN", "model": "Z", "d": 128, "ln": False, "clip": True, "grid": "G8", "stream": "L"},
    "Z768":     {"pipe": "frame", "key": "Z768", "model": "Z", "d": 768, "ln": True, "clip": True, "grid": "G8", "stream": "L"},
    "Z768noLN": {"pipe": "frame", "key": "Z768noLN", "model": "Z", "d": 768, "ln": False, "clip": True, "grid": "G8", "stream": "L"},
    "ZB32":     {"pipe": "frame", "key": "Z", "model": "Z", "d": 128, "ln": True, "clip": True, "grid": "B32", "stream": "L"},
    "ZUM":      {"pipe": "frame", "key": "ZUM", "model": "Z", "d": 128, "ln": True, "clip": True, "grid": "G8", "stream": "UM"},
    "ZW":       {"pipe": "zw", "key": "ZW", "model": "ZW", "d": 128, "ln": True, "clip": True, "grid": "G8", "stream": "W13"},
}
GROUPS = {"pooled": ("P0", "P0c", "P0W", "P0UM", "P0B32"),
          "frameL": ("Z", "B0", "Znoclip", "ZnoLN", "Z768", "Z768noLN", "ZB32"),
          "frameUM": ("ZUM",), "zw": ("ZW",)}


def candidates(grid: str) -> list[dict]:
    """G8 = p x lr (h 256, weighted). B32 = h x weighting x p x lr; its first 8 cells are exactly G8 (tie-breaking)."""
    hs, ws = ((CFG.head_hidden,), (True,)) if grid == "G8" else (CFG.b32_hidden, CFG.b32_class_weighted)
    return [{"hidden": h, "class_weighted": w, "dropout": p, "lr": lr}
            for h in hs for w in ws for p in CFG.grid_dropout for lr in CFG.grid_lr]


def s_name(key: str, hp: dict) -> str:
    return f"S_{key}_h{hp['hidden']}_w{int(hp['class_weighted'])}_d{hp['dropout']}_lr{hp['lr']}.json"


def run_hash(ds, store, key, rep) -> str:
    blob = json.dumps({"g": asdict(CFG), "key": key, "spec": {k: v for k, v in ARMS[key].items() if k != "grid"},
                       "rep": str(rep), "cache": store.meta["cache_id"],
                       "split": hashlib.sha256(split_path(ds).read_bytes()).hexdigest()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def derived_dir(store) -> Path:
    return store.root.parent / f"derived_G_{store.meta['cache_id']}"


# --------------------------------------------------------------------------- pooled pipeline (E2 code)
def pooled_run(pooled_arr, rep: str, fr, hp: dict, clip: bool, seed: int) -> dict:
    (Xtr, Xva), _, shape = E2.make_inputs(pooled_arr, fr.rows["train"], [fr.rows["train"], fr.rows["val"]], rep, CFG.pooling)
    cfg = TrainConfig(lr=hp["lr"], weight_decay=CFG.weight_decay, batch_size=CFG.batch_size, max_epochs=CFG.max_epochs,
                      patience=CFG.patience, class_weighted=hp["class_weighted"], seed=seed,
                      grad_clip=CFG.grad_clip if clip else None)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    res = train_classifier(lambda: E2.build_model(shape, rep, hp["hidden"], hp["dropout"]),
                           (Xtr, fr.labels["train"]), (Xva, fr.labels["val"]), cfg)
    model = res["model"]
    out = {"val_macro_f1": res["best_val_macro_f1"], "val": compute_metrics(fr.labels["val"], predict(model, Xva)),
           "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], "train_seconds": res["train_seconds"],
           "n_params": n_params(model), "history": res["history"], "gpu_peak_mib": gpu_peak_mib(),
           "process_ram_mib": process_ram_mib(), "train_cfg": cfg.to_dict()}
    if rep == "weighted_all":
        out["layer_weights"] = torch.softmax(model.layer_weighting.logits.detach().cpu(), 0).tolist()
    return out


# --------------------------------------------------------------------------- frame pipelines
def b0_pooled_stats(data, rows, chunk=64):
    U = []
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            U.append(PooledFrameHead.pooled(data.batch(rows[i:i + chunk])).double().cpu().numpy())
    U = np.concatenate(U)
    z = FrameStandardizer().fit([U])                      # E2's pooled standardiser (same eps) on TRAINING recordings
    return z.mean_, z.std_, U


def frame_model_fn(spec, hp, b0_stats=None):
    if spec["model"] == "B0":
        return lambda: PooledFrameHead(b0_stats[0], b0_stats[1], hp["hidden"], hp["dropout"]).to(device())
    if spec["model"] == "ZW":
        return lambda: LayerWeightedFrameModel(spec["d"], spec["ln"], hp["dropout"], hp["hidden"]).to(device())
    return lambda: build_frame_model(spec["d"], spec["ln"], hp["dropout"], hp["hidden"]).to(device())


def frame_run(data, fr, spec, hp: dict, seed: int, b0_stats=None) -> dict:
    data.fit(fr.rows["train"], None)
    cfg = SeqTrainConfig(lr=hp["lr"], weight_decay=CFG.weight_decay, batch_size=CFG.batch_size, max_epochs=CFG.max_epochs,
                         patience=CFG.patience, class_weighted=hp["class_weighted"], seed=seed,
                         grad_clip=CFG.grad_clip if spec["clip"] else NO_CLIP, bucket=CFG.bucket)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    io0 = dict(getattr(data, "io", {}))
    res = train_sequence_classifier(frame_model_fn(spec, hp, b0_stats), data, (fr.rows["train"], fr.labels["train"]),
                                    (fr.rows["val"], fr.labels["val"]), cfg)
    model = res["model"]
    out = {"val_macro_f1": res["best_val_macro_f1"], "val": compute_metrics(fr.labels["val"], predict_rows(model, data, fr.rows["val"])),
           "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], "train_seconds": res["train_seconds"],
           "n_params": n_params(model), "history": res["history"], "gpu_peak_mib": gpu_peak_mib(),
           "process_ram_mib": process_ram_mib(), "train_cfg": cfg.to_dict()}
    if spec["model"] == "ZW":
        out["layer_weights"] = torch.softmax(model.mix.logits.detach().cpu(), 0).tolist()
        io = data.io
        out["io"] = {k: io[k] - io0.get(k, 0) for k in io}
    return out


# --------------------------------------------------------------------------- search, gates
def search(arm, ds, store, fr_sel, d: Path, runner, rep_label, log_stage, cells=None) -> list[dict]:
    """cells: optional subset of the arm's candidates (gates-only mode); the files are the same as in the full
    search (same run hash), so the pilot reuses them without retraining."""
    spec = ARMS[arm]
    rh = run_hash(ds, store, spec["key"], rep_label)
    out = []
    for hp in candidates(spec["grid"]):
        if cells is not None and hp not in cells:
            continue
        p = d / s_name(spec["key"], hp)
        if p.exists() and json.loads(p.read_text()).get("run_hash") == rh:
            out.append(json.loads(p.read_text())); continue
        log_stage(f"phase S {arm} {hp} (training)")
        r = runner(hp)
        r.update({"arm": arm, "key": spec["key"], "hp": hp, "seed": CFG.selection_seed, "run_hash": rh,
                  "rep": str(rep_label), "hit_max_epochs": r["epochs_run"] >= CFG.max_epochs})
        E3.write_atomic(p, r)
        print(f"    S {arm:9s} {hp}  val F1 {r['val_macro_f1']:.4f}  best ep {r['best_epoch']:3d} run {r['epochs_run']:3d}  "
              f"{r['train_seconds']:6.1f}s  GPU {r['gpu_peak_mib']} MiB  RAM {r['process_ram_mib']}", flush=True)
        out.append(r)
    return out


FIELDS = ("val_macro_f1", "best_epoch", "epochs_run")


def same_run(g: dict, ref: dict) -> dict:
    c = {f: [ref[f], g[f]] for f in FIELDS}
    c["history_val_f1_identical"] = [h["val_macro_f1"] for h in ref["history"]] == [h["val_macro_f1"] for h in g["history"]]
    c["history_train_loss_identical"] = [h["train_loss"] for h in ref["history"]] == [h["train_loss"] for h in g["history"]]
    c["val_confusion_identical"] = ref["val"]["confusion"] == g["val"]["confusion"]
    c["exact"] = bool(all(ref[f] == g[f] for f in FIELDS) and c["history_val_f1_identical"]
                      and c["history_train_loss_identical"] and c["val_confusion_identical"])
    return c


def find(runs, hp):
    return next(r for r in runs if r["hp"] == hp)


def gates_pooled(ds, k, L, S) -> list[dict]:
    e2d = PATHS.results / "E2" / ds / f"fold{k}"
    e2sel = json.loads((e2d / "selection.json").read_text())["selected"]
    G = []
    base = {"hidden": 256, "class_weighted": True, "dropout": 0.2, "lr": 1e-3}
    G.append({"gate": "R1", "cell": base, "ref": f"A_layer{L}_mean_std",
              **same_run(find(S["P0"], base), json.loads((e2d / f"A_layer{L}_mean_std.json").read_text()))})
    G.append({"gate": "R2", "cell": base, "ref": "A_weighted_all_mean_std",
              **same_run(find(S["P0W"], base), json.loads((e2d / "A_weighted_all_mean_std.json").read_text()))})
    if e2sel["rep"] == f"layer{L}" and e2sel["pooling"] == "mean_std":
        for h, gate, arm in ((256, "R3", "P0"), (512, "R4", "P0B32")):
            for p in (0.2, 0.4):
                for lr in (1e-3, 3e-4):
                    hp = {"hidden": h, "class_weighted": True, "dropout": p, "lr": lr}
                    ref = json.loads((e2d / f"B_h{h}_d{p}_lr{lr}.json").read_text())
                    if ref["epochs_run"] >= 50:
                        raise SystemExit(f"gate {gate}: E2 reference {ref} reached its 50-epoch cap - not comparable")
                    G.append({"gate": gate, "cell": hp, "ref": f"B_h{h}_d{p}_lr{lr}", **same_run(find(S[arm], hp), ref)})
        c = json.loads((e2d / "C_unweighted.json").read_text())
        hp = {k_: c["candidate"][k_] for k_ in ("hidden", "class_weighted", "dropout", "lr")}
        G.append({"gate": "R5", "cell": hp, "ref": "C_unweighted", **same_run(find(S["P0B32"], hp), c)})
        a = json.loads((e2d / f"A_layer{L}_mean_std.json").read_text()); b = json.loads((e2d / "B_h256_d0.2_lr0.001.json").read_text())
        G.append({"gate": "E2-internal", "cell": base, "ref": "A vs B_h256_d0.2_lr0.001", **same_run(a, b)})
    return G


def gates_z(ds, k, S) -> list[dict]:
    e4d = PATHS.results / "E4" / ds / f"fold{k}"
    return [{"gate": "R6", "cell": hp, "ref": f"S_concat_H_d{hp['dropout']}_lr{hp['lr']}",
             **same_run(find(S["Z"], hp), json.loads((e4d / f"S_concat_H_d{hp['dropout']}_lr{hp['lr']}.json").read_text()))}
            for hp in candidates("G8")]


def stop_if_failed(gates, what):
    bad = [f"{g['gate']} {g['cell']} vs {g['ref']}" for g in gates if not g["exact"]]
    if bad:
        raise SystemExit(f"IMPLEMENTATION GATE FAILED ({what}): {bad}")


# --------------------------------------------------------------------------- pilot
def pilot(ds, group, gates_only=False):
    if ds not in ("RAVDESS", "CREMA-D") or group not in GROUPS:
        raise SystemExit("bad arguments")
    if gates_only and group not in ("pooled", "frameL"):
        raise SystemExit("gates-only mode covers the pooled and frameL groups")
    if group == "zw" and ds not in CFG.zw_datasets:
        raise SystemExit(f"Z-W is run on {CFG.zw_datasets} only (team decision)")
    stage = lambda s: print(f"  [stage] {ds} fold 0 {group}: {s}", flush=True)  # noqa: E731
    index = pd.read_csv(PATHS.metadata / "index.csv")
    stage("opening feature store")
    store = FeatureStore.open(ds, index)
    k = 0
    fr = fold_rows(store, index, ds, k)
    fr_sel = without_test(fr)                                   # the ONLY fold object passed to any training code
    layer_info = E3.fold_layer(ds, k)
    L = layer_info["layer"]
    e4_layer = json.loads((PATHS.results / "E4" / ds / f"fold{k}" / "selection.json").read_text())["layer"]
    if L != e4_layer:
        raise SystemExit(f"layer rule mismatch: {L} vs E4 {e4_layer}")
    d = PATHS.results / "G" / ds / f"fold{k}"
    d.mkdir(parents=True, exist_ok=True)
    rep = {"group": group, "dataset": ds, "fold": k, "layer": L, "layer_info": layer_info, "test_predictions_made": False,
           "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "numpy": np.__version__, "python": sys.version.split()[0],
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
           "deterministic_algorithms": None, "gates": [], "checks": {}, "t0": time.time()}
    t_start = time.perf_counter()
    S = {}
    if group == "pooled":
        pooled = np.asarray(store.array("hubert_pooled"))
        reps = {"L": (pooled, f"layer{L}"), "weighted_all": (pooled, "weighted_all")}
        if not gates_only:
            reps["UM"] = (np.load(derived_dir(store) / "um_pooled.npy"), "layer0")
        base = {"hidden": 256, "class_weighted": True, "dropout": 0.2, "lr": 1e-3}
        gate_cells = {"P0": None, "P0W": [base], "P0B32": []}
        e2d = PATHS.results / "E2" / ds / f"fold{k}"
        e2sel = json.loads((e2d / "selection.json").read_text())["selected"]
        if e2sel["rep"] == f"layer{L}" and e2sel["pooling"] == "mean_std":
            c = json.loads((e2d / "C_unweighted.json").read_text())["candidate"]
            gate_cells["P0B32"] = [{"hidden": 512, "class_weighted": True, "dropout": p, "lr": lr} for p in (0.2, 0.4) for lr in (1e-3, 3e-4)] + \
                                  [{k_: c[k_] for k_ in ("hidden", "class_weighted", "dropout", "lr")}]
        for arm in (("P0", "P0W", "P0B32") if gates_only else GROUPS["pooled"]):
            spec = ARMS[arm]
            arr, r_ = reps[spec["rep"]]
            S[arm] = search(arm, ds, store, fr_sel, d, lambda hp, arr=arr, r_=r_, spec=spec: pooled_run(arr, r_, fr_sel, hp, spec["clip"], 0),
                            r_ if spec["rep"] != "UM" else "UM", stage, cells=gate_cells[arm] if gates_only else None)
        stage("implementation gates R1-R5")
        rep["gates"] = gates_pooled(ds, k, L, S)
        stop_if_failed(rep["gates"], "pooled")
        stage("determinism probe (P0 cell re-run)")
        base = {"hidden": 256, "class_weighted": True, "dropout": 0.2, "lr": 1e-3}
        again = pooled_run(pooled, f"layer{L}", fr_sel, base, False, 0)
        rep["checks"]["determinism_probe_P0"] = same_run(again, find(S["P0"], base))
        if not rep["checks"]["determinism_probe_P0"]["exact"]:
            raise SystemExit("DETERMINISM PROBE FAILED (pooled): non-deterministic training")
    elif group in ("frameL", "frameUM"):
        if group == "frameL":
            stage(f"loading HuBERT layer {L}")
            H, rep_label = load_hubert_layer(store, L), f"layer{L}"
        else:
            stage("loading uniform 13-layer average (derived)")
            H, rep_label = np.load(derived_dir(store) / "um_frames.npy"), "UM"
        data = SequenceData(store, L if group == "frameL" else "UM", device(), hubert=H, need_disvoice=False)
        del H
        print(f"    data on {device()}, RAM {process_ram_mib()}", flush=True)
        if group == "frameL":
            # Z first: gate R6 and the determinism probe before any other frame arm
            S["Z"] = search("Z", ds, store, fr_sel, d, lambda hp: frame_run(data, fr_sel, ARMS["Z"], hp, 0), rep_label, stage)
            stage("implementation gate R6 (Z == E4 concat_H)")
            rep["gates"] = gates_z(ds, k, S)
            stop_if_failed(rep["gates"], "Z")
            stage("determinism probe (Z cell re-run)")
            base = {"hidden": 256, "class_weighted": True, "dropout": 0.2, "lr": 1e-3}
            again = frame_run(data, fr_sel, ARMS["Z"], base, 0)
            rep["checks"]["determinism_probe_Z"] = same_run(again, find(S["Z"], base))
            if not rep["checks"]["determinism_probe_Z"]["exact"]:
                raise SystemExit("DETERMINISM PROBE FAILED (frame): non-deterministic training")
            stage("B0 input check")
            data.fit(fr_sel.rows["train"], None)
            mu, sd, _ = b0_pooled_stats(data, fr_sel.rows["train"])
            pooled = np.asarray(store.array("hubert_pooled"))
            rows = np.concatenate([fr_sel.rows["train"], fr_sel.rows["val"]])
            (ref,), _, _ = E2.make_inputs(pooled, fr_sel.rows["train"], [rows], f"layer{L}", CFG.pooling)
            mine = []
            with torch.no_grad():
                for i in range(0, len(rows), 64):
                    b = data.batch(rows[i:i + 64])
                    mine.append(((PooledFrameHead.pooled(b).double().cpu().numpy() - mu) / sd))
            dmat = np.abs(np.concatenate(mine) - ref.double().cpu().numpy())
            diff = float(dmat.max())
            rep["checks"]["B0_input_max_abs_diff"] = diff
            rep["checks"]["B0_input_share_above_tol"] = float((dmat > CFG.b0_input_tolerance).mean())
            rep["checks"]["B0_input_pass"] = diff <= CFG.b0_input_tolerance
            print(f"    B0 input max |diff| vs E2 standardised pooled input: {diff:.2e} (tolerance {CFG.b0_input_tolerance})", flush=True)
            if gates_only:
                rep["note"] = "gates-only mode: the B0 input check is recorded, not acted on; no other frame arm is run"
            elif diff > CFG.b0_input_tolerance:
                raise SystemExit(f"B0 INPUT CHECK FAILED: max abs diff {diff} > {CFG.b0_input_tolerance}")
            b0_stats = (mu, sd)
            for arm in (() if gates_only else ("B0", "Znoclip", "ZnoLN", "Z768", "Z768noLN", "ZB32")):
                spec = ARMS[arm]
                S[arm] = search(arm, ds, store, fr_sel, d,
                                lambda hp, spec=spec: frame_run(data, fr_sel, spec, hp, 0, b0_stats if spec["model"] == "B0" else None),
                                rep_label, stage)
        else:
            S["ZUM"] = search("ZUM", ds, store, fr_sel, d, lambda hp: frame_run(data, fr_sel, ARMS["ZUM"], hp, 0), "UM", stage)
    elif group == "zw":
        dd = derived_dir(store)
        lsum = np.load(dd / "layer_sum.npy", mmap_mode="r")
        lsq = np.load(dd / "layer_sumsq.npy", mmap_mode="r")
        man = store.manifest
        data = StreamedLayers(store.root / "hubert.npy", man.frame_offset.values, man.n_frames.values, lsum, lsq, device())
        S["ZW"] = search("ZW", ds, store, fr_sel, d, lambda hp: frame_run(data, fr_sel, ARMS["ZW"], hp, 0), "W13", stage)
        data.close()
        rep["checks"]["zw_io_total"] = data.io
    rep["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    rep["runs"] = {arm: [{k_: r[k_] for k_ in ("hp", "val_macro_f1", "best_epoch", "epochs_run", "hit_max_epochs",
                                               "train_seconds", "n_params", "gpu_peak_mib", "process_ram_mib")}
                         | ({"io": r["io"]} if "io" in r else {}) | ({"layer_weights": r["layer_weights"]} if "layer_weights" in r else {})
                         for r in runs] for arm, runs in S.items()}
    rep["selected_validation_only"] = {arm: max(runs, key=lambda r: r["val_macro_f1"])["hp"] if runs else None
                                       for arm, runs in S.items()}
    rep["group_seconds"] = time.perf_counter() - t_start
    rep["gpu_peak_mib_process"] = gpu_peak_mib()
    rep["process_ram_mib"] = process_ram_mib()
    rep["mode"] = "gates-only" if gates_only else "pilot"
    E3.write_atomic(PATHS.results / "G" / f"{'gates' if gates_only else 'pilot'}_{ds}_{group}.json", rep, indent=1)
    stage(f"done in {rep['group_seconds'] / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--gates-only", action="store_true", help="reproduction gates R1-R6, determinism probes, B0 input check")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", required=True)
    args = ap.parse_args()
    if not args.pilot:
        raise SystemExit("Only the validation-only fold-0 pilot is implemented/approved; phase D (test) is not available.")
    cfg_path = PATHS.configs / "g_study.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != json.loads(json.dumps(asdict(CFG))):
        raise SystemExit("configs/g_study.json differs from GConfig - protocol changed? stopped.")
    cfg_path.write_text(json.dumps(asdict(CFG), indent=1))
    pilot(args.dataset, args.group, gates_only=args.gates_only)


if __name__ == "__main__":
    main()
