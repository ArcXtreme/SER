"""
E4 audit (read-only with respect to results, splits and caches; writes results/E4/audit.json).

Checks are made from the SAVED artifacts, independently of the code path that produced them:
  1 completion   per fold: 32 S files (4 models x 8) + selection.json + 7 variants x 3 seeds D files,
                 one run hash per fold, every checkpoint exists and its metadata matches its D file
  2 isolation    S files contain validation metrics only; selections re-derivable from S validation
                 F1 alone; every S file and selection.json older than every D file and checkpoint;
                 D test rows == fold test rows; HuBERT layer == the E2 phase-A rule (SHA-256 unchanged);
                 D hyper-parameters == selection (ablations use xattn_HMP's)
  3 training-only preprocessing: checkpoint standardisers == TRAIN-row statistics (recomputed);
                 class weights == recomputed from training labels
  4 reproduction every reloaded checkpoint re-predicts its saved test predictions exactly; seed-0
                 finals reproduce the selection validation F1; E3-comparable S cells reproduce E3
                 exactly in every fold; where a re-searched model selects E3's setting its D test
                 predictions equal E3's
  5 pilot reuse  fold-0 S runs are the pilot's (same run hash, files not newer than the pilot report)
  6 capacity     E4 = 769,286 parameters; control within 5 %
  7 integrity    split SHA-256, Stage 4 cache SHA-256, prosody SHA-256, E1-E3 snapshot (1,432 files)
  info (not pass/fail, pre-registered as reported limitations): runs reaching the 150-epoch cap,
                 selections at the lower / upper edge of the learning-rate grid
Run: python scripts/stage5_e4_audit.py
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.models.e3_fusion import E3Fusion, n_params  # noqa: E402
from src.models.e4_cross_attention import E4CrossAttention  # noqa: E402
from src.preprocessing.splits import DEFAULT_FOLDS, split_path  # noqa: E402
from src.training.data import fold_rows  # noqa: E402
from src.training.preprocessing import FrameStandardizer  # noqa: E402
from src.training.seq_trainer import predict_rows  # noqa: E402
from src.training.trainer import class_weights  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


drv = _load("e4drv", "stage5_e4_run.py")
SNAPSHOT = PATHS.results / "E4" / "snapshot_before_e4_pilot.json"


def chk(report, name, cond, detail=None):
    report.setdefault("checks", {})[name] = bool(cond)
    if detail is not None:
        report.setdefault("details", {})[name] = detail
    print(("PASS " if cond else "FAIL ") + name, flush=True)


def audit_fold(k, d: Path, fr, data, store, expected_layer: int, e3_dir: Path | None, cfg=None):
    """All per-fold checks from saved artifacts. Returns (ok: dict[str, bool], detail: dict)."""
    cfg = cfg or drv.CFG
    variants = cfg.tuned + cfg.ablations
    cands = [{"dropout": dr, "lr": lr} for dr in cfg.grid_dropout for lr in cfg.grid_lr]
    ok = {n: True for n in ("complete", "one_hash", "test_free", "sel_repro", "order", "rows", "layer", "hp",
                            "seed0", "cw", "ck_meta", "std", "repredict", "e3_S", "e3_D")}
    det = {"std_max_abs_diff": {}, "repredict_mismatch": [], "e3_S_mismatch": [], "e3_D": [], "cap_hits": []}
    S_files = {(v, h["dropout"], h["lr"]): d / drv.s_name(v, h) for v in cfg.tuned for h in cands}
    D_files = {(v, s): d / f"D_{v}_seed{s}.json" for v in variants for s in cfg.final_seeds}
    ok["complete"] = (d / "selection.json").exists() and all(p.exists() for p in S_files.values()) and \
        all(p.exists() for p in D_files.values())
    if not ok["complete"]:
        return {n: False for n in ok}, det
    sel = json.loads((d / "selection.json").read_text())
    Sj = {key: json.loads(p.read_text()) for key, p in S_files.items()}
    Dj = {key: json.loads(p.read_text()) for key, p in D_files.items()}
    ok["one_hash"] = len({sel["run_hash"]} | {r["run_hash"] for r in Sj.values()} | {r["run_hash"] for r in Dj.values()}) == 1
    for r in Sj.values():
        if '"test' in json.dumps(r) or r["val"]["n"] != len(fr.rows["val"]):
            ok["test_free"] = False
        if r["epochs_run"] >= cfg.max_epochs:
            det["cap_hits"].append(f"S {r['variant']} {r['hp']}")
    for v in cfg.tuned:
        runs = [Sj[(v, h["dropout"], h["lr"])] for h in cands]
        best = runs[max(range(len(runs)), key=lambda j: (runs[j]["val_macro_f1"], -j))]
        ok["sel_repro"] &= best["hp"] == sel["selected"][v] and best["val_macro_f1"] == sel["selected_val_macro_f1"][v]
    t_sel = max([p.stat().st_mtime for p in S_files.values()] + [(d / "selection.json").stat().st_mtime])
    t_d = min([p.stat().st_mtime for p in D_files.values()] + [Path(r["checkpoint"]).stat().st_mtime for r in Dj.values()])
    ok["order"] = t_sel <= t_d
    det["order"] = {"last_selection_artifact": t_sel, "first_test_artifact": t_d}
    ok["rows"] = all(r["test_rows"] == fr.rows["test"].tolist() and r["test"]["n"] == len(fr.rows["test"]) for r in Dj.values())
    ok["layer"] = sel["layer"] == expected_layer and all(r["layer"] == expected_layer for r in Dj.values())
    cw = class_weights(fr.labels["train"])
    for (v, s), r in Dj.items():
        ok["hp"] &= r["hp"] == sel["selected"][drv.hp_source(v)]
        if v in cfg.tuned and s == cfg.selection_seed:
            ok["seed0"] &= abs(max(h["val_macro_f1"] for h in r["history"]) - sel["selected_val_macro_f1"][v]) < 1e-12
        ok["cw"] &= np.allclose(r["class_weights"], cw)
        if r["hit_max_epochs"]:
            det["cap_hits"].append(f"D {v} seed{s}")
    # training-only statistics, recomputed from the raw arrays (independent of SequenceData.fit)
    off, n = store.manifest.frame_offset.values, store.manifest.n_frames.values
    ref = {}
    for st, arr in (("hubert", data._np["hubert"]), ("mfcc", np.asarray(store.array("mfcc"))),
                    ("prosody", np.asarray(store.array("prosody_frames")))):
        ref[st] = FrameStandardizer(passthrough=(1, 2) if st == "prosody" else ()).fit(
            [arr[off[r]: off[r] + n[r]] for r in fr.rows["train"]])
    data.fit(fr.rows["train"], None)
    for (v, s), r in Dj.items():
        ck = torch.load(r["checkpoint"], map_location="cpu", weights_only=False)
        ok["ck_meta"] &= (ck["fold"], ck["seed"], ck["variant"], ck["layer"], ck["run_hash"], ck["hp"]) == \
            (k, s, v, sel["layer"], sel["run_hash"], r["hp"])
        for st, z in ref.items():
            diff = float(max(np.abs(ck["standardizers"][st]["mean"] - z.mean_).max(), np.abs(ck["standardizers"][st]["std"] - z.std_).max()))
            det["std_max_abs_diff"][st] = max(det["std_max_abs_diff"].get(st, 0.0), diff)
            ok["std"] &= diff == 0.0
        model, _ = drv.load_checkpoint(r["checkpoint"])
        if predict_rows(model, data, fr.rows["test"]).tolist() != r["test_pred"]:
            ok["repredict"] = False
            det["repredict_mismatch"].append([k, v, s])
    # E3 reproduction (S cells in every fold; D where the re-searched model selected E3's setting)
    if e3_dir is not None:
        for v in drv.E3_COMPARABLE:
            for h in cands:
                if h["lr"] not in cfg.e3_reproduction_lrs:
                    continue
                a, b = json.loads((e3_dir / drv.s_name(v, h)).read_text()), Sj[(v, h["dropout"], h["lr"])]
                same = (a["val_macro_f1"], a["best_epoch"], a["epochs_run"], [x["val_macro_f1"] for x in a["history"]],
                        [x["train_loss"] for x in a["history"]], a["val"]["confusion"]) == \
                    (b["val_macro_f1"], b["best_epoch"], b["epochs_run"], [x["val_macro_f1"] for x in b["history"]],
                     [x["train_loss"] for x in b["history"]], b["val"]["confusion"])
                if not same:
                    ok["e3_S"] = False
                    det["e3_S_mismatch"].append(f"{v} {h}")
        e3_sel = json.loads((e3_dir / "selection.json").read_text())["selected"]
        for v, e3_hp in (("concat_HMP", e3_sel["concat_HMP"]), ("sa_HMP", e3_sel["sa_HMP"]), ("concat_H", e3_sel["concat_HMP"])):
            if sel["selected"][v] != e3_hp:
                det["e3_D"].append({"variant": v, "comparable": False})
                continue
            eq = all(Dj[(v, s)]["test_pred"] == json.loads((e3_dir / f"D_{v}_seed{s}.json").read_text())["test_pred"]
                     for s in cfg.final_seeds)
            ok["e3_D"] &= eq
            det["e3_D"].append({"variant": v, "comparable": True, "test_predictions_identical": eq})
    return ok, det


CHECK_NAMES = {
    "complete": "every fold complete (32 S + selection + 21 D files + checkpoints)",
    "one_hash": "one run hash per fold across S, selection and D",
    "test_free": "S (selection) files contain validation metrics only",
    "sel_repro": "selected hyper-parameters re-derivable from S validation F1 alone",
    "order": "all S files and selection.json older than every D file and checkpoint",
    "rows": "D test rows == the fold's test rows",
    "layer": "HuBERT layer == rule on the fold's own E2 phase-A file (SHA-256 unchanged)",
    "hp": "D hyper-parameters == selection (ablations use xattn_HMP's)",
    "seed0": "seed-0 finals reproduce the selection validation F1",
    "cw": "class weights == recomputed from training labels",
    "ck_meta": "checkpoint metadata matches D files",
    "std": "standardiser statistics == TRAIN-row statistics (all streams, all checkpoints)",
    "repredict": "every reloaded checkpoint re-predicts its saved test predictions exactly",
    "e3_S": "E3-comparable S cells (concat/control x lr {1e-3,3e-4}) reproduce E3 exactly in every fold",
    "e3_D": "where a re-searched model selected E3's setting, its D test predictions equal E3's",
}


def audit_dataset(ds, index):
    from src.features.store import ADOPTABLE, FeatureStore, sha256_file
    e3audit = _load("e3audit", "stage5_e3_audit.py")
    R = {}
    store = FeatureStore.open(ds, index)
    root = PATHS.results / "E4" / ds
    dc = drv.DataCache(store)
    agg = {n: True for n in CHECK_NAMES}
    details = {"per_fold": {}}
    layer_ok = True
    for k in range(DEFAULT_FOLDS[ds]):
        d = root / f"fold{k}"
        fr = fold_rows(store, index, ds, k)
        sel = json.loads((d / "selection.json").read_text())
        e2p = PATHS.results / "E2" / ds / f"fold{k}" / "selection.json"
        li = drv.E3.select_layer(json.loads(e2p.read_text())["A"])
        layer_ok &= hashlib.sha256(e2p.read_bytes()).hexdigest() == sel["layer_info"]["source_sha256"]
        ok, det = audit_fold(k, d, fr, dc.get(sel["layer"]), store, li["layer"], PATHS.results / "E3" / ds / f"fold{k}")
        for n in agg:
            agg[n] &= ok[n]
        details["per_fold"][k] = det
        print(f"  fold {k} audited: {[n for n, v in ok.items() if not v] or 'all pass'}", flush=True)
    agg["layer"] &= layer_ok
    for n, text in CHECK_NAMES.items():
        chk(R, f"{ds}: {text}", agg[n])
    R["details"] = details
    # pilot reuse
    pil = json.loads((PATHS.results / "E4" / f"pilot_{ds}.json").read_text())
    t_pil = (PATHS.results / "E4" / f"pilot_{ds}.json").stat().st_mtime
    s0 = [json.loads(p.read_text()) for p in (root / "fold0").glob("S_*.json")]
    chk(R, f"{ds}: fold-0 S runs are the pilot's (same run hash, not rewritten after the pilot)",
        len(s0) == 32 and all(r["run_hash"] == pil["run_hash"] for r in s0) and
        all(p.stat().st_mtime <= t_pil for p in (root / "fold0").glob("S_*.json")))
    # capacity
    e4 = n_params(E4CrossAttention()); sa = n_params(E3Fusion(n_sa_blocks=drv.CFG.n_sa_blocks))
    chk(R, f"{ds}: E4 has 769,286 parameters; control within 5 %", e4 == 769_286 and abs(sa - e4) / e4 < 0.05,
        {"xattn_HMP": e4, "sa_HMP": sa, "rel_diff": (sa - e4) / e4})
    # integrity
    sha = hashlib.sha256(split_path(ds).read_bytes()).hexdigest().upper()
    chk(R, f"{ds}: Stage 3 split file unchanged (SHA-256)", sha == e3audit.EXPECTED_SPLIT_SHA[ds], sha)
    ref = store.meta["adopted_from"]["sha256"]
    now = {n: sha256_file(store.root / n) for n in ADOPTABLE}
    chk(R, f"{ds}: Stage 4 audio/manifest/HuBERT/pooled/MFCC unchanged since adoption (SHA-256)",
        all(now[n] == ref[n] for n in ADOPTABLE) and sha256_file(store.root / "mfcc.npy") == ref["mfcc.npy (source, for verification)"])
    e2a = json.loads((PATHS.results / "E2" / "audit.json").read_text())[ds]["prosody_sha256_now"]
    chk(R, f"{ds}: prosody arrays unchanged since the E2 audit (SHA-256)", all(sha256_file(store.root / f) == h for f, h in e2a.items()))
    # info: cap hits and grid edges (pre-registered as reported limitations, not failures)
    sm = json.loads((root / "summary.json").read_text()) if (root / "summary.json").exists() else None
    R["info"] = {"cap_hits": {k: v["cap_hits"] for k, v in details["per_fold"].items()},
                 "selection_edges": sm["selection_edges"] if sm else None}
    return R


def snapshot_check(R):
    snap = json.loads(SNAPSHOT.read_text())

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    bad = [p for p, h in snap.items() if not Path(p).exists() or sha(p) != h]
    chk(R, f"E1-E3 results/checkpoints/configs/splits unchanged ({len(snap)}-file snapshot)", len(snap) == 1432 and not bad, bad[:20])


def main():
    index = pd.read_csv(PATHS.metadata / "index.csv")
    out = {ds: audit_dataset(ds, index) for ds in ("RAVDESS", "CREMA-D")}
    out["integrity"] = {}
    snapshot_check(out["integrity"])
    out["all_checks_passed"] = all(all(v["checks"].values()) for v in out.values() if isinstance(v, dict))
    (PATHS.results / "E4" / "audit.json").write_text(json.dumps(out, indent=1, default=float))
    print("ALL CHECKS PASSED" if out["all_checks_passed"] else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
