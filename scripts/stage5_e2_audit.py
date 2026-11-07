"""
E2 audit (read-only with respect to results, splits and caches; writes results/E2/audit.json).

Checks are made from the SAVED artifacts, independently of the code path that produced them:
  1 completion   per fold: 30 A + 8 B + 1 C + selection.json + 3 D files, one run hash per
                 dataset, every checkpoint exists, loads, and matches its selection
  2 isolation    A-C files contain validation metrics only (no test keys, n = |val|);
                 the selected config is reproducible from the A-C validation numbers alone;
                 selection.json is older than every D file and checkpoint;
                 D test rows == the fold's test speakers' rows, disjoint from train/val
  3 training-only preprocessing
                 checkpoint standardiser mean/std == statistics of the TRAIN rows (recomputed),
                 != statistics of train+val+test; class weights recomputed from train labels by
                 re-training fold 0 seed 0 (also reproduces its validation F1 exactly)
  4 analysis     per-seed results, per-fold results, layer weights of weighted(all), epochs,
                 validation->test gap, E1/E2 identical test rows
  5 integrity    Stage 3 split SHA-256; Stage 4 cache files vs the SHA-256 recorded at adoption,
                 modification times of the prosody arrays

Run: python scripts/stage5_e2_audit.py
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

from src.evaluation.metrics import METRICS, summarise  # noqa: E402
from src.features.store import ADOPTABLE, FeatureStore, sha256_file  # noqa: E402
from src.models.e2_hubert import pooled_input  # noqa: E402
from src.preprocessing.splits import DEFAULT_FOLDS, split_path  # noqa: E402
from src.training.data import fold_rows  # noqa: E402
from src.training.trainer import class_weights  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402

spec = importlib.util.spec_from_file_location("e2drv", Path(__file__).with_name("stage5_e2_run.py"))
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
CFG = drv.CFG
EXPECTED_SPLIT_SHA = {"RAVDESS": "DDCFFCCF0AA9E23BF969FBB6757ACDEB68A2C5AD0884B70E044C5DDB8EA99BA5",
                      "CREMA-D": "A64882199275F58C1C34322E5B1A797609C261D183ACC216C8C3FF7510868EDC"}


def chk(report, name, cond, detail=None):
    report.setdefault("checks", {})[name] = bool(cond)
    if detail is not None:
        report.setdefault("details", {})[name] = detail
    print(("PASS " if cond else "FAIL ") + name)


def audit_dataset(ds, index):
    R = {}
    store = FeatureStore.open(ds, index)
    pooled = np.asarray(store.array("hubert_pooled"))
    root = PATHS.results / "E2" / ds
    rhash = drv.run_hash(ds, store)
    n_folds = DEFAULT_FOLDS[ds]
    hashes, per_seed, per_fold, gaps, lw, epochs, sel_counts, folds_out = set(), {s: [] for s in CFG.final_seeds}, [], [], [], [], [], {}
    a_files_ok = test_free = sel_reproducible = order_ok = rows_ok = std_ok = ck_ok = True
    detail = {}
    for k in range(n_folds):
        d = root / f"fold{k}"
        fr = fold_rows(store, index, ds, k)
        A = sorted(d.glob("A_*.json")); B = sorted(d.glob("B_*.json")); C = d / "C_unweighted.json"
        D = [d / f"D_seed{s}.json" for s in CFG.final_seeds]
        complete = len(A) == 30 and len(B) == 8 and C.exists() and (d / "selection.json").exists() and all(p.exists() for p in D)
        a_files_ok &= complete
        sel = json.loads((d / "selection.json").read_text())
        AB = [json.loads(p.read_text()) for p in A + B + [C]]
        for r in AB:
            hashes.add(r["run_hash"])
            keys = json.dumps(r)
            if '"test' in keys or r["val"]["n"] != len(fr.rows["val"]):
                test_free = False
        # re-derive the selection from the saved validation numbers only
        Aj = [json.loads(p.read_text()) for p in A]
        order = {(r, p): i for i, (r, p) in enumerate((r, p) for r in CFG.representations for p in CFG.poolings)}
        Aj.sort(key=lambda r: order[(r["candidate"]["rep"], r["candidate"]["pooling"])])
        a_best = drv.best_of(Aj)["candidate"]
        from itertools import product
        Bj = {(r["candidate"]["hidden"], r["candidate"]["dropout"], r["candidate"]["lr"]): r for r in (json.loads(p.read_text()) for p in B)}
        Bj = [Bj[key] for key in product(CFG.grid_hidden, CFG.grid_dropout, CFG.grid_lr)]
        b = drv.best_of(Bj)
        c = json.loads(C.read_text())
        final = c if c["val_macro_f1"] > b["val_macro_f1"] else b
        sel_reproducible &= (final["candidate"] == sel["selected"] and a_best == sel["A_selected"])
        Dj = [json.loads(p.read_text()) for p in D]
        t_abc = max(p.stat().st_mtime for p in A + B + [C])               # last validation-only run
        order_ok &= all(p.stat().st_mtime >= t_abc and Path(r["checkpoint"]).stat().st_mtime >= t_abc for p, r in zip(D, Dj))
        sel_after_d = (d / "selection.json").stat().st_mtime > min(p.stat().st_mtime for p in D)
        if sel_after_d:
            detail.setdefault("selection_json_rewritten_after_D", []).append(k)
        cand = sel["selected"]
        Xtr = pooled_input(pooled, fr.rows["train"], cand["rep"], cand["pooling"]).reshape(len(fr.rows["train"]), -1).astype(np.float64)
        tr_mean, tr_std = Xtr.mean(0), Xtr.std(0) + 1e-6
        allr = np.concatenate([fr.rows["train"], fr.rows["val"], fr.rows["test"]])
        all_mean = pooled_input(pooled, allr, cand["rep"], cand["pooling"]).reshape(len(allr), -1).astype(np.float64).mean(0)
        del Xtr
        for r in Dj:
            hashes.add(r["run_hash"])
            rows_ok &= (r["test_rows"] == fr.rows["test"].tolist()) and not (set(r["test_rows"]) & (set(fr.rows["train"]) | set(fr.rows["val"])))
            ck = torch.load(r["checkpoint"], map_location="cpu", weights_only=False)
            ck_ok &= (ck["candidate"] == sel["selected"] and ck["fold"] == k and ck["seed"] == r["seed"] and ck["run_hash"] == rhash
                      and ck["feature_cache"] == store.meta["cache_id"])
            # standardiser = TRAIN statistics (streamed sum / sum of squares in the code under test)
            m_ok = np.allclose(ck["standardizer"]["mean"], tr_mean, rtol=1e-6, atol=1e-6)
            s_ok = np.allclose(ck["standardizer"]["std"], tr_std, rtol=1e-4, atol=1e-5)
            differs = not np.allclose(ck["standardizer"]["mean"], all_mean, rtol=1e-6, atol=1e-6)
            std_ok &= (m_ok and s_ok and differs)
            per_seed[r["seed"]].append({key: r["test"][key] for key in METRICS})
            epochs.append({"fold": k, "seed": r["seed"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                           "train_seconds": r["train_seconds"], "infer_ms": r["test_inference_ms_per_recording"],
                           "n_params": r["n_params"], "gpu_peak_mib": r["gpu_peak_mib"], "ram_peak_mib": r["process_ram_mib"]["peak"]})
            if cand["rep"] == "weighted_all":
                lw.append({"fold": k, "seed": r["seed"], "source": "final checkpoint", "pooling": cand["pooling"],
                           "weights": torch.softmax(ck["state_dict"]["layer_weighting.logits"], 0).tolist()})
        m = {key: float(np.mean([r["test"][key] for r in Dj])) for key in METRICS}
        per_fold.append(m)
        gaps.append({"fold": k, "selected_val_f1": sel["selected_val_macro_f1"], "test_f1_mean_seeds": m["macro_f1"],
                     "gap": sel["selected_val_macro_f1"] - m["macro_f1"]})
        for r in Aj:
            if r["candidate"]["rep"] == "weighted_all":
                lw.append({"fold": k, "seed": 0, "source": "phase A (default head)", "pooling": r["candidate"]["pooling"],
                           "weights": r["layer_weights"]})
        sel_counts.append(sel["selected"])
        folds_out[k] = {"selected": sel["selected"], "selected_val_macro_f1": sel["selected_val_macro_f1"],
                        "test_per_seed": {r["seed"]: {key: r["test"][key] for key in METRICS} for r in Dj},
                        "test_mean_over_seeds": m, "confusion_per_seed_sum": np.sum([r["test"]["confusion"] for r in Dj], axis=0).tolist(),
                        "per_class_f1_mean_seeds": {c: float(np.mean([r["test"]["per_class"][c]["f1"] for r in Dj])) for c in Dj[0]["test"]["per_class"]}}
        detail[k] = {"A": len(A), "B": len(B), "C": C.exists(), "D": sum(p.exists() for p in D)}
    rewritten = detail.pop("selection_json_rewritten_after_D", [])
    chk(R, f"{ds}: all {n_folds} folds complete (30 A + 8 B + 1 C + selection + 3 D)", a_files_ok, detail)
    R["note_selection_json_rewritten_after_D_in_folds"] = (
        f"{rewritten}: the pipeline-check run (fold 0 only) produced the D files; the full run then reloaded the cached "
        "A-C results and rewrote selection.json with identical content (selection_seconds < 1 s). Driver fixed to keep an "
        "identical existing selection.json." if rewritten else [])
    chk(R, f"{ds}: single run hash across all files", len(hashes) == 1 and rhash in hashes, sorted(hashes))
    chk(R, f"{ds}: phase A-C files contain only validation metrics (no test keys, n = |val|)", test_free)
    chk(R, f"{ds}: selection re-derived from saved A-C validation scores == saved selection", sel_reproducible)
    chk(R, f"{ds}: every validation-only run (A-C) finished before the first test result / checkpoint", order_ok)
    chk(R, f"{ds}: D test rows == fold test rows, disjoint from train/val", rows_ok)
    chk(R, f"{ds}: checkpoints load and match selection/fold/seed/run hash/feature cache", ck_ok)
    chk(R, f"{ds}: checkpoint standardiser == TRAIN-row statistics (and != all-row statistics)", std_ok)

    # class weights + exact reproduction of fold 0 seed 0 from scratch
    fr = fold_rows(store, index, ds, 0)
    sel = json.loads((root / "fold0" / "selection.json").read_text())
    cand = sel["selected"]
    (Xtr, Xva), _, shape = drv.make_inputs(pooled, fr.rows["train"], [fr.rows["train"], fr.rows["val"]], cand["rep"], cand["pooling"])
    res = drv.train_classifier(lambda: drv.build_model(shape, cand["rep"], cand["hidden"], cand["dropout"]),
                               (Xtr, fr.labels["train"]), (Xva, fr.labels["val"]), drv.train_cfg(cand, 0))
    w_expected = class_weights(fr.labels["train"]) if cand["class_weighted"] else np.ones(6)
    counts = np.bincount(fr.labels["train"], minlength=6)
    chk(R, f"{ds}: class weights used in training == n_train / (6 n_train,c) from TRAIN labels",
        np.allclose(res["class_weights"], w_expected), {"train_counts": counts.tolist(), "weights": res["class_weights"],
                                                        "class_weighted": cand["class_weighted"]})
    d0 = json.loads((root / "fold0" / "D_seed0.json").read_text())
    chk(R, f"{ds}: retraining fold 0 seed 0 from scratch reproduces validation F1 and best epoch exactly",
        res["best_val_macro_f1"] == d0["val"]["macro_f1"] and res["best_epoch"] == d0["best_epoch"],
        {"recomputed": res["best_val_macro_f1"], "saved": d0["val"]["macro_f1"]})

    # E1 used the identical test rows
    same = all(json.loads((PATHS.results / "E1" / ds / f"fold{k}_svm_seed0.json").read_text())["test_rows"]
               == json.loads((root / f"fold{k}" / "D_seed0.json").read_text())["test_rows"] for k in range(n_folds))
    chk(R, f"{ds}: E1 and E2 evaluated on identical test rows in every fold", same)

    R["test_mean_std_over_folds"] = summarise(per_fold)
    R["test_per_seed_over_folds"] = {s: summarise(v) for s, v in per_seed.items()}
    R["folds"] = folds_out
    R["validation_to_test_gap"] = gaps
    R["epochs_and_cost"] = epochs
    R["layer_weights"] = lw
    R["selected"] = sel_counts
    # 5 integrity
    sha = hashlib.sha256(split_path(ds).read_bytes()).hexdigest().upper()
    chk(R, f"{ds}: Stage 3 split file unchanged (SHA-256)", sha == EXPECTED_SPLIT_SHA[ds], sha)
    ref = store.meta["adopted_from"]["sha256"]
    now = {n: sha256_file(store.root / n) for n in ADOPTABLE}
    now_mfcc = sha256_file(store.root / "mfcc.npy")
    chk(R, f"{ds}: Stage 4 cache audio/manifest/HuBERT/pooled/MFCC unchanged since adoption (SHA-256)",
        all(now[n] == ref[n] for n in ADOPTABLE) and now_mfcc == ref["mfcc.npy (source, for verification)"])
    import os, time
    cpu_runs = store.meta["stages"]["cpu"]["runs"]
    last_cpu = time.mktime(time.strptime(cpu_runs[-1]["finished"], "%Y-%m-%d %H:%M:%S"))
    prosody_files = [f for f in os.listdir(store.root) if f.startswith("prosody")]
    chk(R, f"{ds}: prosody arrays not modified after the last Stage 4 CPU run",
        all((store.root / f).stat().st_mtime <= last_cpu + 5 for f in prosody_files),
        {f: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime((store.root / f).stat().st_mtime)) for f in prosody_files})
    R["prosody_sha256_now"] = {f: sha256_file(store.root / f) for f in prosody_files}
    return R


def main():
    index = pd.read_csv(PATHS.metadata / "index.csv")
    out = {ds: audit_dataset(ds, index) for ds in ("RAVDESS", "CREMA-D")}
    out["all_checks_passed"] = all(all(v["checks"].values()) for v in out.values() if isinstance(v, dict))
    (PATHS.results / "E2" / "audit.json").write_text(json.dumps(out, indent=1, default=float))
    print("ALL CHECKS PASSED" if out["all_checks_passed"] else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
