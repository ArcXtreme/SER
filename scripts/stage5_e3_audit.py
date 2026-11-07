"""
E3 audit (read-only with respect to results, splits and caches; writes results/E3/audit.json).

Checks are made from the SAVED artifacts, independently of the code path that produced them:
  1 completion   per fold: 8 S files + selection.json + 3 seeds x 10 variants D files, one run hash
                 per fold, every checkpoint exists, loads and matches its D file
  2 isolation    S files contain validation metrics only (no test keys, n = |val|); the selected
                 hyper-parameters are reproducible from the S validation numbers alone; every S
                 file and selection.json is older than every D file and checkpoint; D test rows ==
                 the fold's test rows; the HuBERT layer equals the rule applied to the fold's own
                 E2 phase-A file (SHA-256 unchanged)
  3 training-only preprocessing
                 checkpoint standardiser mean/std == statistics of the TRAIN rows (recomputed for
                 every stream, incl. the HuBERT layer); DisVoice fill values == training medians;
                 class weights == recomputed from training labels
  4 reproduction every checkpoint, reloaded, re-predicts its saved test predictions exactly;
                 seed-0 finals reproduce the selection runs' validation F1
  5 capacity     control parameter count within 5 % of the E4 reference
  6 integrity    Stage 3 split SHA-256; Stage 4 cache files vs the SHA-256 recorded at adoption;
                 prosody arrays vs the SHA-256 recorded by the E2 audit

Run: python scripts/stage5_e3_audit.py
"""
import hashlib
import importlib.util
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.features.store import ADOPTABLE, FeatureStore, sha256_file  # noqa: E402
from src.models.e3_fusion import E3Fusion, e4_reference_fusion_params, n_params  # noqa: E402
from src.preprocessing.splits import DEFAULT_FOLDS, split_path  # noqa: E402
from src.training.data import fold_rows  # noqa: E402
from src.training.preprocessing import FrameStandardizer  # noqa: E402
from src.training.seq_trainer import predict_rows  # noqa: E402
from src.training.trainer import class_weights  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402

spec = importlib.util.spec_from_file_location("e3drv", Path(__file__).with_name("stage5_e3_run.py"))
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
CFG = drv.CFG
EXPECTED_SPLIT_SHA = {"RAVDESS": "DDCFFCCF0AA9E23BF969FBB6757ACDEB68A2C5AD0884B70E044C5DDB8EA99BA5",
                      "CREMA-D": "A64882199275F58C1C34322E5B1A797609C261D183ACC216C8C3FF7510868EDC"}


def chk(report, name, cond, detail=None):
    report.setdefault("checks", {})[name] = bool(cond)
    if detail is not None:
        report.setdefault("details", {})[name] = detail
    print(("PASS " if cond else "FAIL ") + name, flush=True)


def audit_dataset(ds, index):
    R = {}
    store = FeatureStore.open(ds, index)
    root = PATHS.results / "E3" / ds
    variants = CFG.tuned + CFG.ablations
    ok = {k: True for k in ("complete", "one_hash", "test_free", "sel_repro", "order", "rows", "layer", "std", "dv",
                            "cw", "ck_meta", "repredict", "seed0", "hp")}
    detail = {"order": {}, "std_max_abs_diff": {}, "repredict_mismatch": []}
    dc = drv.DataCache(store)
    for k in range(DEFAULT_FOLDS[ds]):
        d = root / f"fold{k}"
        fr = fold_rows(store, index, ds, k)
        S_files = [d / f"S_{v}_d{dr}_lr{lr}.json" for v in CFG.tuned for dr, lr in product(CFG.grid_dropout, CFG.grid_lr)]
        D_files = {(v, s): d / f"D_{v}_seed{s}.json" for v in variants for s in CFG.final_seeds}
        ok["complete"] &= (d / "selection.json").exists() and all(p.exists() for p in S_files) and \
            all(p.exists() for p in D_files.values())
        sel = json.loads((d / "selection.json").read_text())
        Sj = [json.loads(p.read_text()) for p in S_files]
        Dj = {key: json.loads(p.read_text()) for key, p in D_files.items()}
        ok["one_hash"] &= len({sel["run_hash"]} | {r["run_hash"] for r in Sj} | {r["run_hash"] for r in Dj.values()}) == 1
        # 2 isolation
        for r in Sj:
            if '"test' in json.dumps(r) or r["val"]["n"] != len(fr.rows["val"]):
                ok["test_free"] = False
        for v in CFG.tuned:
            runs = [r for r in Sj if r["variant"] == v]
            best = runs[max(range(len(runs)), key=lambda j: (runs[j]["val_macro_f1"], -j))]
            ok["sel_repro"] &= best["hp"] == sel["selected"][v] and best["val_macro_f1"] == sel["selected_val_macro_f1"][v]
        t_sel = max([p.stat().st_mtime for p in S_files] + [(d / "selection.json").stat().st_mtime])
        t_d = min([p.stat().st_mtime for p in D_files.values()] + [Path(r["checkpoint"]).stat().st_mtime for r in Dj.values()])
        ok["order"] &= t_sel <= t_d
        detail["order"][k] = {"last_selection_artifact": t_sel, "first_test_artifact": t_d}
        ok["rows"] &= all(r["test_rows"] == fr.rows["test"].tolist() and r["test"]["n"] == len(fr.rows["test"]) for r in Dj.values())
        e2p = PATHS.results / "E2" / ds / f"fold{k}" / "selection.json"
        li = drv.select_layer(json.loads(e2p.read_text())["A"])
        ok["layer"] &= li["layer"] == sel["layer"] == sel["layer_info"]["layer"] and \
            hashlib.sha256(e2p.read_bytes()).hexdigest() == sel["layer_info"]["source_sha256"] and \
            all(r["layer"] == sel["layer"] for r in Dj.values())
        for (v, s), r in Dj.items():
            ok["hp"] &= r["hp"] == sel["selected"][drv.hp_source(v)]
            if v in CFG.tuned and s == CFG.selection_seed:
                ok["seed0"] &= abs(max(h["val_macro_f1"] for h in r["history"]) - sel["selected_val_macro_f1"][v]) < 1e-12
            ok["cw"] &= np.allclose(r["class_weights"], class_weights(fr.labels["train"]))
        # 3 training-only statistics, recomputed from the cache (independent of SequenceData)
        data = dc.get(sel["layer"])
        ref = {}
        for s, arr in (("hubert", data._np["hubert"]), ("mfcc", np.asarray(store.array("mfcc"))),
                       ("prosody", np.asarray(store.array("prosody_frames")))):
            off, n = store.manifest.frame_offset.values, store.manifest.n_frames.values
            ref[s] = FrameStandardizer(passthrough=(1, 2) if s == "prosody" else ()).fit(
                [arr[off[r]: off[r] + n[r]] for r in fr.rows["train"]])
        for (v, s), r in Dj.items():
            ck = torch.load(r["checkpoint"], map_location="cpu", weights_only=False)
            ok["ck_meta"] &= (ck["fold"], ck["seed"], ck["variant"], ck["layer"], ck["run_hash"], ck["hp"]) == \
                (k, s, v, sel["layer"], sel["run_hash"], r["hp"])
            for st, z in ref.items():
                diff = float(max(np.abs(ck["standardizers"][st]["mean"] - z.mean_).max(), np.abs(ck["standardizers"][st]["std"] - z.std_).max()))
                detail["std_max_abs_diff"][st] = max(detail["std_max_abs_diff"].get(st, 0.0), diff)
                ok["std"] &= diff == 0.0
            if ck["disvoice_preprocessor"] is not None:
                X = np.asarray(store.array("prosody103_wide"), np.float64)[fr.rows["train"]]
                imp = np.asarray(store.array("prosody103_wide_imputed"))[fr.rows["train"]]
                med = np.nanmedian(np.where(imp, np.nan, X), axis=0)
                ok["dv"] &= np.allclose(ck["disvoice_preprocessor"]["fill"], np.where(np.isfinite(med), med, 0.0))
        # 4 reproduction of the saved test predictions from every checkpoint
        for (v, s), r in Dj.items():
            spec_ = drv.parse_variant(v)
            data.fit(fr.rows["train"], spec_["disvoice_indicators"])
            model, _ = drv.load_checkpoint(r["checkpoint"])
            p = predict_rows(model, data, fr.rows["test"])
            if p.tolist() != r["test_pred"]:
                ok["repredict"] = False
                detail["repredict_mismatch"].append([k, v, s])
        print(f"  fold {k} audited", flush=True)
    chk(R, f"{ds}: every fold complete (8 S + selection + 30 D files + checkpoints)", ok["complete"])
    chk(R, f"{ds}: one run hash per fold across S, selection and D", ok["one_hash"])
    chk(R, f"{ds}: S (selection) files contain validation metrics only", ok["test_free"])
    chk(R, f"{ds}: selected hyper-parameters re-derivable from S validation F1 alone", ok["sel_repro"])
    chk(R, f"{ds}: all S files and selection.json older than every D file and checkpoint", ok["order"], detail["order"])
    chk(R, f"{ds}: D test rows == the fold's test rows", ok["rows"])
    chk(R, f"{ds}: HuBERT layer == rule on the fold's own E2 phase-A file (SHA-256 unchanged)", ok["layer"])
    chk(R, f"{ds}: D hyper-parameters == selection (ablations use concat_HMP's)", ok["hp"])
    chk(R, f"{ds}: seed-0 finals reproduce the selection validation F1", ok["seed0"])
    chk(R, f"{ds}: class weights == recomputed from training labels", ok["cw"])
    chk(R, f"{ds}: checkpoint metadata matches D files", ok["ck_meta"])
    chk(R, f"{ds}: standardiser statistics == TRAIN-row statistics (all streams, all checkpoints)", ok["std"],
        detail["std_max_abs_diff"])
    chk(R, f"{ds}: DisVoice fill values == training medians", ok["dv"])
    chk(R, f"{ds}: every reloaded checkpoint re-predicts its saved test predictions exactly", ok["repredict"],
        detail["repredict_mismatch"])
    # 5 capacity
    e3 = n_params(E3Fusion(n_sa_blocks=0)); sa = n_params(E3Fusion(n_sa_blocks=CFG.n_sa_blocks))
    e4 = e3 - (3 * CFG.d_model * CFG.d_model + CFG.d_model) + e4_reference_fusion_params()
    chk(R, f"{ds}: control parameters within 5 % of the E4 reference", abs(sa - e4) / e4 < 0.05,
        {"concat_HMP": e3, "sa_HMP": sa, "e4_reference_total": e4, "rel_diff": (sa - e4) / e4})
    # 6 integrity
    sha = hashlib.sha256(split_path(ds).read_bytes()).hexdigest().upper()
    chk(R, f"{ds}: Stage 3 split file unchanged (SHA-256)", sha == EXPECTED_SPLIT_SHA[ds], sha)
    ref = store.meta["adopted_from"]["sha256"]
    now = {n: sha256_file(store.root / n) for n in ADOPTABLE}
    chk(R, f"{ds}: Stage 4 audio/manifest/HuBERT/pooled/MFCC unchanged since adoption (SHA-256)",
        all(now[n] == ref[n] for n in ADOPTABLE) and sha256_file(store.root / "mfcc.npy") == ref["mfcc.npy (source, for verification)"])
    e2a = json.loads((PATHS.results / "E2" / "audit.json").read_text())[ds]["prosody_sha256_now"]
    chk(R, f"{ds}: prosody arrays unchanged since the E2 audit (SHA-256)",
        all(sha256_file(store.root / f) == h for f, h in e2a.items()))
    return R


def main():
    index = pd.read_csv(PATHS.metadata / "index.csv")
    out = {ds: audit_dataset(ds, index) for ds in ("RAVDESS", "CREMA-D")}
    out["all_checks_passed"] = all(all(v["checks"].values()) for v in out.values() if isinstance(v, dict))
    (PATHS.results / "E3" / "audit.json").write_text(json.dumps(out, indent=1, default=float))
    print("ALL CHECKS PASSED" if out["all_checks_passed"] else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
