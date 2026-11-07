"""
E4 controls (run AFTER the E4 runs; never used for any selection or model decision).

1. Label permutation (leakage control, same design as E2/E3): for every fold the selected
   xattn_HMP configuration (same HuBERT layer, same hyper-parameters) is retrained (seed 0) on the
   SAME training frames with permuted training labels (seed 1000 + fold; validation labels intact),
   then the test speakers are predicted once. Without leakage, test macro-F1 must be near 1/6.
   -> results/E4/<DS>/control_label_permutation.json

2. Feature-swap diagnostic (SUPPLEMENTARY, not evidence of feature contribution): every saved
   xattn_HMP and concat_HMP checkpoint (3 seeds x folds) is evaluated on validation and test with
   the MFCC and prosody frames replaced by those of another recording of the same partition
   (fixed derangement, seed 0; donor frames time-stretched to the recipient's length by nearest
   index). HuBERT is intact. Reported: macro-F1 intact vs swapped. Interpretation limits: it
   measures the sensitivity of a TRAINED model to off-distribution input pairs, not what a stream
   contributes to learning (that is answered by the trained ablations).
   -> results/E4/<DS>/diagnostic_feature_swap.json

Run: python scripts/stage5_e4_controls.py --datasets RAVDESS [--only permutation|swap]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.features.store import FeatureStore  # noqa: E402
from src.preprocessing.splits import DEFAULT_FOLDS  # noqa: E402
from src.training.data import fold_rows  # noqa: E402
from src.training.seq_trainer import predict_rows, train_sequence_classifier  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402

spec = importlib.util.spec_from_file_location("e4drv", Path(__file__).with_name("stage5_e4_run.py"))
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
SWAP_SEED = 0
SWAP_MODELS = ("xattn_HMP", "concat_HMP")


def permutation_fold(data, fr, sel, k):
    hp, spec_ = sel["selected"]["xattn_HMP"], drv.parse_variant("xattn_HMP")
    data.fit(fr.rows["train"], None)
    y_perm = np.random.default_rng(1000 + k).permutation(fr.labels["train"])
    r = train_sequence_classifier(lambda: drv.build_model(spec_, hp), data, (fr.rows["train"], y_perm),
                                  (fr.rows["val"], fr.labels["val"]), drv.train_cfg(hp, 0))
    m = compute_metrics(fr.labels["test"], predict_rows(r["model"], data, fr.rows["test"]))
    return {"fold": k, "layer": sel["layer"], "hp": hp, "test_macro_f1": m["macro_f1"], "test_UA": m["UA"],
            "test_accuracy": m["accuracy"], "val_macro_f1": r["best_val_macro_f1"], "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"], "share_train_labels_unchanged": float((y_perm == fr.labels["train"]).mean())}


def swap_fold(data, fr, d: Path, k):
    data.fit(fr.rows["train"], None)
    out = []
    donors = {p: fr.rows[p][drv.derangement(len(fr.rows[p]), SWAP_SEED)] for p in ("val", "test")}
    for v in SWAP_MODELS:
        for s in drv.CFG.final_seeds:
            D = json.loads((d / f"D_{v}_seed{s}.json").read_text())
            model, _ = drv.load_checkpoint(D["checkpoint"])
            row = {"fold": k, "variant": v, "seed": s}
            intact_test = predict_rows(model, data, fr.rows["test"])
            if intact_test.tolist() != D["test_pred"]:
                raise RuntimeError(f"fold{k} {v} seed{s}: checkpoint does not reproduce its saved test predictions")
            for p in ("val", "test"):
                intact = predict_rows(model, data, fr.rows[p])
                sw = drv.predict_swapped(model, data, fr.rows[p], donors[p])
                f_i = compute_metrics(fr.labels[p], intact)["macro_f1"]
                f_s = compute_metrics(fr.labels[p], sw)["macro_f1"]
                row[p] = {"macro_f1_intact": f_i, "macro_f1_swapped": f_s, "delta": f_s - f_i,
                          "share_predictions_changed": float((intact != sw).mean())}
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["RAVDESS", "CREMA-D"])
    ap.add_argument("--only", choices=["permutation", "swap"], default=None)
    args = ap.parse_args()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    for ds in args.datasets:
        store = FeatureStore.open(ds, index)
        dc = drv.DataCache(store)
        perm, swap = [], []
        for k in range(DEFAULT_FOLDS[ds]):
            fr = fold_rows(store, index, ds, k)
            d = PATHS.results / "E4" / ds / f"fold{k}"
            sel = json.loads((d / "selection.json").read_text())
            data = dc.get(sel["layer"])
            if args.only in (None, "permutation"):
                perm.append(permutation_fold(data, fr, sel, k)); print(ds, "perm", perm[-1], flush=True)
            if args.only in (None, "swap"):
                swap += swap_fold(data, fr, d, k); print(ds, "swap fold", k, "done", flush=True)
        if perm:
            f1 = np.array([r["test_macro_f1"] for r in perm])
            out = {"dataset": ds, "control": "training labels permuted (seed 1000+fold), selected xattn_HMP config, seed 0",
                   "folds": perm, "test_macro_f1_mean": float(f1.mean()), "test_macro_f1_std": float(f1.std(ddof=1)),
                   "chance_level": 1 / 6}
            drv.write_atomic(PATHS.results / "E4" / ds / "control_label_permutation.json", out, indent=1)
            print(ds, "permuted-label test macro-F1", f"{f1.mean():.3f} ± {f1.std(ddof=1):.3f}", flush=True)
        if swap:
            agg = {}
            for v in SWAP_MODELS:
                for p in ("val", "test"):
                    per_fold = [np.mean([r[p]["delta"] for r in swap if r["variant"] == v and r["fold"] == k])
                                for k in range(DEFAULT_FOLDS[ds])]
                    agg[f"{v} {p}"] = {"delta_mean_over_folds": float(np.mean(per_fold)), "delta_std_over_folds": float(np.std(per_fold, ddof=1)),
                                       "delta_per_fold": [float(x) for x in per_fold],
                                       "share_predictions_changed_mean": float(np.mean([r[p]["share_predictions_changed"] for r in swap if r["variant"] == v]))}
            out = {"dataset": ds, "status": "SUPPLEMENTARY diagnostic - not definitive evidence of feature contribution",
                   "method": "MFCC+prosody of each recording replaced by those of another recording of the same partition "
                             "(derangement seed 0), time-stretched to the recipient length (nearest index); HuBERT intact; "
                             "saved checkpoints, no training",
                   "runs": swap, "aggregate": agg}
            drv.write_atomic(PATHS.results / "E4" / ds / "diagnostic_feature_swap.json", out, indent=1)
            for n, a in agg.items():
                print(ds, "swap", n, f"delta {a['delta_mean_over_folds']:+.3f} ± {a['delta_std_over_folds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
