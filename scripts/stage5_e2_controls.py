"""
E2 leakage control (run AFTER the E2 runs; never used for any selection).

Label-permutation test: for every fold, the fold's selected E2 configuration is retrained
(seed 0) on the SAME training features but with randomly permuted training labels
(validation labels intact, so early stopping behaves as usual), then the test speakers are
predicted once. Without leakage the model cannot learn emotion and test macro-F1 must be near
chance (~0.17 for 6 classes). A clearly higher value would mean information reaches the model
through some path other than the training labels (e.g. overlapping speakers / recordings).

Run: python scripts/stage5_e2_controls.py --datasets RAVDESS CREMA-D
Writes results/E2/<DS>/control_label_permutation.json
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
from src.training.trainer import predict, train_classifier  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402

spec = importlib.util.spec_from_file_location("e2drv", Path(__file__).with_name("stage5_e2_run.py"))
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["RAVDESS", "CREMA-D"])
    args = ap.parse_args()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    for ds in args.datasets:
        store = FeatureStore.open(ds, index)
        pooled = np.asarray(store.array("hubert_pooled"))
        res = []
        for k in range(DEFAULT_FOLDS[ds]):
            fr = fold_rows(store, index, ds, k)
            cand = json.loads((PATHS.results / "E2" / ds / f"fold{k}" / "selection.json").read_text())["selected"]
            (Xtr, Xva, Xte), _, shape = drv.make_inputs(pooled, fr.rows["train"],
                                                        [fr.rows["train"], fr.rows["val"], fr.rows["test"]],
                                                        cand["rep"], cand["pooling"])
            y_perm = np.random.default_rng(1000 + k).permutation(fr.labels["train"])
            r = train_classifier(lambda: drv.build_model(shape, cand["rep"], cand["hidden"], cand["dropout"]),
                                 (Xtr, y_perm), (Xva, fr.labels["val"]), drv.train_cfg(cand, 0))
            m = compute_metrics(fr.labels["test"], predict(r["model"], Xte))
            res.append({"fold": k, "test_macro_f1": m["macro_f1"], "test_UA": m["UA"], "test_accuracy": m["accuracy"],
                        "val_macro_f1": r["best_val_macro_f1"], "share_train_labels_unchanged": float((y_perm == fr.labels["train"]).mean())})
            print(ds, res[-1])
        f1 = np.array([r["test_macro_f1"] for r in res])
        out = {"dataset": ds, "control": "training labels permuted (seed 1000+fold), selected E2 config, seed 0",
               "folds": res, "test_macro_f1_mean": float(f1.mean()), "test_macro_f1_std": float(f1.std(ddof=1)),
               "chance_level": 1 / 6}
        (PATHS.results / "E2" / ds / "control_label_permutation.json").write_text(json.dumps(out, indent=1))
        print(ds, "permuted-label test macro-F1", f"{f1.mean():.3f} ± {f1.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
