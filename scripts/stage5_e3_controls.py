"""
E3 leakage control (run AFTER the E3 runs; never used for any selection).

Label-permutation test, same design as E2: for every fold the selected concat_HMP configuration
(same HuBERT layer, same hyper-parameters) is retrained (seed 0) on the SAME training frames with
randomly permuted training labels (validation labels intact, so early stopping behaves as usual),
then the test speakers are predicted once. Without leakage, test macro-F1 must be near chance (1/6).

Run: python scripts/stage5_e3_controls.py --datasets RAVDESS CREMA-D
Writes results/E3/<DS>/control_label_permutation.json
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

spec = importlib.util.spec_from_file_location("e3drv", Path(__file__).with_name("stage5_e3_run.py"))
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["RAVDESS", "CREMA-D"])
    args = ap.parse_args()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    for ds in args.datasets:
        store = FeatureStore.open(ds, index)
        dc = drv.DataCache(store)
        res = []
        for k in range(DEFAULT_FOLDS[ds]):
            fr = fold_rows(store, index, ds, k)
            sel = json.loads((PATHS.results / "E3" / ds / f"fold{k}" / "selection.json").read_text())
            hp, spec_ = sel["selected"]["concat_HMP"], drv.parse_variant("concat_HMP")
            data = dc.get(sel["layer"]).fit(fr.rows["train"], None)
            y_perm = np.random.default_rng(1000 + k).permutation(fr.labels["train"])
            r = train_sequence_classifier(lambda: drv.build_model(spec_, hp), data, (fr.rows["train"], y_perm),
                                          (fr.rows["val"], fr.labels["val"]), drv.train_cfg(hp, 0))
            m = compute_metrics(fr.labels["test"], predict_rows(r["model"], data, fr.rows["test"]))
            res.append({"fold": k, "layer": sel["layer"], "test_macro_f1": m["macro_f1"], "test_UA": m["UA"],
                        "test_accuracy": m["accuracy"], "val_macro_f1": r["best_val_macro_f1"],
                        "best_epoch": r["best_epoch"],
                        "share_train_labels_unchanged": float((y_perm == fr.labels["train"]).mean())})
            print(ds, res[-1], flush=True)
        f1 = np.array([r["test_macro_f1"] for r in res])
        out = {"dataset": ds, "control": "training labels permuted (seed 1000+fold), selected concat_HMP config, seed 0",
               "folds": res, "test_macro_f1_mean": float(f1.mean()), "test_macro_f1_std": float(f1.std(ddof=1)),
               "chance_level": 1 / 6}
        (PATHS.results / "E3" / ds / "control_label_permutation.json").write_text(json.dumps(out, indent=1))
        print(ds, "permuted-label test macro-F1", f"{f1.mean():.3f} ± {f1.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
