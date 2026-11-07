"""
E1 figures from results/E1/<DS>/summary.json and runs.csv (run stage5_e1_run.py --summarise first).

  plots/E1_confusion.png              row-normalised confusion (recall %), summed over folds (+ seeds)
  plots/E1_aggregation_comparison.png test macro-F1 per aggregation (DESCRIPTIVE: hyper-parameters
                                      selected on validation within each aggregation)
  plots/E1_per_class_f1.png           per-class F1 (headline models), mean over folds
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.preprocessing.dataset_index import CLASSES  # noqa: E402
from src.utils import plot_style as ps  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402

DATASETS = ("RAVDESS", "CREMA-D")
MODELS = {"svm": "E1-SVM", "rf": "E1-RF"}


def main():
    ps.apply()
    S = {ds: json.loads((PATHS.results / "E1" / ds / "summary.json").read_text()) for ds in DATASETS}
    D = {ds: pd.read_csv(PATHS.results / "E1" / ds / "runs.csv") for ds in DATASETS}

    # --- confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
    for r, ds in enumerate(DATASETS):
        for c, (m, name) in enumerate(MODELS.items()):
            a = axes[r, c]
            cm = np.array(S[ds][m]["headline"]["confusion_summed"], dtype=float)
            pct = 100 * cm / cm.sum(1, keepdims=True)
            a.imshow(pct, cmap=ps.SEQ_BLUE, vmin=0, vmax=100)
            for (i, j), v in np.ndenumerate(pct):
                a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 55 else ps.INK)
            a.set_xticks(range(6), CLASSES, rotation=35); a.set_yticks(range(6), CLASSES); a.grid(False)
            f1 = S[ds][m]["headline"]["test"]["macro_f1"]
            a.set_title(f"{ds} - {name}  (macro-F1 {f1['mean']:.3f} ± {f1['std']:.3f})", fontsize=10)
            a.set_xlabel("predicted"); a.set_ylabel("true")
    fig.suptitle("E1: confusion matrices, % of each true class (summed over all test folds"
                 " and RF seeds)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E1_confusion.png")

    # --- aggregation comparison (descriptive)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    aggs = ["mean", "mean_std", "functionals"]
    for a, ds in zip(axes, DATASETS):
        for k, (m, name) in enumerate(MODELS.items()):
            means, stds = [], []
            for ag in aggs:
                g = D[ds][(D[ds].model == m) & (D[ds].variant == f"agg={ag}")]
                per_fold = g.groupby("fold").test_macro_f1.mean()
                means.append(per_fold.mean()); stds.append(per_fold.std(ddof=1))
            x = np.arange(3) + (k - 0.5) * 0.18
            a.errorbar(x, means, yerr=stds, fmt="o", ms=8, capsize=4, color=ps.MODEL_COLOR[name], label=name)
        a.set_xticks(range(3), ["mean (39)", "mean+std (78)", "functionals (195)"])
        a.set_ylabel("test macro-F1 (mean ± std over folds)")
        a.set_title(ds, fontsize=11); a.legend(loc="lower right")
    fig.suptitle("E1: MFCC aggregation strategies (descriptive; hyper-parameters chosen on validation "
                 "within each aggregation)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E1_aggregation_comparison.png")

    # --- per-class F1
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for a, ds in zip(axes, DATASETS):
        x = np.arange(6)
        for k, (m, name) in enumerate(MODELS.items()):
            v = [S[ds][m]["headline"]["per_class_f1_mean"][c] for c in CLASSES]
            a.bar(x + (k - 0.5) * 0.38, v, 0.36, color=ps.MODEL_COLOR[name], label=name)
        a.set_xticks(x, CLASSES); a.set_title(ds, fontsize=11); a.set_ylim(0, 1)
        a.axhline(1 / 6, color=ps.MUTED, lw=0.8, ls="--", label="chance level (1/6)")
    axes[0].set_ylabel("test F1 (mean over folds)"); axes[0].legend(loc="upper left", ncol=3)
    fig.suptitle("E1: per-class F1", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E1_per_class_f1.png")


if __name__ == "__main__":
    main()
