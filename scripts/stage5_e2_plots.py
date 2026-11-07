"""
E2 figures (run stage5_e2_run.py --summarise first). VALIDATION and TEST figures are separate.

  plots/E2_layerwise_validation.png   phase A validation macro-F1 per hidden state, both poolings
                                      (+ concat(1,9), weighted(all)); mean +- std over folds
  plots/E2_pooling_validation.png     paired validation difference mean_std - mean per representation
  plots/E2_confusion_test.png         test confusion (row %), summed over folds and seeds
  plots/E1_vs_E2_per_class_f1.png     test per-class F1: E1-SVM, E1-RF, E2
  plots/E1_vs_E2_folds_test.png       test macro-F1 per fold (paired lines) E1-SVM, E1-RF, E2
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
POOL_COLOR = {"mean": ps.SERIES[0], "mean_std": ps.SERIES[1]}
POOL_LABEL = {"mean": "mean (768)", "mean_std": "mean + std (1536)"}


def phase_a(ds):
    rows = []
    root = PATHS.results / "E2" / ds
    for d in sorted(root.glob("fold*")):
        for a in json.loads((d / "selection.json").read_text())["A"]:
            rows.append({"fold": int(d.name[4:]), **a})
    return pd.DataFrame(rows)


def main():
    ps.apply()
    S2 = {ds: json.loads((PATHS.results / "E2" / ds / "summary.json").read_text()) for ds in DATASETS}
    S1 = {ds: json.loads((PATHS.results / "E1" / ds / "summary.json").read_text()) for ds in DATASETS}

    # 1 layer-wise (validation)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for a, ds in zip(axes, DATASETS):
        A = phase_a(ds)
        for p in ("mean", "mean_std"):
            g = A[(A.pooling == p) & A.rep.str.startswith("layer")].copy()
            g["l"] = g.rep.str[5:].astype(int)
            s = g.groupby("l").val_macro_f1.agg(["mean", "std"])
            a.errorbar(s.index + (-0.1 if p == "mean" else 0.1), s["mean"], yerr=s["std"], marker="o", ms=5, capsize=2,
                       color=POOL_COLOR[p], label=POOL_LABEL[p], lw=1.6)
            for x, rep in ((13.6, "concat_1_9"), (14.8, "weighted_all")):
                v = A[(A.pooling == p) & (A.rep == rep)].val_macro_f1
                a.errorbar(x + (-0.1 if p == "mean" else 0.1), v.mean(), yerr=v.std(ddof=1), marker="D", ms=6, capsize=2,
                           color=POOL_COLOR[p], ls="none", elinewidth=1.6)
        a.axvline(12.8, color=ps.GRID, lw=1)
        a.set_xticks(list(range(13)) + [13.6, 14.8], [str(i) for i in range(13)] + ["1+9", "all\n(wtd)"])
        for l in (1, 9, 12):
            a.get_xticklabels()[l].set_fontweight("bold")
        a.set_xlabel("HuBERT hidden state (0 = Transformer input; bold = layers 1, 9, 12)")
        a.set_ylabel("VALIDATION macro-F1 (mean ± std over folds)")
        a.set_title(f"{ds} ({len(A.fold.unique())} folds)", fontsize=11)
        a.legend(loc="lower right")
    fig.suptitle("E2 layer-wise probe: validation macro-F1 per hidden state (default head; no test data used)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E2_layerwise_validation.png")

    # 2 pooling (validation, paired)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for a, ds in zip(axes, DATASETS):
        A = phase_a(ds)
        piv = A.pivot_table(index=["fold", "rep"], columns="pooling", values="val_macro_f1")
        d = (piv["mean_std"] - piv["mean"]).reset_index(name="d")
        order = [f"layer{l}" for l in range(13)] + ["concat_1_9", "weighted_all"]
        s = d.groupby("rep").d.agg(["mean", "std"]).reindex(order)
        nb = d.assign(b=d.d > 0).groupby("rep").b.sum().reindex(order)
        x = np.arange(len(order))
        a.bar(x, s["mean"], 0.6, color=ps.SERIES[1])
        a.errorbar(x, s["mean"], yerr=s["std"], fmt="none", ecolor=ps.INK_2, capsize=2, lw=1)
        a.axhline(0, color=ps.INK_2, lw=0.8)
        for xi, (m_, n_) in enumerate(zip(s["mean"], nb)):
            a.text(xi, (m_ if m_ > 0 else 0) + s["std"].max() + 0.004, f"{int(n_)}/{d.fold.nunique()}", ha="center",
                   fontsize=7, color=ps.INK_2)
        a.set_xticks(x, [o.replace("layer", "") .replace("concat_1_9", "1+9").replace("weighted_all", "wtd") for o in order])
        a.set_xlabel("representation (hidden state)"); a.set_title(ds, fontsize=11)
    axes[0].set_ylabel("Δ VALIDATION macro-F1\n(mean+std − mean)")
    fig.suptitle("E2 pooling: paired validation difference per fold (numbers = folds where mean+std is better)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E2_pooling_validation.png")

    # 3 confusion (test)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for a, ds in zip(axes, DATASETS):
        cm = np.array(S2[ds]["confusion_summed_over_folds_and_seeds"], float)
        pct = 100 * cm / cm.sum(1, keepdims=True)
        a.imshow(pct, cmap=ps.SEQ_BLUE, vmin=0, vmax=100)
        for (i, j), v in np.ndenumerate(pct):
            a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 55 else ps.INK)
        a.set_xticks(range(6), CLASSES, rotation=35); a.set_yticks(range(6), CLASSES); a.grid(False)
        f1 = S2[ds]["test"]["macro_f1"]
        a.set_title(f"{ds} - E2  (TEST macro-F1 {f1['mean']:.3f} ± {f1['std']:.3f})", fontsize=10)
        a.set_xlabel("predicted"); a.set_ylabel("true")
    fig.suptitle("E2: test confusion matrices, % of each true class (all folds, 3 seeds)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E2_confusion_test.png")

    # 4 per-class F1 E1 vs E2 (test)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
    for a, ds in zip(axes, DATASETS):
        x = np.arange(6)
        series = [("E1-SVM", S1[ds]["svm"]["headline"]["per_class_f1_mean"]),
                  ("E1-RF", S1[ds]["rf"]["headline"]["per_class_f1_mean"]), ("E2", S2[ds]["per_class_f1_mean"])]
        for k, (name, d) in enumerate(series):
            a.bar(x + (k - 1) * 0.27, [d[c] for c in CLASSES], 0.25, color=ps.MODEL_COLOR[name], label=name)
        a.axhline(1 / 6, color=ps.MUTED, lw=0.8, ls="--", label="chance level (1/6)")
        a.set_xticks(x, CLASSES); a.set_ylim(0, 1); a.set_title(ds, fontsize=11)
    axes[0].set_ylabel("TEST F1 (mean over folds)"); axes[0].legend(loc="upper left", ncol=4)
    fig.suptitle("E1 vs E2: per-class test F1", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E1_vs_E2_per_class_f1.png")

    # 5 per-fold paired (test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for a, ds in zip(axes, DATASETS):
        e1 = pd.read_csv(PATHS.results / "E1" / ds / "runs.csv")
        e1 = e1[e1.variant == "headline"].groupby(["model", "fold"]).test_macro_f1.mean()
        folds = sorted(int(k) for k in S2[ds]["folds"])
        e2 = [S2[ds]["folds"][str(k)]["test_mean_over_seeds"]["macro_f1"] for k in folds]
        xs = np.arange(3)
        for k, f in enumerate(folds):
            a.plot(xs, [e1["rf"][f], e1["svm"][f], e2[k]], color=ps.GRID, lw=1, zorder=1)
        for xi, (name, vals) in enumerate((("E1-RF", [e1["rf"][f] for f in folds]),
                                           ("E1-SVM", [e1["svm"][f] for f in folds]), ("E2", e2))):
            a.scatter([xi] * len(vals), vals, color=ps.MODEL_COLOR[name], s=40, zorder=2, edgecolor=ps.SURFACE)
            a.errorbar(xi + 0.18, np.mean(vals), yerr=np.std(vals, ddof=1), fmt="D", color=ps.INK, capsize=3, ms=5)
        a.set_xticks(xs, ["E1-RF", "E1-SVM", "E2 (HuBERT)"]); a.set_title(ds, fontsize=11)
        a.set_ylabel("TEST macro-F1 (dots = folds; ◆ mean ± std)")
    fig.suptitle("E1 vs E2 on identical speaker-independent folds (lines connect the same fold)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E1_vs_E2_folds_test.png")

    # 6 learned layer weights of weighted(all) (phase A, default head, one run per fold and pooling)
    audit = json.loads((PATHS.results / "E2" / "audit.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for a, ds in zip(axes, DATASETS):
        for r in audit[ds]["layer_weights"]:
            if r["source"].startswith("phase A"):
                a.plot(range(13), r["weights"], color=POOL_COLOR[r["pooling"]], lw=1.2, alpha=0.8, marker="o", ms=3)
        a.axhline(1 / 13, color=ps.INK_2, lw=0.9, ls="--")
        a.text(-0.3, 1 / 13 + 0.0008, "uniform 1/13", fontsize=7, color=ps.INK_2, ha="left", va="bottom")
        a.set_xticks(range(13)); a.set_xlabel("hidden state"); a.set_title(ds, fontsize=11)
        for p in ("mean", "mean_std"):
            a.plot([], [], color=POOL_COLOR[p], label=f"{POOL_LABEL[p]} (one line per fold)")
        a.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("softmax layer weight")
    fig.suptitle("E2 weighted(all): learned layer weights - consistent shape across folds, but close to uniform",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E2_layer_weights.png")


if __name__ == "__main__":
    main()
