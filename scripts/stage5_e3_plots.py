"""
E3 figures (run stage5_e3_run.py --summarise first). TEST figures unless stated otherwise.

  plots/E3_variants_test.png          test macro-F1 of every E3 variant (+ E1-SVM, E2), fold dots + mean ± std
  plots/E3_stream_contributions.png   paired per-fold test deltas: leave-one-stream-out, add-to-HuBERT,
                                      control vs concat, DisVoice; and Shapley values per stream
  plots/E3_confusion_test.png         test confusion (row %) of concat_HMP, summed over folds and seeds
  plots/E1_E2_E3_folds_test.png       per-fold paired lines E1-SVM -> E2 -> E3 -> E3-SA
  plots/E2_vs_E3_per_class_f1.png     per-class test F1 E2, E3 (concat_HMP), E3-SA
  plots/E3_search_validation.png      VALIDATION macro-F1 of the S-phase runs per fold (concat vs control)
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
ORDER = ["E1-SVM", "E2", "concat_M", "concat_P", "concat_MP", "concat_H", "concat_HM", "concat_HP", "concat_HMP",
         "sa_HMP", "concat_HMP_dv", "concat_HMP_dvind"]
LABEL = {"E1-SVM": "E1-SVM\n(MFCC stats)", "E2": "E2\n(pooled HuBERT)", "concat_M": "M", "concat_P": "P",
         "concat_MP": "M+P", "concat_H": "H", "concat_HM": "H+M", "concat_HP": "H+P", "concat_HMP": "H+M+P\n(E3)",
         "sa_HMP": "H+M+P\n+self-attn\n(control)", "concat_HMP_dv": "E3\n+DisVoice", "concat_HMP_dvind": "E3\n+DisVoice\n+indicators"}


def color(v):
    if v in ("E1-SVM", "E2"):
        return ps.MODEL_COLOR[v]
    if v == "sa_HMP":
        return ps.MODEL_COLOR["E3-SA"]
    if v == "concat_HMP":
        return ps.MODEL_COLOR["E3"]
    return ps.MUTED


def fold_scores(ds, S3):
    folds = sorted(int(k) for k in S3["folds"])
    out = {v: [S3["folds"][str(k)]["variants"][v]["test_mean_over_seeds"]["macro_f1"] for k in folds]
           for v in ORDER if v.startswith(("concat", "sa"))}
    e2 = json.loads((PATHS.results / "E2" / ds / "summary.json").read_text())
    out["E2"] = [e2["folds"][str(k)]["test_mean_over_seeds"]["macro_f1"] for k in folds]
    e1 = pd.read_csv(PATHS.results / "E1" / ds / "runs.csv")
    e1 = e1[(e1.variant == "headline") & (e1.model == "svm")].groupby("fold").test_macro_f1.mean()
    out["E1-SVM"] = [float(e1[k]) for k in folds]
    return folds, out


def main():
    ps.apply()
    S3 = {ds: json.loads((PATHS.results / "E3" / ds / "summary.json").read_text()) for ds in DATASETS}

    # 1 all variants
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True)
    for a, ds in zip(axes, DATASETS):
        folds, F = fold_scores(ds, S3[ds])
        x = np.arange(len(ORDER))
        for i, v in enumerate(ORDER):
            vals = np.array(F[v])
            a.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=14, color=color(v),
                      alpha=0.55, edgecolor="none", zorder=2)
            a.errorbar(i + 0.25, vals.mean(), yerr=vals.std(ddof=1), fmt="D", ms=5, color=ps.INK, capsize=3, lw=1.2, zorder=3)
            a.text(i + 0.33, vals.mean(), f"{vals.mean():.3f}", fontsize=7.5, color=ps.INK_2, va="center")
        a.axhline(np.mean(F["concat_HMP"]), color=ps.MODEL_COLOR["E3"], lw=0.8, ls=":", zorder=1)
        for xv in (1.5, 8.5):
            a.axvline(xv, color=ps.GRID, lw=1)
        a.set_ylabel("TEST macro-F1"); a.set_title(f"{ds} ({len(folds)} folds; dots = folds, each the mean of 3 seeds; ◆ mean ± std)", fontsize=10.5)
        a.set_xticks(x, [LABEL[v] for v in ORDER], fontsize=8)
    fig.suptitle("E3: test macro-F1 of every variant on identical speaker-independent folds (H = HuBERT, M = MFCC, P = prosody)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E3_variants_test.png")

    # 2 stream contributions (paired deltas) + Shapley
    comps = [("HuBERT | M+P   (HMP - MP)", "HuBERT\ngiven M+P"), ("MFCC | H+P     (HMP - HP)", "MFCC\ngiven H+P"),
             ("prosody | H+M  (HMP - HM)", "prosody\ngiven H+M"), ("MFCC | H       (HM - H)", "MFCC\ngiven H"),
             ("prosody | H    (HP - H)", "prosody\ngiven H"), ("sa_HMP - concat_HMP", "self-attn\ncontrol\nvs E3"),
             ("DisVoice | HMP (dv - HMP)", "DisVoice\ngiven E3"), ("indicators | dv (dvind - dv)", "indicators\ngiven DisVoice")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.6), gridspec_kw={"width_ratios": [4, 1.2]})
    for r, ds in enumerate(DATASETS):
        a = axes[r, 0]
        P = S3[ds]["paired_test_macro_f1"]
        for i, (key, lab) in enumerate(comps):
            d = np.array(P[key]["diff_per_fold"])
            a.scatter(np.full(len(d), i) + np.linspace(-0.1, 0.1, len(d)), d, s=16, color=ps.SERIES[0], alpha=0.6, edgecolor="none")
            a.errorbar(i + 0.22, d.mean(), yerr=d.std(ddof=1), fmt="D", ms=5, color=ps.INK, capsize=3, lw=1.2)
            # labels in a fixed row at the top of the panel (data x, axes y): never covers dots, bars or the zero line
            a.text(i, 0.985, f"{d.mean():+.3f}  ({P[key]['n_better']}/{P[key]['n']} folds)",
                   transform=a.get_xaxis_transform(), fontsize=7.5, color=ps.INK_2, ha="center", va="top")
        a.axhline(0, color=ps.INK_2, lw=0.8, zorder=1)
        lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.12 * (hi - lo))
        a.set_xticks(range(len(comps)), [c[1] for c in comps], fontsize=8)
        a.set_ylabel("Δ TEST macro-F1 (paired, per fold)"); a.set_title(ds, fontsize=11)
        b = axes[r, 1]
        sh = S3[ds]["shapley_test_macro_f1"]["values"]
        b.bar(range(3), [sh[s] for s in "HMP"], 0.6, color=ps.SERIES[0])
        for i, s in enumerate("HMP"):
            b.text(i, sh[s] + 0.005, f"{sh[s]:.3f}", ha="center", fontsize=8, color=ps.INK_2)
        b.set_xticks(range(3), ["HuBERT", "MFCC", "prosody"], fontsize=8)
        b.set_title("Shapley value\n(sum = F1(H+M+P) − 1/6)", fontsize=9)
    fig.suptitle("E3 ablations: contribution of each feature stream (test, paired on identical folds)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E3_stream_contributions.png")

    # 3 confusion E3
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for a, ds in zip(axes, DATASETS):
        cm = np.array(S3[ds]["variants"]["concat_HMP"]["confusion_summed_over_folds_and_seeds"], float)
        pct = 100 * cm / cm.sum(1, keepdims=True)
        a.imshow(pct, cmap=ps.SEQ_BLUE, vmin=0, vmax=100)
        for (i, j), v in np.ndenumerate(pct):
            a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 55 else ps.INK)
        a.set_xticks(range(6), CLASSES, rotation=35); a.set_yticks(range(6), CLASSES); a.grid(False)
        f1 = S3[ds]["variants"]["concat_HMP"]["test"]["macro_f1"]
        a.set_title(f"{ds} - E3  (TEST macro-F1 {f1['mean']:.3f} ± {f1['std']:.3f})", fontsize=10)
        a.set_xlabel("predicted"); a.set_ylabel("true")
    fig.suptitle("E3 (concatenation, H+M+P): test confusion matrices, % of each true class (all folds, 3 seeds)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E3_confusion_test.png")

    # 4 per-fold paired lines
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    seq = [("E1-SVM", "E1-SVM"), ("E2", "E2"), ("concat_HMP", "E3"), ("sa_HMP", "E3-SA")]
    for a, ds in zip(axes, DATASETS):
        folds, F = fold_scores(ds, S3[ds])
        for j in range(len(folds)):
            a.plot(range(4), [F[v][j] for v, _ in seq], color=ps.GRID, lw=1, zorder=1)
        for i, (v, name) in enumerate(seq):
            a.scatter([i] * len(folds), F[v], color=ps.MODEL_COLOR[name], s=40, zorder=2, edgecolor=ps.SURFACE)
            a.errorbar(i + 0.18, np.mean(F[v]), yerr=np.std(F[v], ddof=1), fmt="D", color=ps.INK, capsize=3, ms=5)
        a.set_xticks(range(4), ["E1-SVM", "E2 (pooled\nHuBERT)", "E3 (concat\nH+M+P)", "E3 control\n(self-attn)"])
        a.set_title(ds, fontsize=11); a.set_ylabel("TEST macro-F1 (dots = folds; ◆ mean ± std)")
    fig.suptitle("E1 vs E2 vs E3 on identical speaker-independent folds (lines connect the same fold)", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E1_E2_E3_folds_test.png")

    # 5 per-class F1
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
    for a, ds in zip(axes, DATASETS):
        e2 = json.loads((PATHS.results / "E2" / ds / "summary.json").read_text())["per_class_f1_mean"]
        series = [("E2", e2), ("E3", S3[ds]["variants"]["concat_HMP"]["per_class_f1_mean"]),
                  ("E3-SA", S3[ds]["variants"]["sa_HMP"]["per_class_f1_mean"])]
        x = np.arange(6)
        for k, (name, d) in enumerate(series):
            a.bar(x + (k - 1) * 0.27, [d[c] for c in CLASSES], 0.25, color=ps.MODEL_COLOR[name], label=name)
        a.set_xticks(x, CLASSES); a.set_ylim(0, 1); a.set_title(ds, fontsize=11)
    axes[0].set_ylabel("TEST F1 (mean over folds and seeds)")
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ps.MODEL_COLOR[n]) for n in ("E2", "E3", "E3-SA")],
                   labels=["E2 (pooled HuBERT)", "E3 (concat H+M+P)", "E3 control (self-attn)"], loc="upper left", ncol=3)
    fig.suptitle("E2 vs E3: per-class test F1", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E2_vs_E3_per_class_f1.png")

    # 6 search (validation)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=False)
    for a, ds in zip(axes, DATASETS):
        folds = sorted(int(k) for k in S3[ds]["folds"])
        # marker shape as secondary encoding (the two hues are close); one shared legend outside the plot area
        for v, name, off, mk in (("concat_HMP", "E3", -0.15, "o"), ("sa_HMP", "E3-SA", 0.15, "^")):
            for i, k in enumerate(folds):
                vals = [r["val_macro_f1"] for r in S3[ds]["folds"][str(k)]["search"][v]]
                a.scatter(np.full(len(vals), i + off), vals, s=22, marker=mk, color=ps.MODEL_COLOR[name], alpha=0.85,
                          edgecolor="none")
        a.set_xticks(range(len(folds)), [f"fold {k}\nL{S3[ds]['folds'][str(k)]['layer']}" for k in folds])
        a.set_ylabel("VALIDATION macro-F1"); a.set_title(ds, fontsize=11)
    handles = [plt.Line2D([], [], ls="none", marker="o", color=ps.MODEL_COLOR["E3"], label="E3 concat (4 configs)"),
               plt.Line2D([], [], ls="none", marker="^", color=ps.MODEL_COLOR["E3-SA"], label="self-attn control (4 configs)")]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("E3 hyper-parameter search (validation only; L = HuBERT layer chosen from the fold's E2 validation)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); ps.save(fig, PATHS.plots / "E3_search_validation.png")


if __name__ == "__main__":
    main()
