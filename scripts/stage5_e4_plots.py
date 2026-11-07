"""
E4 figures (run after stage5_e4_run.py --summarise and stage5_e4_controls.py). TEST figures unless stated.

  plots/E4_models_test.png         test macro-F1 of every model (E1-SVM, E2, E3 as run, the 4 re-searched tuned
                                   models, 3 E4 ablations): fold dots + mean ± std
  plots/E4_comparisons_test.png    paired per-fold test deltas: key comparisons, secondary, ablations, re-search effects
  plots/E4_search_validation.png   VALIDATION macro-F1 vs learning rate (mean over folds; per dropout) + selected-lr counts
  plots/E4_confusion_test.png      E4 test confusion (row %), summed over folds and seeds
  plots/E4_per_class_f1.png        per-class test F1: E2, HuBERT-only, concat, control, E4
  plots/E4_feature_swap.png        SUPPLEMENTARY: change in macro-F1 when MFCC+prosody are swapped between recordings
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
COL = {"E1-SVM": ps.MODEL_COLOR["E1-SVM"], "E2": ps.MODEL_COLOR["E2"], "E3:concat_HMP": ps.MODEL_COLOR["E3"],
       "concat_H": ps.SERIES[5], "concat_HMP": ps.MODEL_COLOR["E3"], "sa_HMP": ps.MODEL_COLOR["E3-SA"],
       "xattn_HMP": ps.MODEL_COLOR["E4"]}
MARK = {"concat_H": "s", "concat_HMP": "o", "sa_HMP": "^", "xattn_HMP": "D"}
NAME = {"concat_H": "HuBERT-only", "concat_HMP": "concat", "sa_HMP": "self-attn control", "xattn_HMP": "E4 cross-attn"}
ORDER = ["E1-SVM", "E2", "E3:concat_HMP", "concat_H", "concat_HMP", "sa_HMP", "xattn_HMP", "xattn_HM", "xattn_HP", "xattn_H_MPcat"]
LABEL = {"E1-SVM": "E1-SVM", "E2": "E2\npooled HuBERT", "E3:concat_HMP": "E3 concat\n(as run, E3 grid)",
         "concat_H": "HuBERT-only\n(frame model)", "concat_HMP": "concat\nH+M+P", "sa_HMP": "self-attn\ncontrol",
         "xattn_HMP": "E4\ncross-attn", "xattn_HM": "E4 abl.\nH↔M", "xattn_HP": "E4 abl.\nH↔P", "xattn_H_MPcat": "E4 abl.\nH↔concat(M,P)"}


def fold_scores(ds, S):
    folds = sorted(int(k) for k in S["folds"])
    out = {v: [S["folds"][str(k)]["variants"][v]["test_mean_over_seeds"]["macro_f1"] for k in folds]
           for v in ORDER if v.startswith(("concat", "sa", "xattn"))}
    e2 = json.loads((PATHS.results / "E2" / ds / "summary.json").read_text())
    out["E2"] = [e2["folds"][str(k)]["test_mean_over_seeds"]["macro_f1"] for k in folds]
    e3 = json.loads((PATHS.results / "E3" / ds / "summary.json").read_text())
    out["E3:concat_HMP"] = [e3["folds"][str(k)]["variants"]["concat_HMP"]["test_mean_over_seeds"]["macro_f1"] for k in folds]
    e1 = pd.read_csv(PATHS.results / "E1" / ds / "runs.csv")
    e1 = e1[(e1.variant == "headline") & (e1.model == "svm")].groupby("fold").test_macro_f1.mean()
    out["E1-SVM"] = [float(e1[k]) for k in folds]
    return folds, out


def delta_panel(a, P, comps, title):
    for i, (key, lab) in enumerate(comps):
        d = np.array(P[key]["diff_per_fold"])
        a.scatter(np.full(len(d), i) + np.linspace(-0.1, 0.1, len(d)), d, s=16, color=ps.SERIES[0], alpha=0.6, edgecolor="none")
        a.errorbar(i + 0.22, d.mean(), yerr=d.std(ddof=1), fmt="D", ms=5, color=ps.INK, capsize=3, lw=1.2)
        a.text(i, 0.985, f"{d.mean():+.3f}\n({P[key]['n_better']}/{P[key]['n']})", transform=a.get_xaxis_transform(),
               fontsize=7.5, color=ps.INK_2, ha="center", va="top")
    a.axhline(0, color=ps.INK_2, lw=0.8, zorder=1)
    lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.22 * (hi - lo))
    a.set_xticks(range(len(comps)), [c[1] for c in comps], fontsize=7.8)
    a.set_title(title, fontsize=10.5)


def main():
    ps.apply()
    S = {ds: json.loads((PATHS.results / "E4" / ds / "summary.json").read_text()) for ds in DATASETS}

    # 1 all models
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.4), sharex=True)
    for a, ds in zip(axes, DATASETS):
        folds, F = fold_scores(ds, S[ds])
        for i, v in enumerate(ORDER):
            vals = np.array(F[v])
            a.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=14, color=COL.get(v, ps.MUTED),
                      alpha=0.6, edgecolor="none", zorder=2)
            a.errorbar(i + 0.25, vals.mean(), yerr=vals.std(ddof=1), fmt="D", ms=5, color=ps.INK, capsize=3, lw=1.2, zorder=3)
            a.text(i + 0.33, vals.mean(), f"{vals.mean():.3f}", fontsize=7.5, color=ps.INK_2, va="center")
        for xv in (2.5, 6.5):
            a.axvline(xv, color=ps.GRID, lw=1)
        a.set_ylabel("TEST macro-F1")
        a.set_title(f"{ds} ({len(folds)} folds; dots = folds, each the mean of 3 seeds; ◆ mean ± std)", fontsize=10.5)
        a.set_xticks(range(len(ORDER)), [LABEL[v] for v in ORDER], fontsize=8)
    fig.suptitle("E4: test macro-F1 on identical speaker-independent folds (models between the grey lines share one 8-candidate "
                 "validation search)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E4_models_test.png")

    # 2 paired comparisons
    key = [("E4 - concat (primary)", "E4 − concat\n(primary)"), ("E4 - control", "E4 −\ncontrol"),
           ("E4 - HuBERT-only", "E4 −\nHuBERT-only"), ("control - concat", "control −\nconcat"),
           ("concat - HuBERT-only (M+P given H, fair search)", "concat −\nHuBERT-only"), ("E4 - E2", "E4 − E2")]
    abl = [("prosody | E4 (HMP - HM)", "prosody\ngiven E4\n(HMP − H↔M)"), ("MFCC | E4 (HMP - HP)", "MFCC\ngiven E4\n(HMP − H↔P)"),
           ("stage-1 MFCC<->prosody attention (HMP - H_MPcat)", "stage-1\nM↔P attn"),
           ("cross-attention vs concat, same inputs (H_MPcat - concat)", "H↔concat(M,P)\n− concat"),
           ("concat (E4 search) - concat (E3)", "concat:\nE4 grid −\nE3 grid"), ("control (E4 search) - control (E3)", "control:\nE4 grid −\nE3 grid")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for r, ds in enumerate(DATASETS):
        P = S[ds]["paired_test_macro_f1"]
        delta_panel(axes[r, 0], P, key, f"{ds}: key and secondary comparisons")
        delta_panel(axes[r, 1], P, abl, f"{ds}: E4 ablations and effect of the wider search")
        axes[r, 0].set_ylabel("Δ TEST macro-F1 (paired per fold)")
    fig.suptitle("E4 paired comparisons (test; mean Δ and folds better shown above each column)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E4_comparisons_test.png")

    # 3 search (validation)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for a, ds in zip(axes, DATASETS):
        folds = sorted(int(k) for k in S[ds]["folds"])
        lrs = S[ds]["config"]["grid_lr"]
        for v in ("concat_H", "concat_HMP", "sa_HMP", "xattn_HMP"):
            for dr, ls in ((0.2, "-"), (0.4, ":")):
                m = [np.mean([next(c["val_macro_f1"] for c in S[ds]["folds"][str(k)]["search"][v]
                                   if c["lr"] == lr and c["dropout"] == dr) for k in folds]) for lr in lrs]
                a.plot(lrs, m, ls=ls, marker=MARK[v], color=COL[v], lw=1.6, ms=6)
        a.set_xscale("log"); a.invert_xaxis()
        a.set_xticks(lrs, [f"{x:g}" for x in lrs]); a.minorticks_off()
        a.set_xlabel("learning rate (grid; lowest on the right)"); a.set_ylabel("VALIDATION macro-F1 (mean over folds)")
        e = S[ds]["selection_edges"]
        a.set_title(f"{ds}\nfolds selecting the lowest lr (3e-5): " +
                    ", ".join(f"{NAME[v]} {e[v]['lower_edge_folds']}/{e[v]['n_folds']}" for v in e), fontsize=8.5, loc="left")
    handles = [plt.Line2D([], [], color=COL[v], marker=MARK[v], label=NAME[v]) for v in MARK] + \
        [plt.Line2D([], [], color=ps.INK_2, ls="-", label="dropout 0.2"), plt.Line2D([], [], color=ps.INK_2, ls=":", label="dropout 0.4")]
    fig.legend(handles=handles, loc="upper right", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("E4 search (validation only; seed 0)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); ps.save(fig, PATHS.plots / "E4_search_validation.png")

    # 4 confusion E4
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for a, ds in zip(axes, DATASETS):
        cm = np.array(S[ds]["variants"]["xattn_HMP"]["confusion_summed_over_folds_and_seeds"], float)
        pct = 100 * cm / cm.sum(1, keepdims=True)
        a.imshow(pct, cmap=ps.SEQ_BLUE, vmin=0, vmax=100)
        for (i, j), v in np.ndenumerate(pct):
            a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 55 else ps.INK)
        a.set_xticks(range(6), CLASSES, rotation=35); a.set_yticks(range(6), CLASSES); a.grid(False)
        f1 = S[ds]["variants"]["xattn_HMP"]["test"]["macro_f1"]
        a.set_title(f"{ds} - E4  (TEST macro-F1 {f1['mean']:.3f} ± {f1['std']:.3f})", fontsize=10)
        a.set_xlabel("predicted"); a.set_ylabel("true")
    fig.suptitle("E4 (cross-attention, H+M+P): test confusion matrices, % of each true class (all folds, 3 seeds)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E4_confusion_test.png")

    # 5 per-class F1
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.0), sharey=True)
    names = [("E2", "E2 (pooled HuBERT)"), ("concat_H", NAME["concat_H"]), ("concat_HMP", NAME["concat_HMP"]),
             ("sa_HMP", NAME["sa_HMP"]), ("xattn_HMP", NAME["xattn_HMP"])]
    for a, ds in zip(axes, DATASETS):
        e2 = json.loads((PATHS.results / "E2" / ds / "summary.json").read_text())["per_class_f1_mean"]
        x = np.arange(6)
        for k, (v, lab) in enumerate(names):
            d = e2 if v == "E2" else S[ds]["variants"][v]["per_class_f1_mean"]
            a.bar(x + (k - 2) * 0.17, [d[c] for c in CLASSES], 0.15, color=COL[v], label=lab)
        a.set_xticks(x, CLASSES); a.set_ylim(0, 1); a.set_title(ds, fontsize=11)
    axes[0].set_ylabel("TEST F1 (mean over folds and seeds)")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[v]) for v, _ in names], labels=[n for _, n in names],
               loc="upper right", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("E4: per-class test F1", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); ps.save(fig, PATHS.plots / "E4_per_class_f1.png")

    # 6 feature swap (supplementary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))          # own y-range per dataset (CREMA-D deltas are larger)
    for a, ds in zip(axes, DATASETS):
        p = PATHS.results / "E4" / ds / "diagnostic_feature_swap.json"
        if not p.exists():
            a.set_title(f"{ds}: no diagnostic file"); continue
        agg = json.loads(p.read_text())["aggregate"]
        cols = [("xattn_HMP val", "E4\nvalidation"), ("xattn_HMP test", "E4\ntest"),
                ("concat_HMP val", "concat\nvalidation"), ("concat_HMP test", "concat\ntest")]
        for i, (k, lab) in enumerate(cols):
            d = np.array(agg[k]["delta_per_fold"])
            c = COL["xattn_HMP"] if k.startswith("xattn") else COL["concat_HMP"]
            a.scatter(np.full(len(d), i) + np.linspace(-0.1, 0.1, len(d)), d, s=18, color=c, alpha=0.7, edgecolor="none")
            a.errorbar(i + 0.22, d.mean(), yerr=d.std(ddof=1), fmt="D", ms=5, color=ps.INK, capsize=3)
            a.text(i, 0.985, f"{d.mean():+.3f}", transform=a.get_xaxis_transform(), fontsize=8, color=ps.INK_2, ha="center", va="top")
        a.axhline(0, color=ps.INK_2, lw=0.8)
        lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.12 * (hi - lo))
        a.set_xticks(range(4), [c[1] for c in cols]); a.set_title(ds, fontsize=11)
    axes[0].set_ylabel("Δ macro-F1, swapped − intact (per fold)")
    fig.suptitle("SUPPLEMENTARY diagnostic: MFCC + prosody swapped between recordings at evaluation (HuBERT intact) - "
                 "sensitivity of trained models, not feature contribution", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(); ps.save(fig, PATHS.plots / "E4_feature_swap.png")


if __name__ == "__main__":
    main()
