"""
Final comparative analysis of E1-E4 (READ-ONLY with respect to every existing result, checkpoint, config and
dataset). No training, no test evaluation: only the saved per-run files are read.

Writes (new files only):
  results/final_analysis/final_analysis.json
  plots/FINAL_per_class_E2_vs_E4.png, plots/FINAL_confusion_E2_E4.png, plots/FINAL_error_overlap.png,
  plots/FINAL_accuracy_vs_cost.png

Sections of the JSON
  verification   test macro-F1 per model recomputed (a) from the per-run metric fields, (b) from the per-run
                 confusion matrices, (c) for E1/E2/E4 from saved predictions + manifest labels; compared with the
                 summary files and with the numbers printed in reports/stage5_models.md
  per_class      precision / recall / F1 per class (mean over folds and seeds), per-fold E4-E2, E3-E2 deltas
  confusion      summed confusion matrices (row %) and E4 - E2 difference; top confusions; predicted-class shares
  agreement      E2 vs E4 (and other pairs) on identical test recordings: both right / only one right / both wrong
  pooling        E2 vs the HuBERT-only frame model per fold, with E2's selected representation and both
                 models' validation F1
  efficiency     parameters, train seconds per run, total training seconds of every saved run (search +
                 finals), inference ms per recording, GPU and process RAM peaks
  statistics     per-fold ranks, seed std, validation->test gaps, permutation controls, audits; an EXPLORATORY
                 (not pre-registered) sign-flip test of E2 vs E4 over all 11 folds
"""
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.features.store import FeatureStore  # noqa: E402
from src.preprocessing.dataset_index import CLASSES  # noqa: E402
from src.utils import plot_style as ps  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402

R = PATHS.results
OUT = R / "final_analysis"
DSN = {"RAVDESS": 6, "CREMA-D": 5}
SEEDS = (0, 1, 2)
# model key -> (label, experiment, variant)
MODELS = {
    "E1": ("E1 SVM (MFCC statistics)", "E1", "svm"),
    "E2": ("E2 pooled HuBERT (MLP)", "E2", None),
    "E3": ("E3 concat H+M+P (E3 grid)", "E3", "concat_HMP"),
    "E3-SA": ("E3 self-attn control (E3 grid)", "E3", "sa_HMP"),
    "E3-H": ("E3 HuBERT-only (concat hp)", "E3", "concat_H"),
    "E4": ("E4 cross-attention H+M+P", "E4", "xattn_HMP"),
    "E4-SA": ("self-attn control (E4 grid)", "E4", "sa_HMP"),
    "E4-concat": ("concat H+M+P (E4 grid)", "E4", "concat_HMP"),
    "E4-H": ("HuBERT-only frame model (E4 grid)", "E4", "concat_H"),
}
# numbers printed in reports/stage5_models.md (test macro-F1 mean over folds) - checked, not trusted
REPORTED = {"RAVDESS": {"E1": 0.537, "E2": 0.762, "E3": 0.729, "E3-SA": 0.726, "E3-H": 0.747, "E4": 0.723,
                        "E4-SA": 0.730, "E4-concat": 0.726, "E4-H": 0.736},
            "CREMA-D": {"E1": 0.503, "E2": 0.713, "E3": 0.695, "E3-SA": 0.669, "E3-H": 0.685, "E4": 0.674,
                        "E4-SA": 0.686, "E4-concat": 0.694, "E4-H": 0.688}}


# --------------------------------------------------------------------------- loading
def runs(ds, key):
    """List of per-run dicts {fold, seed, test (metrics), val_f1, pred, rows, train_s, infer_ms, gpu, ram, n_params}."""
    _, exp, var = MODELS[key]
    out = []
    for k in range(DSN[ds]):
        if exp == "E1":
            j = json.loads((R / "E1" / ds / f"fold{k}_svm_seed0.json").read_text())
            h = j["headline"]
            out.append({"fold": k, "seed": 0, "test": h["test"], "val_f1": h["val"]["macro_f1"], "pred": h["test_pred"],
                        "rows": j["test_rows"], "train_s": h["train_seconds"], "infer_ms": h["test_inference_ms_per_recording"],
                        "gpu": None, "ram": None, "n_params": None, "size": h["model_size"]})
            continue
        for s in SEEDS:
            p = (R / "E2" / ds / f"fold{k}" / f"D_seed{s}.json") if exp == "E2" else (R / exp / ds / f"fold{k}" / f"D_{var}_seed{s}.json")
            j = json.loads(p.read_text())
            out.append({"fold": k, "seed": s, "test": j["test"], "val_f1": j["val"]["macro_f1"], "pred": j["test_pred"],
                        "rows": j["test_rows"], "train_s": j["train_seconds"], "infer_ms": j["test_inference_ms_per_recording"],
                        "gpu": j.get("gpu_peak_mib"), "ram": (j.get("process_ram_mib") or {}).get("peak"),
                        "n_params": j.get("n_params")})
    return out


def f1_from_confusion(cm):
    cm = np.asarray(cm, float)
    tp = np.diag(cm); p = tp / np.maximum(cm.sum(0), 1e-12); r = tp / np.maximum(cm.sum(1), 1e-12)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(f.mean())


def macro_f1(y, p):
    cm = np.zeros((6, 6)); np.add.at(cm, (np.asarray(y), np.asarray(p)), 1)
    return f1_from_confusion(cm), cm


def fold_mean(rs, fn):
    return {k: float(np.mean([fn(r) for r in rs if r["fold"] == k])) for k in sorted({r["fold"] for r in rs})}


def sign_flip_p(d):
    d = np.asarray(d, float)
    signs = np.array(list(product((-1, 1), repeat=len(d))))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


# --------------------------------------------------------------------------- analysis
def main():
    OUT.mkdir(exist_ok=True)
    index = pd.read_csv(PATHS.metadata / "index.csv")
    A = {"verification": {}, "per_class": {}, "confusion": {}, "agreement": {}, "pooling": {}, "efficiency": {},
         "statistics": {}}
    RUNS, LAB = {}, {}
    for ds in DSN:
        LAB[ds] = FeatureStore.open(ds, index).manifest.label_id.values.astype(int)      # read-only
        RUNS[ds] = {m: runs(ds, m) for m in MODELS}

    # ---------------- 1 verification
    for ds in DSN:
        V = {}
        sums = {e: json.loads((R / e / ds / "summary.json").read_text()) for e in ("E2", "E3", "E4")}
        for m, rs in RUNS[ds].items():
            f_stored = fold_mean(rs, lambda r: r["test"]["macro_f1"])
            f_cm = fold_mean(rs, lambda r: f1_from_confusion(r["test"]["confusion"]))
            f_pred = fold_mean(rs, lambda r: macro_f1(LAB[ds][r["rows"]], r["pred"])[0])
            vals = np.array(list(f_stored.values()))
            exp, var = MODELS[m][1], MODELS[m][2]
            if exp == "E1":
                e1 = pd.read_csv(R / "E1" / ds / "runs.csv")
                e1 = e1[(e1.model == "svm") & (e1.variant == "headline")].groupby("fold").test_macro_f1.mean()
                summ = float(e1.mean())
            elif exp == "E2":
                summ = sums["E2"]["test"]["macro_f1"]["mean"]
            else:
                summ = sums[exp]["variants"][var]["test"]["macro_f1"]["mean"]
            V[m] = {"per_fold": f_stored, "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                    "max_abs_diff_metric_vs_confusion": max(abs(f_stored[k] - f_cm[k]) for k in f_stored),
                    "max_abs_diff_metric_vs_predictions": max(abs(f_stored[k] - f_pred[k]) for k in f_stored),
                    "summary_mean": summ, "matches_summary": abs(summ - vals.mean()) < 1e-9,
                    "reported_3dp": REPORTED[ds][m], "matches_report_3dp": round(vals.mean(), 3) == REPORTED[ds][m]}
        A["verification"][ds] = V

    # ---------------- 2 per-class
    for ds in DSN:
        P = {}
        for m in ("E1", "E2", "E3", "E4", "E4-SA", "E4-concat", "E4-H"):
            rs = RUNS[ds][m]
            P[m] = {met: {c: float(np.mean([r["test"]["per_class"][c][met] for r in rs])) for c in CLASSES}
                    for met in ("precision", "recall", "f1")}
            P[m]["per_fold_f1"] = {c: fold_mean(rs, lambda r, c=c: r["test"]["per_class"][c]["f1"]) for c in CLASSES}
        deltas = {}
        for a, b in (("E4", "E2"), ("E3", "E2"), ("E4-H", "E2"), ("E4", "E4-concat"), ("E4-concat", "E4-H"), ("E4", "E4-H"),
                     ("E2", "E1")):
            deltas[f"{a} - {b}"] = {}
            for c in CLASSES:
                for met in ("precision", "recall", "f1"):
                    pa = fold_mean(RUNS[ds][a], lambda r: r["test"]["per_class"][c][met])
                    pb = fold_mean(RUNS[ds][b], lambda r: r["test"]["per_class"][c][met])
                    d = np.array([pa[k] - pb[k] for k in pa])
                    deltas[f"{a} - {b}"].setdefault(c, {})[met] = {"mean": float(d.mean()), "n_better": int((d > 0).sum()),
                                                                   "n": len(d), "per_fold": [round(x, 4) for x in d]}
        P["deltas"] = deltas
        A["per_class"][ds] = P

    # ---------------- 3 confusion
    for ds in DSN:
        C = {}
        for m in ("E1", "E2", "E3", "E4", "E4-concat", "E4-H"):
            cm = np.sum([np.array(r["test"]["confusion"]) for r in RUNS[ds][m]], axis=0).astype(float)
            pct = 100 * cm / cm.sum(1, keepdims=True)
            off = [(CLASSES[i], CLASSES[j], float(pct[i, j])) for i in range(6) for j in range(6) if i != j]
            C[m] = {"row_pct": pct.round(2).tolist(), "counts": cm.astype(int).tolist(),
                    "top_confusions": sorted(off, key=lambda x: -x[2])[:6],
                    "predicted_share_pct": (100 * cm.sum(0) / cm.sum()).round(2).tolist(),
                    "true_share_pct": (100 * cm.sum(1) / cm.sum()).round(2).tolist(),
                    "sad_column_share_pct": float(100 * cm[:, CLASSES.index("sad")].sum() / cm.sum())}
        d = np.array(C["E4"]["row_pct"]) - np.array(C["E2"]["row_pct"])
        C["E4_minus_E2_row_pct"] = d.round(2).tolist()
        C["largest_changes_E4_minus_E2"] = sorted([(CLASSES[i], CLASSES[j], float(d[i, j])) for i in range(6) for j in range(6) if i != j],
                                                  key=lambda x: -abs(x[2]))[:8]
        A["confusion"][ds] = C

    # ---------------- 4 agreement on identical test recordings (seed s of model a vs seed s of model b)
    for ds in DSN:
        G = {}
        for a, b in (("E2", "E4"), ("E2", "E4-H"), ("E4", "E4-concat"), ("E4-concat", "E4-H"), ("E2", "E1")):
            cats = []
            for k in range(DSN[ds]):
                ra = [r for r in RUNS[ds][a] if r["fold"] == k]
                rb = [r for r in RUNS[ds][b] if r["fold"] == k]
                for x in ra:
                    for y in rb:
                        if b == "E1" and y["seed"] != 0:
                            continue
                        if b != "E1" and x["seed"] != y["seed"]:
                            continue
                        assert x["rows"] == y["rows"]
                        t = LAB[ds][x["rows"]]
                        pa, pb = np.array(x["pred"]), np.array(y["pred"])
                        ca, cb = pa == t, pb == t
                        cats.append([np.mean(ca & cb), np.mean(ca & ~cb), np.mean(~ca & cb), np.mean(~ca & ~cb),
                                     np.mean((~ca & ~cb) & (pa == pb)), np.mean(pa == pb)])
            c = np.array(cats).mean(0)
            G[f"{a} vs {b}"] = {"both_correct": c[0], "only_" + a: c[1], "only_" + b: c[2], "both_wrong": c[3],
                                "both_wrong_same_label": c[4], "prediction_agreement": c[5], "n_pairs": len(cats)}
        A["agreement"][ds] = {k: {kk: (float(vv) if not isinstance(vv, int) else vv) for kk, vv in v.items()} for k, v in G.items()}

    # ---------------- 5 pooling: E2 vs HuBERT-only frame model per fold
    for ds in DSN:
        e2 = json.loads((R / "E2" / ds / "summary.json").read_text())
        e4 = json.loads((R / "E4" / ds / "summary.json").read_text())
        rows = []
        for k in range(DSN[ds]):
            f2, f4 = e2["folds"][str(k)], e4["folds"][str(k)]
            rows.append({"fold": k, "E2_selected": f2["selected"], "E2_val_selected": f2["val_macro_f1_selection"],
                         "E2_test": f2["test_mean_over_seeds"]["macro_f1"], "E2_n_params": f2["n_params"],
                         "H_layer": f4["layer"], "H_selected": f4["selected"]["concat_H"],
                         "H_val_selected": f4["selected_val_macro_f1"]["concat_H"],
                         "H_test": f4["variants"]["concat_H"]["test_mean_over_seeds"]["macro_f1"],
                         "E4_val_selected": f4["selected_val_macro_f1"]["xattn_HMP"],
                         "E4_test": f4["variants"]["xattn_HMP"]["test_mean_over_seeds"]["macro_f1"],
                         "same_single_layer": f2["selected"]["rep"] == f"layer{f4['layer']}"})
        d_test = np.array([r["H_test"] - r["E2_test"] for r in rows])
        d_val = np.array([r["H_val_selected"] - r["E2_val_selected"] for r in rows])
        same = [r for r in rows if r["same_single_layer"]]
        A["pooling"][ds] = {"folds": rows,
                            "H_minus_E2_test": {"mean": float(d_test.mean()), "n_better": int((d_test > 0).sum()), "n": len(d_test)},
                            "H_minus_E2_val_selected": {"mean": float(d_val.mean()), "n_better": int((d_val > 0).sum()), "n": len(d_val)},
                            "same_layer_folds": [r["fold"] for r in same],
                            "H_minus_E2_test_same_layer_folds": float(np.mean([r["H_test"] - r["E2_test"] for r in same])) if same else None,
                            "E2_pooled_input_dim_note": "E2 mean_std pooling of a 768-d layer = 1,536 inputs (mean: 768); the frame "
                                                        "models pool 128-d projected frames -> 256 (mean+std)"}

    # ---------------- 6 efficiency
    def all_train_seconds(exp, ds):
        tot, n = 0.0, 0
        for p in (R / exp / ds).rglob("*.json"):
            if p.name in ("summary.json", "selection.json", "control_label_permutation.json", "diagnostic_feature_swap.json") or p.parent == R / exp / ds and exp != "E1":
                continue
            j = json.loads(p.read_text())
            if exp == "E1":
                if "headline" in j:
                    tot += float(j["headline"]["train_seconds"]) + float(j.get("selection_seconds") or 0); n += 1
                continue
            if "train_seconds" in j:
                tot += float(j["train_seconds"]); n += 1
        return tot, n
    for ds in DSN:
        E = {}
        for m in MODELS:
            rs = RUNS[ds][m]
            E[m] = {"n_params": rs[0]["n_params"], "train_s_median": float(np.median([r["train_s"] for r in rs])),
                    "infer_ms_mean": float(np.mean([r["infer_ms"] for r in rs])),
                    "gpu_peak_mib_max": max((r["gpu"] or 0) for r in rs) or None,
                    "process_ram_peak_mib_max": max((r["ram"] or 0) for r in rs) or None}
            if m == "E1":
                E[m]["model_size"] = rs[0]["size"]
            if m == "E2":
                E[m]["n_params_per_fold"] = sorted({r["n_params"] for r in rs})
        E["total_training_seconds_all_saved_runs"] = {e: dict(zip(("seconds", "n_runs"), all_train_seconds(e, ds))) for e in ("E1", "E2", "E3", "E4")}
        A["efficiency"][ds] = E

    # ---------------- 7 statistics
    for ds in DSN:
        S = {}
        F = {m: A["verification"][ds][m]["per_fold"] for m in MODELS}
        main_models = ("E1", "E2", "E3", "E4", "E4-SA", "E4-concat", "E4-H")
        ranks = {m: [] for m in main_models}
        for k in range(DSN[ds]):
            order = sorted(main_models, key=lambda m: -F[m][k])
            for i, m in enumerate(order):
                ranks[m].append(i + 1)
        S["rank_per_fold"] = ranks
        S["mean_rank"] = {m: float(np.mean(v)) for m, v in ranks.items()}
        S["best_model_per_fold"] = [max(main_models, key=lambda m: F[m][k]) for k in range(DSN[ds])]
        S["seed_std_within_fold_mean"] = {m: float(np.mean([np.std([r["test"]["macro_f1"] for r in RUNS[ds][m] if r["fold"] == k], ddof=1)
                                                           for k in range(DSN[ds])])) for m in MODELS if m != "E1"}
        S["val_to_test_gap"] = {m: float(np.mean([r["val_f1"] - r["test"]["macro_f1"] for r in RUNS[ds][m]])) for m in MODELS}
        S["permutation_control"] = {e: json.loads((R / e / ds / "control_label_permutation.json").read_text())["test_macro_f1_mean"]
                                    for e in ("E2", "E3", "E4")}
        A["statistics"][ds] = S
    d_all = np.concatenate([[A["verification"][ds]["E4"]["per_fold"][k] - A["verification"][ds]["E2"]["per_fold"][k]
                             for k in range(DSN[ds])] for ds in DSN])
    d_h = np.concatenate([[A["verification"][ds]["E4-H"]["per_fold"][k] - A["verification"][ds]["E2"]["per_fold"][k]
                           for k in range(DSN[ds])] for ds in DSN])
    A["statistics"]["EXPLORATORY_pooled_11_folds"] = {
        "note": "post hoc, NOT pre-registered; folds of two datasets pooled as 11 paired units; descriptive only",
        "E4_minus_E2": {"mean": float(d_all.mean()), "n_better": int((d_all > 0).sum()), "n": 11, "sign_flip_p": sign_flip_p(d_all)},
        "HuBERTonly_minus_E2": {"mean": float(d_h.mean()), "n_better": int((d_h > 0).sum()), "n": 11, "sign_flip_p": sign_flip_p(d_h)}}
    A["statistics"]["audits"] = {e: {"all_checks_passed": json.loads((R / e / "audit.json").read_text())["all_checks_passed"],
                                     "n_checks": sum(len(v["checks"]) for v in json.loads((R / e / "audit.json").read_text()).values()
                                                     if isinstance(v, dict) and "checks" in v)} for e in ("E2", "E3", "E4")}

    (OUT / "final_analysis.json").write_text(json.dumps(A, indent=1, default=float))
    figures(A)
    print_summary(A)


# --------------------------------------------------------------------------- figures
def figures(A):
    ps.apply()
    C2, C4 = ps.MODEL_COLOR["E2"], ps.MODEL_COLOR["E4"]
    # per-class P / R / F1, E2 vs E4, + per-fold F1 deltas
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for r, ds in enumerate(DSN):
        P = A["per_class"][ds]
        for c_, met in enumerate(("precision", "recall", "f1")):
            a = axes[r, c_]
            x = np.arange(6)
            a.bar(x - 0.2, [P["E2"][met][c] for c in CLASSES], 0.38, color=C2, label="E2 pooled HuBERT")
            a.bar(x + 0.2, [P["E4"][met][c] for c in CLASSES], 0.38, color=C4, label="E4 cross-attention")
            a.set_xticks(x, [c[:4] for c in CLASSES], fontsize=8); a.set_ylim(0.4, 0.95)
            a.set_title(f"{ds}: {met}", fontsize=10)
        a = axes[r, 3]
        D = P["deltas"]["E4 - E2"]
        for i, c in enumerate(CLASSES):
            d = np.array(D[c]["f1"]["per_fold"])
            a.scatter(np.full(len(d), i) + np.linspace(-0.1, 0.1, len(d)), d, s=16, color=C4, alpha=0.65, edgecolor="none")
            a.errorbar(i + 0.22, d.mean(), yerr=d.std(ddof=1), fmt="D", ms=5, color=ps.INK, capsize=3)
            a.text(i, 0.985, f"{d.mean():+.3f}\n({D[c]['f1']['n_better']}/{D[c]['f1']['n']})", transform=a.get_xaxis_transform(),
                   fontsize=7.2, color=ps.INK_2, ha="center", va="top")
        a.axhline(0, color=ps.INK_2, lw=0.8)
        lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.3 * (hi - lo))
        a.set_xticks(range(6), [c[:4] for c in CLASSES], fontsize=8)
        a.set_title(f"{ds}: per-fold Δ F1 (E4 − E2)", fontsize=10)
    axes[0, 0].set_ylabel("TEST (mean over folds and seeds)"); axes[1, 0].set_ylabel("TEST (mean over folds and seeds)")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C2), plt.Rectangle((0, 0), 1, 1, color=C4)],
               labels=["E2 pooled HuBERT", "E4 cross-attention"], loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("Per-class test precision / recall / F1: E2 vs E4 (folds better shown above each class)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); ps.save(fig, PATHS.plots / "FINAL_per_class_E2_vs_E4.png")

    # confusion E2, E4, E4 - E2 (diverging: two hues, neutral grey midpoint)
    div = LinearSegmentedColormap.from_list("div", [ps.SERIES[0], "#f2f1ec", ps.SERIES[7]])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.4))
    for r, ds in enumerate(DSN):
        C = A["confusion"][ds]
        for c_, (m, title) in enumerate((("E2", "E2 pooled HuBERT"), ("E4", "E4 cross-attention"))):
            pct = np.array(C[m]["row_pct"]); a = axes[r, c_]
            a.imshow(pct, cmap=ps.SEQ_BLUE, vmin=0, vmax=100)
            for (i, j), v in np.ndenumerate(pct):
                a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 55 else ps.INK)
            a.set_title(f"{ds} - {title} (row %)", fontsize=10)
        d = np.array(C["E4_minus_E2_row_pct"]); a = axes[r, 2]
        lim = max(8, float(np.abs(d).max()))
        a.imshow(d, cmap=div, vmin=-lim, vmax=lim)
        for (i, j), v in np.ndenumerate(d):
            a.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5, color=ps.INK)
        a.set_title(f"{ds} - E4 − E2 (percentage points; red = E4 more)", fontsize=10)
        for a in axes[r]:
            a.set_xticks(range(6), CLASSES, rotation=35); a.set_yticks(range(6), CLASSES); a.grid(False)
            a.set_xlabel("predicted"); a.set_ylabel("true")
    fig.suptitle("Test confusion matrices summed over folds and seeds: E2 vs E4", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / "FINAL_confusion_E2_E4.png")

    # error overlap
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=True)
    pairs = ["E2 vs E4", "E2 vs E4-H", "E4 vs E4-concat", "E4-concat vs E4-H", "E2 vs E1"]
    cols = [ps.SERIES[2], ps.SERIES[0], ps.SERIES[3], ps.SERIES[7]]
    for a, ds in zip(axes, DSN):
        G = A["agreement"][ds]
        for i, p in enumerate(pairs):
            g = G[p]; x, y = p.split(" vs ")
            vals = [g["both_correct"], g["only_" + x], g["only_" + y], g["both_wrong"]]
            left = 0
            for v, col in zip(vals, cols):
                a.barh(i, v, left=left, color=col, height=0.6, edgecolor=ps.SURFACE, linewidth=1)
                if v > 0.04:
                    a.text(left + v / 2, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8, color="white" if col != ps.SERIES[3] else ps.INK)
                left += v
        a.set_yticks(range(len(pairs)), [p.replace("E4-H", "HuBERT-only").replace("E4-concat", "concat") for p in pairs], fontsize=8.5)
        a.set_xlim(0, 1); a.set_title(ds, fontsize=11); a.set_xlabel("share of test recordings (same seed)")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols],
               labels=["both correct", "only first correct", "only second correct", "both wrong"], loc="upper right", ncol=4,
               frameon=False, bbox_to_anchor=(0.995, 0.995), fontsize=8.5)
    fig.suptitle("Error overlap on identical test recordings (%)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); ps.save(fig, PATHS.plots / "FINAL_error_overlap.png")

    # accuracy vs cost
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    show = [("E2", "E2", ps.MODEL_COLOR["E2"], "o"), ("E3", "E3 concat", ps.MODEL_COLOR["E3"], "o"),
            ("E4-H", "HuBERT-only", ps.SERIES[5], "s"), ("E4-concat", "concat (E4 grid)", ps.MODEL_COLOR["E3"], "s"),
            ("E4-SA", "self-attn control", ps.MODEL_COLOR["E3-SA"], "^"), ("E4", "E4 cross-attn", ps.MODEL_COLOR["E4"], "D")]
    # identity by colour + marker shape, one legend outside the data area (points are too close for direct labels)
    for a, ds in zip(axes, DSN):
        for m, lab, col, mk in show:
            v = A["verification"][ds][m]; e = A["efficiency"][ds][m]
            a.errorbar(e["train_s_median"], v["mean"], yerr=v["std"], fmt=mk, color=col, ms=8, capsize=3, mec=ps.SURFACE)
        a.set_xscale("log"); a.set_xlabel("median training seconds per final run (log)"); a.set_ylabel("TEST macro-F1 (mean ± std over folds)")
        a.set_title(ds, fontsize=11)
    fig.legend(handles=[plt.Line2D([], [], ls="none", marker=mk, color=col, ms=8, label=lab) for _, lab, col, mk in show],
               loc="upper right", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("Accuracy vs training cost per run (classifier stage only; HuBERT feature extraction not included)", x=0.01,
                 y=0.9, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86)); ps.save(fig, PATHS.plots / "FINAL_accuracy_vs_cost.png")


def print_summary(A):
    for ds in DSN:
        print(f"===== {ds}")
        for m, v in A["verification"][ds].items():
            print(f"  {m:10s} {v['mean']:.4f} ± {v['std']:.4f}  summary ok {v['matches_summary']}  report ok {v['matches_report_3dp']}  "
                  f"|cm diff| {v['max_abs_diff_metric_vs_confusion']:.1e} |pred diff| {v['max_abs_diff_metric_vs_predictions']:.1e}")


if __name__ == "__main__":
    main()
