"""
Stage 2 driver: index -> integrity check -> summary statistics -> figures.

Run:   C:\\SER_data\\.venv\\Scripts\\python.exe scripts/stage2_prepare_data.py

Outputs (all small, kept in the repo for reproducibility):
  data/metadata/index.csv              one row per audio file (labels, speaker, audio stats)
  data/metadata/stage2_summary.json    verified counts used in the report
  plots/data_class_distribution.png
  plots/data_speaker_distribution.png
  plots/data_duration_distribution.png
  plots/data_cremad_demographics.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.preprocessing.audio_check import check_index  # noqa: E402
from src.preprocessing.dataset_index import CLASSES, build_index  # noqa: E402
from src.utils import plot_style as ps  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402


def summarise(df: pd.DataFrame) -> dict:
    """Compute every number the report will quote about the data (no hand-typed counts)."""
    out = {}
    for ds, g in df.groupby("dataset"):
        inc = g[g.included & (g.status == "ok")]
        spk = g.drop_duplicates("speaker")
        out[ds] = {
            "files_total": int(len(g)),
            "files_included_6class_ok": int(len(inc)),
            "files_excluded_by_label": {k: int(v) for k, v in
                                        g[~g.included].emotion_orig.value_counts().items()},
            "status_counts": {k: int(v) for k, v in g.status.value_counts().items()},
            "bad_files": g.loc[g.status != "ok", ["rel_path", "status", "error"]].to_dict("records"),
            "speakers": int(g.speaker.nunique()),
            "speakers_by_gender": {k: int(v) for k, v in spk.gender.value_counts().items()},
            "class_counts_included": {c: int((inc.label == c).sum()) for c in CLASSES},
            "sample_rates": {str(int(k)): int(v) for k, v in g.sample_rate.dropna().value_counts().items()},
            "channels": {str(int(k)): int(v) for k, v in g.channels.dropna().value_counts().items()},
            "duration_s": {q: round(float(g.duration_s.quantile(p)), 3) for q, p in
                           [("min", 0), ("p05", .05), ("median", .5), ("p95", .95), ("max", 1)]},
            "duration_total_h": round(float(g.duration_s.sum() / 3600), 3),
            "files_with_clipping_gt_0.1pct": int((g.clip_frac > 0.001).sum()),
            "clips_per_speaker_included": {
                "min": int(inc.groupby("speaker").size().min()),
                "max": int(inc.groupby("speaker").size().max())},
        }
        if ds == "CREMA-D":
            out[ds]["age"] = {"min": float(spk.age.min()), "median": float(spk.age.median()),
                              "max": float(spk.age.max())}
    return out


# ------------------------------------------------------------------ figures
def present(df: pd.DataFrame) -> list[str]:
    """Datasets present in the index, in a fixed display order."""
    return [d for d in ["RAVDESS", "CREMA-D"] if d in set(df.dataset)]


def plot_class_distribution(df: pd.DataFrame, path: Path) -> None:
    """Two small multiples (independent y-scales: corpus sizes differ ~7x)."""
    ok = df[df.status == "ok"]
    dsets = present(df)
    fig, axes = plt.subplots(1, len(dsets), figsize=(5 * len(dsets), 3.6), squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, dsets):
        g = ok[ok.dataset == ds]
        cats = CLASSES + (["calm", "surprised"] if ds == "RAVDESS" else [])
        counts = [int(((g.label == c) if c in CLASSES else (g.emotion_orig == c)).sum())
                  for c in cats]
        colors = [ps.DATASET_COLOR[ds] if c in CLASSES else ps.AXIS for c in cats]
        bars = ax.bar(cats, counts, color=colors, width=0.7)
        for b, n in zip(bars, counts):
            ax.annotate(str(n), (b.get_x() + b.get_width() / 2, n), ha="center", va="bottom",
                        fontsize=8, color=ps.INK_2, xytext=(0, 2), textcoords="offset points")
        ax.set_title(f"{ds}  (n={sum(counts[:6])} used)")
        ax.set_ylabel("recordings")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Emotion class distribution", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.text(0.01, -0.02, "Grey bars: RAVDESS classes excluded from the shared 6-class task "
             "(calm, surprised).", fontsize=8, color=ps.MUTED)
    ps.save(fig, path)


def plot_speaker_distribution(df: pd.DataFrame, path: Path) -> None:
    """Per-speaker counts stacked by emotion; speakers sorted by gender then ID."""
    inc = df[df.included & (df.status == "ok")]
    dsets = present(df)
    fig, axes = plt.subplots(len(dsets), 1, figsize=(12, 3.4 * len(dsets)), squeeze=False)
    axes = axes[:, 0]
    for ax, ds in zip(axes, dsets):
        g = inc[inc.dataset == ds]
        # male speakers first (the "male"/"female" labels below assume this order)
        spk = g.drop_duplicates("speaker").assign(m_first=lambda s: s.gender != "M")
        order = spk.sort_values(["m_first", "speaker"]).speaker.tolist()
        tab = g.pivot_table(index="speaker", columns="label", values="rel_path",
                            aggfunc="count", fill_value=0).reindex(index=order, columns=CLASSES)
        bottom = np.zeros(len(tab))
        x = np.arange(len(tab))
        for c in CLASSES:
            ax.bar(x, tab[c].values, bottom=bottom, color=ps.EMOTION_COLOR[c], width=0.8,
                   label=c, edgecolor=ps.SURFACE, linewidth=0.5)
            bottom += tab[c].values
        n_male = int((g.drop_duplicates("speaker").gender == "M").sum())
        ax.axvline(n_male - 0.5, color=ps.MUTED, lw=0.8, ls="--")
        ax.text(n_male / 2, bottom.max() * 1.02, "male", ha="center", color=ps.INK_2, fontsize=9)
        ax.text(n_male + (len(tab) - n_male) / 2, bottom.max() * 1.02, "female",
                ha="center", color=ps.INK_2, fontsize=9)
        ax.set_xlim(-0.6, len(tab) - 0.4)
        ax.set_ylim(0, bottom.max() * 1.12)
        ax.set_title(f"{ds}: {len(tab)} speakers, recordings per speaker (6-class subset)")
        ax.set_ylabel("recordings")
        if ds == "RAVDESS":
            ax.set_xticks(x, [s.split("_")[1] for s in tab.index], fontsize=7)
        else:
            ax.set_xticks([])
            ax.set_xlabel("speaker (sorted by gender, then ID)")
    axes[0].legend(ncol=6, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8)
    fig.tight_layout()
    ps.save(fig, path)


def plot_duration_distribution(df: pd.DataFrame, path: Path) -> None:
    ok = df[df.included & (df.status == "ok")]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.arange(0, ok.duration_s.max() + 0.25, 0.25)
    for ds in present(df):
        d = ok[ok.dataset == ds].duration_s
        ax.hist(d, bins=bins, density=True, histtype="step", lw=2, color=ps.DATASET_COLOR[ds],
                label=f"{ds} (median {d.median():.2f} s)")
    ax.set_xlabel("duration (s), before silence trimming")
    ax.set_ylabel("density")
    ax.set_title("Recording duration")
    ax.legend()
    ps.save(fig, path)


def plot_cremad_demographics(df: pd.DataFrame, path: Path) -> None:
    spk = df[df.dataset == "CREMA-D"].drop_duplicates("speaker")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.arange(15, 80, 5)
    ax.hist([spk[spk.gender == "M"].age, spk[spk.gender == "F"].age], bins=bins, stacked=True,
            color=[ps.SERIES[0], ps.SERIES[1]], label=["male", "female"], edgecolor=ps.SURFACE)
    ax.set_xlabel("actor age (years)")
    ax.set_ylabel("actors")
    ax.set_title(f"CREMA-D actor age and gender (n={len(spk)})")
    ax.legend()
    ps.save(fig, path)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["RAVDESS", "CREMA-D"])
    args = ap.parse_args()

    ps.apply()
    PATHS.metadata.mkdir(parents=True, exist_ok=True)
    PATHS.plots.mkdir(parents=True, exist_ok=True)

    print("Indexing", args.datasets, "...")
    df = build_index(PATHS.raw, args.datasets)
    print(f"  {len(df)} files indexed; checking audio integrity (decodes every file) ...")
    df = check_index(df, PATHS.raw)
    df.to_csv(PATHS.metadata / "index.csv", index=False)

    summary = summarise(df)
    (PATHS.metadata / "stage2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    plot_class_distribution(df, PATHS.plots / "data_class_distribution.png")
    plot_speaker_distribution(df, PATHS.plots / "data_speaker_distribution.png")
    plot_duration_distribution(df, PATHS.plots / "data_duration_distribution.png")
    if "CREMA-D" in present(df):
        plot_cremad_demographics(df, PATHS.plots / "data_cremad_demographics.png")
    print("Figures written to", PATHS.plots)


if __name__ == "__main__":
    main()
