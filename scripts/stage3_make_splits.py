"""
Stage 3 driver: create, verify, save and visualise speaker-independent splits.

Run:   C:\\SER_data\\.venv\\Scripts\\python.exe scripts/stage3_make_splits.py [--datasets RAVDESS CREMA-D]

Outputs
  data/splits/<DS>_speaker_folds.json      speaker lists per fold + seed + index fingerprint
  data/splits/<DS>_split_summary.csv       speakers / recordings / class counts per fold & partition
  data/splits/cross_corpus_<A>_to_<B>.json (only when both corpora are indexed)
  plots/splits_<DS>.png                    speaker x fold assignment grid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.preprocessing.dataset_index import CLASSES  # noqa: E402
from src.preprocessing.splits import (DEFAULT_FOLDS, PARTITIONS, cross_corpus_split,  # noqa: E402
                                      load_fold, make_speaker_folds, partition_files,
                                      save_folds, usable_files, verify_folds)
from src.utils import plot_style as ps  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402

PART_COLOR = {"train": ps.SERIES[0], "val": ps.SERIES[1], "test": ps.SERIES[2]}


def summary_table(files: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        parts = partition_files(files, fold)
        for p in PARTITIONS:
            d = parts[p]
            spk = d.drop_duplicates("speaker")
            row = {"fold": fold["fold"], "partition": p,
                   "speakers": d.speaker.nunique(),
                   "male": int((spk.gender == "M").sum()), "female": int((spk.gender == "F").sum()),
                   "recordings": len(d)}
            row.update({c: int((d.label == c).sum()) for c in CLASSES})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_assignment(files: pd.DataFrame, folds: list[dict], dataset: str, path: Path) -> None:
    """Grid: rows = folds, columns = speakers (male first); colour = partition."""
    spk = files.drop_duplicates("speaker")
    spk = spk.assign(m_first=spk.gender != "M").sort_values(["m_first", "speaker"])
    speakers = spk.speaker.tolist()
    n_male = int((spk.gender == "M").sum())
    code = {p: i for i, p in enumerate(PARTITIONS)}
    grid = np.array([[code[next(p for p in PARTITIONS if s in f[p])] for s in speakers] for f in folds])

    width = max(8.0, 0.32 * len(speakers) + 3) if len(speakers) <= 30 else 14.0
    fig, ax = plt.subplots(figsize=(width, 0.42 * len(folds) + 1.8))
    ax.pcolormesh(grid, cmap=ListedColormap([PART_COLOR[p] for p in PARTITIONS]),
                  vmin=-0.5, vmax=2.5, edgecolors=ps.SURFACE, linewidth=1.0 if len(speakers) <= 30 else 0.3)
    ax.set_yticks(np.arange(len(folds)) + 0.5, [f"fold {f['fold']}" for f in folds])
    ax.invert_yaxis()
    if len(speakers) <= 30:
        ax.set_xticks(np.arange(len(speakers)) + 0.5, [s.split("_")[1] for s in speakers], fontsize=7)
    else:
        ax.set_xticks([])
    ax.axvline(n_male, color=ps.INK, lw=1.2)
    ax.set_xlabel(f"speakers  (left of line: {n_male} male  |  right: {len(speakers) - n_male} female)")
    ax.grid(False)
    for i, f in enumerate(folds):
        ax.text(len(speakers) + 0.4, i + 0.5,
                f"train {len(f['train'])} · val {len(f['val'])} · test {len(f['test'])}",
                va="center", fontsize=8, color=ps.INK_2)
    ax.set_xlim(0, len(speakers))
    ax.set_title(f"{dataset}: speaker-independent folds (each speaker is in exactly one partition per fold)")
    ax.legend(handles=[Patch(color=PART_COLOR[p], label=p) for p in PARTITIONS],
              ncol=3, loc="upper left", bbox_to_anchor=(0, -0.18 if len(speakers) <= 30 else -0.08))
    ps.save(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["RAVDESS", "CREMA-D"])
    args = ap.parse_args()
    ps.apply()

    index = pd.read_csv(PATHS.metadata / "index.csv")
    for ds in args.datasets:
        files = usable_files(index, ds)
        folds = make_speaker_folds(files, DEFAULT_FOLDS[ds])

        # Reproducibility: regenerating with the same seed must give identical folds.
        assert make_speaker_folds(files, DEFAULT_FOLDS[ds]) == folds, "split is not deterministic"

        report = verify_folds(files, folds)          # raises LeakageError on any violation
        path = save_folds(ds, files, folds, report)

        # Round trip: reload exactly as an experiment would (fingerprint + re-verification).
        for f in folds:
            parts = load_fold(index, ds, f["fold"])
            assert {p: sorted(parts[p].speaker.unique()) for p in PARTITIONS} == \
                   {p: f[p] for p in PARTITIONS}

        table = summary_table(files, folds)
        table.to_csv(PATHS.splits / f"{ds}_split_summary.csv", index=False)
        plot_assignment(files, folds, ds, PATHS.plots / f"splits_{ds}.png")

        print(f"\n=== {ds}: {report['n_files']} recordings, {report['n_speakers']} speakers, "
              f"{report['n_folds']} folds -> {path.name}")
        print("Verification passed:", ", ".join(report["checks_passed"]))
        print(table.to_string(index=False))

    # Cross-corpus protocol (separate files, separate results tables later)
    if {"RAVDESS", "CREMA-D"} <= set(index.dataset):
        for src, tgt in [("CREMA-D", "RAVDESS"), ("RAVDESS", "CREMA-D")]:
            split = cross_corpus_split(usable_files(index, src), usable_files(index, tgt))
            out = PATHS.splits / f"cross_corpus_{src}_to_{tgt}.json"
            out.write_text(json.dumps(split, indent=2), encoding="utf-8")
            print(f"\ncross-corpus {src} -> {tgt}: train {len(split['train'])} / "
                  f"val {len(split['val'])} {src} speakers, test {len(split['test'])} {tgt} speakers")


if __name__ == "__main__":
    main()
