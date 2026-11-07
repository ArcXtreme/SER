"""
Stage 3 - Speaker-independent train / validation / test splits.

WHY SPEAKER-INDEPENDENT?
------------------------
A voice carries speaker identity (pitch range, timbre, accent) much more strongly than
emotion. If clips of the same actor appear in both training and test data, a model can
partly "recognise the actor" instead of the emotion, and test accuracy overestimates how
well it works on new people. This is DATA LEAKAGE. We therefore split by SPEAKER: every
recording of a speaker goes to exactly one partition (train, val or test) within a fold.

PROTOCOL (within-corpus, used for E1-E4)
----------------------------------------
Outer loop - K-fold over speakers (GroupKFold family, groups = speaker ID):
    each speaker is in the TEST partition of exactly one fold.
Inner split - among the remaining (non-test) speakers, a second group split selects
    VALIDATION speakers (used only for early stopping / hyper-parameters).
    The rest are TRAIN speakers.

We use scikit-learn's StratifiedGroupKFold, which is GroupKFold (never splits a group)
plus an attempt to keep a label's proportion equal across folds. We stratify on GENDER,
so each test/validation partition contains male and female speakers - otherwise a fold
could test only on women, which would confound "unseen speakers" with "unseen gender".
Emotion does not need stratification: in both corpora every speaker records every class.

    RAVDESS : 24 speakers, 6 outer folds -> test 4, val 4, train 16 speakers per fold
    CREMA-D : 91 speakers, 5 outer folds -> test ~18, val ~15, train ~58 speakers per fold

CROSS-CORPUS (separate protocol, never mixed with the above)
------------------------------------------------------------
cross_corpus_split(): train+val on all speakers of corpus A, test on all of corpus B.
It is saved to a different file and must be reported in a separate results table.

REPRODUCIBILITY
---------------
The split is defined by speaker lists saved as JSON in data/splits/, together with the
seed and a SHA-256 fingerprint of the exact file set. Every experiment (E1-E4) loads the
JSON instead of re-splitting; load_fold() refuses to run if the index has changed.
Note: the split seed is fixed and separate from the model-training seeds, so running a
model with seeds 0, 1, 2 changes the initialisation, never the data partition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing.dataset_index import CLASSES

PARTITIONS = ("train", "val", "test")
DEFAULT_FOLDS = {"RAVDESS": 6, "CREMA-D": 5}
INNER_VAL_SPLITS = 5          # validation = ~1/5 of the non-test speakers
SPLIT_SEED = 42


class LeakageError(AssertionError):
    """Raised when a split violates speaker independence or coverage rules."""


def _check(condition: bool, message: str) -> None:
    # Explicit raise instead of `assert`, because `python -O` strips assert statements.
    if not condition:
        raise LeakageError(message)


# ----------------------------------------------------------------------------- inputs
def usable_files(index: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Rows of `dataset` that belong to the 6-class task AND passed the integrity check."""
    df = index[(index.dataset == dataset) & index.included & (index.status == "ok")]
    return df.sort_values("rel_path").reset_index(drop=True)


def fingerprint(files: pd.DataFrame) -> str:
    """
    SHA-256 over the sorted (path, speaker, label) triples.
    Any change to the file set or labels changes the hash, so a stale split is detected.
    """
    lines = sorted(f"{p}|{s}|{l}" for p, s, l in
                   zip(files.rel_path, files.speaker, files.label_id))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ----------------------------------------------------------------------------- splitting
def make_speaker_folds(files: pd.DataFrame, n_folds: int, seed: int = SPLIT_SEED,
                       inner_splits: int = INNER_VAL_SPLITS) -> list[dict]:
    """
    Build K speaker-independent folds with train/val/test speaker lists.

    Input : files  - one row per recording (needs columns speaker, gender)
            n_folds - number of outer folds K
    Output: list of K dicts {"fold": k, "train": [...], "val": [...], "test": [...]}
            (sorted speaker-ID lists; JSON-serialisable)

    The splitter sees RECORDINGS (rows of `files`) with groups = speaker ID, so the
    library itself guarantees that all recordings of a speaker land on the same side.
    y = gender is used only for stratification (balancing), not as a learning target.
    """
    files = files.sort_values("rel_path").reset_index(drop=True)   # deterministic row order
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (rest_idx, test_idx) in enumerate(outer.split(files, files.gender, groups=files.speaker)):
        rest = files.iloc[rest_idx].reset_index(drop=True)
        # Inner split on the non-test recordings only -> validation never touches test speakers.
        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + 1 + k)
        train_idx, val_idx = next(inner.split(rest, rest.gender, groups=rest.speaker))
        folds.append({
            "fold": k,
            "train": sorted(rest.speaker.iloc[train_idx].unique()),
            "val": sorted(rest.speaker.iloc[val_idx].unique()),
            "test": sorted(files.speaker.iloc[test_idx].unique()),
        })
    return folds


def partition_files(files: pd.DataFrame, fold: dict) -> dict[str, pd.DataFrame]:
    """Map a fold's speaker lists to recording-level DataFrames {train, val, test}."""
    return {p: files[files.speaker.isin(fold[p])].reset_index(drop=True) for p in PARTITIONS}


# ----------------------------------------------------------------------------- verification
def verify_folds(files: pd.DataFrame, folds: list[dict], require_all_classes: bool = True) -> dict:
    """
    Raise LeakageError on any violation; otherwise return a report of what was checked.

    Checks (per fold):
      1. speaker sets of train/val/test are pairwise disjoint;
      2. file sets of train/val/test are pairwise disjoint;
      3. coverage: every usable file is in exactly one partition, no unknown files;
      4. all 6 classes present in every partition (when require_all_classes);
      5. every partition contains both genders.
    Checks (across folds):
      6. each speaker is a TEST speaker in exactly one fold (proper K-fold);
      7. validation speakers are never test speakers of the same fold (covered by 1).
    """
    all_speakers = set(files.speaker)
    all_files = set(files.rel_path)
    _check(len(all_files) == len(files), "duplicate rel_path rows in the input index")
    test_count = {s: 0 for s in all_speakers}

    for fold in folds:
        k = fold["fold"]
        S = {p: set(fold[p]) for p in PARTITIONS}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            _check(not (S[a] & S[b]), f"fold {k}: speakers in both {a} and {b}: {sorted(S[a] & S[b])}")
        _check(S["train"] | S["val"] | S["test"] == all_speakers,
               f"fold {k}: speaker coverage mismatch "
               f"(missing {sorted(all_speakers - S['train'] - S['val'] - S['test'])})")

        parts = partition_files(files, fold)
        F = {p: set(parts[p].rel_path) for p in PARTITIONS}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            _check(not (F[a] & F[b]), f"fold {k}: {len(F[a] & F[b])} files in both {a} and {b}")
        _check(sum(len(F[p]) for p in PARTITIONS) == len(all_files)
               and (F["train"] | F["val"] | F["test"]) == all_files,
               f"fold {k}: not every usable file is in exactly one partition")

        for p in PARTITIONS:
            _check(len(parts[p]) > 0, f"fold {k}: empty {p} partition")
            if require_all_classes:
                missing = set(CLASSES) - set(parts[p].label)
                _check(not missing, f"fold {k}: {p} partition lacks classes {sorted(missing)}")
            _check(parts[p].gender.nunique() == 2, f"fold {k}: {p} partition is single-gender")
        for s in S["test"]:
            test_count[s] += 1

    bad = {s: c for s, c in test_count.items() if c != 1}
    _check(not bad, f"speakers not tested exactly once across folds: {bad}")
    return {"n_folds": len(folds), "n_speakers": len(all_speakers), "n_files": len(all_files),
            "checks_passed": ["speaker_disjoint", "file_disjoint", "full_coverage",
                              "all_classes_present" if require_all_classes else "classes_not_required",
                              "both_genders_present", "each_speaker_tested_once"]}


# ----------------------------------------------------------------------------- cross-corpus
def cross_corpus_split(source: pd.DataFrame, target: pd.DataFrame, seed: int = SPLIT_SEED) -> dict:
    """
    SEPARATE protocol: train/val on the source corpus (val = held-out SOURCE speakers),
    test on ALL speakers of the target corpus. Source and target speaker IDs are
    dataset-prefixed (RAV_/CRE_), so they can never collide.
    """
    src_ds, tgt_ds = source.dataset.unique(), target.dataset.unique()
    _check(len(src_ds) == 1 and len(tgt_ds) == 1 and src_ds[0] != tgt_ds[0],
           "cross-corpus split needs two different single-dataset inputs")
    src = source.sort_values("rel_path").reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=INNER_VAL_SPLITS, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(src, src.gender, groups=src.speaker))
    split = {"protocol": "cross-corpus", "source": src_ds[0], "target": tgt_ds[0],
             "train": sorted(src.speaker.iloc[tr].unique()), "val": sorted(src.speaker.iloc[va].unique()),
             "test": sorted(target.speaker.unique())}
    _check(not (set(split["train"]) & set(split["val"])), "cross-corpus: train/val overlap")
    _check(not (set(split["test"]) & (set(split["train"]) | set(split["val"]))),
           "cross-corpus: test speakers overlap source speakers")
    return split


# ----------------------------------------------------------------------------- save / load
def split_path(dataset: str) -> Path:
    from src.utils.paths import PATHS
    return PATHS.splits / f"{dataset}_speaker_folds.json"


def save_folds(dataset: str, files: pd.DataFrame, folds: list[dict], report: dict,
               seed: int = SPLIT_SEED) -> Path:
    payload = {
        "dataset": dataset,
        "protocol": "within-corpus, speaker-independent",
        "method": "StratifiedGroupKFold (groups=speaker, stratified by gender); "
                  f"validation = first of {INNER_VAL_SPLITS} inner group splits of non-test speakers",
        "seed": seed,
        "n_folds": len(folds),
        "index_sha256": fingerprint(files),
        "verification": report,
        "folds": folds,
    }
    path = split_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_fold(index: pd.DataFrame, dataset: str, fold: int) -> dict[str, pd.DataFrame]:
    """
    THE function every experiment (E1-E4) uses to get its data.

    Loads the saved speaker lists, checks the fingerprint against the current index
    (refuses a stale split), re-runs the leakage verification, and returns
    {"train": df, "val": df, "test": df} of recordings.
    """
    payload = json.loads(split_path(dataset).read_text(encoding="utf-8"))
    files = usable_files(index, dataset)
    _check(fingerprint(files) == payload["index_sha256"],
           f"{dataset}: index has changed since the split was created - regenerate splits")
    verify_folds(files, payload["folds"])
    return partition_files(files, payload["folds"][fold])
