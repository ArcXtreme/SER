"""
Stage 5 - mapping the approved Stage 3 splits onto rows of the Stage 4 feature cache.

Every experiment (E1-E4) gets its data ONLY through fold_rows(); it never re-splits.
load_fold() (Stage 3) re-checks the index fingerprint and re-runs all leakage checks on
every call, so a stale or modified split cannot be used silently.

Returned row indices refer to the cache manifest (= FeatureStore row order).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.store import FeatureStore
from src.preprocessing.dataset_index import CLASSES
from src.preprocessing.splits import PARTITIONS, load_fold


@dataclass(frozen=True)
class FoldRows:
    dataset: str
    fold: int
    rows: dict          # partition -> np.ndarray of manifest row indices
    labels: dict        # partition -> np.ndarray of label ids (0..5, order = CLASSES)
    speakers: dict      # partition -> sorted list of speaker ids


def without_test(fr: FoldRows) -> FoldRows:
    """Copy of a fold WITHOUT the test partition, for model-selection code: any attempt to
    read fr.rows['test'] / fr.labels['test'] raises KeyError instead of silently leaking."""
    keep = ("train", "val")
    return FoldRows(fr.dataset, fr.fold, {p: fr.rows[p] for p in keep}, {p: fr.labels[p] for p in keep},
                    {p: fr.speakers[p] for p in keep})


def fold_rows(store: FeatureStore, index: pd.DataFrame, dataset: str, fold: int) -> FoldRows:
    parts = load_fold(index, dataset, fold)                     # verified speaker-independent
    pos = {p: i for i, p in enumerate(store.manifest.rel_path)}
    rows, labels, speakers = {}, {}, {}
    for p in PARTITIONS:
        r = np.array([pos[x] for x in parts[p].rel_path], dtype=np.int64)
        y = store.manifest.label_id.values[r].astype(np.int64)
        if not np.array_equal(y, parts[p].label_id.values):
            raise ValueError(f"{dataset} fold {fold} {p}: cache labels disagree with the index")
        rows[p], labels[p], speakers[p] = r, y, sorted(parts[p].speaker.unique())
    # defensive re-check at the level the models actually see (row indices)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(rows[a]) & set(rows[b]) or set(speakers[a]) & set(speakers[b]):
            raise AssertionError(f"{dataset} fold {fold}: overlap between {a} and {b}")
    if set(np.unique(store.manifest.label_id)) != set(range(len(CLASSES))):
        raise ValueError("label ids are not 0..5")
    return FoldRows(dataset, fold, rows, labels, speakers)
