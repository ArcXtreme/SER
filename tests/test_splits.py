"""
Tests for Stage 3 splitting. A synthetic index mimics RAVDESS (24 speakers, 12 M / 12 F,
every speaker records every class) so the tests are fast and independent of real data.

Most important: the NEGATIVE tests deliberately inject leakage and check that
verify_folds() catches it - a checker that never fails proves nothing.
"""
import copy

import pandas as pd
import pytest

from src.preprocessing.dataset_index import CLASSES
from src.preprocessing.splits import (LeakageError, cross_corpus_split, fingerprint,
                                      make_speaker_folds, partition_files, verify_folds)


def synthetic(prefix="RAV", n_speakers=24, per_class=2, dataset="RAVDESS"):
    rows = []
    for s in range(1, n_speakers + 1):
        for c in CLASSES:
            for r in range(per_class):
                rows.append(dict(dataset=dataset, speaker=f"{prefix}_{s:02d}",
                                 gender="M" if s % 2 else "F", label=c,
                                 label_id=CLASSES.index(c), rel_path=f"{prefix}/{s:02d}/{c}_{r}.wav"))
    return pd.DataFrame(rows)


@pytest.fixture
def files():
    return synthetic()


@pytest.fixture
def folds(files):
    return make_speaker_folds(files, n_folds=6)


def test_valid_split_passes_and_has_expected_sizes(files, folds):
    report = verify_folds(files, folds)
    assert report["n_folds"] == 6
    for f in folds:
        assert (len(f["train"]), len(f["val"]), len(f["test"])) == (16, 4, 4)


def test_split_is_deterministic(files, folds):
    assert make_speaker_folds(files, n_folds=6) == folds


def test_different_seed_gives_different_split(files, folds):
    assert make_speaker_folds(files, n_folds=6, seed=7) != folds


def test_gender_balanced_test_and_val(files, folds):
    for f in folds:
        for p in ("val", "test"):
            g = files[files.speaker.isin(f[p])].drop_duplicates("speaker").gender
            assert (g == "M").sum() == (g == "F").sum() == 2


def test_partition_files_keeps_all_recordings_of_a_speaker_together(files, folds):
    parts = partition_files(files, folds[0])
    for p, d in parts.items():
        for s in d.speaker.unique():
            assert len(d[d.speaker == s]) == len(files[files.speaker == s])


# ---------------------------------------------------------------- negative tests
def test_detects_speaker_in_train_and_test(files, folds):
    bad = copy.deepcopy(folds)
    bad[0]["train"].append(bad[0]["test"][0])
    with pytest.raises(LeakageError, match="speakers in both train and test"):
        verify_folds(files, bad)


def test_detects_validation_overlapping_test(files, folds):
    bad = copy.deepcopy(folds)
    bad[1]["val"].append(bad[1]["test"][0])
    with pytest.raises(LeakageError, match="val and test"):
        verify_folds(files, bad)


def test_detects_missing_speaker(files, folds):
    bad = copy.deepcopy(folds)
    bad[2]["train"].pop()
    with pytest.raises(LeakageError, match="coverage"):
        verify_folds(files, bad)


def test_detects_speaker_tested_twice(files, folds):
    bad = copy.deepcopy(folds)
    # move one speaker from fold 0's test into fold 1's test (and out of fold 1 train/val)
    s = bad[0]["test"][0]
    for p in ("train", "val"):
        if s in bad[1][p]:
            bad[1][p].remove(s)
    bad[1]["test"].append(s)
    with pytest.raises(LeakageError, match="tested exactly once"):
        verify_folds(files, bad)


def test_detects_missing_class(files, folds):
    no_disgust = files[~((files.label == "disgust") & files.speaker.isin(folds[0]["test"]))]
    with pytest.raises(LeakageError, match="lacks classes"):
        verify_folds(no_disgust, folds)


def test_fingerprint_changes_when_files_change(files):
    assert fingerprint(files) != fingerprint(files.iloc[1:])


# ---------------------------------------------------------------- cross-corpus
def test_cross_corpus_split_is_disjoint_and_complete():
    src = synthetic("CRE", 20, dataset="CREMA-D")
    tgt = synthetic("RAV", 8, dataset="RAVDESS")
    split = cross_corpus_split(src, tgt)
    assert set(split["test"]) == set(tgt.speaker)
    assert set(split["train"]) | set(split["val"]) == set(src.speaker)
    assert not set(split["train"]) & set(split["val"])
