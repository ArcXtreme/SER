"""
Stage 4 prosody tests. The key test compares our re-implementation of the DisVoice-103
features against the ORIGINAL DisVoice functions (tests/reference/, MIT licence) on the
same F0 contour and audio. Agreement shows we reproduce the paper's feature definitions;
the only intended difference (energy floor for all-zero frames) is avoided by using
non-silent test audio.
"""
import warnings

import numpy as np
import pytest

from src.features.config import DEFAULT
from src.features.prosody import (DISVOICE_NAMES, disvoice_static, duration_features,
                                  energy_features, f0_features, segments)
from tests.reference import disvoice_prosody_functions as ref

SR = 16000
STEP, FRAME = 160.0, 320.0
CFG = DEFAULT.prosody


def make_case(seed: int):
    """Random voiced/unvoiced/pause pattern on a 10 ms grid + matching noisy audio."""
    rng = np.random.default_rng(seed)
    F0 = []
    while len(F0) < 300:
        if rng.random() < 0.6:                          # voiced run with a pitch glide
            L = int(rng.integers(2, 40))
            F0 += list(np.linspace(rng.uniform(90, 300), rng.uniform(90, 300), L) + rng.normal(0, 3, L))
        else:                                           # unvoiced (short) or pause (> 14 frames)
            F0 += [0.0] * int(rng.integers(1, 30))
    F0 = np.asarray(F0[:300])
    y = rng.normal(0, 0.05, int(len(F0) * STEP + 400)) * (1 + 0.5 * np.sin(np.arange(int(len(F0) * STEP + 400)) / 900))
    y = (y - y.mean()) / np.abs(y - y.mean()).max()
    return F0, y


def reference_103(F0, y):
    """Exactly the composition in DisVoice Prosody.prosody_static (after F0 tracking)."""
    segV = ref.V_UV(F0, y, type_seg="Voiced", size_stepS=STEP)
    segUP = ref.V_UV(F0, y, type_seg="Unvoiced", size_stepS=STEP)
    segP = [s for s in segUP if len(s) > 0.14 * SR]
    segU = [s for s in segUP if len(s) <= 0.14 * SR]
    return np.hstack((ref.F0feat(F0), ref.energy_feat(segV, SR, FRAME, STEP),
                      ref.energy_feat(segU, SR, FRAME, STEP), ref.duration_feat(segV, segU, segP, y, SR)))


@pytest.mark.parametrize("seed", range(8))
def test_disvoice103_matches_original_implementation(seed):
    F0, y = make_case(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = reference_103(F0, y)
        ours = disvoice_static(y, F0, CFG)
    assert ours.shape == expected.shape == (103,)
    bad = ~np.isclose(ours, expected, rtol=1e-7, atol=1e-9, equal_nan=True)
    assert not bad.any(), [(DISVOICE_NAMES[i], ours[i], expected[i]) for i in np.flatnonzero(bad)]


def test_segmentation_matches_original_v_uv():
    F0, y = make_case(3)
    for voiced, kind in [(True, "Voiced"), (False, "Unvoiced")]:
        a = segments(F0, y, voiced, STEP)
        b = ref.V_UV(F0, y, type_seg=kind, size_stepS=STEP)
        assert len(a) == len(b) and all(np.array_equal(s, t) for s, t in zip(a, b))


def test_feature_groups_have_documented_sizes():
    F0, y = make_case(1)
    segs = segments(F0, y, True, STEP)
    assert len(f0_features(F0, 0.01)) == 30
    assert len(energy_features(segs, FRAME, STEP, SR)) == 24
    assert len(duration_features(segs, segs, segs, len(y), SR)) == 25
    assert len(DISVOICE_NAMES) == len(set(DISVOICE_NAMES)) == 103


def test_no_voicing_gives_zero_f0_features():
    assert f0_features(np.zeros(100), 0.01) == [0.0] * 30
