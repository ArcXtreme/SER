"""
Stage 4 tests for the Praat F0 tracker (approved switch from pYIN, 2026-11-06).
Covers exact grid alignment, known-F0 synthetic tones (the only ground truth we have),
the documented sub-harmonic behaviour above the ceiling, and the periodicity channel.
"""
import numpy as np
import pytest

from src.features.config import DEFAULT, HOP, WIN, PraatPitchConfig, n_frames
from src.features.prosody import _praat_padding, f0_praat, prosody103, prosody_frames

SR = 16000
PC = PraatPitchConfig()
RNG = np.random.default_rng(3)


def harmonic(f0: float, seconds: float = 1.5, noise: float = 0.02) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    y = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6) if k * f0 < SR / 2)
    return y / np.abs(y).max() + noise * RNG.standard_normal(len(t))


@pytest.mark.parametrize("hop,first_centre,window", [(HOP, WIN // 2, 800), (160, 0, 800),
                                                     (HOP, WIN // 2, 640), (160, 0, 1600)])
def test_padding_formula_puts_praat_frames_on_our_grid(hop, first_centre, window):
    """Re-derive Praat's frame placement (Sampled_shortTermAnalysis) for the padded signal."""
    for N in RNG.integers(400, 100_000, 300):
        n = n_frames(N) if hop == HOP else 1 + N // hop
        L, R, m = _praat_padding(int(N), n, hop, first_centre, window)
        M = L + N + R
        assert L >= 0 and R >= 0
        n_p = int(np.floor((M - window) / hop)) + 1
        first = (M - (n_p - 1) * hop) / 2                     # Praat's first frame centre (samples)
        assert n_p >= m + n
        assert first + m * hop - L == first_centre             # our frame 0
        assert first + (m + n - 1) * hop + window / 2 <= M      # last window inside the signal


def test_f0_praat_checks_alignment_for_many_lengths():
    """f0_praat asserts Praat's actual frame times on every call; exercise both grids."""
    for N in RNG.integers(500, 80_000, 40):
        y = 0.1 * RNG.standard_normal(int(N))
        assert len(f0_praat(y, n_frames(N), HOP, WIN // 2, 60, 800, PC)[0]) == n_frames(N)
        assert len(f0_praat(y, 1 + N // 160, 160, 0, 60, 350, PC)[0]) == 1 + N // 160


def test_non_integer_window_is_rejected():
    with pytest.raises(ValueError):
        f0_praat(np.zeros(16000), 10, HOP, WIN // 2, 70, 800, PC)   # 3*16000/70 not an integer


@pytest.mark.parametrize("f0", [70, 100, 200, 300, 450, 600, 750])
def test_known_f0_is_tracked_with_the_configured_range(f0):
    y = harmonic(f0)
    pf = prosody_frames(y, DEFAULT.prosody, n_frames(len(y)))
    est = 100 * 2 ** (pf[5:-5, 0] / 12)
    assert pf[5:-5, 1].all()
    np.testing.assert_allclose(np.median(est), f0, rtol=0.01)


@pytest.mark.parametrize("f_lo,dur", [(150, 0.1), (250, 0.05)])
def test_octave_jump_cost_does_not_suppress_fast_real_glides(f_lo, dur):
    """An octave rise within 50-100 ms (at / beyond the human limit) must be followed, not
    flattened, by the configured tracker (octave_jump_cost = 0.7)."""
    n_s = 24000
    t = np.arange(n_s) / SR
    f_true = f_lo * 2 ** np.clip((t - 0.5) / dur, 0, 1)
    y = sum(np.sin(k * 2 * np.pi * np.cumsum(f_true) / SR) / k for k in range(1, 6))
    n = n_frames(n_s)
    pf = prosody_frames(y / np.abs(y).max(), DEFAULT.prosody, n)
    truth = np.interp((HOP * np.arange(n) + WIN / 2) / SR, t, f_true)
    v = pf[:, 1] > 0.5
    err = np.abs(pf[v, 0] - 12 * np.log2(truth[v] / 100))
    assert v.mean() > 0.95 and (err > 1).mean() == 0


def test_disvoice_ceiling_produces_subharmonics_not_unvoiced():
    """Documented limitation of the faithful 60-350 Hz variant: a 600 Hz voice is reported
    as voiced at ~300 Hz (half), so the high-pitch information is silently distorted."""
    y = harmonic(600)
    f0, _ = f0_praat(y, 1 + len(y) // 160, 160, 0, 60, 350, PC)
    v = f0 > 0
    assert v.mean() > 0.9
    np.testing.assert_allclose(np.median(f0[v]), 300, rtol=0.02)


def test_periodicity_is_bounded_and_separates_tone_from_noise():
    tone = harmonic(200, noise=0.0)
    noise = RNG.standard_normal(len(tone))
    n = n_frames(len(tone))
    _, p_tone = f0_praat(tone, n, HOP, WIN // 2, 60, 800, PC)
    _, p_noise = f0_praat(noise, n, HOP, WIN // 2, 60, 800, PC)
    assert p_tone.min() >= 0 and p_tone.max() <= 1 and p_noise.min() >= 0 and p_noise.max() <= 1
    assert np.median(p_tone[5:-5]) > 0.95 and np.median(p_noise) < 0.5


def test_tracker_is_deterministic_and_prosody103_shape():
    y = harmonic(220, seconds=2.0) * (np.arange(32000) % 8000 < 6000)   # voiced bursts + pauses
    a = prosody_frames(y, DEFAULT.prosody, n_frames(len(y)))
    b = prosody_frames(y, DEFAULT.prosody, n_frames(len(y)))
    assert np.array_equal(a, b)
    for fmax in (DEFAULT.prosody.disvoice_f0_max, DEFAULT.prosody.f0_max_wide):
        v = prosody103(y, DEFAULT.prosody, fmax)
        assert v.shape == (103,)
