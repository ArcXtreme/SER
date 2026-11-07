"""
Stage 4 tests: shared time grid, MFCC pipeline (vs independent references), temporal
alignment, padding masks and aligned cropping. Synthetic signals only - no dataset needed.
"""
import librosa
import numpy as np
import pytest
import scipy.fft
import torch
from transformers import HubertConfig

from src.features.batching import crop_window, masked_mean, masked_std, pad_sequences
from src.features.config import DEFAULT, HOP, SAMPLE_RATE, WIN, frame_centers_s, n_frames
from src.features.mfcc import MFCCExtractor, dct_matrix, deltas, frame_signal, mel_filterbank
from src.features.prosody import prosody_frames

SR = SAMPLE_RATE
RNG = np.random.default_rng(0)


def tone(f, seconds, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


# ------------------------------------------------------------------ shared time grid
def hubert_conv_frames(n_samples: int) -> int:
    """Frame count from HuBERT's own conv-stack definition (no weights needed)."""
    cfg = HubertConfig()                    # hubert-base architecture defaults
    L = n_samples
    for k, s in zip(cfg.conv_kernel, cfg.conv_stride):
        L = (L - k) // s + 1
    return L


@pytest.mark.parametrize("n", [400, 401, 719, 720, 721, 16000, 31999, 112000, 123457])
def test_frame_count_formula_matches_hubert_conv_stack(n):
    assert n_frames(n) == hubert_conv_frames(n)


def test_frame_signal_matches_manual_slicing_and_grid():
    y = RNG.standard_normal(5000).astype(np.float32)
    F = frame_signal(y)
    assert F.shape == (n_frames(5000), WIN)
    for t in [0, 1, len(F) - 1]:
        np.testing.assert_array_equal(F[t], y[HOP * t: HOP * t + WIN])
    np.testing.assert_allclose(frame_centers_s(3), [0.0125, 0.0325, 0.0525])


# ------------------------------------------------------------------ MFCC components
def test_mel_filterbank_matches_librosa_htk():
    ours = mel_filterbank(SR, 512, 40, 0.0, 8000.0)
    ref = librosa.filters.mel(sr=SR, n_fft=512, n_mels=40, fmin=0.0, fmax=8000.0, htk=True, norm=None)
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_dct_matrix_matches_scipy_orthonormal_dct2():
    x = RNG.standard_normal((7, 40))
    np.testing.assert_allclose(x @ dct_matrix(40).T, scipy.fft.dct(x, type=2, norm="ortho", axis=1), atol=1e-10)


def test_mel_power_spectrum_matches_librosa_on_identical_samples():
    """
    librosa with n_fft=512, win_length=400, center=False puts the 400-sample window in the
    MIDDLE of a 512 frame (offset 56). Pre-padding the signal by 56 makes librosa analyse
    exactly our samples y[320t : 320t+400]; |FFT| is invariant to where the zero padding sits.
    """
    y = RNG.standard_normal(8000).astype(np.float32) * 0.1
    ex = MFCCExtractor(DEFAULT.mfcc)
    ours = (ex.power_spectrum(frame_signal(y)) @ ex.fbank.T) * DEFAULT.mfcc.n_fft   # undo the 1/N
    ref = librosa.feature.melspectrogram(y=np.pad(y, (56, 0)), sr=SR, n_fft=512, hop_length=HOP,
                                         win_length=WIN, window="hamming", center=False, power=2.0,
                                         n_mels=40, fmin=0.0, fmax=8000.0, htk=True, norm=None).T
    T = min(len(ours), len(ref))
    np.testing.assert_allclose(ours[:T], ref[:T], rtol=1e-4, atol=1e-6)


def test_deltas_of_ramp_equal_slope_and_of_constant_are_zero():
    t = np.arange(50, dtype=np.float64)[:, None]
    d = deltas(3.0 * t + 1.0)
    np.testing.assert_allclose(d[2:-2], 3.0)          # interior: exact slope
    np.testing.assert_allclose(deltas(np.ones((10, 2))), 0.0)


def test_mfcc_output_layout():
    y = tone(220, 1.0) + 0.01 * RNG.standard_normal(SR).astype(np.float32)
    out = MFCCExtractor(DEFAULT.mfcc)(y)
    assert out.shape == (n_frames(len(y)), 39) and out.dtype == np.float32
    assert np.isfinite(out).all()
    energy = np.log((frame_signal(y).astype(np.float64) ** 2).sum(1))
    np.testing.assert_allclose(out[:, 12], energy, rtol=1e-5)       # column 12 = log energy
    np.testing.assert_allclose(out[:, 13:26], deltas(out[:, :13].astype(np.float64)), atol=1e-4)


# ------------------------------------------------------------------ temporal alignment
def test_energy_onset_lands_on_the_same_frame_in_mfcc_and_prosody():
    """Silence then a tone starting at sample s0: the first frame whose window reaches s0
    must be the first high-energy frame in BOTH the MFCC energy and the prosody energy."""
    s0 = 5 * HOP + 150                                   # inside frames 4 and 5
    y = np.concatenate([np.zeros(s0, np.float32), tone(200, 1.0)])
    y = y + 1e-6 * RNG.standard_normal(len(y)).astype(np.float32)
    T = n_frames(len(y))
    first_expected = int(np.ceil((s0 - WIN + 1) / HOP))  # smallest t with 320t + 400 > s0
    mf = MFCCExtractor(DEFAULT.mfcc)(y)
    pf = prosody_frames(y, DEFAULT.prosody, T)
    assert pf.shape == (T, 4) and mf.shape == (T, 39)
    first_mfcc = int(np.argmax(mf[:, 12] > mf[0, 12] + 10))
    first_pros = int(np.argmax(pf[:, 3] > pf[0, 3] + 40))
    assert first_mfcc == first_pros == first_expected


def test_f0_frames_are_centred_on_the_hubert_grid():
    """F0 jumps 150 -> 300 Hz at sample s; the jump must appear at the frame whose CENTRE
    (320t + 200) crosses s, within the smoothing of the 50 ms Praat window (3 periods of 60 Hz)."""
    s = int(1.0 * SR)
    y = np.concatenate([tone(150, 1.0), tone(300, 1.0)])
    T = n_frames(len(y))
    pf = prosody_frames(y, DEFAULT.prosody, T)
    f0 = 100 * 2 ** (pf[:, 0] / 12)
    jump = int(np.argmax(f0 > 225))
    centre_frame = int(np.ceil((s - WIN / 2) / HOP))
    assert abs(jump - centre_frame) <= 2
    mid = slice(10, 40)
    np.testing.assert_allclose(f0[mid], 150, rtol=0.02)          # correct pitch
    assert pf[mid, 1].all()                                      # voiced


# ------------------------------------------------------------------ padding & masks
def test_pad_sequences_mask_and_masked_statistics():
    seqs = [RNG.standard_normal((n, 3)).astype(np.float32) for n in (5, 2, 7)]
    x, mask = pad_sequences(seqs)
    assert x.shape == (3, 7, 3) and mask.shape == (3, 7)
    assert mask.sum(1).tolist() == [5, 2, 7]
    assert (x[~mask] == 0).all()                                  # padding is exactly zero
    for b, s in enumerate(seqs):
        np.testing.assert_allclose(masked_mean(x, mask)[b], s.mean(0), atol=1e-6)
        np.testing.assert_allclose(masked_std(x, mask)[b], s.std(0), atol=1e-5)
    # Changing values in padded positions must not change the masked statistics
    x2 = x.clone()
    x2[~mask] = 1e6
    torch.testing.assert_close(masked_mean(x2, mask), masked_mean(x, mask))


def test_crop_window_is_shared_and_bounded():
    assert crop_window(100, None) == (0, 100) and crop_window(100, 200) == (0, 100)
    assert crop_window(100, 40, "center") == (30, 70)
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = crop_window(100, 40, "random", rng)
        assert 0 <= a and b <= 100 and b - a == 40
