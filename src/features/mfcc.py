"""
Stage 4 - MFCC + log-energy + deltas (39 features per frame), implemented step by step.

WHY MFCC FOR EMOTION?
  MFCCs summarise the short-term spectral ENVELOPE (vocal-tract shape / timbre) on a
  perceptual (mel) frequency scale. Emotions change articulation and voice quality
  (e.g. tense, bright voice in anger; lax, dark voice in sadness), which shows up in the
  envelope. Deltas add how the envelope CHANGES over time.

PIPELINE (paper Eqs. 5-8; one frame t = samples [320t, 320t+400) of the shared grid)
  1. framing          x_t[n] = y[320 t + n],  n = 0..399                        (T, 400)
  2. window           x_t[n] * w[n]          (Hamming)                           (T, 400)
  3. DFT              Y_t(k) = sum_n x_t[n] w[n] e^{-j 2 pi k n / N}, N = 512     (T, 257)
  4. power spectrum   P_t(k) = |Y_t(k)|^2 / N                                   (T, 257)
  5. mel filterbank   M_t(m) = sum_k H_m(k) P_t(k),  40 triangular filters      (T, 40)
  6. log              L_t(m) = ln(max(M_t(m), 1e-10))                           (T, 40)
  7. DCT-II (ortho)   c_t(i) = sum_m L_t(m) cos(pi i (m + 0.5) / 40) * a_i      (T, 40)
                      keep i = 1..12 (c0 ~ overall level, replaced by energy)   (T, 12)
  8. log energy       E_t = ln(sum_n x_t[n]^2)   (raw, un-windowed frame, HTK "E") (T, 1)
  9. deltas           d_t = sum_{n=1..2} n (c_{t+n} - c_{t-n}) / (2 sum n^2)    (T, 13)
                      delta-delta = delta of delta                              (T, 13)
  output = [c1..c12, E, d(c1..c12, E), dd(c1..c12, E)]                          (T, 39)

Differences from the paper's text (documented in reports/feature_extraction.md):
  the paper gives 40 ms frames / 20 ms shift / 30 ms Hamming (internally inconsistent);
  we use 25 ms / 20 ms so that MFCC frames coincide exactly with HuBERT frames.
No normalisation is applied here: features are cached raw and z-scored later with
TRAINING-FOLD statistics only (normalising with all data would leak test information).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from src.features.config import MFCCConfig, SAMPLE_RATE, WIN, HOP


# ----------------------------------------------------------------------------- framing
def frame_signal(y: np.ndarray, win: int = WIN, hop: int = HOP) -> np.ndarray:
    """
    Slice y into overlapping frames on the shared grid (no padding, no centering).
    Input (N,) -> output (T, win) with T = 1 + (N - win) // hop. Returns a read-only view.
    """
    if len(y) < win:
        return np.zeros((0, win), dtype=y.dtype)
    t = 1 + (len(y) - win) // hop
    return np.lib.stride_tricks.sliding_window_view(y, win)[::hop][:t]


# ----------------------------------------------------------------------------- mel scale
def hz_to_mel(f):
    """HTK mel scale: perceived pitch grows ~linearly below 1 kHz, logarithmically above."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """
    Triangular filters H_m(k), shape (n_mels, n_fft//2 + 1), peak height 1 (no area norm).

    The n_mels + 2 edge frequencies are equally spaced on the MEL axis, so filters are
    narrow at low frequencies (where hearing resolves pitch finely) and wide at high ones.
    Filter m rises linearly from edge m to edge m+1 and falls to zero at edge m+2.
    """
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    widths = np.diff(edges)                                  # (n_mels + 1,)
    ramps = edges[:, None] - fft_freqs[None, :]               # (n_mels + 2, n_bins)
    lower = -ramps[:-2] / widths[:-1, None]                   # rising slope
    upper = ramps[2:] / widths[1:, None]                      # falling slope
    return np.maximum(0.0, np.minimum(lower, upper)).astype(np.float32)


def dct_matrix(n_in: int) -> np.ndarray:
    """
    Orthonormal DCT-II matrix D (n_in, n_in): c = D @ L.
    Decorrelates the (highly correlated) log filter-bank energies; low-order coefficients
    describe the smooth spectral envelope.
    """
    i = np.arange(n_in)[:, None]
    m = np.arange(n_in)[None, :]
    D = np.cos(np.pi * i * (m + 0.5) / n_in) * np.sqrt(2.0 / n_in)
    D[0] /= np.sqrt(2.0)
    return D.astype(np.float64)


# ----------------------------------------------------------------------------- deltas
def deltas(x: np.ndarray, width: int = 2) -> np.ndarray:
    """
    Regression deltas along time (axis 0), edges replicated.
    d_t = sum_{n=1..W} n (x_{t+n} - x_{t-n}) / (2 sum_{n=1..W} n^2)
    For a linear ramp x_t = a t + b, d_t = a exactly (tested).
    Computed per utterance BEFORE any padding, so padding can never leak into deltas.
    """
    if len(x) == 0:
        return x.copy()
    pad = np.pad(x, ((width, width), (0, 0)), mode="edge")
    T = len(x)
    num = sum(n * (pad[width + n: width + n + T] - pad[width - n: width - n + T])
              for n in range(1, width + 1))
    return num / (2.0 * sum(n * n for n in range(1, width + 1)))


# ----------------------------------------------------------------------------- main
class MFCCExtractor:
    """Pre-computes window, filterbank and DCT once; __call__ maps (N,) audio -> (T, 39)."""

    def __init__(self, cfg: MFCCConfig, sr: int = SAMPLE_RATE, win: int = WIN, hop: int = HOP):
        self.cfg, self.sr, self.win, self.hop = cfg, sr, win, hop
        self.window = get_window(cfg.window, win, fftbins=True).astype(np.float64)
        self.fbank = mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax).astype(np.float64)
        self.dct = dct_matrix(cfg.n_mels)

    def power_spectrum(self, frames: np.ndarray) -> np.ndarray:
        """(T, win) -> (T, n_fft//2+1): |DFT of windowed, zero-padded frame|^2 / N."""
        spec = np.fft.rfft(frames.astype(np.float64) * self.window, n=self.cfg.n_fft, axis=1)
        return (np.abs(spec) ** 2) / self.cfg.n_fft

    def log_mel(self, frames: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(self.power_spectrum(frames) @ self.fbank.T, self.cfg.log_floor))

    def __call__(self, y: np.ndarray) -> np.ndarray:
        frames = frame_signal(y, self.win, self.hop)                          # (T, 400)
        ceps = self.log_mel(frames) @ self.dct.T                              # (T, 40)
        ceps = ceps[:, 1:self.cfg.n_ceps + 1]                                 # (T, 12)
        energy = np.log(np.maximum((frames.astype(np.float64) ** 2).sum(1), self.cfg.log_floor))
        static = np.concatenate([ceps, energy[:, None]], axis=1)              # (T, 13)
        d1 = deltas(static, self.cfg.delta_width)
        d2 = deltas(d1, self.cfg.delta_width)
        return np.concatenate([static, d1, d2], axis=1).astype(np.float32)    # (T, 39)


MFCC_NAMES = ([f"c{i}" for i in range(1, 13)] + ["logE"]
              + [f"d_c{i}" for i in range(1, 13)] + ["d_logE"]
              + [f"dd_c{i}" for i in range(1, 13)] + ["dd_logE"])
