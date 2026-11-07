"""
Stage 4 - Prosodic features.

WHY PROSODY FOR EMOTION?
  Prosody = the melody (F0 / pitch), loudness (energy) and timing (voicing, pauses,
  rhythm) of speech. Arousal is strongly reflected in prosody: anger, fear and happiness
  typically raise F0 level and variability and energy; sadness lowers them and slows the
  rhythm. MFCCs describe the spectral envelope, not F0, so prosody is complementary.

TWO REPRESENTATIONS (kept strictly separate)
--------------------------------------------
F0 TRACKER: Praat autocorrelation ('To Pitch (ac)', Praat defaults, 60-800 Hz) via
praat-parselmouth. Until 2026-11-06 this module used librosa pYIN; pYIN lost voicing in
10.3 % of CREMA-D recordings (class-dependent: disgust 20.5 %, sad 17.8 %), so the team
approved the switch. f0_pyin() is kept only for the documented tracker comparison.

(A) FRAME-LEVEL contours on the shared HuBERT grid (our adaptation; the paper has no
    frame-level prosody). 4 values per 20 ms frame t (centre 0.02 t + 0.0125 s):
        0  f0_st        log-F0 in semitones re 100 Hz; unvoiced frames linearly interpolated
        1  voiced       1 if Praat's Viterbi path selects a voiced candidate, else 0
        2  periodicity  max normalised-autocorrelation strength of the frame's voiced
                        candidates, in [0, 1] (0 if none). NOT the pYIN voicing probability
                        used before: pYIN's value is a posterior probability from an HMM over
                        YIN thresholds; this is a local periodicity measure, available for
                        every frame and not forced to agree with the voiced flag.
        3  energy_db    10 log10(mean(x^2)) of the same 400-sample frame used by MFCC/HuBERT
    Alignment: Praat centres its frames symmetrically in the signal; f0_praat() zero-pads
    the signal so that Praat frame centres fall exactly on 200 + 320 t (asserted on every
    call). No interpolation or resampling of time is needed.

(B) UTTERANCE-LEVEL "DisVoice-103" static vector (what the reference paper used).
    A clean re-implementation of DisVoice prosody_static() (commit 7bb41a4), including its
    segmentation rules and quirks, verified against the original functions in
    tests/reference/ (see tests/test_prosody.py). Feature order and names follow the
    original code (its docstring order is inaccurate: tilt/MSE stats are interleaved).
      30  F0: contour stats (6), per-voiced-segment linear tilt & MSE stats (12, interleaved),
              first voiced segment (6), last voiced segment (6)
      24  energy of VOICED segments: mean of per-segment mean/std/skew/kurt (4),
              tilt stats (4), MSE stats (4), first segment (6), last segment (6)
      24  same for UNVOICED segments
      25  timing: voiced segments per second (1), voiced / unvoiced / pause duration stats
              (6 each), 6 ratios of MEAN durations (PVU, PU, UVU, VVU, VP, UP)
    Unavoidable / deliberate differences from DisVoice (documented):
      * F0 tracker: Praat autocorrelation (DisVoice's own alternative to RAPT; its script calls
        the same 'To Pitch...' with defaults) instead of RAPT (pysptk has no Windows/Python-3.12
        wheel). Praat values are taken exactly on the 10 ms grid, not snapped by nearest time.
      * log-energy of an all-zero frame: floor at -100 dB instead of DisVoice's -1e30
        (which would produce astronomically large statistics).
      * undefined statistics (e.g. skewness of a constant, mean of an empty list) are NaN in
        DisVoice; we keep them NaN here and impute 0 in the cache, recording an imputation mask.
      * computed twice: F0 range 60-350 Hz (DisVoice default, faithful - but above its ceiling
        Praat locks onto sub-harmonics, e.g. 600 Hz -> 300 Hz) and 60-800 Hz (wide, primary).
"""
from __future__ import annotations

import warnings

import librosa
import numpy as np
import scipy.stats as st

from src.features.config import HOP, SAMPLE_RATE, WIN, PraatPitchConfig, ProsodyConfig
from src.features.mfcc import frame_signal

PROSODY_FRAME_NAMES = ["f0_st", "voiced", "periodicity", "energy_db"]
ENERGY_FLOOR = 1e-10                      # -> -100 dB
PYIN_FRAME_LENGTH = 1024                  # previous tracker's window; comparison scripts only


# ============================================================================ F0 tracking
def _praat_padding(n_samples: int, n: int, hop: int, first_centre: int, window: int) -> tuple[int, int, int]:
    """
    Zero padding (left L, right R) that makes Praat's analysis frames fall EXACTLY on our grid.

    Praat (Sampled_shortTermAnalysis) places n_p = floor((M - W) / h) + 1 frames of window W
    and step h symmetrically around the middle of a signal of M samples; the first frame
    centre is (M - (n_p - 1) h) / 2 samples after the start. If M - W = (n_p - 1) h + h/2
    (the half step makes the floor() immune to floating-point rounding), the first centre is
    W/2 + h/4. With L = W/2 + h/4 - first_centre (+ k h until L >= 0), Praat frame m + t is
    centred at sample first_centre + h t of the unpadded signal.
    Returns (L, R, m).
    """
    base = window // 2 + hop // 4 - first_centre
    L, m = base, 0
    while L < 0:
        L, m = L + hop, m + 1
    M = max(window + hop // 2 + (m + n - 1) * hop, L + n_samples)
    M += (-(M - window - hop // 2)) % hop
    return L, M - L - n_samples, m


def f0_praat(y: np.ndarray, n: int, hop: int, first_centre: int, fmin: float, fmax: float,
             pc: "PraatPitchConfig", sr: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """
    Praat autocorrelation pitch tracker (Boersma 1993; Praat 'Sound: To Pitch (ac)'), sampled
    EXACTLY at frame centres first_centre + hop * t (t = 0..n-1, in samples of y).

    Algorithm (what the AI-course viewer should know): per frame, the windowed signal's
    autocorrelation is divided by the window's own autocorrelation (this removes the window's
    taper bias); local maxima r(tau) in [1/fmax, 1/fmin] become voiced candidates with
    'strength' r in [0, 1] (1 = perfectly periodic); one unvoiced candidate per frame scores
    by the voicing / silence thresholds. A Viterbi search over frames then picks one candidate
    per frame, penalising octave jumps (octave_jump_cost) and voiced/unvoiced switches
    (voiced_unvoiced_cost). Same family of dynamic programming as pYIN's HMM, but the
    observation score is a normalised autocorrelation rather than a YIN-threshold probability.

    Returns
      f0          (n,) Hz of the selected path, 0 = unvoiced
      periodicity (n,) max strength r over the frame's VOICED candidates (0 if none), clipped
                  to [0, 1]. NOT a probability: it is the local degree of periodicity,
                  defined whether or not the path search finally called the frame voiced.
    """
    import parselmouth
    periods = 6 if pc.very_accurate else 3          # Praat: window = periods / fmin
    window_f = periods * sr / fmin
    window = int(round(window_f))
    if abs(window_f - window) > 1e-9 or window % 2 or hop % 4:
        raise ValueError(f"exact grid alignment needs an even integer window ({window_f}) and hop % 4 == 0")
    L, R, m = _praat_padding(len(y), n, hop, first_centre, window)
    z = np.concatenate([np.zeros(L), np.asarray(y, np.float64), np.zeros(R)])
    snd = parselmouth.Sound(z, sampling_frequency=sr)
    p = snd.to_pitch_ac(time_step=hop / sr, pitch_floor=fmin, max_number_of_candidates=pc.max_candidates,
                        very_accurate=pc.very_accurate, silence_threshold=pc.silence_threshold,
                        voicing_threshold=pc.voicing_threshold, octave_cost=pc.octave_cost,
                        octave_jump_cost=pc.octave_jump_cost, voiced_unvoiced_cost=pc.voiced_unvoiced_cost,
                        pitch_ceiling=fmax)
    expected_t0 = (L + first_centre) / sr
    if p.nx < m + n or abs(p.x1 + m * p.dx - expected_t0) > 1e-9 or abs(p.dx - hop / sr) > 1e-12:
        raise RuntimeError(f"Praat frames not on grid: x1={p.x1}, dx={p.dx}, nx={p.nx}, "
                           f"expected t0={expected_t0}, m={m}, n={n}")
    sel = p.selected_array[m:m + n]
    f0 = np.nan_to_num(sel["frequency"]).astype(np.float64)
    cand = p.to_array()[:, m:m + n]                  # (max_candidates, n), row 0 = selected
    freq, strength = np.nan_to_num(cand["frequency"]), np.nan_to_num(cand["strength"])
    periodicity = np.where(freq > 0, strength, 0.0).max(axis=0)
    return f0, np.clip(periodicity, 0.0, 1.0)


def f0_pyin(y: np.ndarray, fmin: float, fmax: float, hop: int, frame_length: int,
            start: int = 0, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """
    Probabilistic YIN F0 tracker (PREVIOUS tracker; no longer used for cached features).
    YIN finds the period tau that minimises the difference between the signal and its
    tau-shifted copy; pYIN turns several threshold choices into candidate probabilities
    and picks a smooth path with an HMM (Viterbi), which reduces octave jumps.

    Returns f0 (Hz, 0 = unvoiced) and voicing probability, frame j centred at start + hop*j.
    """
    f0, voiced, vprob = librosa.pyin(y[start:], fmin=fmin, fmax=fmax, sr=sr,
                                     frame_length=frame_length, hop_length=hop, center=True)
    f0 = np.where(voiced & np.isfinite(f0), f0, 0.0)
    return f0.astype(np.float64), np.nan_to_num(vprob).astype(np.float64)


def interpolate_unvoiced(values: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Linear interpolation across unvoiced gaps; edges hold the nearest voiced value."""
    if not voiced.any():
        return np.zeros_like(values)
    idx = np.flatnonzero(voiced)
    return np.interp(np.arange(len(values)), idx, values[idx])


def track_f0(y: np.ndarray, cfg: ProsodyConfig, n: int, hop: int, first_centre: int,
             fmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the configured tracker. Returns (f0 Hz with 0 = unvoiced, per-frame periodicity)."""
    if cfg.f0_tracker != "praat_ac":
        raise ValueError(f"unsupported f0_tracker {cfg.f0_tracker!r}")
    return f0_praat(y, n, hop, first_centre, cfg.f0_min, fmax, cfg.praat)


def prosody_frames(y: np.ndarray, cfg: ProsodyConfig, n: int) -> np.ndarray:
    """(A) Frame-level prosody, shape (n, 4), aligned to the shared grid (see module doc)."""
    f0, periodicity = track_f0(y, cfg, n, hop=HOP, first_centre=WIN // 2, fmax=cfg.f0_max_wide)
    voiced = f0 > 0
    st_ = np.where(voiced, 12.0 * np.log2(np.maximum(f0, 1e-6) / cfg.f0_ref_hz), 0.0)
    frames = frame_signal(y, WIN, HOP)[:n].astype(np.float64)
    energy_db = 10.0 * np.log10(np.maximum((frames ** 2).mean(1), ENERGY_FLOOR))
    return np.stack([interpolate_unvoiced(st_, voiced), voiced.astype(np.float64),
                     periodicity, energy_db], axis=1).astype(np.float32)


# ============================================================================ DisVoice-103
def _stats6(x) -> list[float]:
    """mean, std, max, min, skewness, kurtosis (scipy defaults, as in DisVoice)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return [np.nan] * 6
    return [x.mean(), x.std(), x.max(), x.min(), _skew(x), _kurt(x)]


def _skew(x):
    # scipy warns ("catastrophic cancellation") for near-constant inputs, e.g. 2-element lists.
    # DisVoice gets the same values; we silence the warning and check value ranges in validation.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(st.skew(x)) if len(x) else np.nan


def _kurt(x):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(st.kurtosis(x)) if len(x) else np.nan


def _mean(x):
    return float(np.mean(x)) if len(x) else np.nan


def _std(x):
    return float(np.std(x)) if len(x) else np.nan


def _linear_fit(t: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Least-squares line x ~ a t + b; returns (slope a = 'tilt', mean squared residual)."""
    a, b = np.polyfit(t, x, 1)
    return float(a), float(np.mean((x - (a * t + b)) ** 2))


def f0_features(F0: np.ndarray, step_s: float) -> list[float]:
    """30 F0 features (DisVoice F0feat). F0 in Hz on a 10 ms grid, 0 = unvoiced."""
    nz = F0[F0 != 0]
    if len(nz) == 0:
        return [0.0] * 30
    # Voiced runs: DisVoice splits after every zero, keeps chunks longer than 1 and drops
    # each chunk's last element (the zero delimiter; for the final chunk a real value).
    chunks = np.split(F0, np.flatnonzero(F0 == 0) + 1)
    runs = [c[:-1] for c in chunks if len(c) > 1]
    tilt, mse = [], []
    for r in runs:
        if len(r) > 1:
            a, e = _linear_fit(np.arange(len(r)) * step_s, r)
            if np.isfinite(a):
                tilt.append(a)
                mse.append(e)
    if not tilt:
        tilt, mse = [0.0], [0.0]
    tilt, mse = np.asarray(tilt), np.asarray(mse)
    interleaved = [tilt.mean(), mse.mean(), tilt.std(), mse.std(), tilt.max(), mse.max(),
                   tilt.min(), mse.min(), _skew(tilt), _skew(mse), _kurt(tilt), _kurt(mse)]
    first = _stats6(runs[0]) if runs else [0.0] * 6     # DisVoice would crash here (no runs)
    last = _stats6(runs[-1]) if runs else [0.0] * 6
    return _stats6(nz) + interleaved + first + last


def segments(F0: np.ndarray, y: np.ndarray, voiced: bool, step: int) -> list[np.ndarray]:
    """
    DisVoice V_UV: audio segments for runs of voiced (F0 != 0) or unvoiced (F0 == 0) frames.
    Run boundaries in samples are frame_index * step + step. As in DisVoice, the FINAL run
    is never appended (the loop only closes runs that are followed by a change).
    """
    idx = np.flatnonzero(F0 != 0) if voiced else np.flatnonzero(F0 == 0)
    if len(idx) == 0:
        return []
    change = np.flatnonzero(np.diff(idx) > 1)
    out, start = [], idx[0] * step + step
    for c in change:
        out.append(y[int(start): int(idx[c] * step + step)])
        start = idx[c + 1] * step + step
    return out


def log_energy_contour(seg: np.ndarray, frame: int, step: int) -> np.ndarray:
    """DisVoice get_energy_segment: 10 log10(mean x^2) on frames of `frame` samples every `step`."""
    n = int(len(seg) / step) - 1                     # = int(len/frame/overlap) - 1 in DisVoice
    out = []
    for l in range(max(n, 0)):
        f = seg[int(l * step): int(l * step + frame)]
        out.append(10.0 * np.log10(max(np.mean(f ** 2), ENERGY_FLOOR)) if len(f) else -100.0)
    return np.asarray(out, dtype=np.float64)


def energy_features(segs: list[np.ndarray], frame: int, step: int, sr: int) -> list[float]:
    """24 energy features of a list of segments (DisVoice energy_feat)."""
    if len(segs) < 2:
        return [0.0] * 24
    first, last = 0, len(segs) - 1
    e_mean, e_std, e_skw, e_ku, tilt, mse = [], [], [], [], [], []
    for j, s in enumerate(segs):
        c = log_energy_contour(s, frame, step)
        if j == first and len(c) == 0:
            first += 1
        if j == last and len(c) == 0:
            last -= 1
        if len(c) <= 1:
            continue
        e_mean.append(c.mean()); e_std.append(c.std()); e_skw.append(_skew(c)); e_ku.append(_kurt(c))
        a, e = _linear_fit(np.arange(len(c)) * step / sr, c)
        if np.isfinite(a):
            tilt.append(a); mse.append(e)
    feats = [_mean(e_mean), _mean(e_std), _mean(e_skw), _mean(e_ku),
             _mean(tilt), _std(tilt), _skew(np.asarray(tilt)), _kurt(np.asarray(tilt)),
             _mean(mse), _std(mse), _skew(np.asarray(mse)), _kurt(np.asarray(mse))]
    feats += [0.0] * 6 if first == len(segs) else _stats6(log_energy_contour(segs[first], frame, step))
    if last == 0:
        feats += [0.0] * 6
    else:
        c = log_energy_contour(segs[last], frame, step)
        feats += _stats6(c) if len(c) else [0.0] * 6
    return feats


def _dur_stats(segs: list[np.ndarray], sr: int) -> list[float]:
    d = np.asarray([len(s) / sr for s in segs])
    if len(d) > 1:
        return [d.mean(), d.std(), _skew(d), _kurt(d), d.max(), d.min()]
    return [0.0] * 6


def duration_features(seg_v, seg_u, seg_p, n_samples: int, sr: int) -> list[float]:
    """25 timing features (DisVoice duration_feat). Ratios use MEAN segment durations."""
    v_rate = sr * len(seg_v) / n_samples
    dv, du, dp = _dur_stats(seg_v, sr), _dur_stats(seg_u, sr), _dur_stats(seg_p, sr)
    V, U, P = dv[0], du[0], dp[0]
    ratios = [P / (V + U), P / U, U / (V + U), V / (V + U), V / P, U / P] if (P and U and V) else [0.0] * 6
    return [v_rate] + dv + du + dp + ratios


def disvoice_static(y: np.ndarray, F0: np.ndarray, cfg: ProsodyConfig, sr: int = SAMPLE_RATE) -> np.ndarray:
    """(B) 103-dim DisVoice-style vector from audio y (DC-free, peak-normalised) and 10 ms F0."""
    step = cfg.disvoice_step_s * sr
    frame = cfg.disvoice_frame_s * sr
    seg_v = segments(F0, y, voiced=True, step=step)
    seg_up = segments(F0, y, voiced=False, step=step)
    seg_p = [s for s in seg_up if len(s) > cfg.disvoice_pause_s * sr]
    seg_u = [s for s in seg_up if len(s) <= cfg.disvoice_pause_s * sr]
    vec = (f0_features(F0, cfg.disvoice_step_s) + energy_features(seg_v, frame, step, sr)
           + energy_features(seg_u, frame, step, sr) + duration_features(seg_v, seg_u, seg_p, len(y), sr))
    return np.asarray(vec, dtype=np.float64)


def prosody103(y: np.ndarray, cfg: ProsodyConfig, f0_max: float) -> np.ndarray:
    """
    Track F0 on the DisVoice 10 ms grid (frame j centred at sample 160 j, 1 + N // 160 frames,
    as before) with the given ceiling, then compute the 103 stats.
    DisVoice's own Praat path snaps Praat's time stamps to a np.linspace grid by nearest
    neighbour (decodeF0); we sample Praat exactly on the grid instead.
    """
    step = int(cfg.disvoice_step_s * SAMPLE_RATE)
    F0, _ = track_f0(y, cfg, 1 + len(y) // step, hop=step, first_centre=0, fmax=f0_max)
    return disvoice_static(y, F0, cfg)


DISVOICE_NAMES = (
    ["F0avg", "F0std", "F0max", "F0min", "F0skew", "F0kurt", "F0tiltavg", "F0mseavg",
     "F0tiltstd", "F0msestd", "F0tiltmax", "F0msemax", "F0tiltmin", "F0msemin", "F0tiltskw",
     "F0mseskw", "F0tiltku", "F0mseku", "1F0mean", "1F0std", "1F0max", "1F0min", "1F0skw",
     "1F0ku", "lastF0avg", "lastF0std", "lastF0max", "lastF0min", "lastF0skw", "lastF0ku"]
    + [f"{s}Evoiced" for s in ["avg", "std", "skw", "kurtosis", "avgtilt", "stdtilt", "skwtilt",
                                "kurtosistilt", "avgmse", "stdmse", "skwmse", "kurtosismse", "avg1",
                                "std1", "max1", "min1", "skw1", "kurtosis1", "avglast", "stdlast",
                                "maxlast", "minlast", "skwlast", "kurtosislast"]]
    + [f"{s}Eunvoiced" for s in ["avg", "std", "skw", "kurtosis", "avgtilt", "stdtilt", "skwtilt",
                                  "kurtosistilt", "avgmse", "stdmse", "skwmse", "kurtosismse", "avg1",
                                  "std1", "max1", "min1", "skw1", "kurtosis1", "avglast", "stdlast",
                                  "maxlast", "minlast", "skwlast", "kurtosislast"]]
    + ["Vrate", "avgdurvoiced", "stddurvoiced", "skwdurvoiced", "kurtosisdurvoiced",
       "maxdurvoiced", "mindurvoiced", "avgdurunvoiced", "stddurunvoiced", "skwdurunvoiced",
       "kurtosisdurunvoiced", "maxdurunvoiced", "mindurunvoiced", "avgdurpause", "stddurpause",
       "skwdurpause", "kurtosisdurpause", "maxdurpause", "mindurpause",
       "PVU", "PU", "UVU", "VVU", "VP", "UP"])
assert len(DISVOICE_NAMES) == 103
