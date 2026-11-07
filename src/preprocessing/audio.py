"""
Stage 4 - Audio preprocessing shared by ALL features (HuBERT, MFCC, prosody).

PIPELINE (original files are only read, never modified)
-------------------------------------------------------
  1. decode            soundfile -> float32, shape (n_samples, channels)
  2. mono              average channels (5 RAVDESS files are stereo, Stage 2 finding)
  3. resample          native rate -> 16 kHz (HuBERT was pre-trained on 16 kHz LibriSpeech)
                       band-limited "soxr_hq" resampler avoids aliasing from 48 kHz
  4. trim              remove leading/trailing silence (> top_db below the clip's peak),
                       keeping a small margin; INTERNAL pauses are kept because pause
                       durations are prosodic features
  5. normalise         remove DC offset, divide by peak amplitude

Why trim? RAVDESS clips contain ~0.5-1 s of silence at each end. Silence carries no
emotion, inflates padding, and would be counted as "pauses" by the prosody features.
Why peak-normalise? Recording gain differs between actors, sessions and corpora, so
absolute loudness is not a reliable emotion cue across speakers; relative energy
dynamics inside the utterance are preserved. HuBERT additionally applies its own
zero-mean / unit-variance normalisation (see src/features/hubert.py).

Input : path to a WAV file + AudioConfig
Output: (y, info) where y is float32 mono 16 kHz with |y|max = 1, and info records the
        trim boundaries (in 16 kHz samples) so every step is traceable.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.features.config import AudioConfig


def load_mono(path: Path) -> tuple[np.ndarray, int, int]:
    """Decode and down-mix. Returns (y, native_sr, n_channels)."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)     # (n, ch)
    return x.mean(axis=1), sr, x.shape[1]


def trim_silence(y: np.ndarray, cfg: AudioConfig) -> tuple[int, int]:
    """
    Return [start, end) sample indices of the speech region.

    librosa.effects.trim computes frame RMS in dB relative to the loudest frame and
    finds the first/last frame within `top_db` of it. We add a margin on both sides.
    """
    _, (start, end) = librosa.effects.trim(y, top_db=cfg.trim_top_db,
                                           frame_length=cfg.trim_frame, hop_length=cfg.trim_hop)
    margin = int(cfg.trim_margin_s * cfg.sample_rate)
    return max(0, int(start) - margin), min(len(y), int(end) + margin)


def preprocess(path: Path, cfg: AudioConfig) -> tuple[np.ndarray, dict]:
    y, sr, ch = load_mono(path)
    n_raw = len(y)
    if sr != cfg.sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sample_rate, res_type=cfg.resample_type)
    n_16k = len(y)
    start, end = trim_silence(y, cfg)
    y = y[start:end]
    y = y - y.mean()
    peak = float(np.abs(y).max())
    if not np.isfinite(peak) or peak == 0.0:
        raise ValueError(f"{path}: silent or invalid audio after trimming")
    y = (y / peak).astype(np.float32)
    info = dict(sr_native=sr, channels=ch, n_samples_native=n_raw,
                n_samples_16k_untrimmed=n_16k,
                trim_start=start, trim_end=end, n_samples=len(y))
    return y, info
