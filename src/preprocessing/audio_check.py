"""
Stage 2 - Audio integrity check.

PURPOSE
-------
Before any feature extraction, open EVERY file and verify it is usable. A single
corrupt or empty file can crash a 20-minute HuBERT extraction run, or worse, yield
NaN features that silently poison training.

CHECKS PER FILE (all recorded as columns, nothing is silently dropped)
---------------------------------------------------------------------
  readable      : soundfile can decode the full file
  sample_rate   : native rate (RAVDESS 48 kHz and CREMA-D 16 kHz are expected; we resample later)
  channels      : 1 = mono; stereo would need down-mixing
  duration_s    : n_samples / sample_rate
  finite        : no NaN/Inf samples
  peak          : max |x|; ~0 means a silent/empty recording
  clip_frac     : fraction of samples with |x| >= 0.999 (amplitude clipping = distortion)
  status        : "ok" or a short reason ("unreadable", "empty", "silent", "too_short", "non_finite")

Input : index DataFrame from dataset_index.build_index() + raw-data root.
Output: same DataFrame with the columns above appended.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

MIN_DURATION_S = 0.5      # shorter than this cannot carry a sentence-level emotion
SILENCE_PEAK = 1e-4       # peak amplitude below this = effectively silent


def check_file(path: Path) -> dict:
    """Decode one file and compute integrity statistics (see module docstring)."""
    out = dict(readable=False, sample_rate=np.nan, channels=np.nan, duration_s=np.nan,
               finite=False, peak=np.nan, clip_frac=np.nan, status="unreadable", error="")
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)   # x: (n_samples, channels)
    except Exception as e:
        out["error"] = repr(e)[:200]
        return out
    n, ch = x.shape
    out.update(readable=True, sample_rate=sr, channels=ch, duration_s=n / sr)
    if n == 0:
        out["status"] = "empty"
        return out
    finite = bool(np.isfinite(x).all())
    peak = float(np.abs(x).max()) if finite else np.nan
    out.update(finite=finite, peak=peak,
               clip_frac=float((np.abs(x) >= 0.999).mean()) if finite else np.nan)
    if not finite:
        out["status"] = "non_finite"
    elif peak < SILENCE_PEAK:
        out["status"] = "silent"
    elif n / sr < MIN_DURATION_S:
        out["status"] = "too_short"
    else:
        out["status"] = "ok"
    return out


def check_index(df: pd.DataFrame, raw_root: Path, workers: int = 8) -> pd.DataFrame:
    """Run check_file on every row (threaded: decoding is I/O-bound)."""
    paths = [raw_root / p for p in df["rel_path"]]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(check_file, paths))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
