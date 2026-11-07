"""
Stage 4 - Single source of truth for every feature-extraction parameter.

Why a frozen dataclass + hash?
  * Every number that affects a cached feature lives here (nothing hard-coded elsewhere).
  * The SHA-256 of this config names the cache directory. Changing ANY parameter
    produces a new directory, so features computed with different settings can never
    be mixed up silently - this is the core of the cache-validation mechanism.

THE COMMON TIME GRID (the most important design decision of Stage 4)
-------------------------------------------------------------------
HuBERT's CNN front-end has a receptive field of 400 samples and a stride of 320 samples
at 16 kHz (verified in Stage 1: 349 frames for 7 s). Frame t therefore covers samples
[320 t, 320 t + 400), i.e. 25 ms windows every 20 ms, centred at
        t_center = (320 t + 200) / 16000 s = 0.020 t + 0.0125 s.
MFCC and frame-level prosody are computed on EXACTLY these windows / centres, so all three
streams share one frame index t without any interpolation:
        n_frames(N) = 1 + floor((N - 400) / 320)      for N >= 400 samples.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

SAMPLE_RATE = 16_000
WIN = 400          # 25 ms  - HuBERT CNN receptive field
HOP = 320          # 20 ms  - HuBERT CNN total stride


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    resample_type: str = "soxr_hq"        # high-quality band-limited resampler (RAVDESS 48k -> 16k)
    mono: str = "mean"                    # stereo files (5 in RAVDESS) are averaged to mono
    trim_top_db: float = 30.0             # leading/trailing silence = > 30 dB below the clip peak
    trim_frame: int = 512
    trim_hop: int = 128
    trim_margin_s: float = 0.05           # keep 50 ms around detected speech (soft onsets)
    normalise: str = "dc_remove+peak"     # x <- (x - mean) / max|x - mean|  (same as DisVoice)


@dataclass(frozen=True)
class FrameConfig:
    win_length: int = WIN
    hop_length: int = HOP


@dataclass(frozen=True)
class MFCCConfig:
    n_fft: int = 512                      # 400-sample frame zero-padded to 512 -> 31.25 Hz bins
    window: str = "hamming"               # periodic Hamming (scipy get_window(..., fftbins=True))
    n_mels: int = 40
    fmin: float = 0.0
    fmax: float = 8000.0                  # Nyquist at 16 kHz
    mel_scale: str = "htk"                # mel(f) = 2595 log10(1 + f/700)
    n_ceps: int = 12                      # c1..c12 (c0 dropped; frame energy used instead)
    log_floor: float = 1e-10
    delta_width: int = 2                  # regression window +-2 frames (HTK default)
    # 12 MFCC + 1 log-energy = 13 static; + delta + delta-delta = 39 per frame


@dataclass(frozen=True)
class PraatPitchConfig:
    """
    Praat 'To Pitch (ac)' parameters. All are Praat defaults (= what DisVoice's plain
    'To Pitch...' call uses) EXCEPT octave_jump_cost, chosen in scripts/stage4_praat_settings.py.
    """
    max_candidates: int = 15
    very_accurate: bool = False           # False: 3-period Hanning window; True: 6-period Gaussian
    silence_threshold: float = 0.03       # frames with peak < 3 % of global peak -> silent
    voicing_threshold: float = 0.45       # min. normalised autocorrelation for a voiced candidate
    octave_cost: float = 0.01             # per-octave favouring of higher candidates
    octave_jump_cost: float = 0.7         # Praat default 0.35. Viterbi cost per octave of F0 change
                                          # between frames. With 0.35, 13 % (RAVDESS) / 26 % (CREMA-D)
                                          # of sampled recordings had >= 1 implausible > 7 st jump in
                                          # 10 ms; 0.7 removes them, lowers disagreement with pYIN and
                                          # still tracks synthetic octave glides in 50-100 ms exactly.
    voiced_unvoiced_cost: float = 0.14    # Viterbi penalty for voiced <-> unvoiced transitions


@dataclass(frozen=True)
class ProsodyConfig:
    # --- F0 tracker (approved switch pYIN -> Praat, 2026-11-06; see reports/feature_extraction.md 5.4) ---
    f0_tracker: str = "praat_ac"          # Praat 'To Pitch (ac)' via praat-parselmouth (Praat 6.1.38)
    praat: PraatPitchConfig = field(default_factory=PraatPitchConfig)
    # --- frame-level contours (our adaptation, aligned to the HuBERT grid) ---
    f0_min: float = 60.0                  # = DisVoice minf0; 75 Hz measurably cut low male F0 (CREMA-D)
    f0_max_wide: float = 800.0            # chosen in scripts/stage4_praat_settings.py (700 Hz still
                                          # clipped 0.8 % of RAVDESS voiced frames; above it Praat
                                          # reports sub-harmonics, not 'unvoiced')
    f0_ref_hz: float = 100.0              # F0 stored in semitones re 100 Hz
    # --- DisVoice-style 103 static features (paper: Zhao et al. use DisVoice) ---
    disvoice_step_s: float = 0.01         # DisVoice: 10 ms F0/energy step
    disvoice_frame_s: float = 0.02        # DisVoice: 20 ms energy frames
    disvoice_f0_max: float = 350.0        # DisVoice default maxf0 (kept ONLY for the faithful variant)
    disvoice_pause_s: float = 0.14        # unvoiced run longer than 140 ms = pause


@dataclass(frozen=True)
class HubertConfig:
    model_id: str = "facebook/hubert-base-ls960"
    revision: str = "dba3bb02fda4248b6e082697eee756de8fe8aa8a"   # 'main' commit, pinned 2026-11-05
    input_normalise: bool = True          # preprocessor_config.json: do_normalize = true
    inference_dtype: str = "float32"      # decided by the fp16-vs-fp32 measurement (see LOG)
    storage_dtype: str = "float16"        # frame-level cache precision (pooled stats kept in fp32)
    n_hidden_states: int = 13             # 0 = Transformer input, 1..12 = layer outputs


@dataclass(frozen=True)
class FeatureConfig:
    version: int = 1
    audio: AudioConfig = field(default_factory=AudioConfig)
    frames: FrameConfig = field(default_factory=FrameConfig)
    mfcc: MFCCConfig = field(default_factory=MFCCConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    hubert: HubertConfig = field(default_factory=HubertConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def cache_id(self) -> str:
        return f"v{self.version}_{self.hash()[:10]}"


DEFAULT = FeatureConfig()


def n_frames(n_samples: int, win: int = WIN, hop: int = HOP) -> int:
    """Number of HuBERT-grid frames for a signal of n_samples (0 if shorter than one window)."""
    return 0 if n_samples < win else 1 + (n_samples - win) // hop


def frame_centers_s(n: int, win: int = WIN, hop: int = HOP, sr: int = SAMPLE_RATE):
    """Centre time (seconds) of each frame on the shared grid: (hop*t + win/2) / sr."""
    import numpy as np
    return (hop * np.arange(n) + win / 2) / sr
