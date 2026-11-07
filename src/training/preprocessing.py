"""
Stage 5 - learned preprocessing, ALWAYS fitted on the training partition of a fold only.

Rule (enforced by the API): every transformer has fit(train data) and transform(any data).
transform() never updates parameters, so validation and test data are mapped with the
statistics of the training speakers. A transformer that has not been fitted refuses to
transform (NotFittedError-like RuntimeError).

1. FrameStandardizer  - frame-level streams (MFCC, prosody contours, HuBERT): per-dimension
   mean/std over all VALID frames of all training recordings (padding never counted).
   Selected dimensions can be passed through unchanged (e.g. the binary 'voiced' flag and
   'periodicity', which is already in [0, 1]).

2. DisVoicePreprocessor - the 103 utterance-level DisVoice statistics (audited in Stage 5,
   results/stage5_audit_stage4.json):
     a) values imputed in Stage 4 (cache stores 0 + a boolean mask) are replaced by the
        TRAINING median of that feature (0 is an arbitrary, possibly extreme, value);
     b) features whose training-set |skewness| > skew_threshold are mapped with the signed
        log  g(x) = sign(x) * log(1 + |x|). A plain log is impossible: 55/103 features take
        negative values (tilts, skewness). g is monotonic, odd and ~linear near 0, so it only
        compresses the heavy tails (e.g. F0msemax, robust z up to 11,387 on CREMA-D);
     c) per-feature standardisation with training mean/std, then clipping to +-clip.
        Clipping bounds the influence of the remaining outliers on a neural network
        (decided a priori; not tuned on test data);
     d) OPTIONAL missingness indicators (103 binary columns): a separate experimental choice,
        because the audit showed imputation is class-dependent (CREMA-D: angry 35 % vs
        sad 62 % of recordings) - indicators may act as a shortcut and must be ablated.
"""
from __future__ import annotations

import numpy as np
import scipy.stats as st


class _Fitted:
    _fitted = False

    def _require_fit(self):
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted on training data first")


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class FrameStandardizer(_Fitted):
    def __init__(self, passthrough: tuple[int, ...] = (), eps: float = 1e-6):
        self.passthrough = tuple(passthrough)
        self.eps = eps

    def fit(self, sequences: list[np.ndarray]) -> "FrameStandardizer":
        """sequences: list of (T_i, D) training arrays (unpadded, so every frame is valid)."""
        d = sequences[0].shape[-1]
        s, s2, n = np.zeros(d), np.zeros(d), 0
        for x in sequences:                                   # streaming, works for memmaps
            x = np.asarray(x, dtype=np.float64).reshape(-1, d)
            s += x.sum(0); s2 += (x ** 2).sum(0); n += len(x)
        self.mean_ = s / n
        self.std_ = np.sqrt(np.maximum(s2 / n - self.mean_ ** 2, 0.0)) + self.eps
        for j in self.passthrough:
            self.mean_[j], self.std_[j] = 0.0, 1.0
        self.n_frames_ = n
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._require_fit()
        return ((np.asarray(x, dtype=np.float32) - self.mean_.astype(np.float32))
                / self.std_.astype(np.float32))


class DisVoicePreprocessor(_Fitted):
    def __init__(self, skew_threshold: float = 2.0, clip: float = 5.0, add_indicators: bool = False):
        self.skew_threshold = skew_threshold
        self.clip = clip
        self.add_indicators = add_indicators

    def fit(self, X: np.ndarray, imputed: np.ndarray) -> "DisVoicePreprocessor":
        X = np.asarray(X, dtype=np.float64)
        imputed = np.asarray(imputed, dtype=bool)
        Xo = np.where(imputed, np.nan, X)
        med = np.nanmedian(Xo, axis=0)
        self.fill_ = np.where(np.isfinite(med), med, 0.0)       # all-imputed column -> 0
        Xf = np.where(imputed, self.fill_, X)
        skew = np.array([st.skew(c) if c.std() > 0 else 0.0 for c in Xf.T])
        self.log_cols_ = np.flatnonzero(np.abs(skew) > self.skew_threshold)
        Xt = self._log(Xf)
        self.mean_ = Xt.mean(0)
        self.std_ = Xt.std(0)
        self.std_[self.std_ == 0] = 1.0
        self.train_skew_ = skew
        self._fitted = True
        return self

    def _log(self, X):
        X = X.copy()
        X[:, self.log_cols_] = signed_log(X[:, self.log_cols_])
        return X

    def transform(self, X: np.ndarray, imputed: np.ndarray) -> np.ndarray:
        self._require_fit()
        X = np.asarray(X, dtype=np.float64)
        imputed = np.asarray(imputed, dtype=bool)
        Z = np.clip((self._log(np.where(imputed, self.fill_, X)) - self.mean_) / self.std_, -self.clip, self.clip)
        if self.add_indicators:
            Z = np.concatenate([Z, imputed.astype(np.float64)], axis=1)
        return Z.astype(np.float32)

    @property
    def n_out(self) -> int:
        return 103 * (2 if self.add_indicators else 1)
