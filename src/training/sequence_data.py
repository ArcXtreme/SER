"""
E3/E4 - frame-level inputs for one dataset, one HuBERT layer and one fold.

MEMORY (RTX 3050, 4 GB): the cache stores all 13 hidden states, (T_total, 13, 768) fp16 =
17.6 GB for CREMA-D - too large for RAM or VRAM. Only ONE layer is loaded, (T_total, 768) fp16
(1.35 GB CREMA-D, 0.16 GB RAVDESS), read from the memory-mapped cache in chunks. MFCC (39) and
prosody (4) frames are small. The arrays are moved to `dev` once; every mini-batch is then
gathered on the device (no per-batch host->device copies).

ALIGNMENT: HuBERT, MFCC and prosody share ONE frame-offset table (Stage 4), so the same index
matrix idx[b, t] = offset[row_b] + t selects frame t of recording row_b in all three streams.
Padded positions (t >= n_frames) are filled with 0 AFTER standardisation and flagged in `mask`.
No recording is cropped (longest = 250 frames = 5 s).

STANDARDISATION: FrameStandardizer (src/training/preprocessing.py) fitted per stream on the
frames of the fold's TRAINING recordings only; prosody dims 'voiced' (binary) and 'periodicity'
([0, 1]) pass through unchanged. DisVoice-103 (ablation only): DisVoicePreprocessor fitted on
training recordings only (median fill, signed log, z-score, clip; indicators optional).
"""
from __future__ import annotations

import numpy as np
import torch

from src.features.prosody import PROSODY_FRAME_NAMES
from src.features.store import FeatureStore
from src.training.preprocessing import DisVoicePreprocessor, FrameStandardizer

PROSODY_PASSTHROUGH = (PROSODY_FRAME_NAMES.index("voiced"), PROSODY_FRAME_NAMES.index("periodicity"))
DISVOICE_ARRAY = "prosody103_wide"          # primary DisVoice variant (60-800 Hz), feature_extraction.md


def load_hubert_layer(store: FeatureStore, layer: int, chunk: int = 10_000) -> np.ndarray:
    """
    (T_total, 768) fp16 copy of one hidden state of the (T_total, 13, 768) fp16 cache.
    Read with plain buffered file reads, `chunk` frames (~200 MB) at a time, instead of a memory map:
    mapping the 17.6 GB CREMA-D file put up to 6.6 GB of mapped pages into the process working set
    (the full run was stopped by the host for low memory while loading). Byte-identical result.
    """
    path = store.root / "hubert.npy"
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = read_header(f)
        if fortran or dtype != np.float16 or len(shape) != 3 or shape[1:] != (13, 768):
            raise ValueError(f"unexpected hubert.npy header {shape} {dtype} fortran={fortran}")
        out = np.empty((shape[0], shape[2]), dtype=np.float16)
        for a in range(0, shape[0], chunk):
            n = min(chunk, shape[0] - a)
            buf = np.fromfile(f, dtype=np.float16, count=n * 13 * 768)
            if buf.size != n * 13 * 768:
                raise IOError(f"short read at frame {a}")
            out[a:a + n] = buf.reshape(n, 13, 768)[:, layer, :]
    return out


class SequenceData:
    def __init__(self, store: FeatureStore, layer: int | None, dev: torch.device, hubert: np.ndarray | None = None,
                 need_disvoice: bool = False):
        man = store.manifest
        self.n = len(man)
        self.layer = layer
        self.dev = dev
        self.lengths_np = man.n_frames.values.astype(np.int64)
        self.offset_np = man.frame_offset.values.astype(np.int64)
        self.offset = torch.as_tensor(self.offset_np, device=dev)
        self.length = torch.as_tensor(self.lengths_np, device=dev)
        self.labels_np = man.label_id.values.astype(np.int64)
        # np.array (copy): the cache memmaps are read-only, torch needs writable host memory
        self._np = {"mfcc": np.array(store.array("mfcc")), "prosody": np.array(store.array("prosody_frames"))}
        if layer is not None:
            self._np["hubert"] = hubert if hubert is not None else load_hubert_layer(store, layer)
            if self._np["hubert"].shape != (int(self.lengths_np.sum()), 768):
                raise ValueError(f"hubert layer array has shape {self._np['hubert'].shape}")
        self.X = {s: torch.from_numpy(a).to(dev) for s, a in self._np.items()}     # hubert stays fp16
        if need_disvoice:
            self._dv = (np.asarray(store.array(DISVOICE_ARRAY)), np.asarray(store.array(DISVOICE_ARRAY + "_imputed")))
        self.fitted_on = None

    # ------------------------------------------------------------------ preprocessing (TRAIN only)
    def frames_np(self, stream: str, row: int) -> np.ndarray:
        a = int(self.offset_np[row]); return self._np[stream][a: a + int(self.lengths_np[row])]

    def fit(self, train_rows: np.ndarray, disvoice_indicators: bool | None = None) -> "SequenceData":
        """Fit every standardiser on the TRAINING recordings of the fold. Must be called again per fold."""
        train_rows = np.asarray(train_rows)
        if self.fitted_on != frozenset(train_rows.tolist()):        # frame statistics depend only on the rows
            self.std = {}
            for s in self._np:
                self.std[s] = FrameStandardizer(passthrough=PROSODY_PASSTHROUGH if s == "prosody" else ()).fit(
                    [self.frames_np(s, r) for r in train_rows])
            self._mu = {s: torch.as_tensor(z.mean_, dtype=torch.float32, device=self.dev) for s, z in self.std.items()}
            self._sd = {s: torch.as_tensor(z.std_, dtype=torch.float32, device=self.dev) for s, z in self.std.items()}
        self.dv = None
        if disvoice_indicators is not None:
            X, imp = self._dv
            self.dv_pre = DisVoicePreprocessor(add_indicators=disvoice_indicators).fit(X[train_rows], imp[train_rows])
            self.dv = torch.from_numpy(self.dv_pre.transform(X, imp)).to(self.dev)   # train statistics applied to all
        self.fitted_on = frozenset(train_rows.tolist())
        return self

    # ------------------------------------------------------------------ batches
    def batch(self, rows) -> dict:
        """Padded, standardised, aligned mini-batch for manifest rows `rows` (any order)."""
        if self.fitted_on is None:
            raise RuntimeError("SequenceData.fit(train_rows) must be called first")
        r = torch.as_tensor(np.asarray(rows), dtype=torch.long, device=self.dev)
        L = self.length[r]
        t = torch.arange(int(L.max()), device=self.dev)
        mask = t[None, :] < L[:, None]                                            # (B, T)
        idx = self.offset[r][:, None] + torch.minimum(t[None, :], L[:, None] - 1)  # clamp: pads read a real frame
        out = {"mask": mask, "rows": r}
        for s, X in self.X.items():
            x = (X[idx].float() - self._mu[s]) / self._sd[s]
            out[s] = torch.where(mask[..., None], x, torch.zeros((), device=self.dev))
        if self.dv is not None:
            out["disvoice"] = self.dv[r]
        return out

    def train_batches(self, rows: np.ndarray, batch_size: int, gen: torch.Generator, bucket: int = 20) -> list:
        """
        Length-bucketed shuffling (seeded): random permutation -> chunks of bucket*batch_size
        recordings -> sort each chunk by length -> cut into batches -> shuffle the batch order.
        Every training recording appears exactly once per epoch; padding is reduced because a
        batch holds recordings of similar length.
        """
        rows = np.asarray(rows)
        perm = rows[torch.randperm(len(rows), generator=gen).numpy()]
        batches, c = [], bucket * batch_size
        for i in range(0, len(perm), c):
            ch = perm[i:i + c]
            ch = ch[np.argsort(self.lengths_np[ch], kind="stable")]
            batches += [ch[j:j + batch_size] for j in range(0, len(ch), batch_size)]
        order = torch.randperm(len(batches), generator=gen).numpy()
        return [batches[k] for k in order]

    def eval_batches(self, rows: np.ndarray, batch_size: int = 64) -> list:
        rows = np.asarray(rows)
        s = rows[np.argsort(self.lengths_np[rows], kind="stable")]
        return [s[j:j + batch_size] for j in range(0, len(s), batch_size)]
