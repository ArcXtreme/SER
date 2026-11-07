"""
G-study Z-W data path (report section 8.3): all 13 HuBERT layers, STREAMED per batch from the existing
(T_total, 13, 768) fp16 cache, because the CREMA-D cache (16.4 GB) fits neither in RAM nor in VRAM.

Same interface as SequenceData (fit / batch / train_batches / eval_batches / fitted_on / n / length /
lengths_np / dev), so the unchanged trainer `train_sequence_classifier` and `predict_rows` drive it.

  READING   each recording's frames are ONE contiguous block of T*13*768 fp16 values; read with buffered
            file reads (no memory map). A background thread prefetches the batches of the list most
            recently returned by train_batches / eval_batches, in order; a batch requested out of order
            is read synchronously (identical result, only slower).
  STANDARDISATION  per layer and dimension with TRAINING-frame statistics, from pre-computed per-recording
            sums (src/features/g_derived.py): mean = S/n, std = sqrt(max(S2/n - mean^2, 0)) + 1e-6, the
            FrameStandardizer formula. Applied on the GPU; padded positions are set to 0 afterwards.
  BATCHING  length-bucketed seeded batches: the SAME algorithm as SequenceData.train_batches.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from src.features.g_derived import DIM, N_LAYERS, npy_header

EPS = 1e-6


class StreamedLayers:
    def __init__(self, npy_path: Path, offsets, lengths, layer_sum, layer_sumsq, dev: torch.device,
                 prefetch: bool = True, max_queue: int = 2):
        shape, dtype, self.data_offset = npy_header(Path(npy_path))
        if dtype != np.float16 or shape[1:] != (N_LAYERS, DIM):
            raise ValueError(f"unexpected cache header {shape} {dtype}")
        self.path = Path(npy_path)
        self.offset_np = np.asarray(offsets, np.int64)
        self.lengths_np = np.asarray(lengths, np.int64)
        self.n = len(self.lengths_np)
        self.dev = dev
        self.length = torch.as_tensor(self.lengths_np, device=dev)
        self._sum, self._sumsq = layer_sum, layer_sumsq            # (n, 13, 768) float64 (array or memmap)
        self.layer = "weighted13"
        self.fitted_on = None
        self.prefetch, self.max_queue = prefetch, max_queue
        self._job = None
        self.io = {"bytes": 0, "read_seconds": 0.0, "sync_reads": 0, "prefetched": 0}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ preprocessing (TRAIN only)
    def stats(self, train_rows):
        rows = np.sort(np.asarray(train_rows))
        n = float(self.lengths_np[rows].sum())
        s = np.zeros((N_LAYERS, DIM)); s2 = np.zeros((N_LAYERS, DIM))
        for i in range(0, len(rows), 256):                         # bounded memory when the sums are memmaps
            r = rows[i:i + 256]
            s += np.asarray(self._sum[r]).sum(0); s2 += np.asarray(self._sumsq[r]).sum(0)
        mean = s / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0)) + EPS
        return mean, std

    def fit(self, train_rows, disvoice_indicators=None) -> "StreamedLayers":
        if disvoice_indicators is not None:
            raise ValueError("Z-W uses HuBERT only")
        key = frozenset(np.asarray(train_rows).tolist())
        if self.fitted_on != key:
            self.mean_, self.std_ = self.stats(train_rows)
            self._mu = torch.as_tensor(self.mean_, dtype=torch.float32, device=self.dev)
            self._sd = torch.as_tensor(self.std_, dtype=torch.float32, device=self.dev)
        self.fitted_on = key
        return self

    # ------------------------------------------------------------------ reading
    def _read(self, f, rows) -> np.ndarray:
        L = self.lengths_np[rows]
        out = np.zeros((len(rows), int(L.max()), N_LAYERS, DIM), np.float16)
        t = time.perf_counter()
        for j, r in enumerate(rows):
            f.seek(self.data_offset + int(self.offset_np[r]) * N_LAYERS * DIM * 2)
            b = np.fromfile(f, dtype=np.float16, count=int(L[j]) * N_LAYERS * DIM)
            if b.size != int(L[j]) * N_LAYERS * DIM:
                raise IOError(f"short read for row {r}")
            out[j, :L[j]] = b.reshape(int(L[j]), N_LAYERS, DIM)
        with self._lock:
            self.io["bytes"] += int(L.sum()) * N_LAYERS * DIM * 2
            self.io["read_seconds"] += time.perf_counter() - t
        return out

    def _cancel(self):
        if self._job is not None:
            self._job["stop"].set()
            try:
                while True:
                    self._job["q"].get_nowait()
            except queue.Empty:
                pass
            self._job["thread"].join()
            self._job = None

    def _schedule(self, batches):
        self._cancel()
        if not self.prefetch or not batches:
            return
        q, stop = queue.Queue(maxsize=self.max_queue), threading.Event()

        def put(item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.2)
                    return
                except queue.Full:
                    continue

        def work():
            try:
                with open(self.path, "rb") as f:
                    for b in batches:
                        if stop.is_set():
                            return
                        put((tuple(int(x) for x in b), self._read(f, b)))
            except Exception as e:                                # surfaced in the training thread, never swallowed
                put(("__error__", e))
        th = threading.Thread(target=work, daemon=True)
        self._job = {"q": q, "stop": stop, "thread": th, "remaining": len(batches)}
        th.start()

    def _get(self, rows) -> np.ndarray:
        key = tuple(int(x) for x in rows)
        if self._job is not None and self._job["remaining"] > 0:
            k, arr = self._job["q"].get()
            self._job["remaining"] -= 1
            if k == "__error__":
                self._cancel()
                raise arr
            if k == key:
                self.io["prefetched"] += 1
                return arr
            self._cancel()                                        # out of order: fall back to a synchronous read
        self.io["sync_reads"] += 1
        with open(self.path, "rb") as f:
            return self._read(f, np.asarray(rows))

    def batch(self, rows) -> dict:
        if self.fitted_on is None:
            raise RuntimeError("StreamedLayers.fit(train_rows) must be called first")
        rows = np.asarray(rows)
        x = torch.from_numpy(self._get(rows)).to(self.dev).float()             # (B, T, 13, 768)
        r = torch.as_tensor(rows, dtype=torch.long, device=self.dev)
        L = self.length[r]
        mask = torch.arange(x.shape[1], device=self.dev)[None, :] < L[:, None]
        x.sub_(self._mu).div_(self._sd)                                          # in place: bounded GPU memory
        x.mul_(mask[..., None, None].to(x.dtype))                                # padded positions -> 0 (values finite)
        return {"hubert13": x, "mask": mask, "rows": r}

    # ------------------------------------------------------------------ batch lists (SequenceData algorithm)
    def train_batches(self, rows, batch_size: int, gen: torch.Generator, bucket: int = 20) -> list:
        rows = np.asarray(rows)
        perm = rows[torch.randperm(len(rows), generator=gen).numpy()]
        batches, c = [], bucket * batch_size
        for i in range(0, len(perm), c):
            ch = perm[i:i + c]
            ch = ch[np.argsort(self.lengths_np[ch], kind="stable")]
            batches += [ch[j:j + batch_size] for j in range(0, len(ch), batch_size)]
        order = torch.randperm(len(batches), generator=gen).numpy()
        out = [batches[k] for k in order]
        self._schedule(out)
        return out

    def eval_batches(self, rows, batch_size: int = 64) -> list:
        rows = np.asarray(rows)
        s = rows[np.argsort(self.lengths_np[rows], kind="stable")]
        out = [s[j:j + batch_size] for j in range(0, len(s), batch_size)]
        self._schedule(out)
        return out

    def close(self):
        self._cancel()
