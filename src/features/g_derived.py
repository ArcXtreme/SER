"""
G-study derived features (report section 8.3), computed in ONE sequential, chunked pass over the existing
(T_total, 13, 768) fp16 HuBERT frame cache. The cache is opened read-only with buffered reads (no memory map).

  um_frames   (T_total, 768) fp16   uniform 13-layer average  x_bar_t = (1/13) sum_l x_t^l, computed in float32
                                    from the fp16 cache, then rounded to fp16 (a fixed, label-free approximation;
                                    NOT equivalent to learned layer weighting)
  um_pooled   (N, 1, 1536) fp32     per recording [mean_t ; std_t] of the float32 average (population std, as the
                                    E2 pooled cache), so P0-UM can use the unchanged E2 code path with rep "layer0"
  layer_sum   (N, 13, 768) float64  per recording, per layer: sum_t x_t^l        (only if with_layer_sums; Z-W)
  layer_sumsq (N, 13, 768) float64  per recording, per layer: sum_t (x_t^l)^2    -> exact training-fold statistics
                                    by summing over training recordings (same formula as FrameStandardizer)
Frames of recording i are rows offsets[i] : offsets[i] + lengths[i] (Stage 4 offset table; contiguous, ordered).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

N_LAYERS, DIM = 13, 768


def npy_header(path: Path):
    """(shape, dtype, data_offset_bytes) of a C-ordered .npy file."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(f)
        if fortran:
            raise ValueError(f"{path}: Fortran order not supported")
        return shape, dtype, f.tell()


def check_offsets(offsets: np.ndarray, lengths: np.ndarray, total: int):
    expect = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    if not np.array_equal(np.asarray(offsets), expect) or int(np.sum(lengths)) != total:
        raise ValueError("frame offset table is not contiguous and ordered")


def compute_derived(npy_path: Path, offsets: np.ndarray, lengths: np.ndarray, with_layer_sums: bool,
                    chunk: int = 2500, progress=None) -> dict:
    shape, dtype, off0 = npy_header(npy_path)
    if dtype != np.float16 or len(shape) != 3 or shape[1:] != (N_LAYERS, DIM):
        raise ValueError(f"unexpected cache header {shape} {dtype}")
    T, n = shape[0], len(lengths)
    offsets, lengths = np.asarray(offsets, np.int64), np.asarray(lengths, np.int64)
    check_offsets(offsets, lengths, T)
    ends = offsets + lengths
    um = np.empty((T, DIM), np.float16)
    psum = np.zeros((n, DIM)); psq = np.zeros((n, DIM))
    lsum = np.zeros((n, N_LAYERS, DIM)) if with_layer_sums else None
    lsq = np.zeros((n, N_LAYERS, DIM)) if with_layer_sums else None
    t0 = time.perf_counter()
    nbytes = 0
    with open(npy_path, "rb") as f:
        f.seek(off0)
        for a in range(0, T, chunk):
            m = min(chunk, T - a)
            buf = np.fromfile(f, dtype=np.float16, count=m * N_LAYERS * DIM)
            if buf.size != m * N_LAYERS * DIM:
                raise IOError(f"short read at frame {a}")
            nbytes += buf.nbytes
            x = buf.reshape(m, N_LAYERS, DIM)
            avg = x.astype(np.float32).mean(axis=1)                       # (m, 768) float32
            um[a:a + m] = avg.astype(np.float16)
            rec = np.searchsorted(ends, np.arange(a, a + m), side="right")  # recording of each frame
            starts = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1]])
            ids = rec[starts]
            a64 = avg.astype(np.float64)
            psum[ids] += np.add.reduceat(a64, starts, axis=0)
            psq[ids] += np.add.reduceat(a64 * a64, starts, axis=0)
            if with_layer_sums:
                x64 = x.astype(np.float64)
                lsum[ids] += np.add.reduceat(x64, starts, axis=0)
                lsq[ids] += np.add.reduceat(x64 * x64, starts, axis=0)
            if progress is not None and (a // chunk) % 40 == 0:
                progress(a + m, T, nbytes, time.perf_counter() - t0)
    cnt = lengths[:, None].astype(np.float64)
    mean = psum / cnt
    std = np.sqrt(np.maximum(psq / cnt - mean ** 2, 0.0))
    out = {"um_frames": um, "um_pooled": np.concatenate([mean, std], axis=1).astype(np.float32)[:, None, :],
           "seconds": time.perf_counter() - t0, "bytes_read": nbytes}
    if with_layer_sums:
        out["layer_sum"], out["layer_sumsq"] = lsum, lsq
    return out


def read_recording(npy_path: Path, data_offset: int, frame_offset: int, length: int) -> np.ndarray:
    """(length, 13, 768) fp16 block of one recording (buffered read, no memory map)."""
    with open(npy_path, "rb") as f:
        f.seek(data_offset + int(frame_offset) * N_LAYERS * DIM * 2)
        b = np.fromfile(f, dtype=np.float16, count=int(length) * N_LAYERS * DIM)
    return b.reshape(int(length), N_LAYERS, DIM)


def verify_derived(npy_path: Path, offsets, lengths, derived: dict, rows) -> dict:
    """Independent recomputation on sampled recordings (report section 8.6 c)."""
    _, _, off0 = npy_header(npy_path)
    worst = {"um_frames_max_abs": 0.0, "um_pooled_max_rel": 0.0, "layer_sum_max_rel": 0.0}
    for r in rows:
        x = read_recording(npy_path, off0, offsets[r], lengths[r])
        avg = x.astype(np.float32).mean(axis=1)
        got = derived["um_frames"][offsets[r]: offsets[r] + lengths[r]].astype(np.float32)
        worst["um_frames_max_abs"] = max(worst["um_frames_max_abs"], float(np.abs(got - avg.astype(np.float16).astype(np.float32)).max()))
        a64 = avg.astype(np.float64)
        ref = np.concatenate([a64.mean(0), a64.std(0)])
        p = derived["um_pooled"][r, 0].astype(np.float64)
        worst["um_pooled_max_rel"] = max(worst["um_pooled_max_rel"], float((np.abs(p - ref) / np.maximum(np.abs(ref), 1e-3)).max()))
        if "layer_sum" in derived:
            x64 = x.astype(np.float64)
            for got_, ref_ in ((derived["layer_sum"][r], x64.sum(0)), (derived["layer_sumsq"][r], (x64 * x64).sum(0))):
                worst["layer_sum_max_rel"] = max(worst["layer_sum_max_rel"],
                                                 float((np.abs(got_ - ref_) / np.maximum(np.abs(ref_), 1e-6)).max()))
    worst["n_recordings_checked"] = len(rows)
    worst["pass"] = (worst["um_frames_max_abs"] == 0.0 and worst["um_pooled_max_rel"] <= 1e-6
                     and worst["layer_sum_max_rel"] <= 1e-9)
    return worst
