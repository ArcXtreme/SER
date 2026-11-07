"""
Stage 4 - Feature cache ("feature store"): build, resume, validate, load.

LAYOUT  C:\\SER_data\\features\\<DATASET>\\<cache_id>\\      cache_id = v<version>_<config-hash>
  manifest.csv           row i = recording i (same order as splits.usable_files):
                         rel_path, speaker, label_id, trim info, n_samples, n_frames,
                         wav_offset, frame_offset
  wav16k.npy             (S_total,)            float32  preprocessed audio, concatenated
  mfcc.npy               (T_total, 39)         float32  frame-level
  prosody_frames.npy     (T_total, 4)          float32  frame-level
  hubert.npy             (T_total, 13, 768)    float16  frame-level, all hidden states
  hubert_pooled.npy      (n, 13, 1536)         float32  per-layer [mean | std] over frames (fp32)
  prosody103_disvoice.npy / _wide.npy  (n, 103) float32 utterance-level (NaN imputed with 0)
  prosody103_*_imputed.npy             (n, 103) bool    where imputation happened
  progress_cpu.npy / progress_hubert.npy (n,)   bool    resumability markers
  meta.json              config + hash, index fingerprint, model revision, library versions,
                         per-stage status and timings

WHY CONCATENATED ARRAYS + OFFSETS?
  Recordings have different lengths. Storing them back-to-back with an offset table
  (recording i = rows offset[i]:offset[i+1]) wastes no space on padding and lets
  np.load(mmap_mode="r") read one recording without loading the whole file into RAM.
  HuBERT, MFCC and prosody frames share ONE frame-offset table: they live on the same
  time grid, so recording i has the same number of frames in every stream BY CONSTRUCTION.
  Padding happens only later, per mini-batch, together with a validity mask.

RESUMABILITY
  Arrays are pre-allocated (their sizes follow from the audio lengths), each worker result is
  written in place, and progress_*.npy marks finished recordings. After an interruption the
  next run skips everything already marked done.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.config import DEFAULT, FeatureConfig, n_frames
from src.preprocessing.splits import fingerprint, usable_files
from src.utils.paths import PATHS

FRAME_ARRAYS = {"mfcc": 39, "prosody_frames": 4}
P103 = ("prosody103_disvoice", "prosody103_wide")


class CacheError(RuntimeError):
    """Raised when a cache is missing, incomplete, or inconsistent with the current config/index."""


def cache_dir(dataset: str, cfg: FeatureConfig = DEFAULT) -> Path:
    return PATHS.features / dataset / cfg.cache_id()


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)                              # atomic on the same filesystem


def _versions() -> dict:
    import librosa, parselmouth, scipy, sklearn, soundfile, torch, transformers  # noqa: E401
    v = {m.__name__: m.__version__ for m in (np, pd, scipy, sklearn, librosa, soundfile, torch, transformers,
                                            parselmouth)}
    v["praat"] = parselmouth.PRAAT_VERSION
    return v


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# Files whose content depends only on the audio / frame / HuBERT parts of the config.
ADOPTABLE = ("manifest.csv", "wav16k.npy", "hubert.npy", "hubert_pooled.npy", "progress_hubert.npy")
UNCHANGED_SECTIONS = ("version", "audio", "frames", "mfcc", "hubert")


def adopt_unchanged(dataset: str, index: pd.DataFrame, source_cache_id: str,
                    cfg: FeatureConfig = DEFAULT) -> Path:
    """
    Start a NEW cache (new config hash) by copying the outputs that the config change cannot
    affect - preprocessed audio, manifest, HuBERT - from an existing cache, verified by SHA-256.
    Allowed only if the source was built from the same index and identical audio/frame/MFCC/
    HuBERT config sections. MFCC is NOT copied: stage_cpu recomputes it and verify_adopted()
    checks it is byte-identical to the source (so 'MFCC unchanged' is tested, not assumed).
    Copying (not re-running) HuBERT keeps it bit-identical: GPU inference is not bit-reproducible.
    """
    import shutil
    src, dst = PATHS.features / dataset / source_cache_id, cache_dir(dataset, cfg)
    if src == dst:
        raise CacheError("source and destination cache are the same")
    smeta = json.loads((src / "meta.json").read_text())
    new = cfg.to_dict()
    diff = [s for s in UNCHANGED_SECTIONS if smeta["config"].get(s) != new[s]]
    if diff:
        raise CacheError(f"cannot adopt from {source_cache_id}: config sections differ: {diff}")
    if smeta["index_sha256"] != fingerprint(usable_files(index, dataset)):
        raise CacheError("cannot adopt: source cache was built from a different index")
    for s in ("prepare", "hubert"):
        if not smeta["stages"].get(s, {}).get("complete"):
            raise CacheError(f"cannot adopt: source stage '{s}' incomplete")
    if (dst / "meta.json").exists():
        raise CacheError(f"{dst} already exists")
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    hashes = {}
    for name in ADOPTABLE:
        shutil.copyfile(src / name, dst / name)
        h_src, h_dst = sha256_file(src / name), sha256_file(dst / name)
        if h_src != h_dst:
            raise CacheError(f"copy of {name} differs from source")
        hashes[name] = h_src
    hashes["mfcc.npy (source, for verification)"] = sha256_file(src / "mfcc.npy")
    meta = {k: smeta[k] for k in ("dataset", "index_sha256", "n_files", "n_samples_total", "n_frames_total")}
    meta.update({
        "cache_id": cfg.cache_id(), "config_hash": cfg.hash(), "config": new, "versions": _versions(),
        "adopted_from": {"cache_id": source_cache_id, "config_hash": smeta["config_hash"],
                         "versions": smeta["versions"], "sha256": hashes,
                         "seconds": round(time.perf_counter() - t0, 1)},
        "stages": {s: {**smeta["stages"][s], "adopted_from": source_cache_id} for s in ("prepare", "hubert")},
    })
    _write_json(dst / "meta.json", meta)
    print(f"[adopt] {dataset}: copied {len(ADOPTABLE)} files from {source_cache_id} "
          f"({time.perf_counter() - t0:.0f}s, SHA-256 verified)")
    return dst


def verify_adopted(dataset: str, cfg: FeatureConfig = DEFAULT) -> dict:
    """After stage_cpu: MFCC must be byte-identical to the source cache, adopted files untouched."""
    root = cache_dir(dataset, cfg)
    meta = json.loads((root / "meta.json").read_text())
    src = meta["adopted_from"]["sha256"]
    res = {"mfcc_identical_to_source": sha256_file(root / "mfcc.npy") == src["mfcc.npy (source, for verification)"]}
    for name in ADOPTABLE:
        res[f"{name}_unchanged"] = sha256_file(root / name) == src[name]
    meta["adopted_from"]["verification"] = res
    _write_json(root / "meta.json", meta)
    if not all(res.values()):
        raise CacheError(f"adopted cache verification failed: {res}")
    return res


# =============================================================================== workers
# Module-level functions so Windows "spawn" multiprocessing can pickle them.
def _prepare_worker(args):
    from src.preprocessing.audio import preprocess
    i, path, audio_cfg = args
    y, info = preprocess(Path(path), audio_cfg)
    return i, y, info


_MFCC = None


def _cpu_worker(args):
    """Compute MFCC (T,39), prosody frames (T,4) and both DisVoice-103 vectors for one recording."""
    global _MFCC
    from src.features.mfcc import MFCCExtractor
    from src.features.prosody import prosody103, prosody_frames
    i, y, cfg = args
    if _MFCC is None:
        _MFCC = MFCCExtractor(cfg.mfcc)
    T = n_frames(len(y))
    mf = _MFCC(y)
    pf = prosody_frames(y, cfg.prosody, T)
    p_dv = prosody103(y, cfg.prosody, cfg.prosody.disvoice_f0_max)
    p_wd = prosody103(y, cfg.prosody, cfg.prosody.f0_max_wide)
    return i, mf, pf, p_dv, p_wd


# =============================================================================== build
def stage_prepare(dataset: str, index: pd.DataFrame, cfg: FeatureConfig = DEFAULT,
                  workers: int = 8) -> Path:
    """Preprocess every usable recording once; write wav16k.npy, manifest.csv, meta.json."""
    root = cache_dir(dataset, cfg)
    files = usable_files(index, dataset)
    meta_path = root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["index_sha256"] != fingerprint(files) or meta["config_hash"] != cfg.hash():
            raise CacheError(f"{root} exists but was built for a different index/config")
        if meta["stages"].get("prepare", {}).get("complete"):
            print(f"[prepare] {dataset}: already complete")
            return root
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    jobs = [(i, str(PATHS.raw / p), cfg.audio) for i, p in enumerate(files.rel_path)]
    wavs, infos, failures = [None] * len(jobs), [None] * len(jobs), []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_prepare_worker, j): j[0] for j in jobs}
        for f in futs:
            try:
                i, y, info = f.result()
                wavs[i], infos[i] = y, info
            except Exception as e:                       # never skip silently
                failures.append((files.rel_path.iloc[futs[f]], repr(e)))
    if failures:
        raise CacheError(f"[prepare] {len(failures)} recordings failed: {failures[:5]}")

    man = files[["rel_path", "speaker", "gender", "label", "label_id"]].copy()
    man = pd.concat([man, pd.DataFrame(infos)], axis=1)
    man["n_frames"] = [n_frames(n) for n in man.n_samples]
    if (man.n_frames <= 0).any():
        raise CacheError(f"recordings shorter than one frame: {man.rel_path[man.n_frames <= 0].tolist()}")
    man["wav_offset"] = np.concatenate([[0], np.cumsum(man.n_samples)[:-1]])
    man["frame_offset"] = np.concatenate([[0], np.cumsum(man.n_frames)[:-1]])
    np.save(root / "wav16k.npy", np.concatenate(wavs).astype(np.float32))
    man.to_csv(root / "manifest.csv", index=False)
    _write_json(meta_path, {
        "dataset": dataset, "cache_id": cfg.cache_id(), "config_hash": cfg.hash(),
        "config": cfg.to_dict(), "index_sha256": fingerprint(files),
        "n_files": len(man), "n_samples_total": int(man.n_samples.sum()),
        "n_frames_total": int(man.n_frames.sum()), "versions": _versions(),
        "stages": {"prepare": {"complete": True, "seconds": round(time.perf_counter() - t0, 1)}},
    })
    print(f"[prepare] {dataset}: {len(man)} recordings, {man.n_frames.sum()} frames, "
          f"{time.perf_counter() - t0:.1f}s")
    return root


def _open_or_create(path: Path, shape, dtype):
    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != tuple(shape) or arr.dtype != np.dtype(dtype):
            raise CacheError(f"{path.name}: shape/dtype {arr.shape}/{arr.dtype} != {shape}/{dtype}")
        return arr
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))


def _update_stage(root: Path, stage: str, info: dict) -> None:
    """Record stage status; every run (full or resumed) is appended to a history list."""
    meta = json.loads((root / "meta.json").read_text())
    st = meta["stages"].get(stage, {})
    st["complete"] = info.pop("complete")
    st.setdefault("runs", []).append({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), **info})
    meta["stages"][stage] = st
    _write_json(root / "meta.json", meta)


def stage_cpu(dataset: str, cfg: FeatureConfig = DEFAULT, workers: int = 8, chunk: int = 64) -> None:
    """MFCC + prosody (CPU, multi-process, resumable)."""
    root = cache_dir(dataset, cfg)
    man = pd.read_csv(root / "manifest.csv")
    n, T = len(man), int(man.n_frames.sum())
    wav = np.load(root / "wav16k.npy", mmap_mode="r")
    arrs = {k: _open_or_create(root / f"{k}.npy", (T, d), np.float32) for k, d in FRAME_ARRAYS.items()}
    for k in P103:
        arrs[k] = _open_or_create(root / f"{k}.npy", (n, 103), np.float32)
        arrs[k + "_imputed"] = _open_or_create(root / f"{k}_imputed.npy", (n, 103), np.bool_)
    done = _open_or_create(root / "progress_cpu.npy", (n,), np.bool_)
    todo = np.flatnonzero(~np.asarray(done))
    print(f"[cpu] {dataset}: {n - len(todo)} done, {len(todo)} to process")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c in range(0, len(todo), chunk):
            ids = todo[c:c + chunk]
            jobs = [(int(i), np.array(wav[man.wav_offset[i]: man.wav_offset[i] + man.n_samples[i]]), cfg)
                    for i in ids]
            for i, mf, pf, p_dv, p_wd in ex.map(_cpu_worker, jobs):
                a, b = man.frame_offset[i], man.frame_offset[i] + man.n_frames[i]
                if mf.shape[0] != b - a or pf.shape[0] != b - a:
                    raise CacheError(f"recording {i}: frame count mismatch {mf.shape}, {pf.shape}, {b - a}")
                arrs["mfcc"][a:b] = mf
                arrs["prosody_frames"][a:b] = pf
                for k, v in zip(P103, (p_dv, p_wd)):
                    bad = ~np.isfinite(v)
                    arrs[k][i] = np.where(bad, 0.0, v)
                    arrs[k + "_imputed"][i] = bad
                done[i] = True
            for a in [*arrs.values(), done]:
                a.flush()
            print(f"  {min(c + chunk, len(todo))}/{len(todo)}  ({time.perf_counter() - t0:.0f}s)")
    _update_stage(root, "cpu", {"complete": bool(np.asarray(done).all()), "n_processed": int(len(todo)),
                                "seconds": round(time.perf_counter() - t0, 1), "workers": workers})


def stage_hubert(dataset: str, cfg: FeatureConfig = DEFAULT, flush_every: int = 50) -> None:
    """All 13 HuBERT hidden states (GPU, one utterance at a time, resumable)."""
    import torch
    from src.features.hubert import HubertExtractor
    root = cache_dir(dataset, cfg)
    man = pd.read_csv(root / "manifest.csv")
    n, T = len(man), int(man.n_frames.sum())
    wav = np.load(root / "wav16k.npy", mmap_mode="r")
    hub = _open_or_create(root / "hubert.npy", (T, 13, 768), np.float16)
    pooled = _open_or_create(root / "hubert_pooled.npy", (n, 13, 1536), np.float32)
    done = _open_or_create(root / "progress_hubert.npy", (n,), np.bool_)
    todo = np.flatnonzero(~np.asarray(done))
    print(f"[hubert] {dataset}: {n - len(todo)} done, {len(todo)} to process")
    if len(todo) == 0:
        return
    ex = HubertExtractor(cfg.hubert)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for k, i in enumerate(todo, 1):
        y = np.array(wav[man.wav_offset[i]: man.wav_offset[i] + man.n_samples[i]])
        hs = ex(y)                                                   # (13, T_i, 768) fp32
        a, b = man.frame_offset[i], man.frame_offset[i] + man.n_frames[i]
        hub[a:b] = hs.permute(1, 0, 2).cpu().numpy().astype(np.float16)
        pooled[i] = ex.pooled_stats(hs)
        done[i] = True
        if k % flush_every == 0 or k == len(todo):
            hub.flush(); pooled.flush(); done.flush()
            print(f"  {k}/{len(todo)}  ({time.perf_counter() - t0:.0f}s)")
    elapsed = time.perf_counter() - t0
    _update_stage(root, "hubert", {
        "complete": bool(np.asarray(done).all()), "n_processed": int(len(todo)), "seconds": round(elapsed, 1),
        "sec_per_recording": round(elapsed / len(todo), 4),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
        "device": ex.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "inference_dtype": cfg.hubert.inference_dtype, "hf_cache_snapshots": ex.cache_snapshots(),
    })


# =============================================================================== load
@dataclass
class FeatureStore:
    """Read-only, validated access to one dataset's cached features (memory-mapped)."""
    root: Path
    meta: dict
    manifest: pd.DataFrame

    @classmethod
    def open(cls, dataset: str, index: pd.DataFrame | None = None, cfg: FeatureConfig = DEFAULT,
             require=("cpu", "hubert")) -> "FeatureStore":
        """
        Cache validation: config hash, index fingerprint, manifest order, completed stages
        and array shapes must all match - otherwise CacheError (never silently stale data).
        """
        root = cache_dir(dataset, cfg)
        if not (root / "meta.json").exists():
            raise CacheError(f"no feature cache at {root}")
        meta = json.loads((root / "meta.json").read_text())
        man = pd.read_csv(root / "manifest.csv")
        if meta["config_hash"] != cfg.hash():
            raise CacheError("feature config changed since the cache was built")
        if index is not None:
            files = usable_files(index, dataset)
            if meta["index_sha256"] != fingerprint(files) or man.rel_path.tolist() != files.rel_path.tolist():
                raise CacheError("dataset index changed since the cache was built")
        for s in require:
            if not meta["stages"].get(s, {}).get("complete"):
                raise CacheError(f"stage '{s}' incomplete - re-run extraction to resume")
            if not np.load(root / f"progress_{s}.npy").all():
                raise CacheError(f"stage '{s}' has unprocessed recordings")
        store = cls(root, meta, man)
        T, n = int(man.n_frames.sum()), len(man)
        expected = {"mfcc": (T, 39), "prosody_frames": (T, 4), "hubert": (T, 13, 768),
                    "hubert_pooled": (n, 13, 1536), "prosody103_disvoice": (n, 103),
                    "prosody103_wide": (n, 103)}
        for k, shape in expected.items():
            if (root / f"{k}.npy").exists() and store.array(k).shape != shape:
                raise CacheError(f"{k}.npy has shape {store.array(k).shape}, expected {shape}")
        return store

    def array(self, name: str) -> np.ndarray:
        return np.load(self.root / f"{name}.npy", mmap_mode="r")

    def frames(self, name: str, i: int) -> np.ndarray:
        """Frame-level stream `name` for recording i: (n_frames_i, ...)."""
        a = int(self.manifest.frame_offset[i])
        return self.array(name)[a: a + int(self.manifest.n_frames[i])]

    def wav(self, i: int) -> np.ndarray:
        a = int(self.manifest.wav_offset[i])
        return self.array("wav16k")[a: a + int(self.manifest.n_samples[i])]
