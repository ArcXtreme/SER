"""
Stage 4 - HuBERT numerical checks that decide the inference precision.

Run:  C:\\SER_data\\.venv\\Scripts\\python.exe scripts/stage4_hubert_checks.py [--n 24]

1. Hidden-state semantics: hidden_states[0] == encoder input, hidden_states[i] == output
   of Transformer layer i (recomputed manually from sub-modules).
2. fp16 vs fp32 INFERENCE on real RAVDESS clips: per-layer cosine similarity and relative
   error, plus time per clip and peak GPU memory.
3. fp16 STORAGE error: cast fp32 features to fp16 and back.
Writes results/stage4_hubert_checks.json.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import PATHS  # noqa: E402  (sets HF_HOME)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.features.config import DEFAULT  # noqa: E402
from src.features.hubert import HubertExtractor  # noqa: E402
from src.preprocessing.audio import preprocess  # noqa: E402
from src.preprocessing.splits import usable_files  # noqa: E402


def per_layer_compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a, b: (13, T, 768). Frame-wise cosine and relative L2 error, per layer."""
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)              # (13, T)
    rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-8)              # (13, T)
    return {"cos_min": cos.min(1).values.tolist(), "rel_err_mean": rel.mean(1).tolist(),
            "rel_err_max": rel.max(1).values.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    files = usable_files(index, "RAVDESS").sample(args.n, random_state=0)
    wavs = [preprocess(PATHS.raw / p, DEFAULT.audio)[0] for p in files.rel_path]

    ex32 = HubertExtractor(replace(DEFAULT.hubert, inference_dtype="float32"))
    report = {"semantics_max_abs_err": ex32.verify_hidden_state_semantics(wavs[0])}

    def run(ex):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ex(wavs[0])                                               # warm-up
        t0 = time.perf_counter()
        outs = [ex(w) for w in wavs]
        torch.cuda.synchronize()
        return outs, (time.perf_counter() - t0) / len(wavs), torch.cuda.max_memory_allocated() / 2**20

    o32, t32, m32 = run(ex32)
    del ex32
    torch.cuda.empty_cache()
    ex16 = HubertExtractor(replace(DEFAULT.hubert, inference_dtype="float16"))
    o16, t16, m16 = run(ex16)

    layers = {k: [] for k in ["cos_min", "rel_err_mean", "rel_err_max"]}
    store = {k: [] for k in ["cos_min", "rel_err_mean", "rel_err_max"]}
    absmax = torch.zeros(13)
    for a, b in zip(o32, o16):
        c = per_layer_compare(a, b)
        s = per_layer_compare(a, a.half().float())
        for k in layers:
            layers[k].append(c[k]); store[k].append(s[k])
        absmax = torch.maximum(absmax, a.abs().amax(dim=(1, 2)).cpu())
    agg = lambda d: {"cos_min": np.min(d["cos_min"], 0).round(6).tolist(),  # noqa: E731
                     "rel_err_mean": np.mean(d["rel_err_mean"], 0).round(6).tolist(),
                     "rel_err_max": np.max(d["rel_err_max"], 0).round(6).tolist()}
    report.update({
        "n_clips": len(wavs),
        "mean_clip_seconds": float(np.mean([len(w) / 16000 for w in wavs])),
        "fp32_inference": {"sec_per_clip": t32, "peak_mem_mib": m32},
        "fp16_inference": {"sec_per_clip": t16, "peak_mem_mib": m16},
        "fp16_vs_fp32_inference_per_layer": agg(layers),
        "fp16_storage_of_fp32_per_layer": agg(store),
        "fp32_abs_max_per_layer": absmax.round(decimals=2).tolist(),
    })
    out = PATHS.results / "stage4_hubert_checks.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
