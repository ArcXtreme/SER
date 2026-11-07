"""
Stage 4 driver: build the feature cache (resumable - just re-run after an interruption).

Run:  C:\\SER_data\\.venv\\Scripts\\python.exe scripts/stage4_extract_features.py --datasets RAVDESS
      [--stages prepare cpu hubert] [--workers 8]

  prepare : decode -> mono -> 16 kHz -> trim -> normalise; wav16k.npy + manifest   (CPU)
  cpu     : MFCC (T,39), prosody frames (T,4), DisVoice-103 x2                    (CPU, parallel)
  hubert  : 13 hidden states (T,13,768) fp16 + pooled stats (13,1536) fp32         (GPU)

  --adopt-from <cache_id> : start the new cache by copying audio, manifest and HuBERT from an
      older cache whose audio/frame/MFCC/HuBERT config is identical (SHA-256 verified), then run
      only 'cpu'; MFCC is recomputed and checked byte-identical to the source. Used for the
      pYIN -> Praat switch:  --stages cpu --adopt-from v1_f9418feedb
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd  # noqa: E402

from src.features.store import (adopt_unchanged, cache_dir, stage_cpu, stage_hubert,  # noqa: E402
                                stage_prepare, verify_adopted)
from src.utils.paths import PATHS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["RAVDESS"])
    ap.add_argument("--stages", nargs="+", default=["prepare", "cpu", "hubert"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--adopt-from", default=None)
    args = ap.parse_args()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    for ds in args.datasets:
        t0 = time.perf_counter()
        if args.adopt_from and not (cache_dir(ds) / "meta.json").exists():
            adopt_unchanged(ds, index, args.adopt_from)
        if "prepare" in args.stages:
            stage_prepare(ds, index, workers=args.workers)
        if "cpu" in args.stages:
            stage_cpu(ds, workers=args.workers)
        if "hubert" in args.stages:
            stage_hubert(ds)
        if args.adopt_from:
            print(f"[verify] {ds}:", verify_adopted(ds))
        print(f"== {ds} done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
