"""
G-study pre-computation (report section 8.3): one chunked pass over the existing HuBERT frame cache per dataset.
Writes NEW files only (refuses to overwrite) under  C:\\SER_data\\features\\<DS>\\derived_G_<cache_id>\\ :
  um_frames.npy  um_pooled.npy  meta.json   [+ layer_sum.npy  layer_sumsq.npy for CREMA-D: Z-W is CREMA-D only]
and verifies them against an independent recomputation on 100 random recordings (fails loudly otherwise).

  python scripts/stage5_g_precompute.py --datasets RAVDESS CREMA-D
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.g_derived import compute_derived, verify_derived  # noqa: E402
from src.features.store import FeatureStore  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402
from src.utils.resources import process_ram_mib  # noqa: E402

ZW_DATASETS = ("CREMA-D",)          # team decision 2026-11-07: learned 13-layer weighting on CREMA-D only


def derived_dir(store) -> Path:
    return store.root.parent / f"derived_G_{store.meta['cache_id']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    for ds in args.datasets:
        print(f"  [stage] {ds}: precompute - opening feature store", flush=True)
        store = FeatureStore.open(ds, index)
        out = derived_dir(store)
        names = ["um_frames.npy", "um_pooled.npy", "meta.json"] + (["layer_sum.npy", "layer_sumsq.npy"] if ds in ZW_DATASETS else [])
        if any((out / n).exists() for n in names):
            raise SystemExit(f"{out} already contains derived files - not overwritten")
        man = store.manifest
        offsets, lengths = man.frame_offset.values, man.n_frames.values
        print(f"  [stage] {ds}: precompute - streaming hubert.npy (chunked)", flush=True)
        prog = lambda a, T, b, s: print(f"    {a}/{T} frames, {b / 1e9:.2f} GB in {s:.0f}s ({b / 1e9 / max(s, 1e-9):.2f} GB/s), "  # noqa: E731
                                        f"RAM {process_ram_mib()}", flush=True)
        D = compute_derived(store.root / "hubert.npy", offsets, lengths, ds in ZW_DATASETS, progress=prog)
        print(f"  [stage] {ds}: precompute - verifying on 100 random recordings", flush=True)
        rows = np.random.default_rng(0).choice(len(lengths), size=min(100, len(lengths)), replace=False)
        ver = verify_derived(store.root / "hubert.npy", offsets, lengths, D, rows)
        print(f"    verification: {ver}", flush=True)
        if not ver["pass"]:
            raise SystemExit(f"{ds}: derived-feature verification FAILED {ver}")
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "um_frames.npy", D["um_frames"])
        np.save(out / "um_pooled.npy", D["um_pooled"])
        if ds in ZW_DATASETS:
            np.save(out / "layer_sum.npy", D["layer_sum"])
            np.save(out / "layer_sumsq.npy", D["layer_sumsq"])
        meta = {"dataset": ds, "source": str(store.root / "hubert.npy"), "cache_id": store.meta["cache_id"],
                "source_bytes": (store.root / "hubert.npy").stat().st_size, "n_recordings": int(len(lengths)),
                "n_frames": int(lengths.sum()), "seconds": D["seconds"], "bytes_read": D["bytes_read"],
                "read_GBps": D["bytes_read"] / 1e9 / D["seconds"], "verification": ver,
                "files": {n: (out / n).stat().st_size for n in names if n != "meta.json"},
                "layer_sums": ds in ZW_DATASETS, "process_ram_mib": process_ram_mib()}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"  [stage] {ds}: precompute done in {D['seconds']:.0f}s; files {meta['files']}", flush=True)


if __name__ == "__main__":
    main()
