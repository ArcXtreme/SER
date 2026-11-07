"""
E2 step 1: confirm the HuBERT cache facts E2 relies on (read-only).

  C1 frame representation shape (T, 13, 768) per recording, fp16
  C2 index 0 = Transformer input, 1..12 = layer outputs (Stage 4 manual recomputation, error 0.0)
  C3 masks / alignment: HuBERT, MFCC and prosody have identical frame counts per recording;
     masked pooling of a zero-padded batch == the pooled cache (mean | population std), and is
     unchanged when padding is filled with 1e4 (padding invariance)
  C4 cache = approved Praat configuration (octave-jump cost 0.7), same as the E1 cache
  C5 layer extraction: frames(i)[:, l] is the l-th hidden state (mean of it == pooled[i, l, :768])

Run: python scripts/stage5_e2_inspect.py   -> results/stage5_e2_inspect.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.features.batching import masked_mean, masked_std, pad_sequences  # noqa: E402
from src.features.store import FeatureStore  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402


def main():
    index = pd.read_csv(PATHS.metadata / "index.csv")
    praat = json.loads((PATHS.configs / "feature_cache_praat_v1_efe9081789.json").read_text())
    sem = json.loads((PATHS.results / "stage4_hubert_checks.json").read_text())["semantics_max_abs_err"]
    out = {"C2_hidden_state_semantics_max_abs_err": sem,
           "C2_ok": all(v == 0.0 for k, v in sem.items() if k.startswith("hs"))}
    rng = np.random.default_rng(0)
    for ds in ("RAVDESS", "CREMA-D"):
        s = FeatureStore.open(ds, index)
        m = s.manifest
        r = {}
        hub = s.array("hubert")
        r["C1_cache_shape"], r["C1_dtype"] = list(hub.shape), str(hub.dtype)
        ids = rng.choice(len(m), 16, replace=False)
        r["C1_per_recording_shapes_ok"] = all(s.frames("hubert", i).shape == (m.n_frames[i], 13, 768) for i in ids)
        r["C3_frame_counts_equal_across_streams"] = all(
            len(s.frames("hubert", i)) == len(s.frames("mfcc", i)) == len(s.frames("prosody_frames", i)) == m.n_frames[i]
            for i in ids)
        pooled = s.array("hubert_pooled")
        rel = {}
        for layer in (1, 9, 12):
            x, mask = pad_sequences([np.asarray(s.frames("hubert", i)[:, layer], np.float32) for i in ids])
            mu, sd = masked_mean(x, mask), masked_std(x, mask)
            ref = torch.from_numpy(np.asarray(pooled[ids, layer], np.float32))
            rel[layer] = float((torch.cat([mu, sd], 1) - ref).abs().max() / ref.abs().max())
            x2 = x.clone(); x2[~mask] = 1e4
            assert torch.equal(masked_mean(x2, mask), mu) and torch.allclose(masked_std(x2, mask), sd)
            # C5: layer slice semantics
            i0 = int(ids[0])
            assert np.allclose(np.asarray(s.frames("hubert", i0)[:, layer], np.float32).mean(0),
                               pooled[i0, layer, :768], rtol=1e-2, atol=2e-3)
        r["C3_masked_pooling_vs_pooled_cache_max_rel_diff(fp16 frames)"] = rel
        r["C3_padding_invariance"] = True
        r["C3_padded_batch_sizes"] = {"lengths": m.n_frames[ids].tolist(), "mask_true": int(mask.sum())}
        r["C4_cache_id"] = s.meta["cache_id"]
        r["C4_matches_approved_praat_record"] = (s.meta["config_hash"] == praat[ds]["config_hash"])
        r["C4_octave_jump_cost"] = s.meta["config"]["prosody"]["praat"]["octave_jump_cost"]
        r["C5_layer_slice_ok"] = True
        out[ds] = r
    print(json.dumps(out, indent=1))
    (PATHS.results / "stage5_e2_inspect.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
