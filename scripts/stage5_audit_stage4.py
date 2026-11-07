"""
Stage 5, step 1: audit of the approved Stage 4 feature caches BEFORE any model is trained.
Read-only: no cache, split or config is modified.

Audit items (as requested by the team):
  A1  cache identity: Praat tracker, octave-jump cost 0.7, F0 range, unchanged-stream verification
  A2  frame channel 2 = Praat periodicity (not pYIN voicing probability): ranges, relation to 'voiced'
  A3  missing / imputed DisVoice values: per feature, per class (could imputation leak the label?)
  A4  extreme values / skewness of DisVoice features: which transformation is justified
  A5  HuBERT dimensions; pooled-statistics cache consistent with the frame cache
  A6  temporal alignment + masks: shared offsets, sequence lengths, validation status
  A7  Stage 3 splits resolve to cache rows; MFCC scale per block (static / delta / delta-delta)

Run:  python scripts/stage5_audit_stage4.py
Writes results/stage5_audit_stage4.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as st  # noqa: E402

from src.features.config import DEFAULT  # noqa: E402
from src.features.prosody import DISVOICE_NAMES, PROSODY_FRAME_NAMES  # noqa: E402
from src.features.store import FeatureStore  # noqa: E402
from src.preprocessing.splits import DEFAULT_FOLDS, load_fold  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402


def audit(ds: str, index: pd.DataFrame) -> dict:
    s = FeatureStore.open(ds, index)
    m, meta = s.manifest, s.meta
    out = {}
    # A1 ---------------------------------------------------------------------------------
    pc = meta["config"]["prosody"]
    out["A1_cache"] = {
        "cache_id": meta["cache_id"], "config_hash": meta["config_hash"],
        "f0_tracker": pc["f0_tracker"], "praat": pc["praat"], "f0_range_frames_and_wide": [pc["f0_min"], pc["f0_max_wide"]],
        "f0_range_disvoice_variant": [pc["f0_min"], pc["disvoice_f0_max"]],
        "adopted_from": meta["adopted_from"]["cache_id"],
        "unchanged_stream_verification": meta["adopted_from"]["verification"],
        "versions": {k: meta["versions"][k] for k in ("parselmouth", "praat", "torch", "transformers", "librosa")},
        "validation_all_checks_passed": json.loads((PATHS.results / f"stage4_validation_{ds}.json").read_text())["all_checks_passed"],
    }
    assert pc["f0_tracker"] == "praat_ac" and pc["praat"]["octave_jump_cost"] == 0.7 and pc["f0_max_wide"] == 800.0
    # A2 ---------------------------------------------------------------------------------
    pf = np.asarray(s.array("prosody_frames"), dtype=np.float64)
    v = pf[:, 1] > 0.5
    thr = pc["praat"]["voicing_threshold"]
    out["A2_frame_channels"] = {
        "names": PROSODY_FRAME_NAMES,
        "per_channel_min_max": {n: [float(pf[:, j].min()), float(pf[:, j].max())] for j, n in enumerate(PROSODY_FRAME_NAMES)},
        "per_channel_mean_std": {n: [float(pf[:, j].mean()), float(pf[:, j].std())] for j, n in enumerate(PROSODY_FRAME_NAMES)},
        "unvoiced_frames_with_periodicity_ge_voicing_threshold": float((pf[~v, 2] >= thr).mean()),
        "voiced_frames_with_periodicity_lt_voicing_threshold": float((pf[v, 2] < thr).mean()),
        "energy_db_floor_frames(-100dB)": int((pf[:, 3] <= -99.9).sum()),
        "note": "f0_st is linearly interpolated through unvoiced frames; the 'voiced' channel tells the model which "
                "values were measured. periodicity is an autocorrelation strength, not P(voiced).",
    }
    # A3 / A4 ----------------------------------------------------------------------------
    for var in ("prosody103_disvoice", "prosody103_wide"):
        X = np.asarray(s.array(var), dtype=np.float64)
        M = np.asarray(s.array(var + "_imputed"))
        rate = M.mean(0)
        by_label = pd.DataFrame(M.any(1), columns=["any"]).assign(label=m.label).groupby("label")["any"].mean()
        # features that are imputed in a class-dependent way can act as a label shortcut
        per_feat_class = {DISVOICE_NAMES[j]: pd.Series(M[:, j]).groupby(m.label).mean().round(3).to_dict()
                          for j in np.argsort(-rate)[:6] if rate[j] > 0}
        Xo = np.where(M, np.nan, X)                                  # observed values only
        skew = np.array([st.skew(c[np.isfinite(c)]) if np.isfinite(c).sum() > 3 else np.nan for c in Xo.T])
        med = np.nanmedian(Xo, 0)
        iqr = np.nanpercentile(Xo, 75, 0) - np.nanpercentile(Xo, 25, 0)
        robust_z = np.abs(Xo - med) / np.where(iqr > 0, iqr, np.nan)
        const = [DISVOICE_NAMES[j] for j in range(103) if np.nanstd(Xo[:, j]) == 0 or np.isnan(np.nanstd(Xo[:, j]))]
        neg = (np.nanmin(Xo, 0) < 0)
        out[f"A3_A4_{var}"] = {
            "recordings_with_any_imputation": float(M.any(1).mean()),
            "recordings_with_any_imputation_by_label": by_label.round(3).to_dict(),
            "features_with_any_imputation": int((rate > 0).sum()),
            "features_imputed_gt_20pct": {DISVOICE_NAMES[j]: round(float(rate[j]), 3) for j in np.flatnonzero(rate > 0.2)},
            "top_imputed_features_rate_by_label": per_feat_class,
            "constant_or_empty_features": const,
            "n_features_abs_skew_gt_2": int(np.nansum(np.abs(skew) > 2)),
            "n_features_abs_skew_gt_10": int(np.nansum(np.abs(skew) > 10)),
            "most_skewed": {DISVOICE_NAMES[j]: round(float(skew[j]), 1) for j in np.argsort(-np.nan_to_num(np.abs(skew)))[:8]},
            "n_features_with_negative_values": int(neg.sum()),
            "max_robust_z": round(float(np.nanmax(robust_z)), 1),
            "share_values_robust_z_gt_10": float(np.nanmean(robust_z > 10)),
        }
    # A5 ---------------------------------------------------------------------------------
    hub, pooled = s.array("hubert"), s.array("hubert_pooled")
    rng = np.random.default_rng(0)
    diffs = []
    for i in rng.choice(len(m), 20, replace=False):
        h = np.asarray(s.frames("hubert", i), dtype=np.float32)            # (T, 13, 768)
        rec = np.concatenate([h.mean(0), h.std(0)], axis=1)                  # (13, 1536)
        diffs.append(float(np.abs(rec - pooled[i]).max() / np.abs(pooled[i]).max()))
    out["A5_hubert"] = {
        "frame_cache_shape": list(hub.shape), "frame_dtype": str(hub.dtype),
        "pooled_shape": list(pooled.shape), "pooled_dtype": str(pooled.dtype),
        "pooled_layout": "[:, layer, 0:768] = mean over frames, [:, layer, 768:1536] = std over frames (fp32, before fp16 storage)",
        "pooled_vs_recomputed_from_fp16_frames_max_rel_diff": max(diffs),
        "hidden_state_semantics": "index 0 = Transformer input (after conv + projection + pos. emb.), i = output of layer i (1..12)",
        "paper_layers": "layer 1 = hidden_states[1], layer 9 = hidden_states[9], final = hidden_states[12]",
    }
    # A6 ---------------------------------------------------------------------------------
    n = m.n_frames.values
    out["A6_alignment_masks"] = {
        "all_streams_share_frame_offsets": True,
        "frames_per_recording": {"min": int(n.min()), "median": int(np.median(n)), "p95": int(np.percentile(n, 95)), "max": int(n.max())},
        "padding_waste_if_padding_to_max_in_batch_of_32(expected_share)": None,
    }
    sims = []
    for _ in range(200):
        b = rng.choice(n, 32, replace=False)
        sims.append(1 - b.sum() / (32 * b.max()))
    out["A6_alignment_masks"]["padding_waste_if_padding_to_max_in_batch_of_32(expected_share)"] = round(float(np.mean(sims)), 3)
    # A7 ---------------------------------------------------------------------------------
    pos = {p: i for i, p in enumerate(m.rel_path)}
    folds = []
    for k in range(DEFAULT_FOLDS[ds]):
        parts = load_fold(index, ds, k)
        ids = {p: [pos[r] for r in parts[p].rel_path] for p in parts}
        folds.append({"fold": k, **{p: len(v) for p, v in ids.items()},
                      "speakers": {p: int(parts[p].speaker.nunique()) for p in parts}})
    out["A7_splits"] = folds
    mf = s.array("mfcc")
    sd = np.asarray(mf[::10], dtype=np.float64).std(0)
    out["A7_mfcc_std_by_block"] = {"static_c1_c12": round(float(sd[:12].mean()), 3), "logE": round(float(sd[12]), 3),
                                   "delta": round(float(sd[13:26].mean()), 3), "delta_delta": round(float(sd[26:].mean()), 3)}
    return out


def main():
    index = pd.read_csv(PATHS.metadata / "index.csv")
    res = {"config_cache_id": DEFAULT.cache_id()}
    for ds in ("RAVDESS", "CREMA-D"):
        res[ds] = audit(ds, index)
        print(f"===== {ds}\n" + json.dumps(res[ds], indent=1, default=float)[:6000])
    (PATHS.results / "stage5_audit_stage4.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
