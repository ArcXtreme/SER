"""
Stage 4 documentation: pYIN (previous tracker) vs Praat (current tracker) on the SAME
300 random recordings per corpus that motivated the switch (rng = default_rng(0), RAVDESS first).

  pYIN  : recomputed exactly as the previous pipeline did (60-700 Hz, 1024-sample window,
          hop 320 on y[200:]); checked bit-identical against the previous cache
          (v1_f9418feedb) when that cache is still on disk.
  Praat : read from the CURRENT feature cache (Praat ac, 60-800 Hz, exact grid).

The pre-switch version of this comparison (Praat 60-700 Hz interpolated at frame centres
with get_value_at_time) is archived as results/stage4_f0_tracker_comparison_pyin_v1_f9418feedb.json.

No feature cache is modified.
Run:  python scripts/stage4_f0_tracker_comparison.py [--n 300]
Writes results/stage4_f0_tracker_comparison.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.config import DEFAULT, HOP, WIN, n_frames  # noqa: E402
from src.features.prosody import PYIN_FRAME_LENGTH, f0_pyin  # noqa: E402
from src.features.store import FeatureStore  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402

PREVIOUS_CACHE = "v1_f9418feedb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    rng = np.random.default_rng(0)
    out = {"pyin": "librosa.pyin 60-700 Hz, frame 1024, hop 320, centred on the HuBERT grid (previous pipeline)",
           "praat": f"Praat ac {DEFAULT.prosody.f0_min:g}-{DEFAULT.prosody.f0_max_wide:g} Hz, defaults, "
                    f"current cache {DEFAULT.cache_id()}"}
    for ds in ("RAVDESS", "CREMA-D"):
        s = FeatureStore.open(ds, index)
        m = s.manifest
        old = PATHS.features / ds / PREVIOUS_CACHE / "prosody_frames.npy"
        old_pf = np.load(old, mmap_mode="r") if old.exists() else None
        rows, n_old_match, v_all, p_all = [], 0, [], []
        for i in rng.choice(len(m), min(args.n, len(m)), replace=False):
            y = np.asarray(s.wav(i))
            T = n_frames(len(y))
            f_py, vprob = f0_pyin(y, 60, 700, hop=HOP, frame_length=PYIN_FRAME_LENGTH, start=WIN // 2)
            f_py, vprob = f_py[:T], vprob[:T]
            if old_pf is not None:
                a = int(m.frame_offset[i])
                n_old_match += bool(np.array_equal((f_py > 0).astype(np.float32), old_pf[a:a + T, 1])
                                    and np.allclose(vprob, old_pf[a:a + T, 2], atol=1e-6))
            pf = s.frames("prosody_frames", i)
            v_pr = pf[:, 1] > 0.5
            f_pr = np.where(v_pr, 100 * 2 ** (pf[:, 0] / 12), 0)
            v_py = f_py > 0
            both = v_py & v_pr
            st_diff = np.abs(12 * np.log2(f_pr[both] / f_py[both])) if both.any() else np.array([])
            v_all.append(vprob); p_all.append(pf[:, 2])
            rows.append(dict(label=m.label[i], gender=m.gender[i],
                             vfrac_pyin=v_py.mean(), vfrac_praat=v_pr.mean(),
                             zero_pyin=not v_py.any(), zero_praat=not v_pr.any(),
                             lt10_pyin=v_py.mean() < 0.1, lt10_praat=v_pr.mean() < 0.1,
                             agree_voicing=(v_py == v_pr).mean(),
                             median_st_diff=float(np.median(st_diff)) if len(st_diff) else np.nan,
                             gross_err=float((st_diff > 12 * np.log2(1.2)).mean()) if len(st_diff) else np.nan,
                             octave_err=float((st_diff > 6).mean()) if len(st_diff) else np.nan,
                             above350_praat=float((f_pr[v_pr] > 350).mean()) if v_pr.any() else np.nan,
                             above350_pyin=float((f_py[v_py] > 350).mean()) if v_py.any() else np.nan))
        r = pd.DataFrame(rows)
        v_all, p_all = np.concatenate(v_all), np.concatenate(p_all)
        out[ds] = {
            "n": len(r),
            "pyin_recomputation_matches_previous_cache": (f"{n_old_match}/{len(r)}" if old_pf is not None
                                                          else "previous cache not on disk"),
            "voiced_fraction_mean": {"pyin": r.vfrac_pyin.mean(), "praat": r.vfrac_praat.mean()},
            "recordings_without_voicing": {"pyin": int(r.zero_pyin.sum()), "praat": int(r.zero_praat.sum())},
            "recordings_with_lt_10pct_voiced": {"pyin": int(r.lt10_pyin.sum()), "praat": int(r.lt10_praat.sum())},
            "voicing_agreement_mean": r.agree_voicing.mean(),
            "f0_median_abs_semitone_diff_on_frames_voiced_in_both": r.median_st_diff.median(),
            "share_of_both_voiced_frames_differing_gt_20pct_mean": r.gross_err.mean(),
            "share_of_both_voiced_frames_differing_gt_half_octave_mean": r.octave_err.mean(),
            "share_voiced_frames_above_350Hz_mean": {"pyin": r.above350_pyin.mean(), "praat": r.above350_praat.mean()},
            "voiced_fraction_by_label": r.groupby("label")[["vfrac_pyin", "vfrac_praat"]].mean().round(3).to_dict(),
            "voiced_fraction_by_gender": r.groupby("gender")[["vfrac_pyin", "vfrac_praat"]].mean().round(3).to_dict(),
            "no_voicing_by_label": r.groupby("label")[["zero_pyin", "zero_praat"]].sum().astype(int).to_dict(),
            # The 3rd frame channel changed meaning: pYIN voicing probability -> Praat periodicity.
            "channel2_pyin_vprob_vs_praat_periodicity": {
                "pearson_r": float(np.corrcoef(v_all, p_all)[0, 1]),
                "mean_abs_diff": float(np.abs(v_all - p_all).mean()),
                "mean": {"pyin_vprob": float(v_all.mean()), "praat_periodicity": float(p_all.mean())}},
        }
        print(ds, json.dumps(out[ds], indent=1, default=float))
    (PATHS.results / "stage4_f0_tracker_comparison.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
