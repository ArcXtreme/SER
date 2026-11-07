"""
Stage 4 validation: verifies the feature cache and produces the Stage 4 figures.

Run:  C:\\SER_data\\.venv\\Scripts\\python.exe scripts/stage4_validate_features.py --dataset RAVDESS
Writes results/stage4_validation_<DS>.json and plots/features_*.png.

Checks
  1  completeness        every usable recording present, all progress flags set
  2  finiteness          no NaN / Inf in any stored array
  3  dimensions          array shapes, frame counts == 1 + (N - 400) // 320, offset table consistent
  4  alignment (exact)   prosody energy_db == 10/ln10 * MFCC logE - 10 log10(400) on every frame
  5  alignment (HuBERT)  lag of max correlation between HuBERT-predicted energy and MFCC energy
  6  masks               padded batch masks identical across streams and equal to frame counts
  7  reproducibility     re-extract random recordings from the raw audio and compare
  8  resumability        wipe some recordings' progress, resume, compare with saved values
  9  DisVoice oracle     our 103 features vs original DisVoice functions on real recordings
 10  trimming            fraction of signal energy removed by silence trimming
 11  prosody analysis    imputation counts, value ranges, share of voiced frames > 350 Hz
 12  HuBERT layers       per-layer norms, linear CKA similarity between layers (label-free)
 13  voicing / pitch     voiced-frame statistics, recordings without voicing, class- and
                         gender-wise voicing, F0 distribution, periodicity channel ranges
 14  unchanged streams   (caches adopted from an older cache) SHA-256 of audio, manifest,
                         HuBERT, pooled HuBERT and MFCC still equal to the source cache,
                         AFTER the resume test (which restores the original HuBERT bytes)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from src.features.batching import pad_sequences  # noqa: E402
from src.features.config import DEFAULT, frame_centers_s, n_frames  # noqa: E402
from src.features.hubert import HubertExtractor  # noqa: E402
from src.features.mfcc import MFCC_NAMES, MFCCExtractor  # noqa: E402
from src.features.prosody import (DISVOICE_NAMES, PROSODY_FRAME_NAMES, f0_praat,  # noqa: E402
                                  disvoice_static, prosody103, prosody_frames)
from src.features.store import FeatureStore, stage_cpu, stage_hubert, verify_adopted  # noqa: E402
from src.preprocessing.audio import load_mono, preprocess  # noqa: E402
from src.preprocessing.dataset_index import CLASSES  # noqa: E402
from src.preprocessing.splits import usable_files  # noqa: E402
from src.utils import plot_style as ps  # noqa: E402
from src.utils.paths import PATHS  # noqa: E402

RNG = np.random.default_rng(0)


def check(cond, msg, report):
    report.setdefault("checks", {})[msg] = bool(cond)
    print(("PASS " if cond else "FAIL ") + msg)
    return cond


# ------------------------------------------------------------------------------ helpers
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment (Kornblith et al., 2019): similarity of two
    representations of the same frames, invariant to rotation and isotropic scaling."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num = np.linalg.norm(Y.T @ X) ** 2
    return float(num / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def lag_correlation(pred: list[np.ndarray], true: list[np.ndarray], max_lag: int = 5) -> dict:
    """Mean within-utterance Pearson r between pred[t] and true[t + lag]."""
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        rs = []
        for p, t in zip(pred, true):
            a, b = (p[:len(p) - lag], t[lag:]) if lag >= 0 else (p[-lag:], t[:len(t) + lag])
            if len(a) > 10 and a.std() > 0 and b.std() > 0:
                rs.append(np.corrcoef(a, b)[0, 1])
        out[lag] = float(np.mean(rs))
    return out


def collect_voicing(store, man) -> dict:
    """Per-recording voicing counts and pooled F0 / periodicity values from the cached contours."""
    rows, f0_by_gender, per_v, per_u = [], {"M": [], "F": []}, [], []
    fmax = DEFAULT.prosody.f0_max_wide
    for i in range(len(man)):
        f = store.frames("prosody_frames", i)
        v = f[:, 1] > 0.5
        f0 = 100 * 2 ** (f[v, 0] / 12)
        f0_by_gender[man.gender[i]].append(f0)
        per_v.append(f[v, 2]); per_u.append(f[~v, 2])
        rows.append(dict(gender=man.gender[i], label=man.label[i], frames=len(f), voiced=int(v.sum()),
                         above350=int((f0 > 350).sum()), near_ceiling=int((f0 > 0.9 * fmax).sum())))
    vr = pd.DataFrame(rows)
    vr["vfrac"] = vr.voiced / vr.frames
    vr["none"] = vr.voiced == 0
    return {"vr": vr, "allf0": {g: np.concatenate(v) if v else np.array([]) for g, v in f0_by_gender.items()},
            "per_v": np.concatenate(per_v), "per_u": np.concatenate(per_u)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="RAVDESS")
    ap.add_argument("--n-reextract", type=int, default=20)
    ap.add_argument("--skip-resume-test", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="only redraw the Stage 4 figures from the cache + saved validation JSON (no checks)")
    args = ap.parse_args()
    ds = args.dataset
    ps.apply()
    index = pd.read_csv(PATHS.metadata / "index.csv")
    files = usable_files(index, ds)
    store = FeatureStore.open(ds, index)                    # raises on any cache inconsistency
    man = store.manifest
    if args.figures_only:
        R = json.loads((PATHS.results / f"stage4_validation_{ds}.json").read_text())
        if R["cache_id"] != store.meta["cache_id"]:
            raise SystemExit(f"saved validation is for {R['cache_id']}, cache is {store.meta['cache_id']}")
        vdat = collect_voicing(store, man)
        above = vdat["vr"].groupby(["gender", "label"])[["voiced", "above350"]].sum()
        above["share_above_350Hz"] = (above.above350 / above.voiced).round(4)
        lag_res = {int(k.removeprefix("layer")): {int(l): r for l, r in d.items()}
                   for k, d in R["hubert_mfcc_energy_lag_correlation"].items()}
        make_figures(store, man, R, np.array(R["hubert_cka"]), lag_res, above, ds)
        voicing_figure(vdat, ds)
        return
    n, T = len(man), int(man.n_frames.sum())
    R: dict = {"dataset": ds, "cache_dir": str(store.root), "cache_id": store.meta["cache_id"]}

    # 1 completeness -------------------------------------------------------------------
    check(man.rel_path.tolist() == files.rel_path.tolist(), f"1 all {len(files)} usable recordings in manifest (same order)", R)
    for s in ("cpu", "hubert"):
        check(np.load(store.root / f"progress_{s}.npy").all(), f"1 progress_{s}: every recording processed", R)
    R["n_recordings"], R["n_frames_total"] = n, T

    # 2 finiteness (chunked for the 2 GB HuBERT array) ---------------------------------
    for name in ["wav16k", "mfcc", "prosody_frames", "hubert_pooled", "prosody103_disvoice", "prosody103_wide"]:
        check(np.isfinite(store.array(name)).all(), f"2 {name}: all values finite", R)
    hub = store.array("hubert")
    finite, absmax = True, 0.0
    for a in range(0, T, 20000):
        blk = np.asarray(hub[a:a + 20000], dtype=np.float32)
        finite &= bool(np.isfinite(blk).all()); absmax = max(absmax, float(np.abs(blk).max()))
    check(finite, "2 hubert: all values finite", R)
    R["hubert_abs_max"] = absmax

    # 3 dimensions ---------------------------------------------------------------------
    check(all(man.n_frames == [n_frames(s) for s in man.n_samples]), "3 n_frames == 1 + (N-400)//320 for every recording", R)
    check((np.diff(man.frame_offset) == man.n_frames[:-1].values).all() and man.frame_offset.iloc[-1] + man.n_frames.iloc[-1] == T,
          "3 frame-offset table contiguous and complete", R)
    check((np.diff(man.wav_offset) == man.n_samples[:-1].values).all(), "3 wav-offset table contiguous", R)
    R["shapes"] = {k: list(store.array(k).shape) for k in
                   ["wav16k", "mfcc", "prosody_frames", "hubert", "hubert_pooled", "prosody103_disvoice", "prosody103_wide"]}
    R["dtypes"] = {k: str(store.array(k).dtype) for k in R["shapes"]}

    # 4 exact energy identity between MFCC and prosody streams --------------------------
    mf, pf = store.array("mfcc"), store.array("prosody_frames")
    pred_db = 10 / np.log(10) * mf[:, 12].astype(np.float64) - 10 * np.log10(400)
    err = np.abs(pred_db - pf[:, 3])
    valid = pf[:, 3] > -99                                  # floor-clipped frames excluded
    R["energy_identity_max_abs_err_db"] = float(err[valid].max())
    check(err[valid].max() < 1e-3, "4 prosody energy_db == affine(MFCC logE) on every frame (same samples)", R)

    # 5 HuBERT <-> MFCC alignment by lagged correlation ---------------------------------
    ids = RNG.permutation(n)
    fit_ids, test_ids = ids[:300], ids[300:600]
    lag_res = {}
    for layer in (0, 9):
        X = np.concatenate([store.frames("hubert", i)[:, layer].astype(np.float32) for i in fit_ids])
        y = np.concatenate([store.frames("mfcc", i)[:, 12] for i in fit_ids])
        Xm, ym = X.mean(0), y.mean()
        W = np.linalg.solve((X - Xm).T @ (X - Xm) + 10.0 * np.eye(768), (X - Xm).T @ (y - ym))
        pred = [(store.frames("hubert", i)[:, layer].astype(np.float32) - Xm) @ W for i in test_ids]
        true = [store.frames("mfcc", i)[:, 12] for i in test_ids]
        lag_res[layer] = lag_correlation(pred, true)
    R["hubert_mfcc_energy_lag_correlation"] = {f"layer{k}": v for k, v in lag_res.items()}
    for layer, d in lag_res.items():
        check(max(d, key=d.get) == 0, f"5 HuBERT layer {layer} -> MFCC energy: correlation peaks at lag 0 "
                                      f"(r={d[0]:.3f}, r(+-1)={d[1]:.3f}/{d[-1]:.3f})", R)

    # 6 masks ---------------------------------------------------------------------------
    b_ids = RNG.choice(n, 16, replace=False)
    streams = {k: [store.frames(k, i) for i in b_ids] for k in ("hubert", "mfcc", "prosody_frames")}
    padded = {k: pad_sequences([np.asarray(s, dtype=np.float32) for s in v]) for k, v in streams.items()}
    masks = [m for _, m in padded.values()]
    check(all(torch.equal(masks[0], m) for m in masks[1:]), "6 padding masks identical across HuBERT/MFCC/prosody", R)
    check(masks[0].sum(1).tolist() == man.n_frames.iloc[b_ids].tolist(), "6 mask sums equal recording frame counts", R)
    check(all((x[~m] == 0).all() for x, m in padded.values()), "6 padded positions are zero", R)

    # 7 reproducibility from raw audio --------------------------------------------------
    ex = HubertExtractor(DEFAULT.hubert)
    mfx = MFCCExtractor(DEFAULT.mfcc)
    rep = {"wav": 0.0, "mfcc": 0.0, "prosody_frames": 0.0, "p103_disvoice": 0.0, "hubert_rel": 0.0}
    for i in RNG.choice(n, args.n_reextract, replace=False):
        y, _ = preprocess(PATHS.raw / man.rel_path[i], DEFAULT.audio)
        rep["wav"] = max(rep["wav"], float(np.abs(y - store.wav(i)).max()))
        rep["mfcc"] = max(rep["mfcc"], float(np.abs(mfx(y) - store.frames("mfcc", i)).max()))
        rep["prosody_frames"] = max(rep["prosody_frames"], float(np.abs(
            prosody_frames(y, DEFAULT.prosody, man.n_frames[i]) - store.frames("prosody_frames", i)).max()))
        p = np.nan_to_num(prosody103(y, DEFAULT.prosody, DEFAULT.prosody.disvoice_f0_max), nan=0, posinf=0, neginf=0)
        rep["p103_disvoice"] = max(rep["p103_disvoice"], float(np.abs(p.astype(np.float32) - store.array("prosody103_disvoice")[i]).max()))
        h = ex(y).permute(1, 0, 2).cpu().numpy()
        c = store.frames("hubert", i).astype(np.float32)
        rep["hubert_rel"] = max(rep["hubert_rel"], float(np.abs(h - c).max() / np.abs(h).max()))
    R["reextraction_max_abs_diff"] = rep
    check(rep["wav"] == 0 and rep["mfcc"] == 0 and rep["prosody_frames"] == 0 and rep["p103_disvoice"] == 0,
          f"7 re-extraction of {args.n_reextract} recordings: audio, MFCC, prosody bit-identical", R)
    check(rep["hubert_rel"] < 2e-3, f"7 re-extraction: HuBERT within fp16 storage tolerance (max rel {rep['hubert_rel']:.1e})", R)
    del ex
    torch.cuda.empty_cache()

    # 8 resumability --------------------------------------------------------------------
    if not args.skip_resume_test:
        rid = RNG.choice(n, 25, replace=False)
        saved = {k: [np.array(store.frames(k, i)) for i in rid] for k in ("mfcc", "prosody_frames", "hubert")}
        saved_p = np.array(store.array("prosody103_wide")[rid])
        saved_pooled = np.array(store.array("hubert_pooled")[rid])
        for s, arrs in (("cpu", ("mfcc", "prosody_frames")), ("hubert", ("hubert",))):
            prog = np.load(store.root / f"progress_{s}.npy", mmap_mode="r+")
            for k in arrs:
                a = np.load(store.root / f"{k}.npy", mmap_mode="r+")
                for i in rid:
                    a[man.frame_offset[i]: man.frame_offset[i] + man.n_frames[i]] = 0
                a.flush()
            prog[rid] = False
            prog.flush()
            del prog
        stage_cpu(ds); stage_hubert(ds)
        store = FeatureStore.open(ds, index)
        same = all(np.array_equal(saved[k][j], store.frames(k, i)) for k in ("mfcc", "prosody_frames") for j, i in enumerate(rid))
        hub_diff = max(float(np.abs(saved["hubert"][j].astype(np.float32) - store.frames("hubert", i).astype(np.float32)).max())
                       for j, i in enumerate(rid))
        R["resume_test"] = {"n_wiped": 25, "cpu_bit_identical": same, "hubert_max_abs_diff": hub_diff,
                            "p103_identical": bool(np.array_equal(saved_p, store.array("prosody103_wide")[rid]))}
        check(same and hub_diff < 0.02 and R["resume_test"]["p103_identical"],
              f"8 resume after wiping 25 recordings restores the same features (HuBERT max diff {hub_diff:.1e})", R)
        # GPU inference is not bit-reproducible: put the ORIGINAL HuBERT bytes back so the cache
        # stays byte-identical to what was extracted (and, if adopted, to its source; check 14).
        hub_w = np.load(store.root / "hubert.npy", mmap_mode="r+")
        pool_w = np.load(store.root / "hubert_pooled.npy", mmap_mode="r+")
        for j, i in enumerate(rid):
            hub_w[man.frame_offset[i]: man.frame_offset[i] + man.n_frames[i]] = saved["hubert"][j]
            pool_w[i] = saved_pooled[j]
        hub_w.flush(); pool_w.flush()
        del hub_w, pool_w

    # 9 DisVoice oracle on real recordings -----------------------------------------------
    from tests.reference import disvoice_prosody_functions as ref
    mism, n_or = [], 30
    for i in RNG.choice(n, n_or, replace=False):
        y = np.asarray(store.wav(i), dtype=np.float64)
        F0, _ = f0_praat(y, 1 + len(y) // 160, 160, 0, DEFAULT.prosody.f0_min,
                         DEFAULT.prosody.disvoice_f0_max, DEFAULT.prosody.praat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segV = ref.V_UV(F0, y, "Voiced", 160.0); segUP = ref.V_UV(F0, y, "Unvoiced", 160.0)
            segP = [s for s in segUP if len(s) > 0.14 * 16000]; segU = [s for s in segUP if len(s) <= 0.14 * 16000]
            try:
                exp = np.hstack((ref.F0feat(F0), ref.energy_feat(segV, 16000, 320.0, 160.0),
                                 ref.energy_feat(segU, 16000, 320.0, 160.0), ref.duration_feat(segV, segU, segP, y, 16000)))
            except Exception as e:                           # original code can crash on edge cases
                mism.append((man.rel_path[i], f"original DisVoice raised {e!r}")); continue
            ours = disvoice_static(y, F0, DEFAULT.prosody)
        bad = ~np.isclose(ours, exp, rtol=1e-6, atol=1e-8, equal_nan=True)
        if bad.any():
            mism.append((man.rel_path[i], [DISVOICE_NAMES[k] for k in np.flatnonzero(bad)]))
    R["disvoice_oracle_real"] = {"n": n_or, "mismatches": mism}
    check(not mism, f"9 DisVoice-103 equals original DisVoice code on {n_or} real recordings (same F0 input)", R)

    # 10 trimming: energy removed --------------------------------------------------------
    removed = []
    for i in range(n):
        y, sr, _ = load_mono(PATHS.raw / man.rel_path[i])
        import librosa
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type=DEFAULT.audio.resample_type)
        y = y - y.mean()
        e = y.astype(np.float64) ** 2
        removed.append(1 - e[man.trim_start[i]: man.trim_end[i]].sum() / e.sum())
    removed = np.asarray(removed)
    man["energy_removed_by_trim"] = removed
    R["trimming"] = {"duration_s_before": man.n_samples_16k_untrimmed.describe().drop("count").div(16000).round(3).to_dict(),
                     "duration_s_after": man.n_samples.describe().drop("count").div(16000).round(3).to_dict(),
                     "energy_removed_frac_median": float(np.median(removed)),
                     "energy_removed_frac_p99": float(np.quantile(removed, 0.99)),
                     "energy_removed_frac_max": float(removed.max()),
                     "files_removing_more_than_1pct": man.loc[removed > 0.01, "rel_path"].tolist()}
    check(np.quantile(removed, 0.99) < 0.01, "10 trimming removes < 1 % of signal energy for 99 % of recordings", R)

    # 11 prosody analysis ----------------------------------------------------------------
    for k in ("prosody103_disvoice", "prosody103_wide"):
        imp = store.array(k + "_imputed")
        vals = store.array(k)
        R[k] = {"recordings_with_any_imputation": int(imp.any(1).sum()),
                "imputed_values_total": int(imp.sum()),
                "most_imputed_features": {DISVOICE_NAMES[j]: int(imp[:, j].sum()) for j in np.argsort(-imp.sum(0))[:8] if imp[:, j].sum()},
                "abs_max_value": float(np.abs(vals).max()),
                "abs_max_feature": DISVOICE_NAMES[int(np.abs(vals).max(0).argmax())]}
    pf_all = store.array("prosody_frames")
    fmax = DEFAULT.prosody.f0_max_wide
    vdat = collect_voicing(store, man)
    rows = vdat["vr"].to_dict("records")
    above = vdat["vr"].groupby(["gender", "label"])[["voiced", "above350"]].sum()
    above["share_above_350Hz"] = (above.above350 / above.voiced).round(4)
    R["voiced_frames_above_350Hz"] = {f"{g}/{l}": float(v) for (g, l), v in above.share_above_350Hz.items()}
    R["voiced_frames_above_350Hz_overall"] = float(above.above350.sum() / above.voiced.sum())
    R["voiced_fraction_overall"] = float((pf_all[:, 1] > 0.5).mean())
    # Recordings where the tracker finds NO voiced frame (F0 features become 0 by DisVoice convention).
    # Kept in the data (never silently dropped) and listed here for manual listening checks.
    R["f0_all_unvoiced"] = {
        "frame_level": man.rel_path[[r["voiced"] == 0 for r in rows]].tolist(),
        "prosody103_disvoice": man.rel_path[store.array("prosody103_disvoice")[:, 0] == 0].tolist(),
        "prosody103_wide": man.rel_path[store.array("prosody103_wide")[:, 0] == 0].tolist()}

    # 13 voicing / pitch statistics -------------------------------------------------------
    vr, allf0, per_v, per_u = vdat["vr"], vdat["allf0"], vdat["per_v"], vdat["per_u"]
    R["voicing"] = {
        "tracker": f"{DEFAULT.prosody.f0_tracker} {DEFAULT.prosody.f0_min:g}-{fmax:g} Hz",
        "voiced_frame_fraction_overall": round(float(vr.voiced.sum() / vr.frames.sum()), 4),
        "voiced_fraction_per_recording": vr.vfrac.describe().drop("count").round(4).to_dict(),
        "recordings_without_voicing": int(vr.none.sum()),
        "recordings_with_lt_10pct_voiced": int((vr.vfrac < 0.10).sum()),
        "voiced_fraction_by_label": vr.groupby("label").vfrac.mean().round(4).to_dict(),
        "no_voicing_by_label": vr.groupby("label").none.sum().astype(int).to_dict(),
        "voiced_fraction_by_gender": vr.groupby("gender").vfrac.mean().round(4).to_dict(),
        "voiced_fraction_by_gender_label": {f"{g}/{l}": round(float(v), 4)
                                            for (g, l), v in vr.groupby(["gender", "label"]).vfrac.mean().items()},
        "f0_hz_quantiles_by_gender": {g: {q: round(float(np.percentile(a, q)), 1) for q in (1, 5, 25, 50, 75, 95, 99)}
                                      for g, a in allf0.items() if len(a)},
        "voiced_frames_near_ceiling_share": round(float(vr.near_ceiling.sum() / max(vr.voiced.sum(), 1)), 5),
        "periodicity_voiced_quantiles": {q: round(float(np.percentile(per_v, q)), 3) for q in (5, 50, 95)},
        "periodicity_unvoiced_quantiles": {q: round(float(np.percentile(per_u, q)), 3) for q in (5, 50, 95)},
        "periodicity_range": [float(pf_all[:, 2].min()), float(pf_all[:, 2].max())],
    }
    check(0 <= R["voicing"]["periodicity_range"][0] and R["voicing"]["periodicity_range"][1] <= 1,
          "13 periodicity channel within [0, 1]", R)
    check(set(np.unique(pf_all[:, 1])) <= {0.0, 1.0}, "13 voiced channel is binary", R)
    fv = np.concatenate([a for a in allf0.values() if len(a)])
    check(fv.min() >= DEFAULT.prosody.f0_min * 0.99 and fv.max() <= fmax * 1.01,
          f"13 voiced F0 within the configured range {DEFAULT.prosody.f0_min:g}-{fmax:g} Hz "
          f"(observed {fv.min():.1f}-{fv.max():.1f})", R)

    # 14 streams that must be unchanged by the prosody switch ------------------------------
    if "adopted_from" in store.meta:
        try:
            res = verify_adopted(ds)
        except Exception as e:                               # record the failure, never hide it
            res = {"verification_error": False}
            R["unchanged_vs_source_cache_error"] = repr(e)
        R["unchanged_vs_source_cache"] = {"source": store.meta["adopted_from"]["cache_id"], **res}
        check(all(res.values()), "14 audio, manifest, HuBERT, pooled HuBERT and MFCC byte-identical (SHA-256) "
                                 f"to source cache {store.meta['adopted_from']['cache_id']}", R)

    # 12 HuBERT layers (label-free) --------------------------------------------------------
    sel = np.sort(RNG.choice(T, 20000, replace=False))
    H = np.asarray(hub[sel], dtype=np.float32)                       # (20000, 13, 768)
    cka = np.array([[linear_cka(H[:, a], H[:, b]) for b in range(13)] for a in range(13)])
    R["hubert_layer_frame_norm_mean"] = np.linalg.norm(H, axis=2).mean(0).round(3).tolist()
    R["hubert_cka"] = cka.round(4).tolist()
    R["hubert_cka_layer1_vs_layer9"] = float(cka[1, 9])

    # sizes --------------------------------------------------------------------------------
    R["storage_MB"] = {p.name: round(p.stat().st_size / 2**20, 1) for p in sorted(store.root.glob("*.npy"))}
    R["storage_total_MB"] = round(sum(R["storage_MB"].values()), 1)
    R["stages"] = store.meta["stages"]
    R["all_checks_passed"] = all(R["checks"].values())

    out = PATHS.results / f"stage4_validation_{ds}.json"
    out.write_text(json.dumps(R, indent=2, default=str))
    make_figures(store, man, R, cka, lag_res, above, ds)
    voicing_figure(vdat, ds)
    print(f"\n{'ALL CHECKS PASSED' if R['all_checks_passed'] else 'SOME CHECKS FAILED'} -> {out}")


# ================================================================================= figures
def make_figures(store, man, R, cka, lag_res, above, ds):
    # example: one female actor, angry strong, statement "kids"
    ex_i = int(man.index[man.rel_path.str.contains("03-01-05-02-01-01-02.wav")][0]) if ds == "RAVDESS" else 0
    y = np.asarray(store.wav(ex_i)); Tn = int(man.n_frames[ex_i]); tc = frame_centers_s(Tn)
    mf, pf, hb = store.frames("mfcc", ex_i), store.frames("prosody_frames", ex_i), store.frames("hubert", ex_i).astype(np.float32)

    # --- A: alignment on the shared time axis
    fig, ax = plt.subplots(5, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.4, 1.4]})
    t = np.arange(len(y)) / 16000
    ax[0].plot(t, y, color=ps.MUTED, lw=0.5); ax[0].set_ylabel("waveform")
    for k in range(0, Tn, 10):
        ax[0].axvline(tc[k], color=ps.GRID, lw=0.6)
    v = pf[:, 1] > 0.5
    ax[1].plot(tc, 100 * 2 ** (pf[:, 0] / 12), color=ps.AXIS, lw=1.2, label="interpolated")
    ax[1].plot(tc[v], 100 * 2 ** (pf[v, 0] / 12), ".", color=ps.SERIES[0], ms=4, label="voiced frames")
    ax[1].axhline(350, color=ps.SERIES[7], lw=0.8, ls="--"); ax[1].text(tc[0], 360, "DisVoice ceiling 350 Hz", fontsize=7, color=ps.INK_2)
    ax[1].set_ylabel("F0 (Hz)"); ax[1].legend(loc="upper right", fontsize=7)
    ax[2].plot(tc, pf[:, 3], color=ps.SERIES[1], label="prosody energy (dB)")
    ax2 = ax[2].twinx(); ax2.plot(tc, mf[:, 12], color=ps.SERIES[2], lw=1, ls="--", label="MFCC logE"); ax2.grid(False)
    ax[2].set_ylabel("energy (dB)"); ax2.set_ylabel("logE", color=ps.INK_2)
    ax[2].legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    ext = [tc[0] - 0.01, tc[-1] + 0.01]
    ax[3].imshow(mf[:, :13].T, aspect="auto", origin="lower", cmap="RdBu_r", extent=[*ext, -0.5, 12.5],
                 vmin=-np.abs(mf[:, :12]).max(), vmax=np.abs(mf[:, :12]).max())
    ax[3].set_ylabel("MFCC c1..c12, logE"); ax[3].grid(False)
    ax[4].imshow(hb[:, 9, :].T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-2, vmax=2, extent=[*ext, 0, 768])
    ax[4].set_ylabel("HuBERT layer 9\n(768 dims)"); ax[4].set_xlabel("time (s)  -  every stream sampled at the same 20 ms frame centres")
    ax[4].grid(False)
    fig.suptitle(f"Temporal alignment of all streams ({man.rel_path[ex_i].split('/')[-1]}, {man.label[ex_i]}, "
                 f"{Tn} frames)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / f"features_alignment_{ds}.png")

    # --- B: MFCC maps per emotion (same actor/statement/intensity where possible)
    if ds == "RAVDESS":
        # Display only: z-score each of the 39 dimensions with dataset-wide mean/std, because
        # deltas are ~10x smaller than static coefficients and would otherwise look blank.
        allm = store.array("mfcc")
        mu, sd = allm.mean(0), allm.std(0) + 1e-8
        fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharey=True)
        for ax_, (c, code) in zip(axes.flat, zip(CLASSES, ["01", "03", "04", "05", "06", "07"])):
            inten = "01" if c == "neutral" else "02"
            i = int(man.index[man.rel_path.str.endswith(f"03-01-{code}-{inten}-01-01-02.wav")][0])
            m = (store.frames("mfcc", i) - mu) / sd
            ax_.imshow(m.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-3, vmax=3,
                       extent=[0, len(m) * 0.02, -0.5, 38.5])
            ax_.set_title(c, color=ps.INK); ax_.grid(False)
            for yb in (12.5, 25.5):
                ax_.axhline(yb, color=ps.INK, lw=0.6)
            ax_.set_xlabel("time (s)")
        for a in axes[:, 0]:
            a.set_yticks([6, 19, 32], ["static\n(c1-12, logE)", "delta", "delta-\ndelta"])
        fig.suptitle("39-dim MFCC feature maps (each dimension z-scored over the dataset; colour -3..+3), "
                     "actor 02 (female), 'kids', strong intensity",
                     x=0.01, ha="left", fontweight="bold")
        fig.tight_layout(); ps.save(fig, PATHS.plots / f"features_mfcc_by_emotion_{ds}.png")

    # --- C: HuBERT layers
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    for k, L in enumerate((0, 1, 9, 12)):
        a = fig.add_subplot(gs[0, k])
        a.imshow(hb[:, L, :].T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-2, vmax=2,
                 extent=[0, Tn * 0.02, 0, 768])
        a.set_title(f"hidden_states[{L}]" + (" (Transformer input)" if L == 0 else f" (layer {L} output)"), fontsize=9)
        a.set_xlabel("time (s)"); a.grid(False)
        if k == 0:
            a.set_ylabel("dimension")
    a = fig.add_subplot(gs[1, 0:2])
    im = a.imshow(cka, cmap=ps.SEQ_BLUE, vmin=0, vmax=1)
    a.set_xticks(range(13)); a.set_yticks(range(13)); a.grid(False)
    for (r, c_), val in np.ndenumerate(cka):
        if r in (1, 9) and c_ in (1, 9):
            a.text(c_, r, f"{val:.2f}", ha="center", va="center", fontsize=7, color=ps.INK if val < 0.6 else "white")
    a.set_title("Linear CKA between layers (20,000 random frames)", fontsize=10)
    a.set_xlabel("hidden state"); a.set_ylabel("hidden state"); fig.colorbar(im, ax=a, fraction=0.04)
    a = fig.add_subplot(gs[1, 2:4])
    for layer, d in lag_res.items():
        lags = sorted(d)
        a.plot(lags, [d[l] for l in lags], marker="o", color=ps.SERIES[0 if layer == 0 else 1], label=f"HuBERT layer {layer}")
    a.axvline(0, color=ps.MUTED, lw=0.8, ls="--")
    a.set_xlabel("lag (frames, 20 ms)"); a.set_ylabel("mean within-utterance r")
    a.set_title("HuBERT-predicted energy vs MFCC energy: peak at lag 0 = aligned", fontsize=10); a.legend()
    fig.tight_layout(); ps.save(fig, PATHS.plots / f"features_hubert_layers_{ds}.png")

    # --- D: prosody distributions per emotion (descriptive)
    p = store.array("prosody103_wide")
    df = pd.DataFrame({"label": man.label, "gender": man.gender,
                       "F0 mean (Hz)": p[:, DISVOICE_NAMES.index("F0avg")],
                       "F0 std (Hz)": p[:, DISVOICE_NAMES.index("F0std")],
                       "voiced segments / s": p[:, DISVOICE_NAMES.index("Vrate")]})
    fr = [store.frames("prosody_frames", i) for i in range(len(man))]
    df["energy std (dB)"] = [f[:, 3].std() for f in fr]
    df["voiced fraction"] = [(f[:, 1] > 0.5).mean() for f in fr]
    cols = ["F0 mean (Hz)", "F0 std (Hz)", "energy std (dB)", "voiced fraction"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    for a, col in zip(axes[:4], cols):
        for k, c in enumerate(CLASSES):
            for g, off, alpha in (("M", -0.18, 0.55), ("F", 0.18, 1.0)):
                vals = df[(df.label == c) & (df.gender == g)][col] if col == "F0 mean (Hz)" else None
                if vals is None:
                    break
                bp = a.boxplot(vals, positions=[k + off], widths=0.3, patch_artist=True, showfliers=False)
                for b in bp["boxes"]:
                    b.set(facecolor=ps.EMOTION_COLOR[c], alpha=alpha, edgecolor=ps.INK_2)
                for med in bp["medians"]:
                    med.set(color=ps.INK)
            if col != "F0 mean (Hz)":
                bp = a.boxplot(df[df.label == c][col], positions=[k], widths=0.6, patch_artist=True, showfliers=False)
                for b in bp["boxes"]:
                    b.set(facecolor=ps.EMOTION_COLOR[c], edgecolor=ps.INK_2)
                for med in bp["medians"]:
                    med.set(color=ps.INK)
        a.set_xticks(range(6), CLASSES, rotation=40); a.set_title(col, fontsize=10)
    n_unv = len(R["f0_all_unvoiced"]["prosody103_wide"])
    axes[0].set_title("F0 mean (Hz)\nleft box = male (light), right = female", fontsize=10)
    fig.text(0.01, 0.005, f"Box plots: outliers hidden. {n_unv} recordings with no voiced frame "
             "(their F0 features are 0 by the DisVoice convention).", fontsize=8, color=ps.INK_2)
    a = axes[4]
    sh = above.share_above_350Hz.unstack(0).reindex(CLASSES)
    x = np.arange(6)
    a.bar(x - 0.2, sh["M"] * 100, 0.38, color=ps.SERIES[0], label="male")
    a.bar(x + 0.2, sh["F"] * 100, 0.38, color=ps.SERIES[1], label="female")
    a.set_xticks(x, CLASSES, rotation=40); a.set_title("voiced frames with F0 > 350 Hz (%)", fontsize=10); a.legend(fontsize=8)
    fig.suptitle(f"{ds}: prosodic features by emotion (descriptive, all recordings; Praat ac "
                 f"{DEFAULT.prosody.f0_min:g}-{DEFAULT.prosody.f0_max_wide:g} Hz)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); ps.save(fig, PATHS.plots / f"features_prosody_{ds}.png")

    # --- E: dimensions overview
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    a = axes[0]
    a.hist(man.n_frames, bins=40, color=ps.SERIES[0])
    a.set_xlabel("frames per recording (20 ms)"); a.set_ylabel("recordings")
    a.set_title(f"{ds}: {len(man)} recordings, {int(man.n_frames.sum()):,} frames "
                f"(median {int(man.n_frames.median())}, max {int(man.n_frames.max())})", fontsize=10)
    a = axes[1]; a.axis("off"); a.grid(False)
    rows = [["stream", "level", "shape", "dtype"],
            ["HuBERT hidden states", "frame", "(T, 13, 768)", "fp16"],
            ["HuBERT pooled mean|std", "utterance", "(13, 1536)", "fp32"],
            ["MFCC 12 + logE, +D, +DD", "frame", "(T, 39)", "fp32"],
            ["prosody contours", "frame", "(T, 4)", "fp32"],
            ["DisVoice-103 (2 F0 ranges)", "utterance", "(103,)", "fp32"]]
    tb = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left",
                 colWidths=[0.46, 0.18, 0.24, 0.12])
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
    for (r, c_), cell in tb.get_celld().items():
        cell.set_edgecolor(ps.GRID)
        if r == 0:
            cell.set_text_props(weight="bold")
    a.set_title("All frame-level streams share T = 1 + (N - 400) // 320", fontsize=10)
    fig.tight_layout(); ps.save(fig, PATHS.plots / f"features_dimensions_{ds}.png")


def voicing_figure(d: dict, ds: str):
    """F: voicing by class x gender, F0 distribution, periodicity channel (voiced vs unvoiced frames)."""
    vr, allf0, per_v, per_u = d["vr"], d["allf0"], d["per_v"], d["per_u"]
    fmin, fmax = DEFAULT.prosody.f0_min, DEFAULT.prosody.f0_max_wide
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    a = axes[0]
    g = vr.groupby(["label", "gender"]).agg(vfrac=("vfrac", "mean"), none=("none", "sum")).reset_index()
    x = np.arange(len(CLASSES))
    for off, gd, col, name in ((-0.2, "M", ps.SERIES[0], "male"), (0.2, "F", ps.SERIES[1], "female")):
        s = g[g.gender == gd].set_index("label").reindex(CLASSES)
        a.bar(x + off, s.vfrac * 100, 0.38, color=col, label=name)
        for xi, (vf, nn) in enumerate(zip(s.vfrac, s.none)):
            if nn:
                a.text(xi + off, vf * 100 + 1, f"{int(nn)}", ha="center", fontsize=7, color=ps.INK_2)
    a.set_xticks(x, CLASSES, rotation=30); a.set_ylabel("mean voiced-frame share (%)")
    a.set_title(f"Voicing by class and gender\n(recordings with no voiced frame: {int(vr.none.sum())})", fontsize=9)
    a.set_ylim(0, a.get_ylim()[1] * 1.18)
    a.legend(fontsize=8, loc="upper left", ncol=2)
    a = axes[1]
    from matplotlib.ticker import NullFormatter
    bins = np.geomspace(fmin, fmax, 60)
    for gd, col, name in (("M", ps.SERIES[0], "male"), ("F", ps.SERIES[1], "female")):
        if len(allf0[gd]):
            a.hist(allf0[gd], bins=bins, histtype="step", lw=2, color=col, label=f"{name} ({len(allf0[gd]):,} frames)")
    a.set_xscale("log")
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xticks([60, 100, 200, 350, 500, 800], ["60", "100", "200", "350", "500", "800"])
    a.axvline(350, color=ps.INK_2, lw=0.9, ls="--")
    a.text(360, a.get_ylim()[1] * 0.55, "DisVoice\nceiling", fontsize=7, color=ps.INK_2)
    a.set_xlabel("F0 of voiced frames (Hz, log scale)"); a.set_ylabel("frames")
    a.set_title(f"F0 distribution\n(Praat ac {fmin:g}-{fmax:g} Hz)", fontsize=9); a.legend(fontsize=8, loc="upper right")
    a = axes[2]
    b = np.linspace(0, 1, 41)
    a.hist(per_u, bins=b, histtype="step", lw=2, color=ps.SERIES[6], density=True, label=f"unvoiced frames ({len(per_u):,})")
    a.hist(per_v, bins=b, histtype="step", lw=2, color=ps.SERIES[2], density=True, label=f"voiced frames ({len(per_v):,})")
    thr = DEFAULT.prosody.praat.voicing_threshold
    a.set_ylim(0, a.get_ylim()[1] * 1.3)
    a.axvline(thr, color=ps.INK_2, lw=0.9, ls="--")
    a.text(thr - 0.01, a.get_ylim()[1] * 0.55, "voicing\nthreshold", fontsize=7, color=ps.INK_2, ha="right")
    a.set_xlabel("periodicity (max autocorrelation strength\nof the frame's voiced candidates)"); a.set_ylabel("density")
    a.set_title("Periodicity channel (not a probability)\nspike at 0 = frames without any voiced candidate", fontsize=9)
    a.legend(fontsize=8, loc="upper left", ncol=2)
    fig.suptitle(f"{ds}: voicing and pitch after the switch to Praat (all {len(vr)} recordings)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); ps.save(fig, PATHS.plots / f"features_voicing_{ds}.png")


if __name__ == "__main__":
    main()
