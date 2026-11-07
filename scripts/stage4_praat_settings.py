"""
Stage 4 investigation: choose the Praat pitch range / settings (no cache is modified).

Same 300 random recordings per corpus as scripts/stage4_f0_tracker_comparison.py
(rng = default_rng(0), RAVDESS drawn first). All trackers run on the 10 ms DisVoice grid
(frame j centred at sample 160 j) of the cached, preprocessed 16 kHz audio.

No ground-truth F0 exists for RAVDESS/CREMA-D, so the settings are judged by proxies:
  * voiced_frac, no_voicing          voicing coverage (the problem pYIN had)
  * octave_jump_rate                 share of consecutive voiced 10 ms pairs with |dF0| > 7 st;
                                     real F0 cannot move half an octave in 10 ms -> tracking errors
                                     (NOTE: octave_jump_cost penalises exactly this, so for that
                                     parameter the jump metric is partly circular; gpe_vs_pyin and
                                     the synthetic glides are the independent checks)
  * recordings_with_jump             recordings with >= 1 such jump (each one inflates F0 std/MSE)
  * near_ceiling                     share of voiced frames > 0.9 x ceiling (ceiling too low?)
  * near_floor                       share of voiced frames < 1.1 x floor (floor too high?)
  * low_energy_voiced                voiced frames > 40 dB below the clip's loudest frame (spurious)
  * gpe_vs_pyin                      gross pitch error vs pYIN 60-700 Hz (|ratio-1| > 20 %) on
                                     frames both call voiced: agreement with an independent tracker
  * f0 quantiles by gender / class   where does the pitch mass actually lie?
Synthetic tones with known F0 are reported too (the only setting with ground truth).

Run:  python scripts/stage4_praat_settings.py [--n 300]
Writes results/stage4_praat_settings.json
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.config import PraatPitchConfig  # noqa: E402
from src.features.mfcc import frame_signal  # noqa: E402
from src.features.prosody import f0_praat, f0_pyin  # noqa: E402

STEP = 160
BASE = PraatPitchConfig()
SETTINGS = {                                  # name -> (fmin, fmax, PraatPitchConfig)
    "ac_60_350": (60, 350, BASE),             # DisVoice default range
    "ac_60_500": (60, 500, BASE),
    "ac_60_600": (60, 600, BASE),
    "ac_60_700": (60, 700, BASE),
    "ac_60_800": (60, 800, BASE),
    "ac_60_1000": (60, 1000, BASE),
    "ac_50_800": (50, 800, BASE),
    "ac_75_800": (75, 800, BASE),
    "ac_60_800_accurate": (60, 800, replace(BASE, very_accurate=True)),
    "ac_60_800_vthr0.35": (60, 800, replace(BASE, voicing_threshold=0.35)),
    "ac_60_800_vthr0.55": (60, 800, replace(BASE, voicing_threshold=0.55)),
    "ac_60_800_ojc0.5": (60, 800, replace(BASE, octave_jump_cost=0.5)),
    "ac_60_800_ojc0.7": (60, 800, replace(BASE, octave_jump_cost=0.7)),
    "ac_60_800_ojc1.0": (60, 800, replace(BASE, octave_jump_cost=1.0)),
}


def _one(args):
    """All settings + pYIN reference for one recording -> list of per-setting metric dicts."""
    y, meta = args
    n = 1 + len(y) // STEP
    fr = frame_signal(np.concatenate([np.zeros(200), y, np.zeros(200)]), 400, STEP)[:n]
    e_db = 10 * np.log10(np.maximum((fr.astype(np.float64) ** 2).mean(1), 1e-10))
    low_e = e_db < e_db.max() - 40
    f_py, _ = f0_pyin(y, 60, 700, hop=STEP, frame_length=1024)
    f_py = f_py[:n]
    out = []
    for name, (lo, hi, pc) in SETTINGS.items():
        f0, per = f0_praat(y, n, STEP, 0, lo, hi, pc)
        v = f0 > 0
        st = 12 * np.log2(np.where(v, f0, 1) / 100)
        pair = v[1:] & v[:-1]
        jumps = np.abs(np.diff(st))[pair]
        both = v & (f_py > 0)
        ratio = f0[both] / f_py[both]
        out.append(dict(meta, setting=name, voiced_frac=v.mean(), no_voicing=not v.any(),
                        n_voiced=int(v.sum()), n_pairs=int(pair.sum()), n_jumps=int((jumps > 7).sum()),
                        n_near_ceiling=int((f0[v] > 0.9 * hi).sum()), n_near_floor=int((f0[v] < 1.1 * lo).sum()),
                        n_low_energy_voiced=int((v & low_e).sum()), n_both=int(both.sum()),
                        n_gpe=int((np.abs(ratio - 1) > 0.2).sum()), f0_voiced=f0[v].astype(np.float32),
                        periodicity_voiced=float(per[v].mean()) if v.any() else np.nan,
                        periodicity_unvoiced=float(per[~v].mean()) if (~v).any() else np.nan))
    out.append(dict(meta, setting="pyin_60_700", voiced_frac=(f_py > 0).mean(), no_voicing=not (f_py > 0).any(),
                    n_voiced=int((f_py > 0).sum()), f0_voiced=f_py[f_py > 0].astype(np.float32),
                    n_low_energy_voiced=int(((f_py > 0) & low_e).sum()),
                    **_jump_counts(f_py)))
    return out


def _jump_counts(f0):
    v = f0 > 0
    st = 12 * np.log2(np.where(v, f0, 1) / 100)
    pair = v[1:] & v[:-1]
    return dict(n_pairs=int(pair.sum()), n_jumps=int((np.abs(np.diff(st))[pair] > 7).sum()))


def summarise(rows: pd.DataFrame) -> dict:
    res = {}
    for name, g in rows.groupby("setting", sort=False):
        f = np.concatenate(g.f0_voiced.values) if g.n_voiced.sum() else np.array([0.0])
        gender = np.concatenate([[x] * len(a) for x, a in zip(g.gender, g.f0_voiced)])
        label = np.concatenate([[x] * len(a) for x, a in zip(g.label, g.f0_voiced)])
        d = {
            "voiced_frac_mean": round(float(g.voiced_frac.mean()), 4),
            "recordings_without_voicing": int(g.no_voicing.sum()),
            "octave_jump_rate": round(float(g.n_jumps.sum() / max(g.n_pairs.sum(), 1)), 5),
            "recordings_with_jump": int((g.n_jumps > 0).sum()),
            "low_energy_voiced_share": round(float(g.n_low_energy_voiced.sum() / max(g.n_voiced.sum(), 1)), 5),
            "f0_quantiles_hz": {
                gd: {q: round(float(np.percentile(f[gender == gd], q)), 1) for q in (1, 5, 50, 95, 99)}
                for gd in ("M", "F") if (gender == gd).any()},
            "f0_p95_female_by_label": {lb: round(float(np.percentile(f[(gender == "F") & (label == lb)], 95)), 1)
                                       for lb in sorted(set(label)) if ((gender == "F") & (label == lb)).any()},
            "voiced_frac_by_label": g.groupby("label").voiced_frac.mean().round(3).to_dict(),
            "no_voicing_by_label": g.groupby("label").no_voicing.sum().astype(int).to_dict(),
        }
        if name != "pyin_60_700":
            d.update({
                "near_ceiling_share": round(float(g.n_near_ceiling.sum() / max(g.n_voiced.sum(), 1)), 5),
                "near_floor_share": round(float(g.n_near_floor.sum() / max(g.n_voiced.sum(), 1)), 5),
                "gross_pitch_error_vs_pyin": round(float(g.n_gpe.sum() / max(g.n_both.sum(), 1)), 5),
                "periodicity_mean_voiced": round(float(g.periodicity_voiced.mean()), 3),
                "periodicity_mean_unvoiced": round(float(g.periodicity_unvoiced.mean()), 3),
            })
        res[name] = d
    return res


def synthetic() -> dict:
    """Harmonic tones (5 partials, 1/k amplitudes, light noise) with known F0, per setting."""
    rng = np.random.default_rng(7)
    t = np.arange(24000) / 16000
    out = {}
    for F in (70, 100, 200, 300, 450, 600, 750, 900):
        y = sum(np.sin(2 * np.pi * k * F * t) / k for k in range(1, 6) if k * F < 8000)
        y = y / np.abs(y).max() + 0.02 * rng.standard_normal(len(t))
        n = 1 + len(y) // STEP
        out[F] = {}
        for name, (lo, hi, pc) in SETTINGS.items():
            f0, _ = f0_praat(y, n, STEP, 0, lo, hi, pc)
            v = f0 > 0
            out[F][name] = round(float(np.median(f0[v])), 1) if v.any() else 0.0
    return out


def synthetic_glides() -> dict:
    """
    Fast REAL pitch movements with known contour: an octave rise (and fall) within D ms.
    Human maximum speed of pitch change is roughly 10-12 semitones per 100 ms (Xu & Sun 2002),
    so D = 100 ms is at the physiological limit and D = 50 ms beyond it. Metric: share of
    voiced frames whose F0 is more than 1 semitone from the true F0 at the frame centre.
    """
    rng = np.random.default_rng(11)
    sr, n_s = 16000, 24000
    t = np.arange(n_s) / sr
    out = {}
    for f_lo, D in ((150, 0.1), (250, 0.1), (150, 0.05), (250, 0.05)):
        ramp = np.clip((t - 0.5) / D, 0, 1) - np.clip((t - 1.0) / D, 0, 1)      # up at 0.5 s, down at 1.0 s
        f_true = f_lo * 2 ** ramp
        phase = 2 * np.pi * np.cumsum(f_true) / sr
        y = sum(np.sin(k * phase) / k for k in range(1, 6))
        y = y / np.abs(y).max() + 0.02 * rng.standard_normal(n_s)
        n = 1 + n_s // STEP
        truth = np.interp(np.arange(n) * STEP / sr, t, f_true)
        key = f"{f_lo}-{2 * f_lo}Hz_in_{int(D * 1000)}ms"
        out[key] = {}
        for name, (lo, hi, pc) in SETTINGS.items():
            f0, _ = f0_praat(y, n, STEP, 0, lo, hi, pc)
            v = f0 > 0
            err = np.abs(12 * np.log2(f0[v] / truth[v]))
            out[key][name] = {"voiced": round(float(v.mean()), 3), "err_gt_1st": round(float((err > 1).mean()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    from src.utils.paths import PATHS
    rng = np.random.default_rng(0)
    out = {"settings": {k: {"fmin": lo, "fmax": hi, **asdict(pc)} for k, (lo, hi, pc) in SETTINGS.items()},
           "synthetic_median_f0": synthetic(), "synthetic_glides": synthetic_glides()}
    for ds in ("RAVDESS", "CREMA-D"):
        root = next(p for p in sorted((PATHS.features / ds).iterdir()) if (p / "wav16k.npy").exists())
        man = pd.read_csv(root / "manifest.csv")
        wav = np.load(root / "wav16k.npy", mmap_mode="r")
        pick = rng.choice(len(man), min(args.n, len(man)), replace=False)
        jobs = [(np.array(wav[man.wav_offset[i]: man.wav_offset[i] + man.n_samples[i]], dtype=np.float64),
                 dict(rel_path=man.rel_path[i], label=man.label[i], gender=man.gender[i])) for i in pick]
        with ProcessPoolExecutor(args.workers) as ex:
            rows = pd.DataFrame([r for rs in ex.map(_one, jobs, chunksize=8) for r in rs])
        out[ds] = summarise(rows)
        out[ds]["_audio_source"] = str(root)
        print(f"\n=== {ds} (n={len(jobs)}) ===")
        tab = pd.DataFrame({k: {c: v[c] for c in ("voiced_frac_mean", "recordings_without_voicing", "octave_jump_rate",
                                                  "recordings_with_jump", "low_energy_voiced_share")} | {
            "near_ceil": v.get("near_ceiling_share"), "near_floor": v.get("near_floor_share"),
            "gpe_pyin": v.get("gross_pitch_error_vs_pyin"), "F_p99": v["f0_quantiles_hz"].get("F", {}).get(99),
            "M_p99": v["f0_quantiles_hz"].get("M", {}).get(99), "M_p1": v["f0_quantiles_hz"].get("M", {}).get(1)}
            for k, v in out[ds].items() if not k.startswith("_")}).T
        print(tab.to_string())
    print("\nsynthetic tones (median F0):")
    print(pd.DataFrame(out["synthetic_median_f0"]).to_string())
    print("\nsynthetic glides (share of voiced frames > 1 st from truth):")
    print(pd.DataFrame({k: {s: v["err_gt_1st"] for s, v in d.items()} for k, d in out["synthetic_glides"].items()}).to_string())
    (PATHS.results / "stage4_praat_settings.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
