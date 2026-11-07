# Feature Extraction (Stage 4) — Design, Rationale and Verification

All numbers in this document come from executed runs. Sources: `results/stage4_hubert_checks.json`,
`results/stage4_validation_RAVDESS.json`, `results/stage4_validation_CREMA-D.json`,
`results/stage4_praat_settings.json`, `results/stage4_f0_tracker_comparison.json` and
`experiments/LOG.md`. Pre-switch (pYIN) and intermediate (Praat default settings) validation results
are archived as `results/stage4_validation_<DS>_pyin_v1_f9418feedb.json` and
`results/stage4_validation_<DS>_praat_ojc035_v1_36607e8614.json`.

**Current caches:** `v1_efe9081789` (F0 by Praat, approved 2026-11-06). The earlier pYIN cache
`v1_f9418feedb` is kept on disk only as the documented "before" state. Results **reported by the reference paper** (Zhao et al., arXiv:2501.10408)
are labelled as such and are never mixed with ours.

---

## 1. Overview

```
 raw WAV (48 kHz RAVDESS / 16 kHz CREMA-D, never modified)
   │  decode → mono (mean of channels) → resample to 16 kHz (soxr_hq)
   │  → trim leading/trailing silence (30 dB below peak, 50 ms margin) → remove DC → peak-normalise
   ▼
 y : float32, 16 kHz, |y|max = 1   ──── cached once: wav16k.npy
   │
   ├── HuBERT-base (frozen, fp32 inference) ── 13 hidden states ── (T, 13, 768)  fp16 cache
   │                                         └ per-layer mean|std ─ (13, 1536)   fp32
   ├── MFCC: 12 cepstra + log-energy, + Δ + ΔΔ ─────────────────── (T, 39)       fp32
   ├── Praat F0 tracker (autocorrelation) ─┬─ prosody contours: F0 (semitones), voiced,
   │                                       │   periodicity, energy (dB) ─────── (T, 4)  fp32
   └───────────────────────────────────────┴─ DisVoice-103 utterance statistics
                                               (F0 ranges 60–350 Hz and 60–800 Hz) ─ (103,) ×2

 shared time grid:  frame t = samples [320t, 320t+400)  ⇒  25 ms windows every 20 ms,
                    centre 0.020·t + 0.0125 s,   T = 1 + ⌊(N − 400) / 320⌋
```

| Representation | Level | Shape per recording | Stored as | Used for |
|---|---|---|---|---|
| HuBERT hidden states 0–12 | frame | (T, 13, 768) | fp16 | E2b, E3, E4 |
| HuBERT pooled mean ‖ std | utterance | (13, 1536) | fp32 | E2a layer probe |
| MFCC (c1–c12, logE, Δ, ΔΔ) | frame | (T, 39) | fp32 | E1 (as statistics), E3, E4 |
| Prosody contours | frame | (T, 4) | fp32 | E3, E4 (our adaptation) |
| DisVoice-103, F0 ≤ 350 Hz | utterance | (103,) | fp32 + imputation mask | DisVoice's default range (paper comparison only; distorts high F0, §5.3) |
| DisVoice-103, F0 ≤ 800 Hz | utterance | (103,) | fp32 + imputation mask | **primary** utterance-level prosody |

---

## 2. Preprocessing decisions

| Step | Decision | Why |
|---|---|---|
| Sampling rate | 16 kHz | HuBERT was pre-trained on 16 kHz LibriSpeech; CREMA-D is already 16 kHz. |
| Resampling | `soxr_hq` (band-limited) | 48 kHz → 16 kHz without aliasing. |
| Channels | mean of channels | 5 RAVDESS files are stereo (Stage 2 finding). |
| Silence trimming | leading/trailing only, 30 dB below peak, 50 ms margin | Silence carries no emotion, inflates padding and would count as "pauses". Internal pauses are kept (they are prosodic features). **Verified:** RAVDESS mean duration 3.72 s → 1.95 s, but the energy removed is only 0.005 % (median) and 0.14 % (max) of the signal energy, so only silence is removed. |
| Amplitude | DC removal + peak normalisation | Recording gain differs between actors, sessions and corpora, so absolute loudness is not a trustworthy cue across speakers. Within-utterance energy dynamics are preserved. DisVoice applies the same normalisation. |
| HuBERT input | additionally z-normalised (zero mean, unit variance) | `preprocessor_config.json` of `facebook/hubert-base-ls960` specifies `do_normalize: true`. |
| Feature normalisation | **none at extraction** | Features are cached raw. z-scoring is done in Stage 5 with **training-fold statistics only**, because normalising with all data would leak test-speaker information. |
| Padding | none at extraction; zero-pad + mask per mini-batch | See §6. The paper instead repeats short clips up to 7 s; repetition duplicates content and biases utterance statistics. |

---

## 3. HuBERT: what it is and what we extract

**Self-supervised pre-training.** HuBERT (Hsu et al., 2021) is trained on 960 h of unlabelled speech.
Frames are masked, and the model must predict the *cluster ID* of the masked frames from the
context. The targets are k-means clusters of MFCCs in the first iteration and of HuBERT's own
features afterwards. The model never sees emotion labels. Whatever emotional information its
layers carry is a by-product of learning to model speech.

**Architecture (verified in the installed `transformers 5.22.1` source and numerically):**

| Index | Component | Output |
|---|---|---|
| — | Feature encoder: 7 conv layers (kernels 10,3,3,3,3,2,2; strides 5,2,2,2,2,2,2), GroupNorm, GELU | (T, 512) |
| — | Feature projection: LayerNorm + Linear 512→768 | (T, 768) |
| `hidden_states[0]` | + convolutional positional embedding, LayerNorm = **input to Transformer layer 1** | (T, 768) |
| `hidden_states[i]`, i=1..12 | output of Transformer layer *i* (post-LN: self-attention 12×64 heads + FFN 768→3072→768) | (T, 768) |

We recomputed every hidden state manually from the model's sub-modules and compared the results
with the library outputs. The maximum absolute difference was **0.0 for all 13 states**
(`stage4_hubert_checks.json`). The CNN output itself (512-dim) is not part of `hidden_states`.

**Layers 1 and 9 (reference paper).** The paper uses "the 1st and 9th transformer layers", citing
a layer-wise analysis (Pasad et al., 2021) that associates lower layers with acoustic information
and intermediate layers with phonetic information. In HuggingFace indexing these are
`hidden_states[1]` and `hidden_states[9]`; `hidden_states[0]` is the Transformer *input*. We store
all 13 states, so either reading of "1st" can be tested.

**Label-free layer analysis (RAVDESS, 20,000 random frames, linear CKA):**
- CKA(layer 1, layer 9) = **0.71**. The two layers the paper combines are substantially different
  representations, which is consistent with the paper's motivation for using both.
- CKA with layer 0 decreases steadily with depth, to 0.62–0.64 for layers 8–12. Adjacent layers are
  very similar (CKA 0.93–0.98), except 0→1 (0.88).
- The mean frame norm rises from 5.3 (layer 1) to 8.2 (layer 11), then jumps to 11.9 at layer 12.

**Which layer is best for emotion is NOT decided here.** Answering that requires training probes
(experiment E2a). Choosing layers by looking at labels of all recordings would leak test
information into model selection. E2a will therefore fit one probe per layer inside each fold, on
training speakers only, select on validation speakers, and report test results.

**Precision decision (measured on 24 RAVDESS clips):**

| | fp32 inference | fp16 inference |
|---|---|---|
| time per clip | 14.5 ms | 11.0 ms |
| peak GPU memory | 506 MiB | 415 MiB |
| deviation from fp32 | — | cosine ≥ 0.9999; relative error mean 0.11–0.18 %, max 1.4 % |

Storing fp32 features as fp16 adds a relative error of only 0.02 % (max 0.03 %). The largest
|activation| is 9.8, far inside the fp16 range. **Decision: fp32 inference, fp16 storage.** fp16
inference is numerically safe, but it would save only about 4 s for the whole of RAVDESS at 7× the
error. Storage is where fp16 matters: it halves the 3.9 GB that RAVDESS would otherwise need.

**Batch size 1, no padding.** HuBERT-base normalises its first conv layer over the whole time axis
(GroupNorm), so zero-padding would change the features of real frames. Processing one utterance at
a time is exact, and at 18 ms per recording it is also fast enough.

---

## 4. MFCC (39 per frame)

**Why MFCC alongside HuBERT?** MFCCs explicitly describe the short-term spectral envelope
(vocal-tract configuration, voice quality) on a perceptual frequency scale. HuBERT is optimised to
predict phonetic units, which may suppress paralinguistic detail in its upper layers. MFCCs are an
explicit, low-level acoustic view that a fusion model can draw on if HuBERT under-represents it.
Whether this complementarity actually helps is research question RQ3, not an assumption.

**Pipeline (implemented step by step in `src/features/mfcc.py`, following paper Eqs. 5–8):**

| Parameter | Value |
|---|---|
| sampling rate | 16 kHz |
| window | 400 samples (25 ms), periodic Hamming |
| hop | 320 samples (20 ms), identical to HuBERT |
| FFT size | 512 (frame zero-padded; 31.25 Hz bins) |
| power spectrum | \|DFT\|² / N |
| mel filterbank | 40 triangular filters, 0–8000 Hz, HTK mel scale (2595·log10(1+f/700)), unit peak height |
| log | natural log, floor 1e-10 |
| DCT | type-II, orthonormal; keep c1–c12 (c0 dropped) |
| energy | ln Σx² of the raw (un-windowed) 400-sample frame |
| deltas | regression over ±2 frames, edges replicated, computed per utterance before padding |
| layout | [c1..c12, logE, Δ(13), ΔΔ(13)] = 39 |

**Independent verification (unit tests):**
- Our mel filterbank equals `librosa.filters.mel(htk=True, norm=None)`.
- Our DCT matrix equals `scipy.fft.dct(type=2, norm="ortho")`.
- Our mel power spectrum equals librosa's on identical samples (rtol 1e-4).
- Deltas of a linear ramp equal its slope.

**Difference from the paper's text:** the paper states 40 ms frames, a 20 ms shift and a 30 ms
Hamming window, which contradict each other. We use 25 ms / 20 ms so that MFCC frames coincide
exactly with HuBERT frames.

---

## 5. Prosody

**Why prosody?** Arousal is expressed through pitch level and range, loudness and timing. MFCCs
describe the envelope, not the pitch. Descriptive statistics (all recordings, no model involved,
Praat 60–800 Hz, median over recordings of the utterance mean F0) show the expected pattern:

| median F0 mean (Hz) | neutral | sad | disgust | happy | fearful | angry |
|---|---|---|---|---|---|---|
| RAVDESS female | 217 | 235 | 250 | 276 | 322 | 290 |
| RAVDESS male | 120 | 142 | 153 | 186 | 176 | 229 |
| CREMA-D female | 192 | 185 | 191 | 258 | 228 | 270 |
| CREMA-D male | 136 | 131 | 148 | 183 | 160 | 199 |

F0 variability (median within-utterance F0 std) is lowest for neutral and highest for angry
(RAVDESS 78 Hz) and happy (CREMA-D 64 Hz).

### 5.1 What the paper used: DisVoice prosody, 103 features

The paper says it uses DisVoice's 103 prosodic features. We read the DisVoice source
(commit `7bb41a4`, MIT licence) and re-implemented `prosody_static()` faithfully in
`src/features/prosody.py`. The grouping follows DisVoice's actual feature order:

| # | Group | Content |
|---|---|---|
| 30 | F0 | contour mean/std/max/min/skew/kurtosis; per-voiced-segment linear **tilt** and **MSE** statistics (interleaved: tilt-mean, MSE-mean, tilt-std, …); first and last voiced segment |
| 24 | energy, voiced segments | means of per-segment mean/std/skew/kurtosis; tilt stats; MSE stats; first/last segment |
| 24 | energy, unvoiced segments | same |
| 25 | timing | voiced segments per second; duration statistics of voiced, unvoiced and pause segments (pause = unvoiced run > 140 ms); 6 ratios of **mean** durations |

Details of the original that we reproduce (and document as quirks):
- The docstring's feature order is wrong: tilt and MSE statistics are interleaved in the code.
- The final voiced/unvoiced segment is never appended.
- The duration "ratios" use mean, not total, durations.
- Features are set to 0 when fewer than 2 segments exist.

**Verification:** our implementation equals the **original DisVoice functions** (vendored, unmodified,
in `tests/reference/` as a test oracle). This holds to rtol 1e-7 on 8 synthetic voiced/unvoiced/pause
patterns and to rtol 1e-6 on 30 real recordings per corpus, given the same F0 input (re-run after the
tracker switch: 0 mismatches on RAVDESS and CREMA-D).

**What we could not reproduce exactly, and the approximation used:**

| Aspect | DisVoice / paper | Ours | Reason |
|---|---|---|---|
| F0 tracker | RAPT (pysptk) by default; Praat `To Pitch...` as built-in alternative | **Praat autocorrelation** (praat-parselmouth 0.4.7 = Praat 6.1.38), 10 ms | pysptk has no Windows/Python 3.12 wheel; pYIN (first choice) failed on CREMA-D (§5.4) |
| Praat settings | defaults (`To Pitch...` = ac, octave-jump cost 0.35) | defaults except **octave-jump cost 0.7** | §5.5 |
| Praat time mapping | `decodeF0` snaps Praat time stamps to a `linspace` grid (nearest neighbour) | Praat frames placed **exactly** on our grid by zero-padding (asserted per call) | removes up to half a frame of jitter |
| log-energy of an all-zero frame | −1e30 | −100 dB floor | −1e30 produces astronomically large statistics |
| undefined statistics (skew of a constant, mean of empty) | NaN | NaN → 0, with a per-value imputation mask stored | models need finite inputs; the mask keeps it traceable |
| F0 range | 60–350 Hz | both 60–350 Hz (DisVoice default) **and** 60–800 Hz (primary) | §5.3, §5.5 |

(DisVoice's README says Praat is the default, but the code (`self.pitch_method = "rapt"`) uses RAPT.
Its Praat script is `To Pitch... time_step low_f0 high_f0`, i.e. Praat's autocorrelation method with
all other parameters at their defaults.)

### 5.2 Our adaptation: frame-level prosody contours (4 per frame)

The paper's prosody is utterance-level only, and cross-attention between one vector and a sequence
is degenerate. So we additionally store, on the HuBERT grid:

| Index | Value | Meaning |
|---|---|---|
| 0 | `f0_st` | log-F0 in semitones re 100 Hz; unvoiced frames linearly interpolated, edges held |
| 1 | `voiced` | 1 if Praat's Viterbi path selects a voiced candidate, else 0 |
| 2 | `periodicity` | max normalised-autocorrelation strength *r* ∈ [0, 1] among the frame's **voiced candidates** (0 if there are none) |
| 3 | `energy_db` | 10·log10(mean x²) of the same 400-sample frame used by MFCC and HuBERT |

**Channel 2 changed meaning with the tracker switch (dimension kept, name changed from `voiced_p`).**
They are not the same quantity and must not be described as such:

| | pYIN `voiced_p` (before) | Praat `periodicity` (now) |
|---|---|---|
| what it is | posterior probability that the frame is voiced, from pYIN's HMM over a distribution of YIN thresholds | local degree of periodicity: the height of the best autocorrelation peak in the pitch range, normalised by the window's own autocorrelation |
| depends on neighbouring frames | yes (HMM smoothing) | no (computed per frame, before the path search) |
| relation to the `voiced` flag | thresholded to produce it | **not** forced to agree: the path search also uses intensity (silence threshold) and transition costs |
| measured (300 recordings) | mean 0.17 (RAVDESS) / 0.06 (CREMA-D) | mean 0.75 / 0.59; Pearson r with pYIN's value 0.57 / 0.51 |
| unvoiced frames (all recordings) | — | median 0.53 (RAVDESS) / 0.43 (CREMA-D); at least 5 % are exactly 0 (no voiced candidate at all) |
| voiced frames (all recordings) | — | median 0.89 / 0.79; 5th percentile 0.60 / 0.54 |

**How the alignment works.** Praat places its analysis frames symmetrically around the middle of the
signal (`Sampled_shortTermAnalysis`: n = ⌊(M − W)/h⌋ + 1 frames, first centre (M − (n−1)·h)/2).
`f0_praat()` zero-pads the signal (left L, right R) so that the padded length satisfies
M − W = (n−1)·h + h/2 (the half step makes the floor immune to rounding) and the first centre
W/2 + h/4 − L lands on our grid. Frame *t* is then centred at sample `200 + 320·t` (20 ms grid) or
`160·j` (DisVoice 10 ms grid). Every call **asserts** Praat's actual frame times against the grid
(to 1e-9 s); the padding formula is unit-tested for 1,200 signal lengths, two grids and three
window sizes. Requirement: the window 3·16000/f_min must be an even integer number of samples
(true for 50, 60, 75, 80, 100 Hz; other floors are rejected with an error).

### 5.3 Measured finding: DisVoice's 350 Hz ceiling distorts emotional female speech

With Praat at 60–800 Hz, **14.5 % of all voiced RAVDESS frames lie above 350 Hz** (6.2 % in CREMA-D):

| share of voiced frames > 350 Hz | neutral | happy | sad | angry | fearful | disgust |
|---|---|---|---|---|---|---|
| RAVDESS female | 1.6 % | 28.8 % | 18.4 % | 26.5 % | **43.7 %** | 7.7 % |
| RAVDESS male | 0.8 % | 2.4 % | 4.3 % | 11.1 % | 6.2 % | 4.7 % |
| CREMA-D female | 0.6 % | 17.2 % | 1.0 % | 16.9 % | 14.4 % | 3.2 % |
| CREMA-D male | 2.3 % | 5.1 % | 1.2 % | 5.1 % | 2.1 % | 3.2 % |

**New finding with Praat:** above its ceiling Praat does *not* report "unvoiced". It locks onto a
**sub-harmonic**. A synthetic 600 Hz voice is tracked as 300 Hz with a 350 Hz ceiling, 750 Hz as
250 Hz (`results/stage4_praat_settings.json`, `tests/test_praat_f0.py`). So the DisVoice-range
features do not just miss high pitch: they report it **an octave too low**, and how often depends on
emotion and gender. We keep this variant only for comparison with the paper; the primary prosody uses
60–800 Hz. *Caveat:* the high-F0 frames have not been verified by ear.

### 5.4 Why the tracker was switched from pYIN to Praat (approved 2026-11-06)

**Problem.** With pYIN (60–700 Hz), CREMA-D had only **37 %** voiced frames (RAVDESS 72 %) and
**767 of 7,441 recordings (10.3 %)** had no voiced frame at all. The failures were concentrated in
quiet, low-arousal classes (disgust 20.5 %, sad 17.8 %, angry 2.8 % of recordings) and in male
speakers (14.1 % vs 6.0 %). This is a **class-dependent measurement bias**: a classifier could learn
"no pitch ⇒ disgust/sad" from a tracker failure.

**Full-corpus result after the switch** (all usable recordings, final settings):

| | RAVDESS pYIN | RAVDESS Praat | CREMA-D pYIN | CREMA-D Praat |
|---|---|---|---|---|
| voiced frames overall | 71.6 % | 65.4 % | 37.0 % | 45.5 % |
| recordings without voicing, frame level | 6 | **0** | 767 | **0** |
| … DisVoice range (60–350 Hz) | 2 | **0** | 717 | **0** |
| … wide range | 4 | **0** | 507 | **0** |
| recordings with < 10 % voiced frames | — | 0 | — | 12 (disgust 9, fearful 2, angry 1; 10 male) |

| mean voiced share by class (Praat) | neutral | happy | sad | angry | fearful | disgust |
|---|---|---|---|---|---|---|
| RAVDESS | 73.9 % | 74.6 % | 64.5 % | 65.9 % | 67.5 % | 59.3 % |
| CREMA-D | 45.9 % | 49.5 % | 40.6 % | 58.3 % | 47.7 % | 40.5 % |

**Same 300 recordings per corpus that motivated the switch** (`results/stage4_f0_tracker_comparison.json`;
pYIN recomputed and checked bit-identical to the old cache for 300/300 recordings in both corpora):

| | RAVDESS pYIN | RAVDESS Praat | CREMA-D pYIN | CREMA-D Praat |
|---|---|---|---|---|
| mean voiced fraction | 0.74 | 0.67 | 0.40 | 0.48 |
| recordings without any voicing | 1 | 0 | 34 | 0 |
| recordings with < 10 % voiced | 1 | 0 | 40 | 1 |
| voiced fraction disgust / sad | 0.66 / 0.74 | 0.61 / 0.66 | 0.31 / 0.29 | 0.40 / 0.40 |
| frame-wise voicing agreement | 0.82 | | 0.79 | |
| F0 difference on frames voiced by both | median 0.15 st; 0.9 % > 20 %; 0.6 % > ½ octave | | median 0.11 st; 1.0 % > 20 %; 0.8 % > ½ octave | |

**Interpretation.** Where both trackers find voicing, they agree on F0 (median difference about a
tenth of a semitone). The switch changes the *voicing decision*: Praat voices fewer frames than pYIN
on RAVDESS and more on CREMA-D, and no longer loses whole recordings in quiet classes. The residual
class differences in voicing (e.g. CREMA-D disgust/sad 41 % vs angry 58 %) may partly be real
(breathy, creaky or low-intensity phonation), but they cannot be separated from tracker behaviour
without ground-truth pitch. **Absence of fully unvoiced outputs does not mean the tracker is
error-free** (see §5.6).

### 5.5 Choosing the Praat settings (`scripts/stage4_praat_settings.py`)

No ground-truth F0 exists for RAVDESS or CREMA-D, so settings were compared on the same 300 random
recordings per corpus using proxies, plus synthetic signals with known F0:
- **octave-jump rate:** consecutive voiced 10 ms frames differing by more than 7 semitones. Real F0
  cannot move half an octave in 10 ms, so such jumps are tracking errors.
- **ceiling saturation:** voiced frames above 0.9 × ceiling.
- **floor saturation:** voiced frames below 1.1 × floor.
- **gross pitch error (GPE) against pYIN:** the share of frames that both trackers call voiced
  where the two F0 values differ by more than 20 %. pYIN is an independent algorithm.
- **known-F0 tones and octave glides:** the only real ground truth.

| setting (60–800 Hz unless stated) | RAVDESS: jumps / recs with jump / ceiling-sat. / GPE vs pYIN | CREMA-D: same | voiced frac. R / C |
|---|---|---|---|
| 60–350 Hz (DisVoice) | 0.46 % / 80 / 5.4 % / 12.0 % | 0.50 % / 108 / 2.5 % / 6.3 % | 0.655 / 0.484 |
| 60–700 Hz | 0.15 % / 40 / 0.76 % / 0.67 % | 0.33 % / 79 / 0.37 % / 0.95 % | 0.669 / 0.479 |
| 60–800 Hz, Praat defaults | 0.14 % / 38 / 0.31 % / 0.67 % | 0.32 % / 77 / 0.16 % / 0.96 % | 0.669 / 0.477 |
| 60–1000 Hz | 0.15 % / 38 / 0.08 % / 0.71 % | 0.31 % / 74 / 0.03 % / 0.97 % | 0.668 / 0.475 |
| floor 75 Hz | 0.14 % / 37 / 0.28 % / 0.87 % | 0.31 % / 74 / 0.13 % / 1.17 % (floor-sat. 2.1 %) | 0.667 / 0.478 |
| voicing threshold 0.35 | 0.26 % / 64 / — / 0.78 % | 0.71 % / 156 / — / 1.28 % | 0.705 / 0.597 |
| voicing threshold 0.55 | 0.07 % / 19 / — / 0.52 % | 0.10 % / 23 / — / 0.42 % (1 rec. unvoiced) | 0.625 / 0.385 |
| **octave-jump cost 0.7 (chosen)** | **0.01 % / 3 / 0.29 % / 0.52 %** | **0.00 % / 0 / 0.16 % / 0.67 %** | **0.661 / 0.464** |
| octave-jump cost 1.0 | 0.00 % / 0 / 0.27 % / 0.57 % | 0.00 % / 0 / 0.16 % / 0.64 % | 0.656 / 0.455 |

Decisions:
- **Ceiling 800 Hz.** 700 Hz still saturates (0.76 % of RAVDESS voiced frames; female 99th percentile
  ≈ 690 Hz), and a 750 Hz tone is halved at 700 Hz. Going to 1000 Hz brings no further improvement.
- **Floor 60 Hz** (= DisVoice). At 75 Hz, 2.1 % of CREMA-D voiced frames pile up at the floor and a
  70 Hz tone is lost. 50 Hz brings no clear gain and needs a longer window.
- **Voicing threshold 0.45** (default). 0.35 doubles the jump rate. 0.55 loses voicing and
  re-creates an unvoiced recording.
- **Octave-jump cost 0.7** instead of Praat's 0.35. With the default, 13 % (RAVDESS) / 26 % (CREMA-D)
  of sampled recordings contained at least one implausible jump (e.g. 656 → 223 Hz within 10 ms),
  which inflated F0 variance statistics (median `F0msemax` doubled relative to pYIN). Because this
  parameter penalises exactly what the jump metric counts, that metric is partly circular. The
  decision rests on the **independent** checks:
  - GPE against pYIN decreases.
  - Octave glides of 150→300 Hz and 250→500 Hz within 100 ms, and even within 50 ms, which is
    faster than the human limit of about 12 st per 100 ms, are tracked with **0 %** of frames more
    than 1 st off.
  - The cost: −0.8 / −1.4 percentage points of voiced frames, with no recording losing all voicing.

  The default-setting caches were built and fully validated (28/28 checks), then superseded; their
  results are archived (`*_praat_ojc035_v1_36607e8614.json`).
- `very_accurate` (6-period Gaussian window) changed little and was not adopted.

### 5.6 Known limitations of the prosody features (after the switch)
- **No pitch ground truth.** All tracker comparisons use proxies or synthetic signals. Remaining
  octave errors are possible: 3 of 300 sampled RAVDESS recordings still contain one jump of more than
  7 semitones. The male 99th F0 percentile is about 595 Hz in both corpora; this could be genuine
  shouting or octave-up errors and has **not been checked by ear**.
- **Pitch above 800 Hz** would be reported as a sub-harmonic (a 900 Hz tone becomes 450 Hz). Only
  0.37 % (RAVDESS) / 0.17 % (CREMA-D) of voiced frames lie above 720 Hz, so this is rare but possible
  in screams.
- **Residual low voicing.** 12 CREMA-D recordings have < 10 % voiced frames (9 disgust, 10 male),
  down from 767 recordings with none at all. They are kept and are listed via the validation report.
- **Imputation increased on CREMA-D.** 54 % (DisVoice range) / 53 % (wide) of CREMA-D recordings
  have at least one imputed value (pYIN: 44 % / 42 %); RAVDESS 32 % / 29 % (pYIN: 38 % / 36 %). The
  increase comes from recordings that previously had *no* voiced segments: under DisVoice's rules
  they got all-zero F0 and voiced-energy features. They now have short voiced segments whose
  first/last-segment energy skewness and kurtosis are undefined (most-imputed features:
  `skw1Evoiced`/`kurtosis1Evoiced`, 1,474 recordings in the wide variant).
- **Heavy tails.** The largest absolute values are now 4,350 / 9,252 (RAVDESS, DisVoice / wide range)
  and 6,639 / 11,477 (CREMA-D), all in `F0msemax`. These are mean-squared residuals in Hz² of the
  longest-varying voiced segments. Median `F0msemax` is 236 (RAVDESS) vs 153 with pYIN. A plausible
  but **untested** explanation is that pYIN's HMM smooths the contour more and that Praat segments
  voicing differently; the remaining octave jumps (3 of 300 recordings) contribute to the extreme
  values. **Stage 5 must use robust
  scaling** (e.g. log or quantile transforms fitted on training folds) for the DisVoice features.

---

## 6. Temporal alignment — how it works and how we verified it

**By construction.** All frame-level streams are computed on the same windows
`[320t, 320t + 400)`. They share **one** offset table in the cache, so recording *i* has exactly the
same number of frames in every stream. Nothing is interpolated or resampled in time.

**By measurement (RAVDESS, all checks passed):**
1. **Frame count.** T = 1 + ⌊(N−400)/320⌋ equals HuBERT's own conv-stack arithmetic (9 boundary
   lengths) and the actual HuBERT output for every recording.
2. **Exact cross-stream identity.** Prosody `energy_db` = (10/ln10)·MFCC logE − 10·log10(400) holds
   on all 101,892 frames (max error 6e-6 dB). Two independently computed streams therefore analyse
   identical samples.
3. **HuBERT ↔ MFCC lag test.** A linear map from HuBERT frames to MFCC energy is fitted on 300
   recordings. On 300 others, the correlation between prediction and true energy is highest at
   **lag 0**: r = 0.981 for layer 0 (r = 0.853/0.854 at ±1 frame) and r = 0.949 for layer 9.
   A one-frame (20 ms) misalignment would move the peak.
4. **Unit tests.** An energy onset lands on the same frame in MFCC and prosody. A pitch jump appears
   at the frame whose centre crosses it (±2 frames; the Praat window is 50 ms). Praat's own frame
   times are asserted against the grid on every call (§5.2).
5. **Masks.** In padded batches, the masks of HuBERT, MFCC and prosody are identical, equal the
   frame counts, and padded values are 0. Masked mean/std ignore padding even when padding is
   filled with 10⁶. Long recordings are cropped with **one shared window** for all streams.

---

**CREMA-D:** all checks passed as well. The lag-0 peak is r = 0.978 (layer 0) and 0.926 (layer 9);
re-extraction matches within 3.4e-4 relative (HuBERT); resume and the DisVoice oracle pass.

**After the tracker switch (both corpora, 28/28 checks each):** all of the above re-run on the new
caches with identical alignment results (they involve HuBERT and MFCC, which did not change). New
checks: periodicity ∈ [0, 1]; `voiced` is binary; every voiced F0 lies within 60–800 Hz (observed
59.8–799.8 Hz); and **check 14**, the SHA-256 of audio, manifest, HuBERT, pooled HuBERT and MFCC
equals the pYIN cache, re-verified *after* the resume test.

## 6a. How HuBERT and MFCC were kept unchanged during the prosody rebuild

Changing any prosody parameter changes the config hash and therefore the cache directory. Rather
than re-running HuBERT, `adopt_unchanged()` (`src/features/store.py`) starts the new cache by
**copying** the preprocessed audio, manifest, HuBERT and pooled-HuBERT arrays from the old cache.
It does so only if the old cache was built from the same index and identical `audio`, `frames`,
`mfcc` and `hubert` config sections, and it checks every copy by SHA-256. Copying is required
because GPU inference is not bit-reproducible (re-extraction differs by up to 4.6e-4 relative).
MFCC is **recomputed** with the CPU stage and then checked to be byte-identical to the old file, so
"MFCC unchanged" is tested, not assumed. The resume test in validation re-runs HuBERT for 25
recordings and then writes back the original bytes.
Result: `mfcc_identical_to_source`, `hubert.npy_unchanged`, `hubert_pooled.npy_unchanged`,
`wav16k.npy_unchanged` and `manifest.csv_unchanged` are all true for both corpora
(`meta.json → adopted_from.verification`).

## 6b. Dataset summary of the caches

| | RAVDESS | CREMA-D |
|---|---|---|
| recordings processed | 1,056 / 1,056 usable | 7,441 / 7,441 usable (1 silent file excluded in Stage 2) |
| duration before → after trimming (mean) | 3.72 s → 1.95 s | 2.54 s → 2.38 s |
| energy removed by trimming (max) | 0.14 % | 0.75 % |
| frames (total / median / max per recording) | 101,892 / 91 / 215 | 879,279 / 118 / 250 |
| CPU stage (8 processes), pYIN | 56 s | 567 s |
| CPU stage (8 processes), Praat | 33 s (measured while another validation job ran; 19.5 s in an uncontended run with default settings) | 104 s |
| copying unchanged files (adoption, incl. SHA-256) | 20 s | 71 s |
| HuBERT stage (original extraction, reused) | 19.3 s (18.3 ms/rec), peak 446 MiB | 191.8 s (25.8 ms/rec), peak 458 MiB |
| cache size | 2.16 GB | 18.5 GB |
| CKA(layer 1, layer 9) | 0.71 | 0.75 |
| recordings with ≥ 1 imputed DisVoice value (DisVoice range) | 32 % (pYIN 38 %) | 54 % (pYIN 44 %) |

## 7. Reproducibility and caching

- Cache directory `C:\SER_data\features\<DATASET>\v1_<config-hash>`: any parameter change creates a
  new directory.
- `meta.json` records:
  - the full config, the dataset index fingerprint (the same SHA-256 as the Stage 3 splits) and the library versions;
  - the HuggingFace snapshots: config from `main@dba3bb02`, weights from `refs/pr/2@af46f65f` (safetensors);
  - every extraction run (time, recordings processed, GPU memory).
- `FeatureStore.open()` refuses a cache whose config, index, manifest order, completion flags or
  array shapes do not match.
- **Resumable:** arrays are pre-allocated, and progress flags mark finished recordings. Test: after
  wiping 25 recordings and resuming, MFCC/prosody were bit-identical and HuBERT matched exactly.
- **Deterministic:** re-extracting 20 random recordings from the raw audio reproduced audio, MFCC
  and prosody bit-identically, and HuBERT within 4.6e-4 relative (fp16 storage rounding).

---

## 8. Reproduced from the paper vs adapted

| Component | Paper (as reported) | This project | Status |
|---|---|---|---|
| SSL model | HuBERT-base, LibriSpeech 960 h | same checkpoint, pinned revision | reproduced |
| HuBERT layers | layers 1 and 9, fused | all 13 stored; selection by E2a probe | extended |
| HuBERT post-processing | conv block 10×18 (text) / 10×15 (figure) | not part of extraction (Stage 5) | — |
| MFCC | 12 MFCC + energy + Δ + ΔΔ = 39 | same 39, 25/20 ms framing | adapted (alignment) |
| Prosody | DisVoice 103 (RAPT, 60–350 Hz) | faithful re-implementation; F0 by Praat ac (DisVoice's alternative tracker, octave-jump cost 0.7); 60–350 Hz and primary 60–800 Hz variants | approximated (F0 tracker) |
| Frame-level prosody | — | 4 aligned contours | our addition |
| Clip length | 7 s, short clips repeated | trimmed, variable length, pad + mask | adapted |
| Sample rate | 16 kHz | 16 kHz | reproduced |

## References
- Zhao, R., et al. (2025). Leveraging Cross-Attention Transformer and Multi-Feature Fusion for Cross-Linguistic Speech Emotion Recognition. arXiv:2501.10408.
- Hsu, W.-N., et al. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM TASLP, 29, 3451–3460.
- Pasad, A., Chou, J.-C., & Livescu, K. (2021). Layer-wise analysis of a self-supervised speech representation model. IEEE ASRU.
- Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound. Proceedings of the Institute of Phonetic Sciences, 17, 97–110.
- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics, 71, 1–15.
- Xu, Y., & Sun, X. (2002). Maximum speed of pitch change and how it may relate to speech. JASA, 111(3), 1399–1413.
- Mauch, M., & Dixon, S. (2014). pYIN: A fundamental frequency estimator using probabilistic threshold distributions. IEEE ICASSP.
- Kornblith, S., et al. (2019). Similarity of neural network representations revisited. ICML.
- Vásquez-Correa, J. C., et al. DisVoice (software), https://github.com/jcvasquezc/DisVoice (MIT licence).
