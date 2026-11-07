# Central Experiment Log

Rules (agreed with the team):
- Every entry records **what was actually executed**: date, command, config, seed(s), git/commit or file hash where possible, and where the outputs were saved.
- Failed or aborted runs are logged too, with the reason. Nothing is silently dropped.
- Numbers in the report must be traceable to an entry here and a file in `results/`.
- Results from the reference paper (Zhao et al., arXiv:2501.10408) are never copied into this log as our own.

---

## 2026-11-05 — Stage 1: Environment setup

- Created isolated virtual environment `C:\SER_data\.venv` (Python 3.12.10) outside OneDrive.
  Reason: system Python contains unrelated packages; a venv inside OneDrive would sync thousands of files.
- PyTorch wheel chosen: `torch==2.14.1+cu126` (driver 592.82 supports CUDA 13.1; cu126 chosen for compatibility with older drivers on a teammate's machine).
- torchaudio intentionally NOT installed (audio I/O via soundfile/librosa; avoids ffmpeg backend issues on Windows).
- Installed direct deps (pinned in `requirements.txt`); `pip check`: no broken requirements.
- `scripts/check_env.py` -> ALL CHECKS PASSED (CUDA 12.6 runtime, RTX 3050 Laptop, 4.00 GiB, fp16 matmul OK).
- `scripts/check_hubert.py` (facebook/hubert-base-ls960, fp16, 7 s random input):
  94.4 M params; 13 hidden states of shape (1, 349, 768); 49.86 frames/s (20 ms hop);
  all finite; peak GPU memory 255 MiB. -> HuBERT-base fits comfortably in 4 GB.
  Note: 349 frames = 1 + (112000 - 400) // 320, i.e. 25 ms receptive field, 20 ms stride.
  MFCC/prosody must use the same window/hop to be frame-aligned (Stage 4).

## 2026-11-05 — Stage 2: Dataset preparation (in progress)

- RAVDESS downloaded from Zenodo record 1188976; MD5 `bc696df654c87fed845eb13823edef8a` verified; 1440 WAVs.
- CREMA-D: repository lists 7442 WAVs. GitHub LFS media server is slow (~22 s per 80 KB file, measured);
  8 workers ≈ 20 files/min, 24 workers ≈ 48 files/min -> download running in background (~2.5 h).
- `scripts/stage2_prepare_data.py --datasets RAVDESS` (all 1440 files decoded):
  - status: 1440 ok, 0 bad. 24 speakers (12 M / 12 F). 48 kHz for all files.
  - **5 files are stereo** (Actor_01 ×2, Actor_05 ×1, Actor_20 ×2; 2 of them in the 6-class subset) -> preprocessing must down-mix to mono.
  - 6-class subset: 1056 files (neutral 96, others 192 each); excluded calm 192, surprised 192.
  - 44 included clips per speaker (perfectly balanced across speakers).
  - duration min 2.94 s / median 3.67 s / max 5.27 s (before trimming); 0 files with >0.1 % clipped samples.
- Bug found by visual inspection and fixed: speaker-distribution plot labelled the gender groups in the wrong order (the data were correct; the plot sort order was not).
- `pytest`: 7 passed (parsing, label mapping, exclusion, malformed names, silent/short/unreadable detection).
- Portability fix: `rel_path` now stored with "/" (`Path.as_posix()`); previously Windows "\" would break on Linux/Colab.
  RAVDESS index regenerated; counts unchanged.

## 2026-11-05 — Stage 3: Speaker-independent splits (RAVDESS)

- Method: `StratifiedGroupKFold` on recordings, `groups = speaker`, stratified by gender, split seed 42
  (fixed, independent of model seeds). Validation = first of 5 inner group splits of the non-test speakers.
- `scripts/stage3_make_splits.py --datasets RAVDESS`:
  6 folds × (train 16 / val 4 / test 4 speakers) = (704 / 176 / 176 recordings); every val/test partition 2 M + 2 F;
  class counts per test/val partition: neutral 16, others 32 each.
- Verification passed: speaker_disjoint, file_disjoint, full_coverage, all_classes_present, both_genders_present,
  each_speaker_tested_once. Determinism re-check and save→load round trip (with fingerprint check) passed.
- Saved: `data/splits/RAVDESS_speaker_folds.json` (index_sha256 `b875a4f2…5331`), `data/splits/RAVDESS_split_summary.csv`,
  `plots/splits_RAVDESS.png`.
- `pytest`: 19 passed (12 new, including 5 negative tests that inject leakage and confirm it is detected).
- CREMA-D download: 1408 / 7442 at time of entry, 0 failures so far.

## 2026-11-05 — Stage 2/3 completed for CREMA-D

- Download finished: 7442 / 7442 WAVs, 0 failures.
- `stage2_prepare_data.py --datasets RAVDESS CREMA-D`: CREMA-D 7442 files, **1 silent file** (`1076_MTI_SAD_XX.wav`,
  status "silent", excluded as unusable and listed in `stage2_summary.json`) -> 7441 usable; 91 speakers (48 M / 43 F),
  ages 20–74 (median 33); classes neutral 1087, sad 1270, others 1271; all 16 kHz mono; 0.66–5.0 s (median 2.50 s);
  **246 files with > 0.1 % clipped samples** (kept, documented). RAVDESS rows unchanged (fingerprint `b875a4f2…` identical).
- `stage3_make_splits.py --datasets CREMA-D`: 5 folds, ~58/15/18 train/val/test speakers; all leakage checks passed.
  Cross-corpus files: CREMA-D→RAVDESS (train 73 / val 18 CREMA-D spk, test 24 RAVDESS spk) and RAVDESS→CREMA-D
  (train 19 / val 5, test 91). RAVDESS split JSON SHA-256 identical before/after (`DDCFFCCF…9BA5`).

## 2026-11-05 — Stage 4: Feature extraction (RAVDESS)

- HuBERT checks (`scripts/stage4_hubert_checks.py`, 24 RAVDESS clips): hidden-state semantics verified by manual
  recomputation (max abs error 0.0 for all 13); fp16 vs fp32 inference cosine >= 0.9999, rel. error mean 0.11–0.18 %,
  max 1.4 %; fp16 storage rel. error 0.02 %; fp32 14.5 ms/clip (506 MiB), fp16 11.0 ms/clip (415 MiB).
  **Decision: fp32 inference, fp16 storage.**
- DisVoice source read (commit 7bb41a4, MIT); original `prosody_functions.py` vendored as test oracle. pysptk (RAPT) has no
  Py3.12/Windows wheel -> pYIN used. Measured on one female angry clip: 60–350 Hz pYIN voiced 14 %, 60–700 Hz voiced 34 %
  (median F0 364 Hz) -> both F0 ranges stored.
- Cache `C:\SER_data\features\RAVDESS\v1_f9418feedb` (rebuilt from scratch for clean metadata; first build gave identical
  validation results): prepare 9.0 s, CPU stage 56.2 s (8 processes), HuBERT 19.3 s = 18.3 ms/recording, peak GPU 446 MiB.
  1056 recordings, 101,892 frames (median 91, max 215 per recording). Storage 2.16 GB (HuBERT 1.94 GB).
- `scripts/stage4_validate_features.py --dataset RAVDESS`: **all 25 checks passed** (completeness, finiteness, shapes, frame
  formula, offsets, exact energy identity max err 6e-6 dB, HuBERT↔MFCC lag-0 correlation peak r=0.981 (L0) / 0.949 (L9),
  masks, bit-identical re-extraction of 20 recordings (HuBERT rel 4.6e-4), resume after wiping 25 recordings,
  DisVoice-103 == original code on 30 real recordings, trimming removes median 0.005 % / max 0.14 % of energy).
- Findings: 13.6 % of voiced frames > 350 Hz (female fearful 43.6 %, happy 28.7 %, angry 25.7 %);
  7 recordings (0.7 %, all male, strong intensity) without any voiced frame in at least one pYIN pass (kept, flagged);
  38 % of recordings have >= 1 imputed DisVoice value (mostly unvoiced-energy skew/kurtosis);
  DisVoice values heavy-tailed (|max| 1677 / 4739) -> robust scaling needed in Stage 5;
  CKA(layer1, layer9) = 0.71.
- Bugs found and fixed during validation: HuBERT panel colour map in alignment figure (one-hue map for signed data),
  MFCC figure scale (deltas invisible), trimming summary divided the count by 16000, stage timings overwritten by the
  resume test (now a run history), non-writable-array warning in batching.
- `pytest`: 49 passed.

## 2026-11-05 — Stage 4: Feature extraction (CREMA-D) + F0-tracker diagnostic

- Cache `C:\SER_data\features\CREMA-D\v1_f9418feedb`: 7441 recordings, 879,279 frames (median 118, min 32, max 250);
  prepare 12.5 s, CPU 567.4 s (8 processes), HuBERT 191.8 s = 25.8 ms/recording, peak GPU 458 MiB; 18.5 GB.
- `stage4_validate_features.py --dataset CREMA-D`: **all 25 checks passed** (lag-0 peak r=0.978 L0 / 0.926 L9;
  HuBERT re-extraction rel 3.4e-4; resume OK; DisVoice oracle OK; trimming removes max 0.75 % energy).
- **Problem found:** pYIN voiced-frame fraction 37 % (RAVDESS 72 %); 767 / 7441 recordings (10.3 %) with no voiced
  frame; by class disgust 20.5 %, sad 17.8 %, happy 7.0 %, fearful 7.7 %, neutral 5.3 %, angry 2.8 %; M 14.1 % / F 6.0 %;
  affected recordings quieter (median raw peak 0.108 vs 0.212).
- `scripts/stage4_f0_tracker_comparison.py --n 300` (praat-parselmouth 0.4.7 installed in venv FOR EVALUATION ONLY,
  not in requirements.txt): CREMA-D recordings without voicing pYIN 34 / Praat 0 (of 300); voiced fraction 0.40 vs 0.49;
  F0 agreement where both voiced: median 0.10 semitones. RAVDESS: 1 vs 0; 0.74 vs 0.68; 0.14 semitones.
- Status: **open decision** - recommended switch of all F0 tracking to Praat; awaiting team approval. Current caches use pYIN.
  -> resolved 2026-11-06 (next entry).
- Also: RAVDESS validation re-run after a figure-table fix; all checks passed again.

## 2026-11-06 — Stage 4: F0 tracker switched from pYIN to Praat (team decision: Option A)

- Reason (team): pYIN's class-dependent voicing failures on CREMA-D (disgust, sad) must not enter the features.
- `praat-parselmouth==0.4.7` (bundles Praat 6.1.38) added to `requirements.txt`; `pip check` clean.
- DisVoice's own Praat route (fetched from GitHub master: `vuv_praat.praat`, `praat_functions.py`) runs plain
  `To Pitch... step 60 350` (= autocorrelation, all defaults) and maps time stamps to a linspace grid by nearest
  neighbour. Note: DisVoice README says Praat is the default, but its code default is RAPT.
- Implementation (`src/features/prosody.py: f0_praat`): Praat ac sampled EXACTLY on our grids by zero-padding
  (formula from Praat's Sampled_shortTermAnalysis; actual frame times asserted on every call to 1e-9 s).
  Frame channel 2 renamed `voiced_p` -> `periodicity` (max autocorrelation strength of voiced candidates;
  NOT a probability; documented in report §5.2). Dimensions unchanged: (T,4) frames, 2 x (103,).
- Settings investigation `scripts/stage4_praat_settings.py --n 300` (same 300 recordings per corpus as the
  tracker comparison; 15 settings; synthetic tones and octave glides) -> `results/stage4_praat_settings.json`:
  - ceiling 800 Hz (700 Hz still clips 0.76 % RAVDESS voiced frames; above the ceiling Praat outputs
    sub-harmonics, e.g. 600 Hz -> 300 Hz with the DisVoice 350 Hz ceiling);
  - floor 60 Hz (75 Hz: 2.1 % CREMA-D frames at floor, 70 Hz tone lost);
  - voicing threshold 0.45 (default);
  - octave_jump_cost 0.7 instead of 0.35: recordings with a > 7 st jump in 10 ms 38 -> 3 (RAVDESS) and
    77 -> 0 (CREMA-D) of 300; GPE vs pYIN 0.67 -> 0.52 % / 0.96 -> 0.67 %; octave glides in 50-100 ms still
    0 % error; voiced fraction -0.8 / -1.4 pp.
- Intermediate run with Praat DEFAULT settings (cache v1_36607e8614): built for both corpora, 28/28 validation
  checks each; superseded because of the octave jumps (median F0msemax 299 vs pYIN 153 on RAVDESS).
  Results archived (`results/stage4_validation_<DS>_praat_ojc035_v1_36607e8614.json`); cache directories deleted.
- Final caches `v1_efe9081789` built with `stage4_extract_features.py --stages cpu --adopt-from v1_f9418feedb`:
  audio, manifest, HuBERT, pooled HuBERT copied from the pYIN cache with SHA-256 verification (RAVDESS 20 s,
  CREMA-D 71 s); MFCC recomputed and verified byte-identical to the pYIN cache; CPU stage RAVDESS 33.0 s
  (concurrent with another validation job), CREMA-D 103.8 s (pYIN: 56 s / 567 s).
- `stage4_validate_features.py`: **28/28 checks passed on both corpora** (new: check 13 voicing/pitch ranges,
  check 14 SHA-256 of unchanged streams vs source cache, after the resume test, which now restores the
  original HuBERT bytes). Alignment unchanged (lag-0 r 0.981/0.949 RAVDESS, 0.978/0.926 CREMA-D).
- Full-corpus voicing, pYIN -> Praat: RAVDESS voiced frames 71.6 -> 65.4 %, recordings without voicing
  (frame / DisVoice / wide) 6/2/4 -> 0/0/0; CREMA-D 37.0 -> 45.5 %, 767/717/507 -> 0/0/0;
  12 CREMA-D recordings still < 10 % voiced (9 disgust, 10 male).
- 300-recording comparison (`scripts/stage4_f0_tracker_comparison.py`, rewritten: pYIN recomputed and checked
  bit-identical to the old cache 300/300, Praat read from the new cache): CREMA-D recordings without voicing
  34 -> 0, < 10 % voiced 40 -> 1; F0 agreement where both voiced median 0.15 st (RAVDESS) / 0.11 st (CREMA-D);
  channel-2 Pearson r (pYIN vprob vs Praat periodicity) 0.57 / 0.51. Pre-switch JSON archived as
  `results/stage4_f0_tracker_comparison_pyin_v1_f9418feedb.json`.
- Remaining limitations (report §5.6): no ground-truth pitch; male F0 p99 about 595 Hz not checked by ear;
  F0 > 800 Hz would become a sub-harmonic; DisVoice heavy tails larger (max |F0msemax| 9,252 RAVDESS /
  11,477 CREMA-D, wide range); recordings with >= 1 imputed DisVoice value RAVDESS 32 %, CREMA-D 54 %
  (DisVoice range).
- New figures `plots/features_voicing_{RAVDESS,CREMA-D}.png`; other Stage 4 figures regenerated. Layout fixes
  (overlapping log-axis labels, legends/notes over data) redrawn with the new
  `stage4_validate_features.py --figures-only` (cache + saved validation JSON, no checks re-run).
- Stage 3 split files: SHA-256 identical before/after (RAVDESS `DDCFFCCF…9BA5`, CREMA-D `A6488219…8EDC`).
- `pytest`: **67 passed** (18 new in `tests/test_praat_f0.py`; pYIN alignment test now runs on Praat).
- pYIN cache `v1_f9418feedb` (20.2 GB) kept as the documented "before" state; it can be deleted once the report
  no longer needs the bit-identity check of the comparison script.

## 2026-11-06 — Stage 5: Stage 4 audit + E1 (MFCC + SVM / Random Forest)

- Stage 4 approved by the team. Separate config records: `configs/feature_cache_pyin_v1_f9418feedb.json`
  (superseded) and `configs/feature_cache_praat_v1_efe9081789.json` (current, used by all of Stage 5).
- Audit `scripts/stage5_audit_stage4.py` -> `results/stage5_audit_stage4.json` (read-only). Key findings:
  unvoiced frames with periodicity >= 0.45: 66 % RAVDESS / 46 % CREMA-D; DisVoice imputation class-dependent on
  CREMA-D (angry 35 % vs sad 62 % of recordings); 21-26 features |skew| > 2, 55 with negative values, robust z up
  to 11,387; pooled HuBERT consistent with frames (rel 1.6e-4); all split rows resolve to cache rows.
- New modules: `src/training/data.py` (fold_rows over load_fold), `src/evaluation/metrics.py`,
  `src/training/preprocessing.py` (FrameStandardizer, DisVoicePreprocessor: median fill, signed log for
  train-|skew| > 2, z-score, clip +-5, optional indicators), `src/models/e1_classical.py`.
- Tests `tests/test_stage5_e1.py`: 11 passed (incl. test rows set to NaN -> identical selection).
- E1 config (`configs/e1_mfcc_classical.json`): aggregations mean (39) / mean_std (78) / functionals (195);
  SVM RBF C {0.1,1,10,100} x gamma {0.1,1,10}/d; RF 300 trees, max_features {sqrt, 0.2}, min_leaf {1,5},
  seeds 0,1,2; class_weight balanced; selection = validation macro-F1 over aggregation x hp; train speakers only.
- Command: `scripts/stage5_e1_run.py --datasets RAVDESS CREMA-D` (run hashes eab09031eb2a / 2a8f123a8356);
  44 runs (RAVDESS 6 folds x [SVM + 3 RF seeds], CREMA-D 5 folds x 4); all completed, none failed.
  Fold 0 RAVDESS ran first as a pipeline check (results kept, identical config).
- Results (test, mean +- std over folds; `results/E1/<DS>/summary.json`):
  RAVDESS SVM macro-F1 0.537 +- 0.059, UA 0.543 +- 0.061, acc 0.549 +- 0.061;
  RAVDESS RF 0.493 +- 0.032 / 0.503 +- 0.039 / 0.512 +- 0.032;
  CREMA-D SVM 0.503 +- 0.021 / 0.509 +- 0.020 / 0.507 +- 0.021;
  CREMA-D RF 0.457 +- 0.028 / 0.481 +- 0.023 / 0.478 +- 0.024.
  SVM > RF in 5/6 and 5/5 folds. mean -> mean_std +0.05..+0.10 macro-F1; functionals vs mean_std +0.023 (RAVDESS
  SVM) / ~0 otherwise. Validation optimism +0.05 (RAVDESS SVM). Angry easiest; neutral<->sad (RAVDESS),
  happy->angry and fearful->sad (CREMA-D) main confusions.
- Checkpoints `C:\SER_data\checkpoints\E1\`; figures `plots/E1_*.png`; documentation `reports/stage5_models.md`.
- Not started: E2-E4 (await approval).

## 2026-11-06 — Stage 5: E2 (frozen HuBERT + MLP head) and E2 audit

- E1 approved. Inspection `scripts/stage5_e2_inspect.py` -> `results/stage5_e2_inspect.json`: frames (T,13,768) fp16;
  index 0 = Transformer input, 1-12 = layer outputs (Stage 4 recomputation error 0.0); frame counts equal across
  streams; masked pooling of padded batches == pooled cache (max rel 1e-4) and padding-invariant; cache = approved
  Praat config (ojc 0.7).
- New: `src/models/e2_hubert.py`, `src/training/trainer.py` (deterministic, model factory -> seed controls init),
  `src/utils/resources.py` (RAM via OS API; psutil not installed), `without_test()` in `src/training/data.py`,
  `scripts/stage5_e2_{inspect,run,controls,audit,plots}.py`, `tests/test_stage5_e2.py` (11 tests).
- Bug found by the reproducibility test BEFORE any E2 run: model built before seeding -> initial weights not
  controlled by the seed. Fixed (trainer takes a factory, seeds first).
- Config `configs/e2_hubert_head.json`: per fold A) 13 layers + concat(1,9) + weighted(all) x {mean, mean_std}, default
  head (h 256, p 0.2, lr 1e-3, weighted CE) = 30 runs; B) h {256,512} x p {0.2,0.4} x lr {1e-3,3e-4} = 8; C) unweighted
  CE (controlled) = 1; D) final config seeds 0,1,2 -> checkpoint, reload, test once. AdamW wd 1e-4, batch 32, <= 50
  epochs, early stopping on val macro-F1 (patience 10). Run hashes RAVDESS fdb1ea30ea4d, CREMA-D 3da552ff44ca.
- Execution: RAVDESS fold 0 pipeline check, then `stage5_e2_run.py --datasets RAVDESS CREMA-D`. The run was
  INTERRUPTED (session end) during CREMA-D fold 3 phase A (29/30 done); all 416 result files parsed OK; resumed with
  `--folds 3 4` (same run hash, cached runs reused). No failed runs, no retries with changed settings.
  RAVDESS fold 0 selection.json was rewritten (identical content) by the full run after its D files existed ->
  driver now keeps an identical selection; result writes made atomic.
- Results (TEST, mean +- std over folds, 3 seeds averaged per fold; `results/E2/<DS>/summary.json`):
  RAVDESS macro-F1 0.762 +- 0.054, UA 0.771 +- 0.052, acc 0.764 +- 0.057, macro-P 0.775 +- 0.055;
  CREMA-D 0.713 +- 0.030, UA 0.714 +- 0.029, acc 0.712 +- 0.029, macro-P 0.716 +- 0.030.
  Per seed (macro-F1): RAVDESS 0.755 / 0.766 / 0.764; CREMA-D 0.714 / 0.716 / 0.709. Within-fold seed std 0.011 / 0.006.
  vs E1-SVM (paired, same test rows): +0.225 +- 0.049 (6/6 folds), +0.210 +- 0.021 (5/5).
- Validation analysis: middle layers 5-7 best (RAVDESS L5 mean+std 0.806; CREMA-D L5-7 ~0.70); L0/L1 weakest; final
  layer 12 not best; concat(1,9) not better than best single layer; weighted(all) best on CREMA-D (0.72). mean+std
  helps only for layers 0-1 (+0.03..+0.08, all folds), ~0 from layer 5 up. Class weighting: |dval| <= 0.018.
  Learned layer weights: consistent peak at layers 4-7 but near-uniform (max 0.08-0.12 vs 1/13; entropy >= 0.985).
- Validation->test gap: RAVDESS +0.059 (folds 0,5: +0.12), CREMA-D +0.018.
- Audit `scripts/stage5_e2_audit.py` -> `results/E2/audit.json`: ALL CHECKS PASSED (completion, single run hash, A-C
  validation-only, selection re-derivable from A-C, A-C before D, test rows, checkpoints, train-only standardiser,
  class weights from train labels, exact re-training of fold 0 seed 0, identical E1/E2 test rows, split SHA-256,
  Stage 4 cache SHA-256, prosody mtimes).
- Label-permutation control: test macro-F1 0.187 +- 0.050 (RAVDESS), 0.186 +- 0.014 (CREMA-D); chance 0.167.
- Cost: final head training 0.8 s / 7.0 s; inference 0.04 / 0.07 ms per recording (excl. HuBERT 18-26 ms/rec);
  GPU peak 85 / 759 MiB; RAM peak 1.5 / 3.5 GB. Head parameters 198k-1.58M (HuBERT 94.4M frozen).
- `pytest`: 89 passed. Checkpoints `C:\SER_data\checkpoints\E2\`; figures `plots/E2_*.png`, `plots/E1_vs_E2_*.png`;
  documentation `reports/stage5_models.md` §4.
- Not started: E3, E4 (await approval).

## 2026-11-06 - Stage 5 / E3: frame-level concatenation fusion (approved decisions; pilot; full run)
- Approved decisions: HuBERT layer per fold = best single layer of that fold's own E2 phase-A validation (no fold
  averaging); parameter-matched self-attention control included; DisVoice-103 only as a separate ablation; fold-0 pilot
  before the full experiment.
- New code: `src/models/attention.py` (explicit Q/K/V multi-head attention, pre-LN residual block; shared with E4),
  `src/models/e3_fusion.py` (projection -> concat -> fusion -> masked mean+std pooling -> head; stream subsets; DisVoice
  input; E4 reference parameter count; matched_sa_blocks), `src/training/sequence_data.py` (one HuBERT layer on GPU,
  aligned frame gathering, train-only standardisers, length-bucketed seeded batches), `src/training/seq_trainer.py`,
  `scripts/stage5_e3_run.py` (L/S/D phases, --pilot, --summarise), `scripts/stage5_e3_controls.py`,
  `scripts/stage5_e3_audit.py`, `scripts/stage5_e3_plots.py`; `tests/test_stage5_e3.py` (31 tests: attention
  dimensions / masks / reference formula / fully-masked rows / large inputs / fp16 autocast / gradient flow / variable
  length; E3 padding invariance, stream subsets, parameter counts and E4 matching; data alignment, train-only
  statistics, bucketing, prediction order; reproducible training; layer rule; test isolation by NaN poisoning).
- One test initially failed (synthetic learning check: 40 optimisation steps too few); fixed by training the synthetic
  model longer (30 epochs, lr 3e-3). Not a code defect. `pytest`: 120 passed.
- Layers selected: RAVDESS 6,10,5,5,5,5; CREMA-D 7,6,12,11,5 (CREMA-D folds 2/3 margins 0.002 - selection noise).
- Control: 4 self-attention blocks, 751,750 params vs E4 reference total 769,286 (-2.3 %); E3 concat 221,574.
- Pilot (fold 0, phases L+S only, NO test predictions; `results/E3/pilot_<DS>.json`): alignment max abs diff 0.0 in all
  streams (24 real recordings each); GPU peak 694 MiB (RAVDESS) / 2,073 MiB (CREMA-D incl. 1.35 GB resident layer);
  RAM 1.8 / 6.6 GB (transient while reading layer); per run 5.7/16 s (RAVDESS concat/control), 24/104 s (CREMA-D);
  estimate 0.5 h + 2.0 h. No non-finite values. FINDING: grad-norm clipping at 1.0 active on 96-100 % of CREMA-D steps
  (mean norms 2-7) -> acts as per-step normalisation, not a safety net; kept unchanged (fixed a priori, shared with E4,
  no extra tuning); documented as optimiser difference vs E2 (no clipping).
- Minor fix after pilot (no numerical effect): MFCC/prosody arrays copied (np.array) instead of read-only memmap views
  (PyTorch non-writable warning).
- Full run launched: `stage5_e3_run.py --datasets RAVDESS CREMA-D` (fold 0 S runs reused from the pilot, same hash).
- INTERRUPTION: the full run was stopped by Claude Code (host critically low on memory) while loading the CREMA-D
  HuBERT layer, i.e. before any CREMA-D final run. State: RAVDESS complete (6 folds x 8 S + selection + 30 D, no
  partial files); CREMA-D only fold-0 S/selection (from the pilot). Not restarted automatically.
- Fix (no numerical effect): `load_hubert_layer` now uses buffered file reads in 10k-frame chunks instead of a memory
  map (mapped pages had put up to 6.6 GB into the working set). Verified byte-identical on RAVDESS layer 5; E3 tests
  31 passed. Resume plan: `stage5_e3_run.py --datasets CREMA-D` in its own process (same run hashes -> RAVDESS and
  CREMA-D fold-0 S runs reused).
- RESUME (approved: CREMA-D only, no setting changed) 2026-11-06 15:05 - 17:13. Pre-resume snapshot of 425 files
  (`results/E3/snapshot_before_cremad_resume.json`); after the run 425/425 byte-identical. CREMA-D fold-0 hash
  927c2c9fa380 matched -> saved S/selection reused. RAM: layer loads 14-16 s; process peak 2.28 GB (first load),
  3.13 GB thereafter; OS max working set 2.98 GB; min system free RAM 4.59 GB (watchdog threshold 1 GB never reached).
  Monitoring caveats: first watchdog tracked the venv launcher stub (second watchdog attached to the interpreter);
  both appended to one CSV -> some samples lost to file locks (2,970 rows kept). stderr empty; no failed/retried runs.
  All 11 folds complete: 330 D files, 330 checkpoints.
- Post-run (no model setting changed): `--summarise` -> `results/E3/<DS>/summary.json`, `runs.csv`;
  `stage5_e3_controls.py` -> permuted-label test F1 0.158 +- 0.036 (RAVDESS), 0.174 +- 0.014 (CREMA-D), chance 0.167;
  `stage5_e3_audit.py` (under RAM guard, peak WS 4.03 GB) -> 36/36 checks PASS incl. exact re-prediction of all 330
  checkpoints; `stage5_e3_plots.py` -> 6 figures (layout-only fixes after visual check: contribution labels moved to a
  top row, search legend moved outside data + marker shapes). pytest 120 passed.
- RESULTS (test macro-F1, mean +- std over folds): RAVDESS E3 0.729 +- 0.064, control 0.726 +- 0.037, E2 0.762, H-only
  0.747, H+P 0.749; CREMA-D E3 0.695 +- 0.027, control 0.669 +- 0.025, E2 0.713, H-only 0.685.
  E3 - E2: -0.033 (0/6, p .031) / -0.018 (0/5, p .062). Control - E3: -0.003 (3/6) / -0.025 (0/5).
  HuBERT | M+P +0.180 / +0.117 (all folds); M+P | H -0.018 (0/6, p .031) / +0.010 (4/5); DisVoice +0.005 / -0.004;
  indicators -0.013 / -0.008. Shapley H/M/P: 0.364/0.131/0.068 (RAVDESS), 0.283/0.156/0.090 (CREMA-D).
  Full analysis: `reports/stage5_models.md` 5.6-5.13.
- OPEN ISSUE for E4 (not acted on): the self-attention control selected the lowest lr of the grid (3e-4) in 11/11 folds
  -> possibly under-tuned; E4 shares the grid. Needs a decision before E4 (validation-only justification).

## 2026-11-06 - E4 revised protocol PROPOSED (not run; awaiting team review)
- Team request: extend lr search below 3e-4; same search and selection for the control and E4; validation macro-F1
  only; architecture/folds/seeds/evaluation fixed; consider a fair search for the concat comparator. E3 results kept
  as run (no E3 file touched).
- Proposal (reports/stage5_models.md section 6): lr {1e-3, 3e-4, 1e-4, 3e-5} x dropout {0.2, 0.4} = 8 candidates;
  max_epochs 50 -> 150 (no scheduler; the 88 E3 S runs never reached 50 epochs, max 43 -> E3 cells must reproduce
  exactly); tuned models with their own search: xattn_HMP (E4), sa_HMP (control), concat_HMP (re-searched), concat_H;
  ablations xattn_HM / xattn_HP / xattn_H_MPcat with E4's hp; context-corruption diagnostic; decision rule and Holm
  correction fixed in advance; results go to results/E4 only.
- E4 parameters by instantiation: 769,286 total (fusion 596,992 = 4 x 132,736 cross blocks + 2 x 32,896 fuse + 256 LN);
  equals e4_reference_fusion_params(); control 751,750 (-2.3 %) -> no re-matching needed.
- Synthetic benchmark (scratch prototype, not project code; RTX 3050, batch 32): E4 / control training step
  1.03-1.06x, activation memory +7-11 %. Joint time-permutation invariance confirmed (max |diff| 9e-8; no positional
  encoding, as in the control). Budget estimate: CREMA-D ~12 h (8-21), RAVDESS ~2 h; fold-0 validation-only pilot
  first. Scripts: scratchpad e4_proto_bench.py, e4_ablation_bench.py, e3_search_stats.py.
- Report: E3 summary statement added to section 5.13 (HuBERT dominant; extra streams no consistent gain). No result changed.

## 2026-11-06 - E4 protocol APPROVED; fold-0 validation-only PILOT run (full E4 NOT launched)
- Team decisions: lr {1e-3,3e-4,1e-4,3e-5} x dropout {0.2,0.4}, max 150 epochs, same search for xattn_HMP / sa_HMP /
  concat_HMP / concat_H; all three E4 ablations; fold-0 validation-only pilot; E1-E3 preserved; no-positional-encoding
  limitation documented; feature swapping = supplementary diagnostic only.
- Code: src/models/e4_cross_attention.py (E4CrossAttention, structures HMP/HM/HP/H_MPcat; 769,286 params = planned
  reference), scripts/stage5_e4_run.py (pilot only; phase D refused until review), configs/e4_cross_attention.json,
  tests/test_stage5_e4.py (19 tests). pytest 139 passed.
- Documentation correction: the proposal claimed E4 ignores cross-stream frame alignment; wrong - the cross-attention
  weights are content-based (no position), but the fuse layers concatenate frame by frame. Unit-tested both; report 6.2 fixed.
- Snapshot of 1,432 protected files (results/E4/snapshot_before_e4_pilot.json); after pilot 1,432/1,432 unchanged, 0 new.
- Pilot (RAM guard, one CSV writer): RAVDESS run hash e584bddc7755, CREMA-D 6c711ebab212; 32 S runs each; no test
  predictions. E3 reproduction 16/16 EXACT (val F1, best epoch, epochs run, per-epoch val-F1/loss history, confusion).
  No cap hits (max 88 epochs). Selected (val F1): RAVDESS E4 0.2/1e-4 (0.813), control 0.2/3e-5 (0.831, lower edge),
  concat 0.2/3e-4 (0.826, = E3), H 0.2/1e-3 (0.827, upper edge); CREMA-D E4 0.2/3e-4 (0.687), control 0.2/3e-5
  (0.706, lower edge), concat 0.4/3e-4 (0.705, = E3), H 0.2/3e-5 (0.700, lower edge; flat landscape).
- Resources: GPU peak E4 716 / 2,097 MiB; process RAM 1.80 / 3.06 GB; min system free RAM 4.98 / 1.84 GB (host load
  outside the worker); CREMA-D layer load 99 s (E3: 14-16 s). Stalls: one RAVDESS epoch 440 s; CREMA-D one epoch 62 s and
  ~20 min of 1.5-2x slower attention epochs - timing only, results deterministic. Revised budget ~6.5 h clean, ~9 h
  with similar contention. Report section 6.10.
- Waiting for review. Not yet implemented: phase D, summary, permutation control, swap diagnostic, audit.

## 2026-11-06 - E4 full run: RAVDESS complete; CREMA-D STOPPED at start (host memory reaper), NOT restarted
- Approved by the team: full E4 with the 4-value grid (no further extension), all 4 tuned models + 3 ablations,
  permutation control, supplementary swap diagnostic, audit; stop and report on unsafe memory or discrepancy.
- Before launch: phase D, guard (E3 reproduction per fold before any test prediction; selection.json and config
  must match), summary (+Holm, pre-registered decision rule), scripts/stage5_e4_controls.py, scripts/stage5_e4_audit.py,
  scripts/stage5_e4_plots.py. pytest 147 passed (27 E4 tests incl. synthetic end-to-end D + audit + controls).
  Closed Spotify and Media Player (free RAM 5.1 -> 6.5 GB); Chrome/Edge/Notepad left open (possible user work).
- Chain (guarded, stop at first failure): full_RAVDESS 20:41:53 -> 21:36:14 exit 0 (min free RAM 4.16 GB, worker WS
  max 1.74 GB). Fold-0 pilot S runs reused (no retraining); all seed-0 finals reproduced their selection runs.
  RAVDESS: 6 folds x (32 S + selection + 21 D), 126 checkpoints, one run hash per fold, all checkpoints load. No
  RAVDESS test result has been analysed or reported yet.
- full_CREMA-D started 21:36:14 and was killed by Claude Code's background-shell memory reaper within ~10 s (log has only
  the start line; guard CSV one sample: free 6.24 GB, worker 0.61 GB) - i.e. while opening the store / starting to
  load HuBERT layer 7 of fold 0, before any training or test prediction. Our own guard (1 GB) did not trigger.
  Worker process tree gone; no CREMA-D D files, no CREMA-D checkpoints, no .tmp files; CREMA-D fold-0 pilot files
  untouched. E1-E3 snapshot 1,432/1,432 unchanged. Not restarted (awaiting the team).
- CREMA-D restart PREPARED (team-approved; to be run by the user outside Claude Code; Claude Code memory protection
  left enabled): scripts/run_e4_cremad_restart.ps1 (-Step run = only `stage5_e4_run.py --datasets CREMA-D`;
  -Step followup = summarise, controls RAVDESS/CREMA-D, audit; -Step selftest). Guard: stop < 1 GB free (kill tree,
  <tag>_ABORT.txt with the stage, exit 2, never restart); 1 s sampling while opening the store / loading a HuBERT
  layer, 5 s otherwise; CSV logs worker WS, system free RAM, current [stage] line and last log line; preflight refuses
  to start if another stage5 python runs, if protected files changed, or if free RAM < 3 GB. Runner now prints
  [stage] lines (log only; E4Config and run hashes unchanged - pilot hash test passes). New snapshot
  results/E4/snapshot_before_cremad_restart.json: 486 files (RAVDESS 6 folds x 54 result files, 126 RAVDESS
  checkpoints, 33 CREMA-D fold-0 pilot files, pilot reports, E4 config); verify OK; E1-E3 1,432 OK. pytest 147 passed.
  Guard self-test: ~1.2 s sampling during "loading", 5 s after; forced stop (threshold 99 GB) killed the dummy worker
  within ~1 s, wrote the stage, exit 2, no python left.
- 2026-11-06/07 CREMA-D RESTART executed by Claude Code on the team's instruction (user away; Claude Code memory
  protection left on). Preflight OK (6.85 GB free, no python running, AC power, snapshots 486 + 1,432 unchanged,
  CREMA-D fold-0 hash 6c711ebab212 recomputed). `run_e4_cremad_restart.ps1 -Step run` 21:59 -> 02:46, exit 0, stderr
  empty; min free RAM 3.68 GB, worker max 3.00 GB; layer loads 13-15 s; folds 54-66 min. Step-3 verification: 231 D
  files, 231 checkpoints load and match D files / selections; no .tmp; snapshots unchanged.
- Follow-up `-Step followup` 02:47 -> 03:03, exit 0: summarise; controls RAVDESS (worker 2.05 GB); controls CREMA-D
  (worker 4.57 GB, min free 2.73 GB - highest of the experiment, above the 1 GB threshold); audit 41/41 PASS
  (incl. exact re-prediction of 231 checkpoints, E3 S-cell reproduction in every fold). Snapshots unchanged before and
  after. pytest 147 passed. Figures generated (layout-only fixes after visual check: swap figure own y-axis per
  dataset - CREMA-D points were clipped; search figure panel titles split onto two lines).
- RESULTS (test macro-F1): RAVDESS E4 0.723 +- 0.061, control 0.730, concat 0.726, HuBERT-only 0.736 (E3 as run 0.729,
  E2 0.762, E1 0.537); CREMA-D E4 0.674 +- 0.023, control 0.686, concat 0.694, HuBERT-only 0.688 (E3 0.695, E2 0.713,
  E1 0.503). Pre-registered verdict for E4 - concat / control / HuBERT-only: "no consistent improvement" (all Deltas
  negative on both datasets; E4 - concat CREMA-D -0.020, 0/5). Permutation control 0.140 / 0.173 (chance 0.167).
  Swap diagnostic (supplementary) test Delta E4 -0.016 / -0.077, concat -0.015 / -0.088. lr edges: control at 3e-5 in
  1/6 + 2/5 folds (E3: 11/11 at its lower edge); HuBERT-only at 1e-3 in 5/6 RAVDESS folds (upper edge, not extended).
  Report sections 6.11-6.14.

## 2026-11-07 - Final comparative analysis E1-E4 (analysis only; no training, no test evaluation, no E5)
- scripts/stage5_final_analysis.py (read-only) -> results/final_analysis/final_analysis.json; plots FINAL_per_class_E2_vs_E4,
  FINAL_confusion_E2_E4, FINAL_error_overlap, FINAL_accuracy_vs_cost (layout-only fix after visual check: direct labels
  replaced by a legend in the accuracy-vs-cost figure). Report section 7 (observations / interpretations / hypotheses
  labelled separately); status line updated.
- Verification: 9 configurations x 2 datasets, test macro-F1 recomputed from stored metrics, confusion matrices and saved
  predictions + manifest labels (max diff 2.2e-16); all equal summary files and report values (3 dp). Audits E2 28/28,
  E3 36/36, E4 41/41.
- Main measured findings: E2 best (0.762 / 0.713; best in 9/11 folds); frame-level models beat E2 in 5/77 fold
  comparisons (max +0.009); E4 0.723 / 0.674 (-0.039 vs E2 on both). Per class: RAVDESS deficit concentrated on neutral
  (E4-E2 -0.100, 0/6) and shared by the HuBERT-only frame model (-0.096) -> frame-level pipeline effect, not fusion;
  CREMA-D all classes lower, happy most (-0.066) and happy is the class E4 loses on in all comparisons (0/5 vs E2,
  HuBERT-only, concat); sad/fearful/disgust confusion cluster common to all models. Error overlap: only-E2-correct
  9.9 / 11.6 %, only-E4-correct 6.8 / 7.8 %. HuBERT-only frame model < E2 with the same layer (RAVDESS 6/6 folds).
  Exploratory pooled 11-fold sign-flip (post hoc): E4-E2 p 0.002, HuBERT-only-E2 p 0.001.
- Integrity: 2,311 existing result/checkpoint/config files hashed before and after - 0 changed, 0 missing; E1-E3
  (1,432) and E4 (486) snapshots verify.

## 2026-11-07 - G-study PROPOSED (E2-vs-frame-level gap; diagnostic, not E5; nothing implemented or run)
- Report section 8. Facts gathered read-only: E2 phase-A validation - best single layer > weighted(all) in 6/6 RAVDESS
  folds, weighted(all) >= best single in 4/5 CREMA-D folds; E2 selected unweighted CE in 3/5 CREMA-D and 1/6 RAVDESS
  folds and head 512 in 7/11 folds (additional confounds F5 / head width); 1 of 462 E2 runs reached its 50-epoch cap;
  hubert.npy 16.4 GB (CREMA-D, 13 layers interleaved) -> frame-level learned layer weighting not feasible, uniform
  13-layer average (one chunked pass, 1.35 GB) is; 755 GB disk free.
- Design: 12 new HuBERT-only arms on a ladder E2 -> P0 (E2 code, single layer, matched grid G8) -> B0 (frame pipeline,
  no per-frame layers) -> Z-768-noLN / Z-768 / Z-noLN -> Z (existing E4 HuBERT-only); clipping on/off at both ends;
  representation (weighted(all) pooled, uniform-mean pooled and frame); budget B32 (G8 x head {256,512} x loss
  {weighted, unweighted}). All S phases for all arms before any test prediction (frozen selection manifest);
  reproduction guards (Z == E4 concat_H cells, P0 == coinciding E2 phase-B cells). Pre-registered pooled 11-fold
  sign-flip with Holm over 9 primary contrasts. Estimated ~6.5 h CREMA-D + ~1.2 h RAVDESS; fold-0 pilot first.
- 2026-11-07 G-study REVISION 1 (team review; still not implemented or run): section 8 updated.
  (1) statistics per dataset (mean / SD / fold wins / exact sign-flip p, Holm within dataset, descriptive);
  classification consistent / dataset-specific / none by fold consistency + effect size; pooled 11-fold analysis
  exploratory only; limitations of fold-level inference listed. (2) Exact reproduction gates R1-R6 separated from the
  P0-vs-E2 comparison: 58 E2 cells (RAVDESS 48: A_layerL 6, A_weighted 6, B_h256 16, B_h512 16, C_unweighted 4;
  CREMA-D 10: A_layerL 5, A_weighted 5; none reached E2's 50-epoch cap, max 42) + 88 E4 concat_H cells.
  (3) Frame-level learned layer weighting restored as arm Z-W (13 layers streamed from hubert.npy, ~14 GB/epoch on
  CREMA-D; per-layer statistics from pre-computed per-recording sums); UM kept as a separate approximation.
  (4) Pilot: validation only, no checkpoints, hard stopping criteria S1-S8; proposed guard metric change to available
  memory (free + standby), 1 GB threshold kept, needs approval. Estimated total 10-15 h CREMA-D + 1.5 h RAVDESS
  (Z-W dominant uncertainty); pilot ~2-2.5 h.

## 2026-11-07 - G-study implementation + reproduction gates; RESEARCH FROZEN AT E1-E4 (pilot NOT run)
- Implemented (not used for any reported result): trainer.py optional grad_clip (default None = unchanged E2 path);
  src/models/g_frame.py; src/features/g_derived.py; src/training/streamed_layers.py; scripts/stage5_g_run.py (pilot /
  gates-only, no test phase); stage5_g_precompute.py; stage5_g_protect.py; run_g_pilot.ps1 (guard on AVAILABLE memory
  via GlobalMemoryStatusEx, 1 GB; worker 6 GB; self-tests of both stop paths passed). tests/test_stage5_g.py: 12 new;
  full suite 159 passed. Environment check: torch 2.14.1+cu126 / numpy 2.5.4 installed 2026-11-05, unchanged since;
  all E2-E4 runs dated 2026-11-06.
- Integrity manifest results/G/manifest_before_g.json (2,338 files incl. feature caches): verified after every gate step
  (quick, 2,312 files) and at the end (full, 2,338): 0 changed/missing. E1-E3 (1,432) and E4 (486) snapshots unchanged.
- Reproduction gates (fold 0, validation only, guarded, exit 0): R1-R6 all exact - RAVDESS 12/12 pooled (R1, R2, R3 4,
  R4 4, R5, E2-internal) + 8/8 R6; CREMA-D 2/2 pooled (R1, R2) + 8/8 R6. Determinism probes (P0, Z) bit-identical on
  both datasets. Peak worker WS <= 1.74 GB, min available RAM >= 4.56 GB.
- B0 input check FAILED its pre-registered 1e-3 tolerance: max 2.96e-3 (RAVDESS), 2.55e-3 (CREMA-D). Preliminary
  diagnostic (before freeze): B0 GPU path vs float64 recomputation from the same fp16 frames <= 6.5e-4; fp16 frame cache
  vs E2's fp32-origin pooled cache up to 3.0e-3. Unresolved; new-pipeline issue only. Not investigated further.
- Team decision: research frozen at E1-E4. B0 quantisation diagnostic NOT run; G-study pilot NOT run; no test data
  accessed. New G files: results/G/ (gate S runs, gate reports, logs, manifest), configs/g_study.json; no derived
  features written; no checkpoints written.
- 2026-11-07 FINAL: G-study marked NOT RUN - CANCELLED BY SCOPE DECISION (report section 8 header + status box; report
  status line). No G-study training or test evaluation was conducted. Validation-only reproduction gates passed (R1-R6
  exact; determinism probes bit-identical). B0 storage-precision discrepancy (max 2.96e-3 / 2.55e-3 vs 1e-3 tolerance)
  documented as unresolved and specific to the unused G-study pipeline. E1-E4 results, checkpoints, configs and
  snapshots unchanged. Documentation-only update; no results, scripts, configurations, checkpoints or figures altered.
