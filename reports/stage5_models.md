# Stage 5 — Models and Experimental Evaluation

Every number in this document comes from executed runs. Sources are named next to each table
(`results/…`). Results reported in the reference paper (Zhao et al., arXiv:2501.10408) are never
mixed with ours.

Status: **audit done; E1, E2, E3 and E4 done and audited** (E4 audit: 41/41 checks). For E4 results and
limitations see §6.11–6.14. The final comparative analysis of E1–E4 is in §7 (no new training). §8 records a
proposed diagnostic G-study that was **not run, cancelled by scope decision**. The research is frozen at E1–E4.

---

## 0. Audit of the Stage 4 feature caches (`scripts/stage5_audit_stage4.py` → `results/stage5_audit_stage4.json`)

| Item | Finding | Consequence for Stage 5 |
|---|---|---|
| Cache identity | Both corpora use `v1_efe9081789`: Praat ac, octave-jump cost 0.7, 60–800 Hz (DisVoice variant 60–350 Hz). Validation passed 28/28. Audio, HuBERT and MFCC are byte-identical to the pYIN cache. | All experiments read this cache only. The pYIN cache is documented separately (`configs/feature_cache_pyin_v1_f9418feedb.json`) and is not used. |
| Periodicity channel | 66 % (RAVDESS) / 46 % (CREMA-D) of **unvoiced** frames have periodicity ≥ the voicing threshold 0.45. Only 0.2 % / 0.5 % of voiced frames lie below it. | Periodicity is a separate signal, not a soft copy of `voiced` and not a probability. It is kept as an input channel (already in [0, 1]). |
| Imputed DisVoice values | Recordings with ≥ 1 imputed value: RAVDESS 32 % / 29 % (DisVoice / wide range), CREMA-D 54 % / 53 %. **Class-dependent on CREMA-D:** angry 35 % vs sad 62 %. No feature is constant. | The cache fills imputed values with 0. Preprocessing replaces them with the **training-fold median**. Missingness indicators could act as a label shortcut, so they are a **separate ablation**, off by default. |
| Extreme values | 21–26 of 103 features have \|skew\| > 2. 55 features take negative values. Robust z (median/IQR) up to 192 (RAVDESS) and 11,387 (CREMA-D). | Signed log for skewed features (selected on training data), standardisation, clipping at ±5 SD (§2). |
| HuBERT | Frames (T, 13, 768) fp16; pooled (n, 13, 1536) fp32 = [mean ‖ std] per hidden state. Pooled vs recomputed from fp16 frames: max relative difference 1.6e-4 (RAVDESS) / 7.2e-5 (CREMA-D). | Pooled cache usable for layer probing (E2a). Index 0 = Transformer input, 1–12 = layer outputs. |
| Alignment / masks | One frame-offset table for all streams. Frames per recording: RAVDESS 61–215 (median 91), CREMA-D 32–250 (median 118). Random batches of 32 waste about 37 % to padding. | Per-batch padding + boolean mask. Length-bucketed batching is an efficiency option for E2–E4. |
| Splits | All rows of the saved Stage 3 splits resolve to cache rows. RAVDESS: 6 folds × 704 / 176 / 176 recordings (16 / 4 / 4 speakers). CREMA-D: 5 folds, about 4,740 / 1,190 / 1,490 (58 / 15 / 18 speakers). | Data access only via `src/training/data.py: fold_rows()`, which re-runs the leakage checks. |
| MFCC scale | Mean per-dimension std: static 3.44, logE 2.93, Δ 0.88, ΔΔ 0.37 (RAVDESS). | Per-dimension standardisation is required for the SVM and the neural models. |

## 1. Common protocol (E1–E4)

- **Data:** the saved Stage 3 split files, never regenerated (SHA-256 checked: RAVDESS `DDCFFCCF…9BA5`).
  Within a fold, **train speakers** fit everything learnable (scalers, imputation values, model
  parameters). **Validation speakers** are used only for model / hyper-parameter selection and early
  stopping. **Test speakers** are predicted once, after selection is written to disk.
- **Labels:** 6 classes in the fixed order neutral, happy, sad, angry, fearful, disgust (ids 0–5).
- **Class imbalance:** RAVDESS neutral has half as many recordings as the other classes. All models
  use inverse-frequency class weights computed from the training fold: `class_weight="balanced"` in
  E1, class-weighted cross-entropy in E2–E4.
- **Metrics** (`src/evaluation/metrics.py`): accuracy, UA (= macro recall, identical by definition),
  macro precision, macro recall, **macro-F1 (primary)**, per-class P/R/F1, confusion matrix.
- **Aggregation:** seeds are averaged within a fold, then mean ± sample std (ddof = 1) are taken
  over folds. Every fold is reported; no fold is selected.

## 2. Learned preprocessing — fitted on training speakers only (`src/training/preprocessing.py`)

| Stream | Operation (parameters fitted on the training partition only) | Used in |
|---|---|---|
| Utterance-level MFCC functionals | `StandardScaler` inside an sklearn `Pipeline` → fitted together with the classifier on train | E1 |
| Frame-level streams | `FrameStandardizer`: per-dimension mean/std over all valid training frames (padding is never counted); pass-through for `voiced` (binary) and `periodicity` (∈ [0, 1]) | E2–E4 |
| DisVoice-103 | `DisVoicePreprocessor`: (a) imputed → training median; (b) signed log `g(x) = sign(x)·log(1+|x|)` for features with training \|skew\| > 2; (c) z-score with training mean/std; (d) clip to ±5; (e) optional 103 missingness indicators (ablation) | E3/E4 ablations |

`transform()` never updates parameters, and an unfitted transformer refuses to transform. Both are
unit-tested (`tests/test_stage5_e1.py`).

---

## 3. E1 — MFCC statistics + SVM / Random Forest

### 3.1 Why these baselines
They are the classical SER pipeline: hand-crafted spectral features plus a kernel or ensemble
classifier. They show how much a large pretrained model adds, and they check the data, split and
metric pipeline cheaply and deterministically. The SVM (RBF) is a maximum-margin classifier that
does well on small, standardised, dense datasets. The Random Forest is a scale-invariant non-linear
ensemble with a different inductive bias.

### 3.2 From a sequence to a vector (aggregation)
A recording is an MFCC matrix **X** ∈ ℝ^{T×39} with variable T. For each dimension j:

| Aggregation | Vector | Dim |
|---|---|---|
| `mean` | μ_j = (1/T) Σ_t X_tj | 39 |
| `mean_std` | [μ_j, σ_j], σ_j = √((1/T) Σ_t (X_tj − μ_j)²) | 78 |
| `functionals` | [μ_j, σ_j, P10_j, P50_j, P90_j] (percentiles over time) | 195 |

These statistics use only the recording itself and have no fitted parameters, so they cannot leak.
They are cached once per feature cache (`C:\SER_data\cache\stage5\E1_mfcc_agg_<DS>_v1_efe9081789.npz`).

### 3.3 Models
- **Input standardisation:** z = (x − m_train) / s_train, per dimension.
- **SVM (one-vs-one, RBF):** decision function f(x) = Σ_i α_i y_i K(x_i, x) + b with
  K(x, x′) = exp(−γ‖x − x′‖²). It is trained by minimising ½‖w‖² + C Σ_i c_{y_i} ξ_i (soft margin,
  hinge loss). The class weights c_k = n / (6 n_k) come from the training fold. Hyper-parameters:
  C ∈ {0.1, 1, 10, 100} and γ = m/d with m ∈ {0.1, 1, 10}, where d is the input dimension (1/d is
  sklearn's "scale" rule after standardisation). 12 configurations. Deterministic.
- **Random Forest:** 300 CART trees, each grown on a bootstrap sample with a random subset of
  max_features candidate features per split. The splitting criterion is class-weighted Gini
  impurity. The prediction is the majority of averaged leaf class probabilities.
  max_features ∈ {√d, 0.2·d}, min_samples_leaf ∈ {1, 5}; 4 configurations; seeds 0, 1, 2.
- **Model size** (they have no fixed parameter count):
  - SVM: number of support vectors. RAVDESS 652 of 704 training recordings; CREMA-D 4,263 of
    about 4,740. Almost all training points are support vectors, which signals heavy class overlap.
  - RF: total tree nodes. RAVDESS 65 k (mean depth 12.9); CREMA-D 586 k (depth 23.5).

### 3.4 Training and evaluation procedure (`scripts/stage5_e1_run.py`)
For every dataset, fold and model (RF: every seed):
1. `select_on_validation()` fits each **aggregation × hyper-parameter** candidate (36 for SVM, 12 for
   RF) on train and scores macro-F1 on validation. The function has no test argument, and its result
   is written to `<run>.selection.json` first.
2. The selected candidate (highest validation macro-F1; ties go to the first in grid order) is
   refitted alone to measure training time. It must reproduce its validation score exactly (checked).
3. The test speakers are predicted once.
4. For the descriptive aggregation comparison, the validation-best candidate *within* each
   aggregation is also evaluated on test. This is reported but never used for selection.

**Isolation test** (`tests/test_stage5_e1.py`): all test-row features are set to NaN. Selection still
succeeds with identical results, so no test row enters scaling, fitting or selection.

### 3.5 Results — speaker-independent (`results/E1/<DS>/summary.json`, `runs.csv`)

Test metrics, mean ± std over folds (RF: 3 seeds averaged per fold first).

| Dataset | Model | Macro-F1 | UA (= macro recall) | Accuracy | Macro precision |
|---|---|---|---|---|---|
| RAVDESS (6 folds) | SVM | **0.537 ± 0.059** | 0.543 ± 0.061 | 0.549 ± 0.061 | 0.559 ± 0.050 |
| RAVDESS (6 folds) | RF | 0.493 ± 0.032 | 0.503 ± 0.039 | 0.512 ± 0.032 | 0.519 ± 0.030 |
| CREMA-D (5 folds) | SVM | **0.503 ± 0.021** | 0.509 ± 0.020 | 0.507 ± 0.021 | 0.508 ± 0.018 |
| CREMA-D (5 folds) | RF | 0.457 ± 0.028 | 0.481 ± 0.023 | 0.478 ± 0.024 | 0.478 ± 0.019 |

Chance level for 6 classes is about 0.167.

**SVM vs RF, paired by fold:** SVM − RF macro-F1 = +0.044 ± 0.047 (RAVDESS, SVM better in 5 of 6
folds) and +0.046 ± 0.020 (CREMA-D, 5 of 5). With only 5–6 folds this is a consistent difference,
not a formal significance claim.

**Seed variability (RF):** the mean within-fold std across seeds is 0.023 (RAVDESS) and 0.005
(CREMA-D). It is small next to the fold-to-fold variability on RAVDESS (± 0.03–0.06; each test fold
has only 4 speakers).

**Validation → test optimism:** mean validation macro-F1 of the selected model minus its test score
is +0.052 (SVM) / +0.036 (RF) on RAVDESS and +0.013 / +0.016 on CREMA-D. Picking the best of
36 candidates on only 4 validation speakers inflates the validation estimate. This is why test is
used only once.

**Aggregation (descriptive; hyper-parameters chosen on validation within each aggregation):**

| Test macro-F1 | mean (39) | mean + std (78) | functionals (195) |
|---|---|---|---|
| RAVDESS SVM | 0.468 ± 0.045 | 0.518 ± 0.059 | 0.541 ± 0.061 |
| RAVDESS RF | 0.434 ± 0.062 | 0.490 ± 0.039 | 0.495 ± 0.033 |
| CREMA-D SVM | 0.405 ± 0.019 | 0.504 ± 0.022 | 0.503 ± 0.015 |
| CREMA-D RF | 0.406 ± 0.017 | 0.456 ± 0.025 | 0.457 ± 0.025 |

- Adding the **temporal standard deviation** helps every time: +0.05 to +0.10 macro-F1, better in
  5 of 6 / 5 of 5 folds for each model. Emotion is expressed in how the spectrum *varies*, not only
  in its average.
- **Percentiles** add little: +0.023 for the SVM on RAVDESS (5 of 6 folds), about 0 on CREMA-D.
- The validation-based selection picked `functionals` in 5 of 6 (RAVDESS) and 3 of 5 (CREMA-D) SVM
  folds.

**Per-class F1** (headline, mean over folds):

| | neutral | happy | sad | angry | fearful | disgust |
|---|---|---|---|---|---|---|
| RAVDESS SVM | 0.40 | 0.60 | 0.44 | **0.70** | 0.47 | 0.61 |
| RAVDESS RF | 0.33 | 0.55 | 0.38 | 0.66 | 0.45 | 0.58 |
| CREMA-D SVM | 0.50 | 0.44 | 0.54 | **0.66** | 0.42 | 0.46 |
| CREMA-D RF | 0.49 | 0.41 | 0.56 | 0.65 | 0.29 | 0.34 |

- **Angry** is the easiest class everywhere: high arousal leaves a strong spectral and energy trace.
- **Confusions are mostly within arousal groups:**
  - RAVDESS: neutral ↔ sad (21 % of neutral recordings predicted sad, 21 % of sad predicted neutral).
  - CREMA-D: happy → angry (18 %) and fearful → sad (22 %).
  - This is consistent with MFCC statistics capturing arousal better than valence.
- **RF on CREMA-D fearful and disgust:** recall is only 0.21 and 0.27, so RF falls behind the SVM
  mainly on these classes.

**Time** (8-core CPU; excludes feature extraction, includes scaling):

| | SVM train | SVM inference | RF train | RF inference |
|---|---|---|---|---|
| RAVDESS | 0.04 s | 0.07 ms / recording | 0.54 s | 0.32 ms / recording* |
| CREMA-D | 1.8 s | 0.45 ms / recording | 1.8 s | 0.05 ms / recording |

\*RF inference per recording is dominated by thread start-up for the small RAVDESS test set (176).
Selection over all candidates took about 1–8 s (RAVDESS) and about 20–50 s (CREMA-D) per run.

Checkpoints: `C:\SER_data\checkpoints\E1\<DS>\fold<k>_<model>_seed<s>.joblib`.
Figures: `plots/E1_confusion.png`, `plots/E1_aggregation_comparison.png`, `plots/E1_per_class_f1.png`.

### 3.6 Limitations of E1
- Hyper-parameter grids are small by design. They are equal in spirit for both models but not
  equal in size (36 vs 12 candidates per run).
- 4 test speakers per RAVDESS fold give large fold-to-fold variance (macro-F1 range 0.46–0.60
  for the SVM).
- RAVDESS uses only 2 sentences, so the lexical content is constant. This favours spectral
  statistics compared with natural speech.

---

## 4. E2 — Frozen HuBERT + lightweight classification head

### 4.1 Research questions
1. How effective are frozen HuBERT representations for SER?
2. Which hidden state carries the most emotion information?
3. Does summarising time beyond the mean (mean + std) help?
4. How does HuBERT compare with E1 on the same folds?

### 4.2 Inputs (verified, `scripts/stage5_e2_inspect.py` → `results/stage5_e2_inspect.json`)
- Frame cache: (T, 13, 768) fp16 per recording. Index 0 = Transformer input (after CNN, projection
  and positional convolution); index *l* = output of Transformer layer *l* (1–12). The Stage 4
  manual recomputation error is 0.0 for all 13 states.
- Pooled cache: (13, 1536) fp32 per recording = [μ_l ‖ σ_l] over the recording's frames. It was
  computed per **unpadded** recording, and σ is the population std. Masked pooling of zero-padded
  real batches reproduces it to ≤ 1e-4 relative (fp16 rounding of the frame cache) and is unchanged
  when the padding is filled with 1e4. HuBERT is therefore never re-run; E2 loads only the pooled
  cache (RAVDESS 80 MB, CREMA-D 570 MB).
- Same cache (`v1_efe9081789`, Praat configuration) and same Stage 3 folds as E1.

### 4.3 Architecture and formulation (`src/models/e2_hubert.py`)

| Representation | Input x | D (mean / mean+std) |
|---|---|---|
| layer *l* (0–12) | x = p_l (pooled vector of hidden state *l*) | 768 / 1536 |
| concat(1, 9) | x = [p_1 ; p_9] (the reference paper's layer pair) | 1536 / 3072 |
| weighted(all) | x = Σ_{l=0}^{12} α_l p̃_l, α = softmax(a), a ∈ ℝ^13 learned, a initialised to 0 (α_l = 1/13) | 768 / 1536 |

**Exactly how weighted(all) is computed:**
1. Each hidden state is pooled per recording: p_l = μ_l (mean) or [μ_l ; σ_l] (mean + std).
2. Each p_l is standardised per layer and per dimension with training-fold statistics:
   p̃_l = (p_l − m_l^train) / s_l^train.
3. The 13 standardised vectors are mixed with softmax weights; the result goes into the MLP.

Pooling happens **before** weighting. For mean pooling this equals pooling a weighted sum of
frames, because both operations are linear. For the std part it does not: the std of a weighted sum
is not the weighted sum of stds. The weights a are trained jointly with the head, with the same
optimiser, learning rate and weight decay.

**Head:** z = Dropout_p(GELU(W₁x + b₁)), logits = W₂z + b₂, with W₁ ∈ ℝ^{h×D} and W₂ ∈ ℝ^{6×h}.
Trainable parameters: h(D + 1) + 6(h + 1), plus 13 for weighted(all). The selected heads have
198,406 – 1,576,454 parameters, against 94.4 M frozen HuBERT parameters that are never updated.

**Loss:** cross-entropy, optionally class-weighted:
L = −Σ_i w_{y_i} log softmax(logits_i)_{y_i} / Σ_i w_{y_i}, with w_c = n_train / (6 · n_train,c)
computed from **training labels only** (`src/training/trainer.py: class_weights`). For example,
RAVDESS fold 0 has training counts [64, 128, 128, 128, 128, 128] → w = [1.833, 0.917, …].

**Optimisation:** AdamW (weight decay 1e-4), batch size 32, at most 50 epochs. Validation macro-F1
is computed after every epoch. The best epoch's weights are kept (ties go to the earliest epoch), and
training stops after 10 epochs without improvement. Training is deterministic: seeds are set
*before* the model is built (`train_classifier` takes a model factory), shuffling uses a seeded
generator, and `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG` is on. The
unit test that checks reproducible training caught a bug in the first version: the model was built
before seeding, so initial weights were not controlled by the seed. It was fixed before any E2 run.

### 4.4 Model selection and evaluation (`scripts/stage5_e2_run.py`)
Per fold, a staged search uses **validation macro-F1 only**:

| Phase | Varies | Runs per fold |
|---|---|---|
| A | 15 representations × {mean, mean+std}; default head h = 256, p = 0.2, lr = 1e-3, weighted CE | 30 (also the layer-wise probe) |
| B | for the selected (representation, pooling): h ∈ {256, 512} × p ∈ {0.2, 0.4} × lr ∈ {1e-3, 3e-4} | 8 |
| C | selected configuration with **unweighted** CE; everything else identical (controlled) | 1 |
| D | final configuration, seeds 0, 1, 2: train on train speakers, early-stop on validation, save checkpoint, reload it, predict the test speakers once | 3 |

- Phases A–C receive a fold object **without** a test partition (`without_test()`: accessing it
  raises `KeyError`). Their files contain only validation metrics.
- Phase D's seed-0 model must reproduce the selection run's validation F1 exactly (checked).
- Test metrics come from the **reloaded** checkpoint, which must predict the validation set
  identically to the in-memory model.
- Each checkpoint stores: weights, candidate configuration, training configuration, standardiser
  mean and std, dataset, fold, seed, run hash, feature-cache id and best epoch.
  Location: `C:\SER_data\checkpoints\E2\<DS>\fold<k>_seed<s>.pt`.

### 4.5 Audit and controls (`scripts/stage5_e2_audit.py` → `results/E2/audit.json`; all checks passed)

| Check | RAVDESS | CREMA-D |
|---|---|---|
| All folds complete: 30 A + 8 B + 1 C + selection + 3 D | 6/6 | 5/5 |
| One run hash across all files (no mixed configurations) | ✓ | ✓ |
| A–C files hold validation metrics only (no test keys; n = \|val\|) | ✓ | ✓ |
| Selection re-derived from the saved A–C validation scores equals the saved selection | ✓ | ✓ |
| Every A–C run finished before the first test result or checkpoint (file times) | ✓ | ✓ |
| D test rows = the fold's test speakers, disjoint from train/val | ✓ | ✓ |
| Checkpoints load and match selection, fold, seed, run hash and cache | ✓ | ✓ |
| Checkpoint standardiser = **train-row** statistics (≠ all-row statistics) | ✓ | ✓ |
| Class weights used = n_train / (6 n_train,c) from training labels | ✓ | ✓ |
| Fold 0, seed 0 retrained from scratch: same validation F1 and best epoch | ✓ | ✓ |
| E1 and E2 tested on identical rows in every fold | ✓ | ✓ |
| Stage 3 split SHA-256 unchanged | ✓ | ✓ |
| Stage 4 audio/manifest/HuBERT/pooled/MFCC SHA-256 = values recorded at adoption | ✓ | ✓ |
| Prosody arrays not modified after the last Stage 4 CPU run | ✓ | ✓ |

**Label-permutation control** (`scripts/stage5_e2_controls.py`): the selected configuration of each
fold is retrained on randomly **permuted training labels** and scored once on test. Test macro-F1 is
**0.187 ± 0.050** (RAVDESS) and **0.186 ± 0.014** (CREMA-D), against chance 0.167. Without real
labels the pipeline learns nothing transferable, so the high E2 scores do not come from leakage
through features, rows or preprocessing.

**Run history:** the first full run was interrupted by a session end during CREMA-D fold 3,
phase A (29 of 30 runs done). All 416 result files were intact (checked by parsing). The run
resumed from the cached results with the same run hash; nothing failed or was retried with a
different configuration. RAVDESS fold 0 came from the earlier pipeline-check run (same run hash).
When the full run started, it reloaded fold 0's cached A–C results and rewrote `selection.json`
with identical content after the D files already existed. File times therefore show
selection.json after D for that fold, but all A–C results predate D. The driver now keeps an
identical existing selection, and result files are written atomically.

### 4.6 Results — test (mean ± std over folds; each fold = mean of 3 seeds)

| Dataset | Macro-F1 | UA | Accuracy | Macro precision |
|---|---|---|---|---|
| RAVDESS (6 folds) | **0.762 ± 0.054** | 0.771 ± 0.052 | 0.764 ± 0.057 | 0.775 ± 0.055 |
| CREMA-D (5 folds) | **0.713 ± 0.030** | 0.714 ± 0.029 | 0.712 ± 0.029 | 0.716 ± 0.030 |

**Per seed** (mean ± std over folds, macro-F1): RAVDESS 0.755 ± 0.057 / 0.766 ± 0.060 /
0.764 ± 0.047; CREMA-D 0.714 ± 0.031 / 0.716 ± 0.028 / 0.709 ± 0.034. The mean within-fold
seed std is 0.011 (RAVDESS) and 0.006 (CREMA-D), far smaller than the fold-to-fold std.

**Per fold** (test macro-F1, mean of 3 seeds; selected configuration; validation F1 of the selection):

| RAVDESS fold | selected (rep / pooling / h / p / lr / weighted CE) | val F1 (selection) | test F1 (seeds) | test F1 | test UA | test acc |
|---|---|---|---|---|---|---|
| 0 | layer 6 / mean+std / 512 / 0.4 / 1e-3 / yes | 0.798 | 0.677, 0.674, 0.688 | 0.680 | 0.689 | 0.680 |
| 1 | layer 10 / mean / 256 / 0.2 / 1e-3 / yes | 0.791 | 0.709, 0.752, 0.737 | 0.733 | 0.738 | 0.729 |
| 2 | layer 5 / mean+std / 512 / 0.2 / 1e-3 / no | 0.830 | 0.786, 0.781, 0.786 | 0.784 | 0.804 | 0.788 |
| 3 | layer 5 / mean+std / 512 / 0.4 / 1e-3 / yes | 0.832 | 0.798, 0.822, 0.808 | 0.810 | 0.818 | 0.807 |
| 4 | layer 5 / mean / 512 / 0.2 / 3e-4 / yes | 0.817 | 0.825, 0.835, 0.810 | 0.824 | 0.821 | 0.835 |
| 5 | layer 5 / mean+std / 256 / 0.2 / 1e-3 / yes | 0.855 | 0.736, 0.733, 0.753 | 0.740 | 0.759 | 0.746 |

| CREMA-D fold | selected | val F1 (selection) | test F1 (seeds) | test F1 | test UA | test acc |
|---|---|---|---|---|---|---|
| 0 | layer 7 / mean / 512 / 0.4 / 3e-4 / no | 0.707 | 0.699, 0.699, 0.684 | 0.694 | 0.693 | 0.691 |
| 1 | weighted(all) / mean+std / 512 / 0.4 / 3e-4 / yes | 0.727 | 0.756, 0.755, 0.755 | 0.756 | 0.758 | 0.755 |
| 2 | weighted(all) / mean / 256 / 0.4 / 3e-4 / yes | 0.726 | 0.673, 0.682, 0.670 | 0.675 | 0.683 | 0.679 |
| 3 | weighted(all) / mean+std / 256 / 0.2 / 3e-4 / no | 0.728 | 0.730, 0.726, 0.711 | 0.722 | 0.720 | 0.720 |
| 4 | concat(1,9) / mean+std / 512 / 0.2 / 3e-4 / no | 0.768 | 0.713, 0.717, 0.723 | 0.718 | 0.718 | 0.716 |

Per-fold confusion matrices and per-class metrics: `results/E2/audit.json` → `folds`; every seed
is in `results/E2/<DS>/fold<k>/D_seed<s>.json`.

**Validation → test gap** (selected validation F1 − test F1): RAVDESS +0.059 on average (+0.12 in
folds 0 and 5, −0.006 in fold 4); CREMA-D +0.018. Choosing among 39 candidates on 4 validation
speakers (RAVDESS) inflates the validation estimate. With about 15 validation speakers (CREMA-D)
the bias is small.

### 4.7 Representation analysis (VALIDATION, phase A, default head; mean ± std over folds)

| Hidden state | RAVDESS mean | RAVDESS mean+std | CREMA-D mean | CREMA-D mean+std |
|---|---|---|---|---|
| 0 (Transformer input) | 0.616 ± 0.053 | 0.695 ± 0.048 | 0.584 ± 0.016 | 0.613 ± 0.021 |
| **1** | 0.670 ± 0.041 | 0.703 ± 0.047 | 0.622 ± 0.022 | 0.645 ± 0.019 |
| 2 | 0.719 ± 0.037 | 0.733 ± 0.041 | 0.649 ± 0.015 | 0.654 ± 0.015 |
| 3 | 0.745 ± 0.040 | 0.756 ± 0.041 | 0.668 ± 0.012 | 0.672 ± 0.019 |
| 4 | 0.751 ± 0.020 | 0.771 ± 0.022 | 0.680 ± 0.022 | 0.686 ± 0.022 |
| 5 | 0.789 ± 0.037 | **0.806 ± 0.029** | 0.701 ± 0.027 | 0.700 ± 0.024 |
| 6 | 0.789 ± 0.023 | 0.778 ± 0.021 | **0.703 ± 0.017** | 0.702 ± 0.022 |
| 7 | 0.771 ± 0.009 | 0.779 ± 0.031 | 0.701 ± 0.018 | **0.704 ± 0.020** |
| 8 | 0.751 ± 0.017 | 0.751 ± 0.033 | 0.696 ± 0.018 | 0.694 ± 0.012 |
| **9** | 0.751 ± 0.031 | 0.751 ± 0.028 | 0.694 ± 0.021 | 0.692 ± 0.018 |
| 10 | 0.767 ± 0.032 | 0.755 ± 0.034 | 0.691 ± 0.020 | 0.696 ± 0.020 |
| 11 | 0.758 ± 0.034 | 0.753 ± 0.039 | 0.696 ± 0.018 | 0.702 ± 0.021 |
| **12** (final layer) | 0.766 ± 0.048 | 0.751 ± 0.037 | 0.693 ± 0.026 | 0.693 ± 0.024 |
| concat(1, 9) | 0.755 ± 0.015 | 0.743 ± 0.023 | 0.706 ± 0.015 | 0.714 ± 0.025 |
| weighted(all) | 0.790 ± 0.026 | 0.793 ± 0.030 | **0.718 ± 0.023** | **0.721 ± 0.021** |

- **Middle layers (5–7) are best** in both corpora. Performance rises steeply from the Transformer
  input (0) to layer 5, then plateaus or declines slightly up to layer 12. The final layer is not
  the best. Layer 1 (paper) is among the weakest; layer 9 (paper) is mid-range.
- One reading, consistent with layer-wise analyses of HuBERT (Pasad et al., 2021): lower layers stay
  close to acoustics, while upper layers specialise towards the pre-training targets (phonetic
  units), which carry less paralinguistic information.
- **concat(1, 9)** is not better than the best single layer: RAVDESS 0.743–0.755 vs 0.806;
  CREMA-D 0.706–0.714 vs 0.704. **weighted(all)** is close to the best single layer on RAVDESS
  and the best representation on CREMA-D. Phase A selected layer 5 in 4/6 RAVDESS folds and
  weighted(all) in 3/5 CREMA-D folds.

**Pooling** (paired validation difference mean+std − mean, `plots/E2_pooling_validation.png`):
- The std clearly helps for the lowest states: layer 0 +0.079 (6/6 folds) / +0.029 (5/5);
  layer 1 +0.033 (6/6) / +0.024 (5/5).
- From layer 5 upwards the difference is about 0 (−0.015 to +0.008, mixed signs).
- So temporal variability adds information only where the frame representation is still close to
  acoustics; contextualised middle and upper layers already encode it in their mean. Selected
  pooling: mean+std 4/6 (RAVDESS) and 3/5 (CREMA-D).

**Class-weighted vs unweighted CE** (phase C, controlled, validation): differences are within
±0.018 (RAVDESS) and ±0.002 (CREMA-D). Weighting was kept in 5/6 and 2/5 folds; ties go to
weighted. CREMA-D is almost balanced, so a small effect is expected there; on RAVDESS
(neutral = ½ of the other classes) the effect is also small. **No evidence that class weighting
matters** for this head.

**Learned layer weights** (`plots/E2_layer_weights.png`):
- **Consistent shape:** across folds and poolings, the weights peak at hidden states 4–7.
  The per-layer std across folds is ≤ 0.01 and the argmax is always in 4–7.
- **But close to uniform:** the largest weight is 0.079–0.087 (RAVDESS) and 0.094–0.122 (CREMA-D),
  against 1/13 = 0.077. Normalised entropy is ≥ 0.985.
- **Likely cause:** the 13 logits start at 0, share the head's learning rate (3e-4 – 1e-3) and its
  weight decay (which pulls them back towards uniform), and training stops after 8–14 epochs on
  average. They can therefore move only a little.
- **Interpretation:** the weights confirm the *direction* found by the probe (middle layers are most
  useful), but their magnitudes should **not** be read as a precise importance ranking.
  weighted(all) behaves mainly like an average of standardised layers with a mild bias towards
  layers 4–7.

### 4.8 E1 vs E2 (TEST, identical folds and test rows)

| Dataset | E1-SVM | E1-RF | **E2** | E2 − E1-SVM (paired by fold) |
|---|---|---|---|---|
| RAVDESS macro-F1 | 0.537 ± 0.059 | 0.493 ± 0.032 | **0.762 ± 0.054** | +0.225 ± 0.049, better in 6/6 folds |
| RAVDESS UA | 0.543 ± 0.061 | 0.503 ± 0.039 | **0.771 ± 0.052** | |
| CREMA-D macro-F1 | 0.503 ± 0.021 | 0.457 ± 0.028 | **0.713 ± 0.030** | +0.210 ± 0.021, better in 5/5 folds |
| CREMA-D UA | 0.509 ± 0.020 | 0.481 ± 0.023 | **0.714 ± 0.029** | |

**Per-class test F1** (mean over folds and seeds):

| | neutral | happy | sad | angry | fearful | disgust |
|---|---|---|---|---|---|---|
| RAVDESS E1-SVM | 0.40 | 0.60 | 0.44 | 0.70 | 0.47 | 0.61 |
| RAVDESS E2 | 0.77 | 0.75 | **0.66** | 0.82 | 0.73 | 0.85 |
| CREMA-D E1-SVM | 0.50 | 0.44 | 0.54 | 0.66 | 0.42 | 0.46 |
| CREMA-D E2 | 0.78 | 0.73 | **0.64** | 0.81 | **0.65** | 0.67 |

- E2 improves **every** class.
- The largest gains are where E1 confused classes of similar arousal: RAVDESS neutral
  +0.36 and fearful +0.26; CREMA-D happy +0.29 and neutral +0.29.
- The hardest classes for E2 are **sad** (both corpora) and **fearful / disgust** (CREMA-D).
- Main confusions (`plots/E2_confusion_test.png`):
  - RAVDESS: fearful → sad 16 %, neutral → sad 12 %, sad → neutral 10 %, happy → fearful 10 %;
  - CREMA-D: fearful → sad 17 %, disgust → sad 12 %, sad → fearful 11 %, sad → disgust 11 %.

  Errors concentrate among the low-arousal and negative-valence classes, where sad acts as a
  "sink".

**Cost:**

| | Final-model training | Inference | GPU peak | Process RAM peak |
|---|---|---|---|---|
| RAVDESS | 0.8 s (best epoch 8.4 on average, range 1–20; max 31 epochs run) | 0.04 ms / recording | 85 MiB | 1.5 GB |
| CREMA-D | 7.0 s (best epoch 14.1, range 7–29; max 40 epochs run) | 0.07 ms / recording | 759 MiB (weighted(all) puts all 13 layers on the GPU) | 3.5 GB |

The 50-epoch limit was never reached by a final model. Selection (phases A–C) took about 30–40 s
per RAVDESS fold and 90–230 s per CREMA-D fold. **Fair-cost note:** E2 needs the HuBERT forward
pass, 18–26 ms per recording (Stage 4) on the GPU. E1's MFCC extraction is a CPU computation of
a few ms. The head's inference time hides this.

### 4.9 Fairness and limitations of the E1 / E2 comparison
- Both use identical folds, test rows, class-weighting rule, metrics and validation-only selection.
- E2 had a larger search: 39 validation runs per fold against 36 (SVM) / 12 (RF). Its
  representation was pre-trained on 960 h of speech.
- The size of the gap (+0.21 to +0.23, every fold), together with the permutation control, makes it
  unlikely to be an artefact of search size. The comparison is still between whole pipelines
  (pre-trained features + MLP vs MFCC statistics + SVM/RF), not between classifiers on equal
  features.
- With 4 test speakers per RAVDESS fold, fold-to-fold variability (0.68–0.82) dominates seed
  variability, and the RAVDESS validation scores are optimistic by about 0.06.
- Only utterance-level pooling (mean, mean+std) was studied. Temporal structure inside the
  utterance is not modelled in E2; that is the role of E3/E4.
- The layer-weight magnitudes are under-trained (§4.7) and should not be over-interpreted.
- No external results are compared here; numbers from the reference paper use a different
  protocol and are not mixed with ours.

Figures: `plots/E2_layerwise_validation.png`, `plots/E2_pooling_validation.png`,
`plots/E2_layer_weights.png`, `plots/E2_confusion_test.png`, `plots/E1_vs_E2_per_class_f1.png`,
`plots/E1_vs_E2_folds_test.png`.
Results: `results/E2/<DS>/summary.json`, `results/E2/audit.json`,
`results/E2/<DS>/control_label_permutation.json`.
Configuration: `configs/e2_hubert_head.json`.

---

## 5. E3 — Frame-level concatenation fusion (HuBERT + MFCC + prosody)

### 5.1 Research questions
1. Does adding frame-level MFCC and prosody to HuBERT help when the streams are fused by plain
   concatenation (no attention)?
2. How much does each stream contribute (leave-one-out, add-one, Shapley values)?
3. Is a sequence model over HuBERT frames better than E2's pooled MLP (same layer family)?
4. Does extra **self-attention capacity**, with parameters matched to E4, help on its own? This
   control lets E4 separate "attention between feature types" from "more attention / more
   parameters".
5. Do utterance-level DisVoice-103 features add anything (separate ablation)?

### 5.2 Inputs and data pipeline (`src/training/sequence_data.py`)
- All three streams come from the Stage 4 cache `v1_efe9081789` on the shared 20 ms grid. Recording
  *i* has the same number of frames in every stream; one offset table indexes all three.
- **HuBERT layer per fold.** Each fold uses one hidden state: the best single layer in **that fold's
  own** E2 phase-A validation results. The score of a layer is the maximum over the two poolings;
  ties go to the lower layer. The layer is never chosen from an average over folds, because fold
  *k*'s test speakers are validation speakers in other folds. The source file's SHA-256 is recorded.
  Selected layers: RAVDESS 6, 10, 5, 5, 5, 5; CREMA-D 7, 6, 12, 11, 5. On CREMA-D the winning margins
  are tiny: in folds 2 and 3, layers 12 and 11 beat layer 7 by 0.002 validation F1. That is
  selection noise, accepted as part of the approved rule.
- **Memory.** All 13 layers would need 17.6 GB for CREMA-D, so only the chosen layer is copied from
  the memory-mapped cache: (T_total, 768) fp16, 1.35 GB for CREMA-D (48 s to read) and 0.16 GB for
  RAVDESS (6 s). It is kept on the GPU together with MFCC (39) and prosody (4). Mini-batches are
  gathered on the GPU. No recording is cropped (the longest is 250 frames = 5 s).
- **Batching.** One index matrix idx[b, t] = offset[row_b] + t selects the same frame of the same
  recording in all three streams, so the streams stay aligned. Padded positions are set to 0 after
  standardisation and marked False in the mask. Training batches are length-bucketed with a seeded
  generator: permute, sort chunks of 20 batches by length, cut, shuffle the batch order. Each
  training recording appears exactly once per epoch.
- **Training-fold preprocessing.** One FrameStandardizer per stream is fitted on the frames of the
  fold's training recordings. Prosody's `voiced` (binary) and `periodicity` ([0, 1]) pass through
  unchanged. For the DisVoice ablation, DisVoicePreprocessor (§2) is fitted on training recordings.
  The primary 60–800 Hz variant is used. Missingness indicators appear only in the `dvind` variant.

### 5.3 Architecture (`src/models/e3_fusion.py`, `src/models/attention.py`)

| Step | Operation | Shape |
|---|---|---|
| 1 projection | p_s(t) = Dropout(LN(W_s x_s(t) + b_s)), s ∈ {H, M, P} | (T, 128) each |
| 2 concatenation | c(t) = [p_H(t) ; p_M(t) ; p_P(t)] | (T, 384) |
| 3 fusion | z(t) = Dropout(GELU(W_f c(t) + b_f)) | (T, 128) |
| 3′ control only | z ← LN(Block₄(…Block₁(z))) with 4 pre-LN self-attention blocks (4 heads, FFN 256), padding masked as keys | (T, 128) |
| 4 pooling | u = [mean_t z(t) ; √(var_t z(t) + 1e-5)] over real frames only | 256 |
| 4′ DisVoice only | u ← [u ; GELU(W_v v)], v = 103 (or 206 with indicators) | 384 |
| 5 head | logits = W₂ Dropout(GELU(W₁ u)), hidden 256 | 6 |

- **What concatenation can and cannot do.** The fusion layer forms a linear combination of the three
  streams *at the same frame t*, followed by one non-linearity. It cannot let one stream decide
  *which frames* of another stream matter. That is what E4's cross-attention adds.
- **Explicit attention, used by the control now and by E4 later.**
  - Formula: Q = X_q W_Q, K = X_c W_K, V = X_c W_V per head (d_k = 32);
    A = softmax(QKᵀ/√d_k + mask); output = [A₁V₁ ; … ; A₄V₄] W_O.
  - Masking: padded keys get the most negative finite float, so even a fully masked row stays finite.
  - Precision: scores are computed in fp32.
  - Block structure: pre-LN, X ← X + Drop(MHA(LN(X))) followed by X ← X + Drop(FFN(LN(X))).
- **Parameter matching.**
  - E3 (concatenation, H+M+P) has 221,574 parameters, of which 49,280 are in the fusion layer.
  - The planned E4 fusion has 596,992 parameters: 4 cross-attention blocks for MFCC↔prosody and
    HuBERT↔fused, 2 fuse layers and a LayerNorm. These are built from the same `AttentionBlock`
    class.
  - The control uses 4 self-attention blocks (751,750 total, against the E4 reference total of
    769,286), a difference of −2.3%. The number of blocks is chosen automatically
    (`matched_sa_blocks`) and checked to be within 5% by a unit test and the audit.
  - If E4's final design changes, the control must be re-matched.

### 5.4 Training, selection and evaluation (`scripts/stage5_e3_run.py`, `src/training/seq_trainer.py`)
- Same rules as E2:
  - AdamW, weight decay 1e-4, batch size 32;
  - class-weighted CE with weights from training labels (fixed rule, given E2's null result for
    weighting);
  - at most 50 epochs, early stopping on validation macro-F1 with patience 10, ties to the earliest
    epoch;
  - seeds set before the model is built; deterministic algorithms; fp32 throughout.
- New for sequence models: gradient-norm clipping at 1.0 (fixed before any run; see §5.5 for what
  the pilot showed), and a non-finite loss or gradient raises an error.

| Phase | Runs per fold | Uses |
|---|---|---|
| L | — | HuBERT layer from the fold's own E2 phase-A validation file |
| S | concat_HMP: dropout {0.2, 0.4} × lr {1e-3, 3e-4} = 4; sa_HMP (control): same grid = 4 | validation only, seed 0, fold **without** test partition |
| — | `selection.json` written | before any test prediction |
| D | 10 variants × seeds 0, 1, 2 = 30 | train → checkpoint → reload → test once |

- **D variants.**
  - Tuned on their own search: `concat_HMP` (E3) and `sa_HMP` (control).
  - Stream ablations, using concat_HMP's hyper-parameters: `concat_{H, M, P, HM, HP, MP}`. The full
    set of 7 stream subsets allows leave-one-out, add-one and Shapley analyses.
  - DisVoice ablations: `concat_HMP_dv` and `concat_HMP_dvind`.
- **Caveat.** The ablations reuse hyper-parameters tuned for H+M+P, so a subset could do slightly
  better with its own tuning. This is mainly a concern for the subsets without HuBERT.
- **Seed-0 reproduction.** The seed-0 final of each tuned model must reproduce its selection
  validation F1 exactly.
- **Test predictions.** Test predictions come from the reloaded checkpoint. Before testing, it must
  predict validation identically, and its standardiser statistics must equal those in use.
- **Checkpoints.** Stored at `C:\SER_data\checkpoints\E3\<DS>\fold<k>_<variant>_seed<s>.pt` with
  weights, specification, hyper-parameters, layer, per-stream standardiser statistics, DisVoice
  preprocessor parameters, run hash, cache id and best epoch.

### 5.5 Pilot (fold 0, validation phases only; `results/E3/pilot_<DS>.json`)
The pilot ran phases L + S of fold 0 **without any test prediction**, so the protocol could still
have been changed afterwards without test data having been seen. The full run reuses these S runs
(same run hash).

| Check | RAVDESS fold 0 (layer 6) | CREMA-D fold 0 (layer 7) |
|---|---|---|
| Alignment on real data (24 recordings): batched frames vs cache frames, all streams | max abs diff **0.0** | **0.0** |
| Padding = 0; mask length = n_frames | ✓ | ✓ |
| MFCC mean vs mean recomputed from training frames | 1e-17 | 9e-16 |
| GPU peak: concat / control | 305 / 694 MiB | 1,599 / 2,073 MiB (incl. 1.35 GB resident HuBERT layer) |
| Process RAM peak | 1.8 GB | 6.6 GB while reading the layer (memory-mapped pages), 3.1 GB afterwards |
| Training time per run: concat / control | 5.7 / 16.0 s | 23.7 / 104 s |
| Estimated full experiment | 0.5 h | 2.0 h (upper bound) |
| Non-finite loss or gradient | none | none |
| Gradient norm (mean per epoch) | 0.03–3.7 | 1.7–6.9 (max single step 31.7) |
| Steps where clipping at 1.0 was active | concat 26–49 %, control 64–85 % | 96–100 % |

**Stability finding:**
- Training is stable: loss falls steadily and there are no spikes or NaNs.
- Clipping at 1.0 is **not** a rarely triggered safety net. On CREMA-D it rescales almost every step,
  so it acts as per-step gradient normalisation inside AdamW.
- It was kept, not retuned:
  - the value was fixed before any run;
  - E3 and E4 must share one optimiser;
  - retuning it would add an extra search dimension that the other models did not get.
- This is an optimiser difference from E2, which had no clipping.
- Validation F1 of the selected configurations, fold 0: RAVDESS concat 0.826 / control 0.804;
  CREMA-D 0.705 / 0.694.

### 5.6 Run history, interruption and integrity
- **RAVDESS** ran to completion in the first full run: 6 folds × (8 S + selection + 30 D). These
  results were never re-run.
- **Interruption.** The same process then began loading the first CREMA-D HuBERT layer. Claude Code
  stopped it because the host was critically low on memory.
  - Cause: the memory-mapped, strided read of the 17.6 GB cache put up to 6.6 GB of mapped pages into
    the working set, while the RAVDESS data was still resident.
  - Stage reached: no CREMA-D final run had started. Only fold 0's S runs and selection existed, from
    the pilot.
  - The run was not restarted automatically.
- **Fix, with no numerical effect.** `load_hubert_layer` now uses buffered reads of 10,000-frame
  chunks instead of a memory map. On RAVDESS layer 5 the result is byte-identical. The CREMA-D layer
  now loads in 14–16 s.
- **Resume (approved).** `stage5_e3_run.py --datasets CREMA-D` ran in a fresh process, without
  changes to the architecture, hyper-parameters, seeds or protocol.
  - Fold 0 had the same run hash (`927c2c9fa380`), so its saved S runs and `selection.json` were
    reused, not re-run.
  - Before the resume, 425 result, selection and checkpoint files were hashed
    (`results/E3/snapshot_before_cremad_resume.json`). After the run all 425 were byte-identical: 0
    changed or missing.
- **Memory during the resume** (`results/E3/ram_monitor_cremad_resume.csv`; a watchdog would have
  stopped the run below 1 GB free RAM):

  | Moment | Process RAM (current / peak) |
  |---|---|
  | After loading layer 7 (fold 0, first load in the process) | 2.16 / 2.28 GB |
  | After loading layers 6, 12, 11 and 5 (folds 1–4) | 2.92 / 3.13 GB each |
  | Whole run, OS-measured working set | max 2.98 GB (OS peak 3.06 GB) |
  | Whole run, minimum system free RAM | **4.59 GB** |

  - Monitoring caveat 1: the first watchdog tracked the venv launcher stub instead of the real
    interpreter. A second watchdog was attached to the interpreter during the run.
  - Monitoring caveat 2: both watchdogs appended to the same CSV, so some samples were lost to file
    locks. 2,970 samples were still recorded, 1,474 of them from the interpreter.
  - For the analysis scripts, peaks were 4.69 GB in-process for the permutation control and 4.03 GB
    working set for the audit (4.2 GB minimum free). These scripts hold one RAVDESS and one CREMA-D
    layer in sequence.
- **Outcome.** All 11 folds completed: 330 D files and 330 checkpoints. There were no failed or
  retried runs and no non-finite losses or gradients.

### 5.7 Audit and controls (`scripts/stage5_e3_audit.py` → `results/E3/audit.json`; 36/36 checks passed)

| Check (each run separately for RAVDESS and CREMA-D) | Result |
|---|---|
| Every fold complete (8 S + selection + 30 D + 30 checkpoints) | ✓ |
| One run hash per fold across S, selection and D | ✓ |
| S files contain validation metrics only; the selection can be re-derived from S validation F1 alone | ✓ |
| All S files and `selection.json` are older than every D file and checkpoint | ✓ |
| D test rows equal the fold's test rows | ✓ |
| HuBERT layer equals the rule applied to the fold's own E2 phase-A file (SHA-256 unchanged) | ✓ |
| D hyper-parameters equal the selection (ablations use concat_HMP's) | ✓ |
| Seed-0 finals reproduce the selection validation F1 | ✓ |
| Class weights recomputed from training labels | ✓ |
| Checkpoint metadata matches the D files | ✓ |
| Standardiser statistics equal **training-row** statistics (every stream incl. HuBERT, every checkpoint) | ✓ |
| DisVoice fill values equal training medians | ✓ |
| **Every reloaded checkpoint (330) re-predicts its saved test predictions exactly** | ✓ |
| Control parameters within 5% of the E4 reference | ✓ |
| Stage 3 split file; Stage 4 audio/manifest/HuBERT/pooled/MFCC; prosody arrays: SHA-256 unchanged | ✓ |

**Label-permutation control** (`scripts/stage5_e3_controls.py` →
`results/E3/<DS>/control_label_permutation.json`):
- Setup: concat_HMP with each fold's selected hyper-parameters (seed 0), trained on **permuted
  training labels** (seed 1000 + fold).
- Test macro-F1: **0.158 ± 0.036** (RAVDESS) and **0.174 ± 0.014** (CREMA-D), against chance 0.167.
- The frame-level pipeline therefore learns nothing transferable without real labels, which rules
  out leakage through rows, frames or standardisation.

### 5.8 Results — test (mean ± std over folds; each fold = mean of 3 seeds)

| Dataset | Model | Macro-F1 | UA | Accuracy | Macro precision | val F1 (selected) | val→test gap |
|---|---|---|---|---|---|---|---|
| RAVDESS | E1 SVM (MFCC statistics) | 0.537 ± 0.059 | | | | | |
| RAVDESS | E2 pooled HuBERT | **0.762 ± 0.054** | 0.771 ± 0.052 | 0.764 ± 0.057 | 0.775 ± 0.055 | | +0.059 |
| RAVDESS | **E3 concat H+M+P** | 0.729 ± 0.064 | 0.735 ± 0.061 | 0.737 ± 0.068 | 0.746 ± 0.067 | 0.777 | +0.048 |
| RAVDESS | self-attention control H+M+P | 0.726 ± 0.037 | 0.733 ± 0.039 | 0.737 ± 0.047 | 0.742 ± 0.042 | 0.763 | +0.036 |
| CREMA-D | E1 SVM (MFCC statistics) | 0.503 ± 0.021 | | | | | |
| CREMA-D | E2 pooled HuBERT | **0.713 ± 0.030** | 0.714 ± 0.029 | 0.712 ± 0.029 | 0.716 ± 0.030 | | +0.018 |
| CREMA-D | **E3 concat H+M+P** | 0.695 ± 0.027 | 0.697 ± 0.026 | 0.694 ± 0.026 | 0.701 ± 0.024 | 0.713 | +0.018 |
| CREMA-D | self-attention control H+M+P | 0.669 ± 0.025 | 0.672 ± 0.026 | 0.669 ± 0.026 | 0.675 ± 0.023 | 0.703 | +0.034 |

**Per seed** (test macro-F1, mean ± std over folds):

| Dataset | Model | seed 0 | seed 1 | seed 2 | mean within-fold seed std |
|---|---|---|---|---|---|
| RAVDESS | E3 concat | 0.732 ± 0.081 | 0.731 ± 0.076 | 0.724 ± 0.045 | 0.027 (E2: 0.011) |
| RAVDESS | control | 0.726 ± 0.034 | 0.731 ± 0.033 | 0.721 ± 0.069 | 0.033 |
| CREMA-D | E3 concat | 0.694 ± 0.028 | 0.690 ± 0.027 | 0.700 ± 0.027 | 0.006 (E2: 0.006) |
| CREMA-D | control | 0.670 ± 0.028 | 0.672 ± 0.024 | 0.666 ± 0.029 | 0.010 |

**Per fold** (test macro-F1; selection by validation only):

| RAVDESS fold | layer | E3 selected (dropout / lr) | E3 val F1 | E3 test (seeds) | E3 | control selected | control test (seeds) | control | E2 | E1 SVM |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 0.2 / 3e-4 | 0.826 | 0.651, 0.614, 0.668 | 0.644 | 0.4 / 3e-4 | 0.675, 0.680, 0.607 | 0.654 | 0.680 | 0.459 |
| 1 | 10 | 0.2 / 1e-3 | 0.770 | 0.700, 0.720, 0.720 | 0.713 | 0.2 / 3e-4 | 0.772, 0.702, 0.743 | 0.739 | 0.733 | 0.594 |
| 2 | 5 | 0.4 / 1e-3 | 0.789 | 0.805, 0.777, 0.734 | 0.772 | 0.2 / 3e-4 | 0.708, 0.739, 0.759 | 0.735 | 0.784 | 0.555 |
| 3 | 5 | 0.4 / 1e-3 | 0.752 | 0.774, 0.753, 0.710 | 0.746 | 0.2 / 3e-4 | 0.748, 0.758, 0.690 | 0.732 | 0.810 | 0.595 |
| 4 | 5 | 0.4 / 3e-4 | 0.801 | 0.827, 0.833, 0.804 | 0.821 | 0.2 / 3e-4 | 0.718, 0.761, 0.809 | 0.763 | 0.824 | 0.546 |
| 5 | 5 | 0.2 / 1e-3 | 0.774 | 0.635, 0.686, 0.711 | 0.677 | 0.2 / 3e-4 | 0.737, 0.747, 0.721 | 0.735 | 0.740 | 0.473 |

| CREMA-D fold | layer | E3 selected | E3 val F1 | E3 test (seeds) | E3 | control selected | control test (seeds) | control | E2 | E1 SVM |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7 | 0.4 / 3e-4 | 0.705 | 0.689, 0.685, 0.700 | 0.691 | 0.2 / 3e-4 | 0.652, 0.676, 0.633 | 0.654 | 0.694 | 0.513 |
| 1 | 6 | 0.2 / 3e-4 | 0.721 | 0.738, 0.726, 0.736 | 0.733 | 0.2 / 3e-4 | 0.717, 0.700, 0.707 | 0.708 | 0.756 | 0.518 |
| 2 | 12 | 0.4 / 1e-3 | 0.697 | 0.666, 0.663, 0.664 | 0.665 | 0.2 / 3e-4 | 0.647, 0.638, 0.645 | 0.643 | 0.675 | 0.468 |
| 3 | 11 | 0.2 / 3e-4 | 0.705 | 0.702, 0.708, 0.712 | 0.708 | 0.4 / 3e-4 | 0.674, 0.684, 0.670 | 0.676 | 0.722 | 0.501 |
| 4 | 5 | 0.4 / 3e-4 | 0.756 | 0.673, 0.669, 0.687 | 0.676 | 0.2 / 3e-4 | 0.660, 0.660, 0.674 | 0.665 | 0.718 | 0.514 |

**Statistical note.** All p-values are exact two-sided sign-flip tests over folds.
- With 6 folds (RAVDESS) the smallest attainable p is 0.031.
- With 5 folds (CREMA-D) it is **0.0625**, so no CREMA-D comparison can reach p < 0.05 however
  consistent it is. "0/5" or "5/5" folds is the strongest evidence available there.
- Seeds are averaged within a fold and never counted as independent samples.

### 5.9 Contribution of each feature stream (all ablations; test macro-F1, paired on identical folds)

| Variant (concat_HMP hyper-parameters) | RAVDESS F1 | RAVDESS UA | CREMA-D F1 | CREMA-D UA | parameters | fusion params | train s / run (R / C) | ms / recording (R / C) |
|---|---|---|---|---|---|---|---|---|
| **H+M+P (E3)** | 0.729 ± 0.064 | 0.735 ± 0.061 | 0.695 ± 0.027 | 0.697 ± 0.026 | 221,574 | 49,280 | 4.7 / 25.1 | 0.07 / 0.07 |
| H+M+P + self-attention (control; own search) | 0.726 ± 0.037 | 0.733 ± 0.039 | 0.669 ± 0.025 | 0.672 ± 0.026 | 751,750 | 579,456 | 17.7 / 121.9 | 0.25 / 0.26 |
| H | 0.747 ± 0.059 | 0.753 ± 0.056 | 0.685 ± 0.028 | 0.688 ± 0.027 | 182,534 | 16,512 | 3.3 / 18.6 | 0.06 / 0.06 |
| H+M | 0.743 ± 0.060 | 0.749 ± 0.058 | 0.690 ± 0.018 | 0.692 ± 0.019 | 204,294 | 32,896 | 3.8 / 22.3 | 0.06 / 0.06 |
| H+P | **0.749 ± 0.055** | 0.751 ± 0.055 | 0.692 ± 0.022 | 0.693 ± 0.022 | 199,814 | 32,896 | 3.8 / 21.6 | 0.06 / 0.06 |
| M+P | 0.549 ± 0.042 | 0.563 ± 0.039 | 0.577 ± 0.019 | 0.581 ± 0.019 | 106,502 | 32,896 | 5.8 / 39.8 | 0.06 / 0.05 |
| M | 0.482 ± 0.071 | 0.498 ± 0.076 | 0.546 ± 0.017 | 0.550 ± 0.019 | 89,222 | 16,512 | 4.3 / 34.7 | 0.05 / 0.05 |
| P | 0.349 ± 0.059 | 0.387 ± 0.067 | 0.411 ± 0.018 | 0.428 ± 0.020 | 84,742 | 16,512 | 4.6 / 29.5 | 0.05 / 0.05 |
| H+M+P + DisVoice-103 | 0.734 ± 0.064 | 0.740 ± 0.063 | 0.691 ± 0.023 | 0.694 ± 0.022 | 267,654 | 49,280 | 4.1 / 22.1 | 0.07 / 0.07 |
| H+M+P + DisVoice-103 + missingness indicators | 0.721 ± 0.068 | 0.728 ± 0.065 | 0.683 ± 0.018 | 0.686 ± 0.018 | 280,838 | 49,280 | 4.2 / 25.8 | 0.07 / 0.07 |

Inference times exclude feature extraction: batched GPU forward pass only, divided by the number
of recordings.

**Paired comparisons** (Δ test macro-F1 per fold; "better" = folds with Δ > 0):

| Comparison | RAVDESS Δ | better | p | CREMA-D Δ | better | p |
|---|---|---|---|---|---|---|
| HuBERT given M+P (HMP − MP), leave-one-out | **+0.180 ± 0.060** | 6/6 | 0.031 | **+0.117 ± 0.016** | 5/5 | 0.062 |
| MFCC given H+P (HMP − HP), leave-one-out | −0.020 ± 0.025 | 1/6 | 0.062 | +0.003 ± 0.009 | 3/5 | 0.500 |
| prosody given H+M (HMP − HM), leave-one-out | −0.014 ± 0.016 | 2/6 | 0.125 | +0.005 ± 0.010 | 3/5 | 0.375 |
| MFCC given H (HM − H), add-one | −0.005 ± 0.015 | 3/6 | 0.500 | +0.005 ± 0.011 | 4/5 | 0.438 |
| prosody given H (HP − H), add-one | +0.002 ± 0.008 | 4/6 | 0.438 | +0.007 ± 0.007 | 5/5 | 0.062 |
| M+P given H (HMP − H) | **−0.018 ± 0.020** | **0/6** | 0.031 | +0.010 ± 0.009 | 4/5 | 0.125 |
| prosody given M (MP − M) | +0.067 ± 0.049 | 6/6 | 0.031 | +0.031 ± 0.013 | 5/5 | 0.062 |
| MFCC given P (MP − P) | +0.200 ± 0.049 | 6/6 | 0.031 | +0.166 ± 0.029 | 5/5 | 0.062 |
| DisVoice given E3 (dv − HMP) | +0.005 ± 0.014 | 5/6 | 0.406 | −0.004 ± 0.012 | 1/5 | 0.562 |
| missingness indicators given DisVoice (dvind − dv) | −0.013 ± 0.018 | 1/6 | 0.125 | −0.008 ± 0.008 | 1/5 | 0.125 |

The UA version of every comparison is in `results/E3/<DS>/summary.json` → `paired_test_UA`. It
agrees in sign everywhere except "prosody given H" on RAVDESS: −0.002 UA against +0.002 F1, both
negligible.

**Shapley values** (test macro-F1 over all 7 stream subsets, v(∅) = 1/6 = chance; the values sum to
F1(H+M+P) − 1/6):

| Stream | RAVDESS | per-fold range | CREMA-D | per-fold range |
|---|---|---|---|---|
| HuBERT | **0.364** | 0.321 – 0.456 | **0.283** | 0.265 – 0.308 |
| MFCC | 0.131 | 0.101 – 0.171 | 0.156 | 0.148 – 0.168 |
| prosody | 0.068 | 0.033 – 0.097 | 0.090 | 0.083 – 0.100 |

**Interpretation.**
1. **HuBERT carries almost all of the usable information.**
   - Removing it from H+M+P costs 0.18 (RAVDESS) and 0.12 (CREMA-D) macro-F1, in every fold.
   - It holds the largest Shapley share in every fold of both datasets.
   - HuBERT alone (0.747 / 0.685) is within 0.02 of the full model on both datasets.
2. **MFCC and prosody are informative on their own but almost entirely redundant with HuBERT.**
   - Without HuBERT they complement each other strongly. MFCC adds 0.17–0.20 to prosody, and prosody
     adds 0.03–0.07 to MFCC, in every fold.
   - Once HuBERT is present, their marginal effect is within ±0.02.
   - Their sizeable Shapley values (0.13–0.16 MFCC, 0.07–0.09 prosody) therefore come mainly from
     coalitions **without** HuBERT. Shapley values measure average credit over all subsets, not the
     gain from adding a stream to the best model. They should not be read as "MFCC adds 0.13 to E3".
3. **The two datasets disagree on the sign of the small effects.**
   - RAVDESS: adding M+P to HuBERT **hurts** in all 6 folds (−0.018, p = 0.031), and the MFCC
     leave-one-out is negative in 5/6 folds.
   - CREMA-D: the same additions are slightly positive (+0.010 for M+P, 4/5 folds; prosody given H
     +0.007, 5/5).
   - Plausible explanation, not tested: RAVDESS has only 704 training recordings per fold (16 actors),
     so extra input dimensions mostly add variance. Consistent with this:
     - the H+M+P validation→test gap is twice that of H alone (0.048 against 0.023);
     - the within-fold seed std is higher than E2's (0.027 against 0.011).
   - CREMA-D has about 4,740 training recordings per fold (91 speakers in total), which allows a small
     positive contribution to be learnt.
     That contribution is about the size of the seed noise.
4. **DisVoice-103 adds nothing consistent:** +0.005 (5/6) on RAVDESS and −0.004 (1/5) on CREMA-D.
   Adding the missingness indicators lowers F1 on both datasets (1/6 and 1/5 folds better). The
   utterance-level DisVoice features carry no information that the frame streams lack.
5. **Caveat.** Every ablation reuses concat_HMP's hyper-parameters.
   - Subsets without HuBERT (M, P, MP) probably favour other settings: they stop at epochs 23–36 on
     average, against 9–18 for subsets with HuBERT. Their absolute scores may be slightly
     pessimistic.
   - The subsets with HuBERT are close to the tuned configuration, so the main conclusions (1, 2 and
     4) are not affected.

### 5.10 Self-attention control vs concatenation (research question 4)
- **Results.**
  - CREMA-D: the control is **worse than concatenation in all 5 folds** (−0.025 ± 0.010 F1; UA
    −0.025).
  - RAVDESS: no difference on average (−0.003 ± 0.043, 3/6), with much larger fold-to-fold swings
    (from −0.059 to +0.058).
  - Its best validation F1 in the search is also below concat's in 9 of 11 folds (exceptions:
    RAVDESS fold 5, +0.003; CREMA-D fold 2, +0.002; figure `E3_search_validation.png`), so the
    CREMA-D deficit is not a test-set artefact.
- **Cost.**
  - It has 3.4× the parameters (751,750) and needs 3.7–4.9× the training time per run.
  - Inference is 3.6× slower (0.25 against 0.07 ms per recording).
- **Conclusion for E4.** Self-attention capacity of E4's size **does not by itself help**. On CREMA-D
  it hurts. If E4's cross-attention beats concatenation, the gain cannot be explained by "more
  attention parameters". If E4 does not beat concatenation, the control shows that capacity alone
  would not have closed the gap either.
- **Limitation, open before E4.**
  - The control selected the **lowest learning rate of the grid (3e-4) in all 11 folds**. Its optimum
    may lie below the grid, so it may be under-tuned. Concatenation chose 3e-4 in 6/11 folds.
  - E4 shares the same grid and would inherit the same limitation.
  - The grid was fixed before E3, and widening it now would be a protocol change. It was **not** done
    and needs a decision before E4. Any change would have to be applied identically to the control
    and to E4 and justified without test data. The validation evidence, a selection at the grid edge,
    is enough to motivate it.
  - *Follow-up:* the team asked for the grid to be extended below 3e-4 for E4 and its comparators.
    The proposal is in §6. The E3 results above are kept exactly as run and are not replaced.

### 5.11 E3 vs E2 and E1 (test; identical folds and test rows)
- **E3 vs E1.** E3 is far above the E1 SVM: +0.192 on RAVDESS (6/6, p = 0.031) and +0.192 on CREMA-D
  (5/5).
- **E3 vs E2.** E3 is **below E2 in every fold of both datasets**: −0.033 ± 0.026 on RAVDESS (0/6,
  p = 0.031) and −0.018 ± 0.015 on CREMA-D (0/5, p = 0.062). UA gives the same result (−0.036 and
  −0.017).
  - The gap is not caused by the extra streams, since H alone is also below E2: −0.015 on RAVDESS
    (2/6, p = 0.19) and −0.028 on CREMA-D (0/5).
  - It is therefore a property of the E3 pipeline relative to E2.
- **Differences between the pipelines.** There are four, and they cannot be separated with these
  runs:
  1. *HuBERT representation.*
     - E3 uses one hidden state per fold (the approved rule).
     - E2's final selection could also choose learned layer weighting or layer concatenation, and on
       CREMA-D it did: weighted(all) in folds 1–3 and concat(1, 9) in fold 4.
     - On CREMA-D, H − E2 is smallest in fold 0, the one fold where E2 also used a single layer
       (layer 7): −0.012, against −0.030 to −0.037 in folds 1–4.
     - On RAVDESS, E2 used exactly the same single layers as E3 in all six folds, and there H − E2 is
       −0.015 and not significant (2/6).
     - This pattern **suggests** that the CREMA-D gap comes mostly from the single-layer restriction,
       not from frame-level modelling. With one comparable fold, this is a hypothesis, not a result.
  2. *Search breadth.* E2 chose among 39 candidates per fold; E3 among 4.
  3. *Optimiser.* E3 clips gradients at 1.0, which is active on 96–100% of CREMA-D steps (§5.5). E2
     has no clipping.
  4. *Model.* E3 is a frame-level projection plus statistics pooling; E2 is an MLP on pooled vectors.
- **What can be claimed.**
  - Supported: frame-level concatenation of MFCC and prosody with a single HuBERT layer does not
    outperform a well-tuned pooled-HuBERT MLP.
  - Not supported: that frame-level fusion is inherently worse.
- **MFCC sequence model vs E1 SVM (`concat_M − E1-SVM`).**
  - The frame-level MFCC model beats the SVM on MFCC statistics on CREMA-D (+0.043, 5/5).
  - It does not beat it on RAVDESS (−0.055, 1/6, std 0.076).
  - This fits the data-size pattern above: the sequence model needs more speakers.

### 5.12 Per-class results, confusions and generalisation gap
| Dataset | Model | neutral | happy | sad | angry | fearful | disgust |
|---|---|---|---|---|---|---|---|
| RAVDESS | E2 | 0.765 | 0.754 | 0.655 | 0.816 | 0.734 | 0.847 |
| RAVDESS | E3 concat | 0.678 | 0.714 | 0.609 | 0.830 | 0.699 | 0.844 |
| RAVDESS | control | 0.642 | 0.717 | 0.642 | 0.820 | 0.719 | 0.818 |
| RAVDESS | H only | 0.714 | 0.728 | 0.637 | 0.824 | 0.728 | 0.852 |
| CREMA-D | E2 | 0.781 | 0.732 | 0.643 | 0.807 | 0.646 | 0.669 |
| CREMA-D | E3 concat | 0.762 | 0.703 | 0.621 | 0.794 | 0.621 | 0.668 |
| CREMA-D | control | 0.733 | 0.679 | 0.576 | 0.776 | 0.601 | 0.651 |
| CREMA-D | H only | 0.765 | 0.693 | 0.606 | 0.790 | 0.609 | 0.644 |

- **Per class.**
  - On RAVDESS, E3 loses most against E2 on **neutral** (−0.087) and gains slightly on angry
    (+0.014).
  - Adding M+P to HuBERT lowers neutral F1 from 0.714 to 0.678, the largest per-class effect of the
    extra streams.
  - On CREMA-D, E3 is 0.01–0.03 below E2 in every class except disgust, where it ties.
- **Confusions** (E3, % of the true class, summed over folds and seeds; `E3_confusion_test.png`).
  "Sad" absorbs most errors on both datasets:
  - RAVDESS: neutral → sad 19%, fearful → sad 15%, sad → fearful 13%.
  - CREMA-D: fearful → sad 19%, disgust → sad 12%, sad → fearful 12%, sad → disgust 11%.
  - Low-arousal and negative-valence classes are where all models struggle, which matches E1 and E2.
- **Validation → test gap** (selected validation F1 − test F1):
  - RAVDESS: +0.048 for E3 and +0.036 for the control, against +0.059 for E2. The gap is smaller
    because there are 4 candidates instead of 39, but it is still large in folds 0 and 5 (+0.18 and
    +0.10).
  - CREMA-D: +0.018 for E3, equal to E2.

### 5.13 Conclusions and limitations of E3
**Summary.**
- **HuBERT remained the dominant representation throughout E3.**
  - Removing it costs 0.18 (RAVDESS) and 0.12 (CREMA-D) macro-F1, in every fold.
  - It has the largest Shapley share in every fold.
  - HuBERT alone is within 0.02 of the full fusion model on both datasets.
- **The additional acoustic streams did not consistently improve on HuBERT.**
  - Adding MFCC and prosody lowered RAVDESS F1 in all 6 folds (−0.018).
  - On CREMA-D they raised F1 by +0.010 (4/5 folds), which is about the size of the seed noise.
  - DisVoice-103 added nothing consistent on either dataset.
- These results are final for E3 as run. The E4 proposal (§6) re-tunes only the comparators it
  needs, in a separate results directory.

**Answers to §5.1:**
1. Plain concatenation of frame-level MFCC and prosody with HuBERT does **not** help reliably. On
   RAVDESS it hurts (0/6 folds). On CREMA-D it gives about +0.01, which is at the level of the seed
   noise.
2. HuBERT dominates, and the hand-crafted streams are largely redundant with it. Detailed effects by
   leave-one-out, add-one and Shapley analyses are in §5.9.
3. With a single HuBERT layer, the frame-level model does not beat E2's pooled MLP. The reason is
   confounded (§5.11).
4. Extra self-attention capacity matched to E4 does not help, and on CREMA-D it hurts in every fold.
5. DisVoice-103 adds nothing consistent, and its missingness indicators slightly hurt.

**Limitations:**
- Ablations reuse concat_HMP's hyper-parameters (§5.9, point 5).
- The control selected the lowest learning rate of the grid in every fold (§5.10). This is unresolved
  and needs a decision before E4.
- Layer selection is noisy: CREMA-D folds 2 and 3 were won by margins of 0.002. The layer was fixed
  per fold by the approved rule and never revised after test.
- Gradient clipping acts as per-step normalisation on CREMA-D (§5.5). It is an optimiser difference
  from E2.
- The E3 vs E2 comparison is confounded by representation, search breadth and optimiser (§5.11).
- RAVDESS has only 6 folds (4 test actors each), and E3's seed variability is about 2.5× E2's. Small
  RAVDESS differences should not be over-interpreted.
- The run was interrupted once and resumed for CREMA-D only (§5.6). All 425 completed files were
  verified unchanged, and the audit re-predicted all 330 checkpoints exactly.
- No setting was changed after any test result was seen.

Figures: `plots/E3_variants_test.png`, `E3_stream_contributions.png`, `E3_confusion_test.png`,
`E1_E2_E3_folds_test.png`, `E2_vs_E3_per_class_f1.png`, `E3_search_validation.png`. Per-run data:
`results/E3/<DS>/runs.csv`, `summary.json`, and `fold<k>/D_<variant>_seed<s>.json`.

---

## 6. E4 — Cross-attention fusion: protocol (approved 2026-11-06; fold-0 validation-only pilot run, full experiment NOT run)

**Team decisions (2026-11-06):**
- 4-value learning-rate grid, dropout {0.2, 0.4}, at most 150 epochs.
- The same search for E4, the control, concatenation and the HuBERT-only model.
- All three E4 ablations.
- A validation-only fold-0 pilot first.
- E1–E3 preserved.
- The positional-encoding limitation stated in the methodology and the interpretation (§6.2).
- Feature swapping treated as a supplementary diagnostic only (§6.5).
- The full run waits for review of the pilot (§6.10).

Nothing in this section has been trained on real data. The parameter counts come from instantiating
the modules. The costs combine (a) E3's measured seconds per epoch with (b) a synthetic-batch
benchmark of a scratch prototype on the RTX 3050. The benchmark gives cost ratios only.

### 6.1 Research questions (fixed before any E4 run)
1. **Primary.** Does bidirectional cross-attention fusion of HuBERT, MFCC and prosody beat
   concatenation fusion, when both get the same hyper-parameter search?
2. Is any gain due to attention *across feature types*, or just to attention capacity?
   Answered by E4 against the parameter-matched self-attention control.
3. Does cross-attention extract information from MFCC and prosody that HuBERT alone lacks?
   Answered by E4 against a HuBERT-only frame model with the same search. In E3, concatenation could
   not do this (§5.13).
4. Which streams and which stage carry E4's effect? Answered by ablations and a context-corruption
   diagnostic.

### 6.2 Architecture (unchanged from the design the control was matched to, §5.3)

| Step | Operation | Shape |
|---|---|---|
| 1 projection | same as E3: p_s = Dropout(LN(W_s x_s)), s ∈ {H, M, P} | (T, 128) each |
| 2 stage 1: MFCC ↔ prosody | m′ = X(p_M ← p_P), p′ = X(p_P ← p_M) | (T, 128) each |
| 3 fuse 1 | a = Dropout(GELU(W_a [m′ ; p′])) | (T, 128) |
| 4 stage 2: HuBERT ↔ acoustic | h′ = X(p_H ← a), a′ = X(a ← p_H) | (T, 128) each |
| 5 fuse 2 | z = LN(Dropout(GELU(W_z [h′ ; a′]))) | (T, 128) |
| 6 pooling | masked mean + std over real frames (as E3) | 256 |
| 7 head | as E3, hidden 256 | 6 |

- **Cross-attention block X(x ← c).** This is the existing `AttentionBlock(cross=True)`, the same
  code the control uses:
  - x ← x + Drop(MHA(LN_q(x), LN_c(c), mask_c));
  - then x ← x + Drop(FFN(LN_f(x)));
  - 4 heads with d_k = 32, FFN width 256;
  - padded context frames are masked as keys.
- **Queries and context.** Queries come from one stream; keys and values come from the other.
  Each frame of HuBERT can therefore weight *which* frames of the acoustic streams it reads, and the
  other way round. Concatenation cannot do this.
- **Methodological limitation: no positional encoding.** This is part of the approved design. It is
  kept unchanged and stated here so that results are not over-interpreted.
  - Like the control, E4 has no positional encoding.
  - **The cross-attention is content-based.** Its weights depend only on what the frames contain.
    A HuBERT frame at time t cannot prefer the MFCC or prosody frame at the same time t, or
    neighbouring ones, because it does not know where they are. The output of an attention step
    does not depend on the order of the context frames (unit test).
  - **The shared frame grid is used only after attention.** The residual path and the two fusion
    layers concatenate the outputs frame by frame ([h′(t) ; a′(t)]), as E3's concatenation does.
    Misaligning the acoustic streams relative to HuBERT therefore changes E4's output (unit test).
    The draft of this section said E4 does not use the alignment at all. That was wrong and is
    corrected here.
  - Pooling is order-free, so the whole model is invariant to a *joint* permutation of the frames of
    all streams (unit test; max |Δ| 9e-8 on the prototype). Like E3, E4 does not model temporal order
    or dynamics.
  - **Interpretation consequence.** A gain of E4 over concatenation would show the value of
    content-based selection of frames across streams. It would **not** show that E4 exploits the
    temporal correspondence between the streams. A null result does not rule out a
    position-aware cross-attention model, which was not tested.

**Parameters** (by instantiation; identical d_model, heads, FFN, head):

| Component | E3 concat | Self-attention control | **E4** |
|---|---|---|---|
| Projections H / M / P | 98,688 / 5,376 / 896 | same | same |
| Fusion | 49,280 (1 linear layer) | 579,456 (fuse + 4 self-attention blocks + LN) | **596,992** (4 × 132,736 cross blocks + 2 × 32,896 fuse + 256 LN) |
| Head | 67,334 | 67,334 | 67,334 |
| **Total** | 221,574 | 751,750 (−2.3% vs E4) | **769,286** = `e4_reference_fusion_params()` + E3 non-fusion |

The control was matched to exactly this design, so it does **not** need re-matching.

### 6.3 Computational cost

**Synthetic benchmark** (scratch prototype; batch 32; deterministic algorithms; one training step =
forward + backward + clip + AdamW):

| Frames T | Training ms/batch (concat / control / E4) | Inference ms/batch | Activation memory MiB (concat / control / E4) |
|---|---|---|---|
| 91 (RAVDESS median) | 6.2 / 20.8 / 22.1 | 0.8 / 5.1 / 6.4 | 18 / 132 / 145 |
| 118 (CREMA-D median) | 4.1 / 22.2 / 23.4 | 1.2 / 6.9 / 7.3 | 24 / 182 / 203 |
| 250 (longest) | 5.3 / 55.1 / 56.8 | 1.6 / 17.3 / 18.1 | 51 / 538 / 573 |

- **E4 vs control.** E4 costs **1.03–1.06×** the control per step and needs 7–11% more activation
  memory. Attention costs 2T²·128 multiply-adds per block (3.6 M at T = 118).
- **Projected for real runs**, using E3's measured seconds per epoch × the ratios above:

  | Model | CREMA-D (s/epoch) | RAVDESS (s/epoch) |
  |---|---|---|
  | E4 | ≈ 5.1 | ≈ 0.66 |
  | Control | 4.85 (measured) | 0.63 (measured) |
  | Concat | 1.08 (measured) | 0.17 (measured) |
  | HuBERT-only | ≈ 0.80 | ≈ 0.12 |

- **Memory.**
  - GPU: CREMA-D peak ≈ 2.1–2.2 GB. This is the control's measured 2,073 MiB, which includes the
    resident 1.35 GB HuBERT layer, plus E4's extra activation memory. It is below the 4 GB card.
  - RAM: the data path is the same as E3's chunked loader, so the process peak should be about the
    3.1 GB measured in E3.

### 6.4 Revised hyper-parameter search (the only protocol change)

| Item | E3 | **Proposed for E4 and all its tuned comparators** |
|---|---|---|
| Learning rate | {1e-3, 3e-4} | **{1e-3, 3e-4, 1e-4, 3e-5}** (half-decade steps; the old values are kept) |
| Dropout | {0.2, 0.4} | {0.2, 0.4} (unchanged) |
| Candidates per model per fold | 4 | **8** |
| Maximum epochs | 50 | **150** (patience 10, unchanged) |
| Selection | validation macro-F1, seed 0, fold without test partition; ties → earlier candidate | unchanged; grid order (dropout outer; lr 1e-3 → 3e-5 inner) keeps E3's tie-breaking |

**Why the maximum epochs must rise with the grid.**
- In E3, the control's best epochs at 3e-4 were 14 on average and up to 32. At 1e-4 and 3e-5, about
  3× and 10× more epochs are expected. A cap of 50 would truncate the low-learning-rate candidates and bias
  selection against them, recreating the boundary problem in another form.
- Raising the cap **does not change any run that E3 could have produced**:
  - there is no scheduler (constant-rate AdamW);
  - none of the 88 E3 S runs reached epoch 50 (the largest best epoch was 32 and the most epochs
    run was 43).
- So the 1e-3 and 3e-4 cells must reproduce E3's S validation F1 exactly. This is checked in the
  audit (§6.7).

**Fixed and unchanged:**
- architecture; folds and test rows; HuBERT layer per fold (same E2 phase-A rule and files);
- final seeds 0, 1, 2; selection seed 0;
- AdamW with weight decay 1e-4; batch size 32; class-weighted CE; gradient clip 1.0; length
  bucketing; fp32;
- `selection.json` written before any test prediction;
- checkpoint → reload → test once.

**Pre-registered edge rule.**
- If a model selects the new lower edge (3e-5) in most folds of a dataset, this is reported as a
  limitation. The grid is **not** extended again.
- Any S run that reaches 150 epochs without early stopping is flagged in the audit and not re-run.
- Concatenation chose the upper edge (1e-3) in 5/11 E3 folds. It split 6/5 between the two values,
  which points to an optimum inside the range rather than beyond it. The grid is therefore not
  extended upwards. This is a judgement, recorded so that it can be challenged.

### 6.5 Comparison plan

**Tuned models.** Each gets its own 8-candidate search, then 3 seeds. They are stored under
`results/E4/<DS>/`, so **nothing in `results/E3/` is touched**.

| Name | Model | Role |
|---|---|---|
| `xattn_HMP` | **E4** | model under test |
| `sa_HMP` | self-attention control (E3 code, 4 blocks) | capacity control (RQ2) |
| `concat_HMP` | E3 concatenation (E3 code) | **primary comparator** (RQ1); re-searched for fairness |
| `concat_H` | E3 code, HuBERT stream only | HuBERT-only frame model (RQ3) |

- **Why the concatenation model is re-searched.** If E4 and the control get the extended grid but
  concatenation keeps its 4-candidate E3 search, a win for either attention model could come from
  the wider search alone. The same applies to the HuBERT-only model. Both are cheap: together about
  16% of the CREMA-D search time.
- **What happens if the old setting is still the best.** If the re-searched concatenation model
  selects the same setting as in E3, its D runs are identical to E3's by construction. This is an
  extra reproducibility check.
- **The original E3 numbers stay in §5** as the record of the E3 protocol.

**E4 ablations.** These reuse `xattn_HMP`'s selected hyper-parameters (3 seeds each, as in E3) and
the same code with fewer blocks:

| Name | Structure | Parameters | Question |
|---|---|---|---|
| `xattn_HM` | one stage: H ↔ M | 470,022 | prosody removed |
| `xattn_HP` | one stage: H ↔ P | 465,542 | MFCC removed |
| `xattn_H_MPcat` | stage 1 replaced by concatenation of M, P; then H ↔ fused | 503,814 | does the MFCC ↔ prosody stage matter? |

- **Caveat.** As in E3, these ablations are not tuned separately, and they also have fewer parameters
  than E4. They are interpreted as "what the trained E4 recipe loses", not as tuned alternatives.
- **Not planned for E4:**
  - DisVoice-103, which gave no consistent effect in E3;
  - unidirectional variants.
  - Either can be added on request.

**Supplementary diagnostic: feature swapping / context corruption** (evaluation only, no training,
no decisions).
- Take each trained `xattn_HMP` and `concat_HMP` checkpoint.
- Replace the MFCC and prosody inputs with those of a different recording, using a fixed
  derangement (seed 0). HuBERT stays intact.
- Measure the drop in validation and test F1.
- **Status, as decided by the team: supplementary, not definitive evidence of feature
  contribution.**
  - It measures how sensitive a *trained* model is to the swapped inputs. That is not what the
    streams contribute to learning.
  - The swapped inputs are off-distribution: they are real features, but paired with the wrong
    HuBERT frames. A drop can reflect confusion as well as reliance.
  - Feature contributions are established by the trained ablations and comparators (§6.5 tables).
    The swap results are reported beside them as supporting evidence only.

**Controls and audit.**
- Label permutation for `xattn_HMP`, using the E3 recipe (seed 1000 + fold).
- The full E3-style audit (§6.7).

### 6.6 Statistical analysis and decision rule (fixed now, before any E4 result)
- **Unit of analysis.** Paired Δ test macro-F1 per fold, with seeds averaged within a fold. UA is
  reported alongside. Each comparison gets an exact two-sided sign-flip p over folds.
- **Key comparisons:**
  - E4 − concat (primary);
  - E4 − control;
  - E4 − HuBERT-only.
- **Secondary comparisons:**
  - control − concat (re-tests E3's finding under the fair grid);
  - re-searched concat − E3 concat (effect of the wider grid);
  - E4 vs E2 and E1, with E3's confounds (§5.11);
  - ablations;
  - context corruption.
- **Multiplicity.**
  - The three key comparisons are reported with Holm-adjusted p.
  - With 6 and 5 folds, the smallest adjusted p is 0.094 and 0.19, so **statistical significance
    cannot be reached**. This is stated in advance.
- **Decision rule.** "E4 improves on X" will be claimed only if all of the following hold:
  - mean Δ > 0 on **both** datasets;
  - Δ > 0 in ≥ 5/6 RAVDESS folds and ≥ 4/5 CREMA-D folds;
  - mean Δ larger than the mean within-fold seed std of E4.
  - Otherwise the result is "no consistent improvement".

### 6.7 Integrity checks (in addition to the 36 E3 audit checks, adapted)
1. Every model and fold has 8 S runs. Each grid cell's validation F1 is in `selection.json`, and the
   selection can be re-derived from validation data alone.
2. **The E3 cells are reproduced:**
   - `concat_HMP` and `sa_HMP` at lr ∈ {1e-3, 3e-4} reproduce E3's S validation F1 and best epoch
     exactly;
   - where the selection is unchanged, the D test F1 is exactly equal.
3. Every run that hit the 150-epoch cap is listed.
4. E4's parameter count equals 769,286, and the control is within 5% of it.
5. **E1, E2 and E3 stay byte-identical.** All `results/E1`–`results/E3` files and E3 checkpoints
   are SHA-256 snapshotted before the run and compared after it.
6. RAM watchdog as in the CREMA-D resume, writing to **one** CSV with one writer. Below 1 GB free it
   stops the run and reports; it never restarts automatically.

### 6.8 Run budget (estimate; confirmed by the pilot before the full launch)
- **Assumed epochs per run:**
  - 1e-3 and 3e-4: about 25 (E3 measured best epoch 11–20, plus patience);
  - 1e-4: about 50;
  - 3e-5: about 120 (range 80–150).

| Phase | CREMA-D (5 folds) | RAVDESS (6 folds) |
|---|---|---|
| S: 4 models × 8 candidates | ≈ 7.2 h (5.6–8.5) | ≈ 1.2 h |
| D: 4 tuned × 3 seeds + 3 ablations × 3 seeds | ≈ 4.1 h if the attention models select 1e-4 (≈ 2 h at 3e-4; up to ≈ 12 h at 3e-5) | ≈ 0.7 h |
| Permutation control, corruption diagnostic, audit | ≈ 0.5 h | ≈ 0.1 h |
| **Total** | **≈ 12 h** (8–21) | **≈ 2 h** (1.5–3.5) |

- **Where the time goes.** Most of it is the 3e-5 candidates of the two attention models.
- **Cheaper alternative.** A 3-value grid {1e-3, 3e-4, 1e-4} would roughly halve the search, to
  about 3.3 h on CREMA-D. It would leave a real risk of a new lower-edge selection, so it is **not**
  recommended.

### 6.9 Order of work after approval
1. **Implement** `src/models/e4_cross_attention.py`, reusing `AttentionBlock` and
   `StreamProjection`. Unit tests:
   - parameter count = 769,286;
   - padding invariance;
   - key masking;
   - time-permutation invariance;
   - gradients reach every block;
   - control matching.
2. **Write** `scripts/stage5_e4_run.py` and `configs/e4_cross_attention.json`, with the same runner
   logic as E3, a new run hash and a new results directory.
3. **Run a validation-only pilot on fold 0** (RAVDESS and CREMA-D), with all 4 tuned models × 8
   candidates and no test predictions. Report:
   - the actual epochs at 1e-4 and 3e-5;
   - any cap hits;
   - time and memory;
   - exact reproduction of the E3 cells.
   Stop for review before continuing.
4. **Run the full experiment**, then the controls, audit, figures and report §6 results.
5. **Snapshot verification** of E1–E3 after every step.

### 6.10 Fold-0 pilot (validation only; `results/E4/pilot_<DS>.json`, `results/E4/<DS>/fold0/`)
- **Setup.**
  - `scripts/stage5_e4_run.py --pilot`, one process per dataset, under a RAM guard: one writer, stop
    below 1 GB free, no restart.
  - 32 S runs per dataset: 4 models × 8 candidates, seed 0, on the fold without its test partition.
  - `selection.json` was written. **No test prediction was made.** Phase D is disabled in the code
    until review.
- **Before the run.**
  - 19 new unit tests; whole suite 139 passed.
  - SHA-256 snapshot of 1,432 protected files (`results/E4/snapshot_before_e4_pilot.json`): E1–E3
    results, E1–E3 checkpoints, E1–E3 configs and the split files.
- **After the run:** 1,432/1,432 unchanged and no new files in protected directories.

**E3 reproduction (16 cells).** All runs are **exact**: `concat_HMP` and `sa_HMP` × dropout {0.2, 0.4}
× lr {1e-3, 3e-4} × both datasets. Each one matches E3 on:
- validation F1;
- best epoch;
- epochs run;
- the per-epoch validation-F1 and training-loss histories;
- the validation confusion matrix.

This confirms that raising the cap to 150 epochs and using the new runner leave E3's runs unchanged.
Alignment check on real data: maximum difference 0.0 for all streams.

**Epoch counts.**
- **No run reached the 150-epoch cap.** The most epochs run was 88 (RAVDESS `concat_H`, 3e-5).
- At 3e-5, the runs needed 30–88 epochs, far fewer than the ≈ 120 assumed in §6.8.
- On CREMA-D, E4's best epoch was early at every learning rate (1–19). Its selected run peaked at the
  second epoch (index 1) and never improved afterwards.

**Selections** (validation macro-F1; fold 0 only):

| Model | RAVDESS selected (dropout / lr) | val F1 | CREMA-D selected | val F1 | E3 selection (val F1) R / C |
|---|---|---|---|---|---|
| E4 `xattn_HMP` | 0.2 / 1e-4 | 0.813 | 0.2 / 3e-4 | 0.687 | — |
| Control `sa_HMP` | 0.2 / **3e-5 (lower edge)** | 0.831 | 0.2 / **3e-5 (lower edge)** | 0.706 | 0.4 / 3e-4 (0.804); 0.2 / 3e-4 (0.694) |
| Concat `concat_HMP` | 0.2 / 3e-4 (= E3) | 0.826 | 0.4 / 3e-4 (= E3) | 0.705 | identical |
| HuBERT-only `concat_H` | 0.2 / **1e-3 (upper edge)** | 0.827 | 0.2 / **3e-5 (lower edge)** | 0.700 | — |

- **The control was under-tuned in E3, at least on fold 0.**
  - The extended grid raises its best validation F1 by +0.027 (RAVDESS) and +0.011 (CREMA-D).
  - It again selects the new lower edge, and its validation F1 is still rising there:
    - 1e-4 → 3e-5: 0.822 → 0.831 (RAVDESS);
    - 0.700 → 0.706 (CREMA-D).
  - Under the pre-registered rule (§6.4), the grid is **not** extended again. If most folds behave
    like this, it is reported as a limitation.
- **HuBERT-only lands on opposite edges on the two datasets.** Its validation landscape is flat:
  - on CREMA-D the four learning rates at dropout 0.2 lie within 0.008;
  - the 3e-5 cell wins by 0.001 over 3e-4.
  - The edge selections reflect noise, not a missing grid region.
- **Validation ranking on fold 0** (descriptive only; one fold; nothing changed because of it):
  - E4 has the lowest validation F1 of the four tuned models on both datasets.
  - RAVDESS: control 0.831 > HuBERT-only 0.827 ≈ concat 0.826 > E4 0.813.
  - CREMA-D: control 0.706 ≈ concat 0.705 > HuBERT-only 0.700 > E4 0.687.

**Training stability.**
- No non-finite loss or gradient.
- On CREMA-D, clipping at 1.0 was active on 95–100% of steps for every model, as in E3.
- The control's largest pre-clipping gradient norms grow as the learning rate falls: 7 at 1e-3, up
  to 81 at 3e-5 (CREMA-D). E4's stay between 8 and 26. Training stayed stable, but this is noted.

**Resources.**

| | RAVDESS | CREMA-D |
|---|---|---|
| HuBERT layer load | 2 s | **99 s** (E3 resume: 14–16 s) |
| GPU peak, E4 / control / concat / H | 716 / 694 / 311 / 310 MiB | **2,097** / 2,073 / 1,606 / 1,604 MiB (incl. the 1.35 GB resident layer; §6.3 projected 2.1–2.2 GB) |
| Process RAM peak (in-process) | 1.80 GB | 3.06 GB |
| Worker working set, max (guard) | 1.75 GB | 2.99 GB |
| System free RAM, min (guard) | 4.98 GB | **1.84 GB** (58 of 554 samples < 3 GB) |
| Clean seconds per epoch: E4 / control / concat / H | 0.77 / 0.72 / 0.15 / 0.15 | 4.8–5.0 / 4.4–5.8 / 0.9–1.1 / 0.7–0.8 |
| Wall time of the pilot | 15.5 min | 51.4 min |

**Deviations and incidents** (none affects any result: training is deterministic, and the 16 reproduced cells
are exact):
1. **Machine stalls.**
   - One RAVDESS E4 epoch took 440 s instead of 0.8 s.
   - On CREMA-D, one control epoch took 62 s.
   - From about 19:49 to 20:08, attention-model epochs ran 1.5–2× slower (E4 at dropout 0.2: about
     10 s per epoch). This coincides with the lowest free-RAM readings, while the worker's own working
     set stayed at or below 3 GB. Other processes on the host were using memory.
   - The recorded `train_seconds` include these stalls, so timing comparisons from such runs must be
     read with this in mind.
2. **Low free RAM.** Free system RAM fell to 1.84 GB on CREMA-D, 0.84 GB above the guard threshold,
   caused by host load outside the worker. The 99 s layer load is probably related.
3. **Correction to the §6.2 draft text.** This is a documentation correction, not a protocol
   change. The draft said E4 ignores the cross-stream frame alignment; in fact, the fusion layers
   use it.

**Revised budget from fold 0.**
- At clean speed, about 6.5 h in total:
  - CREMA-D: S ≈ 3.3 h, D ≈ 1.8 h;
  - RAVDESS: S ≈ 0.8 h, D ≈ 0.5 h.
- With host contention like during the pilot, about 9 h. The proposal estimated about 14 h.
- Other folds use other layers and will stop at different epochs.

**Still to implement before the full run** (after review): phase D (final seeds, checkpoint → reload
→ test), the summary, the label-permutation control, the feature-swap diagnostic and the audit. All of
these are adapted from E3 and unit-tested before use. *(Done: see §6.11.)*

### 6.11 Full run: history, integrity, resources
- **Code before launch.** Before the full run, the following were added to `scripts/stage5_e4_run.py`:
  - Phase D: 3 seeds, then checkpoint → reload → test once from the reloaded checkpoint.
  - Stop conditions, all checked before any test prediction in a fold:
    - the fold's E3-comparable cells must reproduce E3 exactly;
    - `selection.json` must match the search;
    - the config file must equal the approved protocol.
  - The summary, with Holm correction and the pre-registered decision rule.
- **New scripts:**
  - `scripts/stage5_e4_controls.py`: the label-permutation control and the supplementary swap diagnostic.
  - `scripts/stage5_e4_audit.py`: the audit.
  - `scripts/stage5_e4_plots.py`: the figures.
- **Tests: 147 passed.** The 27 E4 tests include:
  - a synthetic end-to-end phase D, on which the audit passes and which it catches when tampered with;
  - a run with an E3 discrepancy, which stops before any test prediction;
  - a re-run of a fold, which retrains nothing.
- **RAVDESS** (2026-11-06 20:41–21:36, exit 0).
  - The fold-0 pilot S runs were reused.
  - All seed-0 finals reproduced their selection runs.
  - Minimum free RAM 4.16 GB; worker at most 1.74 GB.
- **First CREMA-D attempt** (21:36): **stopped by Claude Code's background-shell memory protection**,
  about 10 s after start.
  - It was opening the store and starting to load HuBERT layer 7, before any training.
  - Our 1 GB guard did not trigger: its only sample showed 6.24 GB free.
  - No CREMA-D D file or checkpoint existed, and nothing was restarted automatically.
- **Restart, approved by the team** (`scripts/run_e4_cremad_restart.ps1 -Step run`; CREMA-D only;
  Claude Code's memory protection left on).
  - Guard: stop below 1 GB free; samples every 1 s during store opening and layer loading, every 5 s
    otherwise.
  - The runner prints `[stage]` lines. These are log-only and do not change the run hash.
  - Preflight passed:
    - 6.8 GB free;
    - no experiment process running;
    - the 486-file E4 snapshot and the 1,432-file E1–E3 snapshot unchanged;
    - CREMA-D fold-0 run hash `6c711ebab212` recomputed and matching.
  - Ran 21:59 → 02:46 (4 h 47 min), **exit 0**, with no stderr output.
  - Minimum free RAM **3.68 GB**; worker at most **3.00 GB**.
  - The five HuBERT layer loads took 13–15 s each (99 s in the pilot).
  - Folds took 54–66 min each.
- **Follow-up** (`-Step followup`, 02:47 → 03:03, exit 0; steps run one after another under the guard):

  | Step | Duration | Peak worker memory | Minimum free RAM |
  |---|---|---|---|
  | Summary | 9 s | | |
  | Controls, RAVDESS | 1.6 min | 2.05 GB | 4.75 GB |
  | Controls, CREMA-D | 11.2 min | **4.57 GB** | **2.73 GB** |
  | Audit | 2.4 min | 2.77 GB | 4.69 GB |

  The CREMA-D controls step is the highest memory use of the whole experiment. It holds the layer plus
  a full model retrain plus swap evaluation. It stayed 1.7 GB above the stop threshold.
- **Integrity after every step.**
  - 486/486 E4 snapshot files unchanged.
  - 1,432/1,432 E1–E3 files unchanged.
  - 231 D files (11 folds × 21), 231 checkpoints. Each one loads, its metadata matches its D file, its
    hyper-parameters match the selection, and every `selection.json` is older than its fold's D files.
  - No `.tmp` files, no stop files.
- **Audit: `results/E4/audit.json`, 41/41 checks PASS.**
  - Completion, one run hash per fold, validation-only S files, and selections that can be re-derived.
  - Selection older than test, test rows, layer rule, hyper-parameters, seed-0 reproduction, class
    weights, checkpoint metadata.
  - Standardisers equal training-row statistics. Every one of the 231 reloaded checkpoints re-predicts
    its saved test predictions exactly.
  - E3 S cells reproduce exactly in **every** fold. Where re-searched concatenation selected E3's
    setting (fold 0 of both datasets), its test predictions are identical to E3's.
  - The fold-0 S runs are the pilot's. Parameter counts are correct. Split, Stage 4 and prosody
    SHA-256 unchanged; E1–E3 snapshot unchanged.
- **Epoch cap.** No run reached 150 epochs.

### 6.12 Results (test; mean ± std over folds; each fold = mean of 3 seeds)

| Dataset | Model | Macro-F1 | UA | Accuracy | val F1 | Parameters |
|---|---|---|---|---|---|---|
| RAVDESS | E1 SVM | 0.537 ± 0.059 | | | | |
| RAVDESS | E2 pooled HuBERT | **0.762 ± 0.054** | 0.771 | 0.764 | | |
| RAVDESS | E3 concat (as run, E3 grid) | 0.729 ± 0.064 | 0.735 | 0.737 | | 221,574 |
| RAVDESS | **E4 cross-attention** | 0.723 ± 0.061 | 0.729 | 0.733 | 0.773 | 769,286 |
| RAVDESS | control (E4 grid) | 0.730 ± 0.043 | 0.739 | 0.743 | 0.766 | 751,750 |
| RAVDESS | concat (E4 grid) | 0.726 ± 0.061 | 0.733 | 0.735 | 0.777 | 221,574 |
| RAVDESS | HuBERT-only (E4 grid) | 0.736 ± 0.046 | 0.740 | 0.745 | 0.782 | 182,534 |
| CREMA-D | E1 SVM | 0.503 ± 0.021 | | | | |
| CREMA-D | E2 pooled HuBERT | **0.713 ± 0.030** | 0.714 | 0.712 | | |
| CREMA-D | E3 concat (as run, E3 grid) | 0.695 ± 0.027 | 0.697 | 0.694 | | 221,574 |
| CREMA-D | **E4 cross-attention** | 0.674 ± 0.023 | 0.676 | 0.674 | 0.697 | 769,286 |
| CREMA-D | control (E4 grid) | 0.686 ± 0.026 | 0.689 | 0.686 | 0.708 | 751,750 |
| CREMA-D | concat (E4 grid) | 0.694 ± 0.028 | 0.696 | 0.693 | 0.713 | 221,574 |
| CREMA-D | HuBERT-only (E4 grid) | 0.688 ± 0.028 | 0.691 | 0.688 | 0.709 | 182,534 |

**Pre-registered decision rule (§6.6)** (`results/E4/decision.json`):

| Comparison | RAVDESS Δ F1 (folds better, p, Holm p) | CREMA-D Δ F1 (folds better, p, Holm p) | Verdict |
|---|---|---|---|
| E4 − concat (primary) | −0.003 (3/6, 0.750, 1.000) | −0.020 (**0/5**, 0.062, 0.188) | **no consistent improvement** |
| E4 − control | −0.007 (1/6, 0.625, 1.000) | −0.012 (1/5, 0.188, 0.375) | **no consistent improvement** |
| E4 − HuBERT-only | −0.013 (1/6, 0.188, 0.562) | −0.014 (1/5, 0.188, 0.375) | **no consistent improvement** |

- **All three key differences are negative on both datasets.** E4 is never better on average.
- E4's mean within-fold seed std is 0.034 (RAVDESS) and 0.012 (CREMA-D). The RAVDESS differences are
  within the seed noise.
- **The CREMA-D deficit against concatenation is consistent**: −0.020, with concatenation better in all
  5 folds. The UA results agree in sign everywhere.

**Secondary comparisons** (Δ test macro-F1; folds better; full table with UA in
`results/E4/<DS>/summary.json`):

| Comparison | RAVDESS | CREMA-D |
|---|---|---|
| control − concat | +0.004 (4/6) | −0.008 (1/5) |
| concat − HuBERT-only (M+P given H, fair search) | −0.010 (1/6) | +0.006 (4/5) |
| control − HuBERT-only | −0.006 (1/6) | −0.002 (2/5) |
| E4 − E2 | −0.039 (1/6, p 0.062) | −0.039 (0/5, p 0.062) |
| concat − E2 | −0.036 (0/6, p 0.031) | −0.019 (0/5) |
| HuBERT-only − E2 | −0.025 (0/6, p 0.031) | −0.025 (0/5) |
| E4 − E1 SVM | **+0.186 (6/6, p 0.031)** | **+0.171 (5/5)** |
| concat: E4 grid − E3 grid | −0.003 (1/6) | −0.001 (0/5) |
| control: E4 grid − E3 grid | +0.004 (2/6) | **+0.016 (4/5)** |
| HuBERT-only (own search) − H (E3, concat hyper-parameters) | −0.011 (1/6) | +0.003 (3/5) |

**E4 ablations** (E4's selected hyper-parameters; not separately tuned):

| Comparison | RAVDESS | CREMA-D |
|---|---|---|
| E4 ablation H↔M (no prosody) | 0.734 ± 0.053 | 0.680 ± 0.029 |
| E4 ablation H↔P (no MFCC) | 0.721 ± 0.045 | 0.674 ± 0.024 |
| E4 ablation H↔concat(M, P) (no stage-1 attention) | 0.730 ± 0.052 | 0.672 ± 0.025 |
| prosody given E4 (HMP − HM) | −0.011 (1/6) | −0.006 (0/5) |
| MFCC given E4 (HMP − HP) | +0.002 (3/6) | −0.000 (2/5) |
| stage-1 M↔P attention (HMP − H↔concat(M,P)) | −0.007 (2/6) | +0.002 (3/5) |
| H↔concat(M,P) − concat (cross-attention vs concatenation, same inputs) | +0.004 (3/6) | −0.021 (0/5) |
| H↔M − HuBERT-only | −0.003 (3/6) | −0.007 (3/5) |
| H↔P − HuBERT-only | −0.015 (2/6) | −0.013 (0/5) |

**Per fold** (test macro-F1; selected dropout / lr from validation):

| Dataset | Fold | Layer | E4 (seeds) | E4 | control | concat | HuBERT-only | E3 concat | E2 | E1 |
|---|---|---|---|---|---|---|---|---|---|---|
| RAVDESS | 0 | 6 | 0.675, 0.613, 0.647 | 0.645 (0.2 / 1e-4) | 0.649 | 0.644 | 0.665 | 0.644 | 0.680 | 0.459 |
| RAVDESS | 1 | 10 | 0.685, 0.675, 0.755 | 0.705 (0.4 / 1e-3) | 0.739 | 0.713 | 0.716 | 0.713 | 0.733 | 0.594 |
| RAVDESS | 2 | 5 | 0.763, 0.702, 0.689 | 0.718 (0.2 / 3e-5) | 0.735 | 0.748 | 0.748 | 0.772 | 0.784 | 0.555 |
| RAVDESS | 3 | 5 | 0.766, 0.745, 0.705 | 0.739 (0.2 / 1e-3) | 0.748 | 0.746 | 0.748 | 0.746 | 0.810 | 0.595 |
| RAVDESS | 4 | 5 | 0.852, 0.828, 0.808 | 0.830 (0.2 / 3e-4) | 0.775 | 0.821 | 0.805 | 0.821 | 0.824 | 0.546 |
| RAVDESS | 5 | 5 | 0.732, 0.666, 0.711 | 0.703 (0.2 / 1e-3) | 0.735 | 0.684 | 0.735 | 0.677 | 0.740 | 0.473 |
| CREMA-D | 0 | 7 | 0.666, 0.635, 0.672 | 0.658 (0.2 / 3e-4) | 0.685 | 0.691 | 0.688 | 0.691 | 0.694 | 0.513 |
| CREMA-D | 1 | 6 | 0.717, 0.702, 0.699 | 0.706 (0.2 / 1e-4) | 0.716 | 0.733 | 0.723 | 0.733 | 0.756 | 0.518 |
| CREMA-D | 2 | 12 | 0.634, 0.657, 0.655 | 0.649 (0.2 / 1e-4) | 0.643 | 0.661 | 0.645 | 0.665 | 0.675 | 0.468 |
| CREMA-D | 3 | 11 | 0.694, 0.684, 0.687 | 0.688 (0.2 / 3e-5) | 0.689 | 0.708 | 0.689 | 0.708 | 0.722 | 0.501 |
| CREMA-D | 4 | 5 | 0.686, 0.659, 0.665 | 0.670 (0.2 / 3e-4) | 0.695 | 0.675 | 0.693 | 0.676 | 0.718 | 0.514 |

**Per seed** (test macro-F1, mean over folds; seed 0 / 1 / 2; mean within-fold seed std):

| Model | RAVDESS | CREMA-D |
|---|---|---|
| E4 | 0.745 / 0.705 / 0.719; 0.034 | 0.679 / 0.667 / 0.676; 0.012 |
| control | 0.732 / 0.732 / 0.728; 0.026 | 0.687 / 0.683 / 0.687; 0.007 |
| concat | 0.730 / 0.729 / 0.719; 0.018 | 0.693 / 0.691 / 0.697; 0.006 |
| HuBERT-only | 0.725 / 0.751 / 0.733; 0.020 | 0.687 / 0.684 / 0.692; 0.009 |

E4 has the largest seed variability of the four tuned models on both datasets.

**Learning-rate selections** (validation only):

| Model | RAVDESS: lowest (3e-5) / highest (1e-3) | CREMA-D: lowest / highest |
|---|---|---|
| E4 | 1/6 / 3/6 | 1/5 / 0/5 |
| control | 1/6 / 0/6 | 2/5 / 0/5 |
| concat | 0/6 / 2/6 | 0/5 / 0/5 |
| HuBERT-only | 0/6 / **5/6** | 1/5 / 2/5 |

**Controls.**
- **Label permutation** (E4, selected configuration, seed 0, labels permuted with seed 1000 + fold):
  - RAVDESS: test macro-F1 **0.140 ± 0.014**; every fold 0.120–0.161, all below chance (0.167).
  - CREMA-D: **0.173 ± 0.007**.
  - No sign of leakage: leakage would push the score *above* chance. The RAVDESS values below 1/6 are
    expected when a model trained on shuffled labels collapses towards a few classes.
- **Feature swap — SUPPLEMENTARY.** Δ macro-F1, swapped − intact (mean over folds; share of predictions
  changed):

  | Model | Partition | RAVDESS | CREMA-D |
  |---|---|---|---|
  | E4 | validation | −0.031 (10%) | −0.087 (25%) |
  | E4 | test | −0.016 (11%) | −0.077 (26%) |
  | concat | validation | −0.062 (14%) | −0.099 (26%) |
  | concat | test | −0.015 (15%) | −0.088 (27%) |

  - Both trained models are sensitive to mismatched MFCC/prosody, more so on CREMA-D.
  - E4 is *not* more dependent on these streams than concatenation.
  - As decided by the team, this diagnostic is supporting evidence only. It measures how a trained
    model reacts to off-distribution input pairs, not what the streams contribute.

**Cost** (medians over D runs):

| Model | s/epoch (RAVDESS / CREMA-D) | s/run (RAVDESS / CREMA-D) | ms per test recording | GPU peak (RAVDESS / CREMA-D) |
|---|---|---|---|---|
| E4 | 0.62 / 4.78 | 15.1 / 91.0 | 0.26–0.27 | 715 / 2,098 MiB |
| control | 0.59 / 4.57 | 18.5 / 129.5 | 0.25–0.26 | 693 / 2,074 MiB |
| concat | 0.16 / 1.00 | 5.3 / 27.0 | 0.06–0.07 | 316 / 1,614 MiB |
| HuBERT-only | 0.13 / 0.79 | 2.8 / 22.9 | 0.05 | 316 / 1,613 MiB |

- Process RAM peaks are per process, not per model: 2.1 GB (RAVDESS) and 3.1 GB (CREMA-D).
- E4 costs 3.5× the parameters of concatenation and 4–5× the time per epoch.

### 6.13 Interpretation
1. **Cross-attention fusion did not improve on concatenation** (primary question), nor on the
   capacity-matched control or the HuBERT-only frame model.
   - On RAVDESS the differences lie inside the seed noise.
   - On CREMA-D, E4 is below concatenation in all 5 folds (−0.020).
   - The pre-registered verdict is "no consistent improvement" for all three key comparisons.
2. **Attention capacity does not explain the result either way.** E4 and the matched control differ by
   −0.007 / −0.012, and neither beats concatenation consistently.
3. **HuBERT remains the dominant representation, as in E3.**
   - HuBERT-only is within 0.014 of every fusion model on both datasets.
   - None of the ways of adding MFCC and prosody consistently improves on it: concatenation, the
     control or cross-attention.
   - Inside E4, removing prosody slightly *raises* F1 (H↔M − HMP = +0.011 RAVDESS, +0.006 CREMA-D).
     The full E4 is better than H↔M in only 1/6 and 0/5 folds. Removing MFCC changes nothing.
4. **Where the deficit comes from.** Replacing concatenation by cross-attention with the *same* inputs
   costs 0.021 on CREMA-D in all 5 folds (H↔concat(M,P) − concat). The stage-1 MFCC↔prosody
   attention adds nothing (+0.002 / −0.007).
5. **The limitations of §6.2 apply.**
   - The attention is content-based, without positional information.
   - A null result therefore says nothing about a position-aware cross-attention model, which was not
     tested.
6. **E2 is still the best model.** All frame-level models (E3 and E4) are 0.02–0.04 below E2 in nearly
   every fold. The confounds of §5.11 still apply:
   - a single layer per fold, against E2's learned layer weighting on CREMA-D;
   - the optimiser (clipping);
   - the search breadth.

### 6.14 Limitations
- **Learning-rate boundary (reported as decided; the grid was not extended again).**
  - The extended grid largely resolved E3's issue. The control now selects the lowest learning rate in
    only 1/6 RAVDESS and 2/5 CREMA-D folds, against 11/11 at E3's lower edge.
  - The re-searched control improved on CREMA-D (+0.016, 4/5).
  - Three control folds and 3 other model-fold selections still sit at 3e-5, so their optimum may lie
    lower.
  - **New:** HuBERT-only selected the *highest* rate (1e-3) in 5/6 RAVDESS folds, and E4 in 3/6. By
    the pre-registered rule the upper edge was not extended, so these models may be under-tuned upwards
    on RAVDESS.
- **Ablations reuse E4's hyper-parameters** and have fewer parameters than E4.
- **Few folds.** With 5–6 folds, no comparison can reach significance after Holm correction, as stated
  in §6.6.
- **Wider search, no test benefit.** For concatenation and HuBERT-only, the wider search did not help
  on test (−0.011 to +0.003). More candidates mainly increase selection noise.
- **Timing.** Timings from the pilot's contention window are inflated (§6.10). The D-run medians above
  match the clean speeds measured in the pilot.
- **One interruption** (§6.11). It was resolved by a guarded CREMA-D-only restart, with no effect on
  any result: the audit passed and both snapshots are unchanged.

Figures: `plots/E4_models_test.png`, `E4_comparisons_test.png`, `E4_search_validation.png`,
`E4_confusion_test.png`, `E4_per_class_f1.png`, `E4_feature_swap.png` (supplementary).
Results: `results/E4/<DS>/summary.json`, `runs.csv`, `fold<k>/`, `control_label_permutation.json`,
`diagnostic_feature_swap.json`; `results/E4/decision.json`; `results/E4/audit.json`.
Logs: `results/E4/*.log` and the RAM CSVs (`ram_monitor_*.csv`). Checkpoints:
`C:\SER_data\checkpoints\E4\<DS>\`.

---

## 7. Final comparative analysis of E1–E4 (analysis only — no new training or test evaluation)

**Source.** `scripts/stage5_final_analysis.py` reads only the saved per-run files. It writes:
- `results/final_analysis/final_analysis.json`;
- `plots/FINAL_per_class_E2_vs_E4.png`, `FINAL_confusion_E2_E4.png`, `FINAL_error_overlap.png` and
  `FINAL_accuracy_vs_cost.png`.

**Integrity of the analysis.** All 2,311 existing files in `results/`, the checkpoints and `configs/`
were SHA-256 hashed before and after the analysis. None changed (§7.10).

**Labels used in this section:**
- **[O] Observation:** measured directly from the saved results.
- **[I] Interpretation:** a conclusion reasoned from observations.
- **[H] Hypothesis:** a possible explanation that these experiments did *not* test.

**Statistics.**
- p-values are exact two-sided sign-flip tests over folds, as before.
- The smallest attainable p is 0.031 with 6 folds (RAVDESS) and 0.062 with 5 folds (CREMA-D).
- Seeds are averaged within a fold.

### 7.1 Verification of the reported numbers
- **[O] Test macro-F1 recomputed three ways.** For 9 model configurations × 2 datasets, each fold's
  score was recomputed:
  1. from the stored metric;
  2. from each run's confusion matrix;
  3. from the saved test predictions with the true labels taken from the Stage 4 manifest.
  - The largest disagreement between the three is 2.2e-16.
  - Every fold mean equals its `summary.json` value and the value printed earlier in this report, to
    3 decimals.
- **[O] Audits:** E2 28/28, E3 36/36, E4 41/41 checks passed.
  - E1 has no separate audit file. Its test isolation is covered by the NaN-poisoning unit test and
    the validation-reproduction check (§3.4).
- **[O] Permutation controls** (training labels shuffled; test macro-F1, chance 0.167):

  | Experiment | RAVDESS | CREMA-D |
  |---|---|---|
  | E2 | 0.187 | 0.186 |
  | E3 | 0.158 | 0.174 |
  | E4 | 0.140 | 0.173 |

### 7.2 Consolidated comparison (test macro-F1, mean ± std over folds; Δ = paired difference from E2 per fold)

| Model | Input → fusion | Trainable params | RAVDESS F1 | Δ vs E2 (folds better) | CREMA-D F1 | Δ vs E2 (folds better) |
|---|---|---|---|---|---|---|
| E1 SVM | MFCC statistics | (SVM; 615 / 4,508 support vectors, fold 0) | 0.537 ± 0.059 | −0.225 (0/6, p 0.031) | 0.503 ± 0.021 | −0.210 (0/5) |
| **E2 pooled HuBERT** | pooled HuBERT (single, weighted or concat layers) → MLP | 198,406 – 1,576,454 (per fold) | **0.762 ± 0.054** | — | **0.713 ± 0.030** | — |
| E3 concat (E3 grid) | H+M+P frames → concat | 221,574 | 0.729 ± 0.064 | −0.033 (0/6, p 0.031) | 0.695 ± 0.027 | −0.018 (0/5) |
| E3 self-attention control (E3 grid) | H+M+P → concat + 4 self-attention blocks | 751,750 | 0.726 ± 0.037 | −0.035 (1/6) | 0.669 ± 0.025 | −0.044 (0/5) |
| E3 HuBERT-only (concat hp) | H frames | 182,534 | 0.747 ± 0.059 | −0.014 (2/6) | 0.685 ± 0.028 | −0.028 (0/5) |
| HuBERT-only (E4 grid) | H frames | 182,534 | 0.736 ± 0.046 | −0.025 (0/6, p 0.031) | 0.688 ± 0.028 | −0.025 (0/5) |
| concat (E4 grid) | H+M+P → concat | 221,574 | 0.726 ± 0.061 | −0.036 (0/6, p 0.031) | 0.694 ± 0.028 | −0.019 (0/5) |
| self-attention control (E4 grid) | H+M+P → concat + 4 self-attention blocks | 751,750 | 0.730 ± 0.043 | −0.031 (1/6) | 0.686 ± 0.026 | −0.027 (0/5) |
| **E4 cross-attention** | H+M+P → 2-stage bidirectional cross-attention | **769,286** | 0.723 ± 0.061 | −0.039 (1/6) | 0.674 ± 0.023 | −0.039 (0/5) |

- **[O] E2 is the strongest model.**
  - Best in 4/6 RAVDESS folds (fold 1: the control; fold 4: E4) and in 5/5 CREMA-D folds.
  - Mean rank among the 7 main models: RAVDESS E2 1.33, HuBERT-only 2.83, control 3.5, E3 4.0,
    E4 4.5, concat 4.83, E1 7.0. CREMA-D E2 1.0, E3 2.4, concat 3.4, HuBERT-only 4.2, control 4.4,
    E4 5.6, E1 7.0.
- **[O] No frame-level model beats E2 reliably.** Across the 7 frame-level configurations, a
  frame-level model beat E2 in **5 of 77** fold comparisons, all on RAVDESS, by at most +0.009.
  These comparisons share folds and are not independent, so this count is descriptive.
- **[O] E4 is the most architecturally sophisticated model and the least accurate HuBERT model on
  CREMA-D.** It is 0.039 below E2 on both datasets.
- **[O] Exploratory, post hoc and not pre-registered:** pooling the 11 folds of both datasets as
  paired units:
  - E4 − E2 = −0.039, better in 1/11 folds, sign-flip p = 0.002;
  - HuBERT-only − E2 = −0.025, 0/11, p = 0.001.
  - The pre-registered analyses (per dataset, §6.6) cannot reach p < 0.05 after correction. These
    pooled values describe how consistent the direction is; they are not a confirmatory test.

### 7.3 Per-class analysis: E2 vs E4 (`FINAL_per_class_E2_vs_E4.png`)

Per-class test metrics, mean over folds and seeds; "folds" = folds in which E4 is better. Precision
(P), recall (R) and F1 are shown for each model.

| Dataset | Class | E2 P / R / F1 | E4 P / R / F1 | ΔF1 E4 − E2 (folds) | ΔF1 E4 − HuBERT-only (folds) | ΔF1 E4 − concat (folds) |
|---|---|---|---|---|---|---|
| RAVDESS | neutral | 0.716 / 0.851 / 0.765 | 0.672 / 0.684 / 0.664 | **−0.100 (0/6)** | −0.005 (4/6) | −0.002 (3/6) |
| RAVDESS | happy | 0.830 / 0.715 / 0.754 | 0.784 / 0.681 / 0.712 | −0.042 (2/6) | −0.019 (2/6) | −0.005 (3/6) |
| RAVDESS | sad | 0.644 / 0.679 / 0.655 | 0.661 / 0.618 / 0.628 | −0.027 (2/6) | −0.009 (2/6) | **+0.028 (6/6)** |
| RAVDESS | angry | 0.839 / 0.807 / 0.816 | 0.811 / 0.837 / 0.816 | +0.000 (3/6) | −0.002 (2/6) | −0.012 (2/6) |
| RAVDESS | fearful | 0.757 / 0.733 / 0.734 | 0.723 / 0.715 / 0.704 | −0.030 (1/6) | −0.011 (3/6) | +0.004 (3/6) |
| RAVDESS | disgust | 0.863 / 0.844 / 0.847 | 0.804 / 0.840 / 0.814 | −0.033 (2/6) | −0.034 (1/6) | −0.030 (1/6) |
| CREMA-D | neutral | 0.773 / 0.799 / 0.781 | 0.735 / 0.771 / 0.749 | −0.032 (1/5) | −0.012 (1/5) | −0.012 (2/5) |
| CREMA-D | happy | 0.744 / 0.723 / 0.732 | 0.696 / 0.648 / 0.666 | **−0.066 (0/5)** | **−0.029 (0/5)** | **−0.036 (0/5)** |
| CREMA-D | sad | 0.626 / 0.663 / 0.643 | 0.590 / 0.626 / 0.605 | −0.038 (0/5) | −0.008 (2/5) | −0.016 (1/5) |
| CREMA-D | angry | 0.794 / 0.821 / 0.807 | 0.765 / 0.795 / 0.779 | −0.028 (0/5) | −0.014 (1/5) | −0.011 (2/5) |
| CREMA-D | fearful | 0.669 / 0.628 / 0.646 | 0.618 / 0.586 / 0.599 | −0.047 (0/5) | −0.011 (1/5) | −0.020 (1/5) |
| CREMA-D | disgust | 0.692 / 0.651 / 0.669 | 0.674 / 0.631 / 0.647 | −0.022 (0/5) | −0.006 (3/5) | −0.022 (1/5) |

**Observations.**
1. **[O] CREMA-D: E4 is below E2 for every class.** It is lower in at least 4 of 5 folds for all six
   classes, and in all 5 folds for five of them. Precision and recall both fall. The largest drops
   are happy (−0.066), fearful (−0.047) and sad (−0.038).
2. **[O] RAVDESS: the deficit is concentrated on neutral.**
   - Neutral F1 −0.100, E4 below E2 in 6/6 folds; recall falls from 0.851 to 0.684.
   - Angry is unchanged (+0.000, 3/6), and E4's angry recall is higher (+0.030).
3. **[O] The RAVDESS neutral loss is not caused by fusion.** The HuBERT-only frame model shows almost
   the same deficit against E2 (neutral −0.096, 1/6), and so does E3 (−0.086, 1/6).
   - E4 − HuBERT-only on neutral is only −0.005.
   - [I] The neutral loss is a property of the **frame-level pipeline**, not of adding MFCC and
     prosody, nor of cross-attention.
   - Neutral is the smallest RAVDESS class (9.1% of test recordings, 64 of 704 training recordings
     per fold) and is up-weighted by the class-weighted loss in every model.
4. **[O] CREMA-D happy is disproportionately affected by cross-attention fusion.** It is the class
   E4 loses on in *every* comparison: against E2 (0/5), HuBERT-only (0/5) and concat (0/5).
   - Its recall drops 0.723 → 0.648 against E2.
   - The extra errors go to fearful (+2.4 pp), angry (+1.6 pp), neutral (+1.5 pp) and disgust
     (+1.4 pp).
5. **[O] RAVDESS: fusion moves errors within the sad / fearful cluster without a net gain.**
   - Against HuBERT-only, E4 raises sad *precision* in 6/6 folds (+0.042) but lowers sad *recall*
     (−0.050, 1/6). It raises fearful recall (+0.035, 5/6) but lowers fearful precision (−0.055, 1/6).
   - Against E2: true sad predicted as fearful +4.9 pp, true fearful predicted as sad −4.5 pp.
   - E4 predicts "sad" for 17.3% of recordings, against 19.3% for E2 and 18.2% true.
   - Against concatenation, E4's sad F1 is higher in 6/6 folds (+0.028), but its disgust F1 is lower
     (−0.030, 1/6).
   - [I] Cross-attention changes *which* low-arousal and negative emotions are confused, rather than
     resolving the confusion.

**[O] Difficult classes: the same pattern in every model** (`FINAL_confusion_E2_E4.png`).
- The largest off-diagonal cells are the same for E1, E2 and E4 on both datasets:

  | Dataset | Model | Largest confusions (% of the true class) |
  |---|---|---|
  | RAVDESS | E2 | fearful → sad 15.8, neutral → sad 11.8 |
  | RAVDESS | E4 | neutral → sad 17.0, sad → fearful 14.1, fearful → sad 11.3 |
  | RAVDESS | E1 | sad → neutral 21.4, neutral → sad 20.8, fearful → sad 19.3 |
  | CREMA-D | E2 | fearful → sad 17.5, disgust → sad 12.0, sad → fearful 11.3, sad → disgust 10.7 |
  | CREMA-D | E4 | fearful → sad 19.0, sad → fearful 13.1, disgust → sad 12.9 |

- Sad, fearful and disgust (plus neutral on RAVDESS) form the low-F1 cluster in every model. On
  CREMA-D, E2's three lowest F1 values are sad 0.643, fearful 0.646 and disgust 0.669.
- [I] This confusion structure is a property of the task and data. HuBERT makes it smaller than
  MFCC statistics do: E2 − E1 is +0.10 to +0.36 F1 per class, better in 5/5 or 6/6 folds for every
  class except RAVDESS angry (5/6). Neither fusion method changes the structure.
- [H] A plausible acoustic reason is shared low-arousal or negative-valence cues, but these
  experiments did not test it.

**[O] Error overlap** (`FINAL_error_overlap.png`; same test recordings, same seed):

| Comparison | RAVDESS | CREMA-D |
|---|---|---|
| Recordings correct only for E2 | 9.9% | 11.6% |
| Recordings correct only for E4 | 6.8% | 7.8% |
| Both wrong | 16.7% | 21.0% |
| …of which, both predicted the same wrong label | 12.7% | 15.6% |
| Prediction agreement, E2 vs E4 | 79% | 75% |
| Prediction agreement, E2 vs HuBERT-only | 83% | 78% |
| Prediction agreement, E2 vs E1 | 57% | 56% |

- [I] E4 corrects fewer of E2's errors than it introduces.
- [I] Most errors are shared, and usually with the *same* wrong label. Both models fail on the same
  recordings in the same way.
- [H] The 7–8% of recordings that only E4 gets right show some non-redundancy. Whether this could be
  exploited, for example by an ensemble, can only be judged with a design selected on validation
  data. Measuring it here on test would amount to test-set model selection, so it is not done.

### 7.4 Why did E2 outperform E3 and E4?

**(a) Temporal pooling.**
- **[O]** E2 pools *before* its classifier: mean or mean+std over frames of the raw 768-d layer,
  standardised per dimension with training statistics.
- **[O]** The frame models pool *after* a per-frame projection: Linear 768 → 128 → per-frame
  LayerNorm → (fusion / attention) → GELU, followed by mean+std.
- **[O]** On RAVDESS, E2 used the *same single layer* as the frame models in all 6 folds. Even so, the
  HuBERT-only frame model is below E2:
  - on test in 6/6 folds (−0.025);
  - on validation in 5/6 folds (−0.035; E2's validation score is the maximum over 39 candidates, so
    this comparison favours E2).
- **[O]** On CREMA-D, E2 used multi-layer inputs in folds 1–4 (weighted(all) three times,
  concat(1, 9) once). HuBERT-only is below E2 in 5/5 folds (−0.025). The gap is smallest in fold 0,
  where both used layer 7 (−0.006).
- **[I] Lost temporal information is *not* supported as the explanation.**
  - E2 discards all temporal structure, yet it wins.
  - The frame models do not model temporal order either: they are invariant to permuting frames
    (§6.2).
  - The comparison therefore contrasts *two order-free summaries*, not "temporal vs pooled".
- **[H] Candidate explanations** (none tested in isolation):
  1. The 128-d per-frame bottleneck plus per-frame LayerNorm discards information that E2's
     1,536-d mean+std of the raw layer keeps.
  2. On CREMA-D, E2 had access to learned multi-layer weighting.
  3. Optimisation differences: gradient clipping active on 95–100% of CREMA-D steps for the frame
     models, none for E2.
  4. Search breadth: 39 candidates per fold for E2, against 8 for each frame model.

**(b) Are MFCC and prosody complementary to HuBERT?**
- **[O] Alone, yes:**
  - In E3, M+P reach 0.549 / 0.577, above the E1 SVM on MFCC statistics (0.537 / 0.503).
  - Without HuBERT they complement each other (+0.07 / +0.03 for prosody given MFCC, 6/6 and 5/5).
- **[O] Given HuBERT, no consistent gain under any fusion method:**

  | Comparison | RAVDESS | CREMA-D |
  |---|---|---|
  | concat − HuBERT-only (E3 grid) | −0.018 (0/6) | +0.010 (4/5) |
  | concat − HuBERT-only (E4 grid) | −0.010 (1/6) | +0.006 (4/5) |
  | E4 − HuBERT-only | −0.013 (1/6) | −0.014 (1/5) |
  | prosody inside E4 (HMP − HM) | −0.011 (1/6) | −0.006 (0/5) |
  | MFCC inside E4 (HMP − HP) | +0.002 | −0.000 |
  | DisVoice-103 added to E3 | +0.005 | −0.004 |

- **[O]** The trained models *do* use M+P. Swapping them between recordings changes 10–27% of
  predictions and lowers F1 (supplementary diagnostic, §6.12).
- **[I]** MFCC and prosody are informative but largely **redundant with HuBERT** for these six
  emotions. The models learn to use them without benefit.
  - On RAVDESS, with only 704 training recordings per fold, adding them tends to *hurt*.
  - On CREMA-D they add at most +0.01, within the seed-noise range.
- **[H]** The extra input dimensions add variance on the small dataset. This fits the larger seed std
  of the fusion models on RAVDESS (§7.6) but was not tested directly.

**(c) Does cross-attention improve feature interaction?**
- **[O] No.** Against concatenation:
  - E4 − concat = −0.003 (RAVDESS, 3/6) and −0.020 (CREMA-D, **0/5**).
  - With identical inputs, replacing concatenation by H↔concat(M,P) cross-attention gives
    −0.021 (CREMA-D, 0/5).
  - The MFCC↔prosody attention stage adds +0.002 / −0.007.
  - Self-attention capacity of the same size does not help either: control − concat +0.004 (4/6) /
    −0.008 (1/5).
- **[I] Methodological limitation.**
  - The cross-attention is content-based, without positional encoding (§6.2), so it cannot exploit
    the frame-level time alignment of the streams.
  - The result applies to content-based cross-attention, not to every form of cross-modal attention.

**(d) Are the additional parameters and training complexity justified?**
- **[O] No.**
  - E4 has 3.5× the parameters of concatenation and 4.2× those of HuBERT-only.
  - It needs 2.9× (RAVDESS) to 3.4× (CREMA-D) the median training time per run of concatenation.
  - It is the least accurate HuBERT model on CREMA-D and the most seed-unstable model on both
    datasets.
  - E2 is more accurate and cheaper to train than every frame model (§7.5).

**(e) Do the fold-wise results support these explanations?**
- **[O] The ranking E2 > frame models is fold-consistent** (§7.2).
- **[O] The effects of adding M+P or of cross-attention are not fold-consistent on RAVDESS.** The
  differences are inside the seed noise and change sign between folds.
- **[O] On CREMA-D they are consistently non-positive for cross-attention:** E4 − concat 0/5, and
  H↔concat(M,P) − concat 0/5.
- [I] The data support "fusion did not help" more firmly than any specific mechanism for *why*.

### 7.5 Computational efficiency (classifier stage only)
HuBERT feature extraction is shared by E2–E4. Its cost was not logged, so it is **not** reported.

| Model | Trainable params | Median training s per final run (R / C) | All saved training runs: search + finals, total s (R / C) | Inference ms per test recording (R / C) | GPU peak MiB (R / C) | Process RAM peak (R / C) |
|---|---|---|---|---|---|---|
| E1 SVM (CPU) | 615 / 4,508 SVs (fold 0) | 0.04 / 1.7 | 145 / 462 (incl. 36-candidate selection) | 0.07 / 0.45 | — | not logged |
| E2 | 198 k – 1.58 M | 0.70 / 6.4 | 196 / 1,102 (42 runs per fold) | 0.04 / 0.07 | 85 / 759 | 1.46 / 3.53 GB |
| E3 concat | 221,574 | 4.3 / 26.2 | 1,485 / 8,070 (whole E3, 38 runs per fold) | 0.07 / 0.07 | 316 / 1,614 | 2.30 / 3.13 GB |
| HuBERT-only (E4 grid) | 182,534 | 2.8 / 22.9 | | 0.05 / 0.05 | 316 / 1,613 | |
| concat (E4 grid) | 221,574 | 5.3 / 27.0 | | 0.07 / 0.06 | 316 / 1,614 | |
| control (E4 grid) | 751,750 | 18.5 / 129.5 | | 0.25 / 0.26 | 693 / 2,074 | |
| **E4** | 769,286 | 15.1 / 91.0 | 4,137 / 20,020 (whole E4, 53 runs per fold) | 0.26 / 0.27 | 715 / 2,098 | 2.11 / 3.14 GB |

- **[O] Measurement notes.**
  - CREMA-D GPU peaks of the frame models include the 1.35 GB resident HuBERT layer.
  - Process RAM peaks are per process.
  - The highest RAM use within E4 was the CREMA-D controls step: 4.57 GB worker working set (§6.11).
    The E3 permutation control reached 4.69 GB in-process (§5.6).
- **[I] Accuracy against complexity.**
  - E2 sits at the best point: highest accuracy at the lowest training cost of any HuBERT model.
    It costs about 14× less per run than E4 on CREMA-D and about 18× less in total experiment
    training time.
  - Attention models cost 2.9–4.8× more training time per run and about 4× more inference time
    than the non-attention frame models, with no accuracy gain.
  - E4's whole experiment took 6.7 h of logged training across both datasets, against 0.4 h for E2.

### 7.6 Statistical and methodological assessment
- **[O] Seed variability** (mean within-fold std of test F1):

  | Model | RAVDESS | CREMA-D |
  |---|---|---|
  | E2 | 0.011 | 0.006 |
  | E3 | 0.027 | 0.006 |
  | E4 | **0.034** | **0.012** |
  | control | 0.026 | 0.007 |
  | concat | 0.018 | 0.006 |
  | HuBERT-only | 0.020 | 0.009 |

  [I] The fusion and attention models are the least stable on the small dataset.
- **[O] Validation → test gap** (mean val F1 of the final runs − test F1):
  - RAVDESS: 0.023–0.052 (E2 0.049, E4 0.050; smallest is E3 HuBERT-only 0.023).
  - CREMA-D: 0.014–0.034 (E2 0.018, E4 0.023).
  - [I] Every model shows a sizeable RAVDESS gap; it does not separate E2 from E4. With 4 test
    actors per fold, speaker variation dominates.
- **[O] Fold-wise consistency.**
  - E2 vs all frame models is consistent (§7.2).
  - Key E4 comparisons are consistent only on CREMA-D (E4 − concat 0/5).
- **[O] Significance.**
  - Per dataset, the smallest attainable p is 0.031 (RAVDESS) and 0.062 (CREMA-D). After Holm
    correction over the three key E4 comparisons, no comparison can be significant (§6.6).
  - Only the uncorrected RAVDESS comparisons with 0/6 or 6/6 folds reach p = 0.031.
  - The pooled 11-fold values in §7.2 are exploratory.
- **[O] Leakage controls.** All permutation controls are at chance and all audits pass (§7.1).
  - Every reloaded checkpoint (E2–E4) re-predicts its saved test predictions exactly.
  - The E3 settings are reproduced exactly inside E4.
- **Limitations of the evidence.**
  - Two acted English corpora, six classes, 6 + 5 speaker-independent folds.
  - No cross-corpus test.
  - One frozen HuBERT model, one layer per fold for the frame models.
  - Search spaces differ between E2 and the frame models.
  - Learning-rate edge selections remain for some models (§6.14).
  - Feature-extraction cost was not measured.

### 7.7 Ablation interpretation across E3 and E4
1. **[O] HuBERT is indispensable and dominant.**
   - Removing it costs 0.18 (RAVDESS) and 0.12 (CREMA-D) in every fold (E3).
   - Shapley shares: HuBERT 0.36 / 0.28, against 0.13 / 0.16 for MFCC and 0.07 / 0.09 for prosody.
2. **[O] MFCC and prosody are redundant given HuBERT** (§7.4 b). Inside E4, prosody is slightly
   harmful and MFCC is neutral.
3. **[O] The fusion method does not change this.** Concatenation, self-attention and cross-attention
   all land within ±0.015 of HuBERT-only. Only cross-attention is consistently *below* concatenation
   (CREMA-D).
4. **[O] DisVoice-103 adds nothing.** Its missingness indicators slightly hurt (E3).
5. **[O] The wider search helped only the under-tuned control** (+0.016 on CREMA-D, 4/5). It did not
   help concatenation or HuBERT-only (−0.011 to +0.003).

### 7.8 Key findings
1. **The strongest model is the simplest HuBERT model.** E2 (pooled HuBERT, one hidden layer) scores
   0.762 / 0.713 and is best in 9 of 11 folds.
2. **The most sophisticated model is not the strongest.** E4 (two-stage bidirectional
   cross-attention, 769 k parameters) scores 0.723 / 0.674. It is 0.039 below E2 on both datasets and
   does not beat concatenation, the capacity-matched control or HuBERT-only (pre-registered verdict:
   no consistent improvement).
3. **HuBERT carries the emotion information.** MFCC and prosody are informative alone but redundant
   given HuBERT, whatever the fusion.
4. **The gap between E2 and the frame-level models is not explained by fusion.** It is equally
   present for the HuBERT-only frame model, which uses the same layer as E2 on RAVDESS. Its cause is
   unresolved (§7.4 a).
5. **Per class:**
   - The RAVDESS gap is mostly neutral (−0.100) and comes from the frame-level pipeline.
   - On CREMA-D every class drops, happy most (−0.066), and happy is uniquely hurt by cross-attention.
   - The sad / fearful / disgust confusion cluster is common to all models and is not reduced by
     fusion.
6. **Cost:** E4 needs 14× (CREMA-D) to 22× (RAVDESS) E2's median training time per run and about 4×
   the inference cost of the non-attention frame models, for lower accuracy.

### 7.9 Future research directions (proposals, not started)
1. **Explain the E2 > frame-model gap before adding complexity.** Use controlled, one-factor,
   validation-selected comparisons on the HuBERT-only frame model:
   - multi-layer (weighted) HuBERT input;
   - no per-frame LayerNorm, or a wider projection (closer to E2's 1,536-d pooled vector);
   - no gradient clipping;
   - matched search breadth.
   [H] If one factor closes most of the gap, the frame-level design was the problem, not frame-level
   modelling as such.
2. **Test complementarity where it is most likely to show.** Add the M+P / DisVoice *utterance*
   statistics to E2's pooled vector. This is cheap and uses the strongest pipeline. A gain is not
   expected from §7.4 b; a null result would close the fusion question for this setting.
3. **Only if fusion is revisited:** use position-aware or windowed cross-attention that can exploit
   the shared frame grid. That is the untested part of the cross-attention hypothesis (§6.2).
4. **Generalisation:** cross-corpus evaluation (RAVDESS ↔ CREMA-D).
   [H] Hand-crafted features may matter more under corpus shift; this is not supported by current
   evidence.
5. **Statistical power:** repeated cross-validation or more speaker folds, and a pre-registered
   pooled analysis across datasets, to make 0.01–0.02 effects decidable.
6. **Class-focused work** on the sad / fearful / disgust cluster and on RAVDESS neutral, through the
   loss or class balance, evaluated per class.

### 7.10 Integrity of this analysis
- **[O] Files unchanged.** 2,311 files (all of `results/` except the new `final_analysis/`, all
  checkpoints and `configs/`) were hashed before the analysis and re-checked after it: none changed.
- **[O] Snapshots.** The E1–E3 snapshot (1,432 files) and the E4 snapshot (486 files) still verify.
- **No new training, test evaluation or model selection was performed.**

Figures: `plots/FINAL_per_class_E2_vs_E4.png`, `FINAL_confusion_E2_E4.png`, `FINAL_error_overlap.png`,
`FINAL_accuracy_vs_cost.png`; analysis data: `results/final_analysis/final_analysis.json`; script:
`scripts/stage5_final_analysis.py`.

---

## 8. Proposal — G-study: why does pooled HuBERT (E2) beat frame-level HuBERT models? (NOT RUN — CANCELLED BY SCOPE DECISION)

> **Final status (2026-11-07): the research is frozen at E1–E4. The G-study was cancelled by scope decision.**
> - **No G-study training or test evaluation was carried out.** No pilot or full run, no test predictions,
>   no checkpoints, and no derived feature files.
> - **Validation-only reproduction gates (fold 0) passed before the cancellation:**
>
>   | Gate set | RAVDESS | CREMA-D |
>   |---|---|---|
>   | Pooled gates (R1–R5 and E2-internal) | 12/12 exact | 2/2 exact (R1, R2; R3–R5 not applicable) |
>   | Frame gate R6 (Z vs E4 `concat_H`) | 8/8 exact | 8/8 exact |
>   | Determinism probes (P0 and Z cells re-run) | bit-identical | bit-identical |
>
>   The prepared G-study code therefore reproduces the saved E2 and E4 runs exactly wherever their settings
>   coincide. Details are in `results/G/gates_<DS>_<group>.json`.
> - **Unresolved B0 discrepancy, specific to the unused G-study pipeline.**
>   - B0's pooled inputs, computed from the fp16 frame cache, differ from E2's standardised pooled inputs
>     (fp32-origin cache) by up to 2.96e-3 on RAVDESS and 2.55e-3 on CREMA-D. This exceeds the pre-registered
>     1e-3 tolerance.
>   - A preliminary measurement points to storage precision: the B0 code path against a float64 recomputation
>     from the same fp16 frames differs by at most 6.5e-4, while the fp16 frames differ from the fp32-origin
>     cache by up to 3.0e-3.
>   - This was not diagnosed further. The tolerance was not changed.
>   - It concerns only the new B0 bridge arm. No evidence links it to any E1–E4 result: E2 used its own pooled
>     cache, E3/E4 used the frame cache directly, and both are reproduced exactly by gates R1–R6.
> - **Existing results remain unchanged.**
>   - The SHA-256 manifest of 2,338 protected files (results, checkpoints, configs and feature caches) verified
>     unchanged after every gate step and at the end.
>   - The E1–E3 snapshot (1,432 files) and the E4 snapshot (486 files) are unchanged.
>   - The only edit to existing code is an optional `grad_clip` field in `src/training/trainer.py`. It defaults
>     to off, which keeps E2's code path identical (confirmed by gates R1–R5).
> - The design below is kept as a record of the proposal only.

This is a diagnostic study, **not** E5. It was not run (see the final status above).

**Status.**
- Direction approved by the team on 2026-11-07: the 12-arm design, the uniform 13-layer average as a
  practical approximation, and a fold-0 validation-only pilot.
- **Revision 1**, made at the team's request:
  - statistics are now per dataset, with pooled results exploratory only (§8.7);
  - E2 reproduction is separated from the comparison with E2 (§8.6);
  - frame-level *learned* layer weighting is restored as arm Z-W (§8.3);
  - explicit pilot safeguards and stopping criteria are added (§8.8).
- **Goal.** Decompose the measured gap between E2 and the HuBERT-only frame model (§7.4 a), one factor
  at a time, under matched search budgets.
- **Gap to explain** (test macro-F1, E2 − HuBERT-only frame model; HuBERT-only lower in every fold):

  | Dataset | Gap | Folds |
  |---|---|---|
  | RAVDESS | −0.025 | 0/6 |
  | CREMA-D | −0.025 | 0/5 |

- **Results location.** All results will go to `results/G/`. No E1–E4 file is touched.

### 8.1 What differs between E2 and the frame models (from the code and saved selections)

| # | Factor | E2 (`src/models/e2_hubert.py`, `src/training/trainer.py`) | HuBERT-only frame model (`E3Fusion(("hubert",))`, `src/training/seq_trainer.py`) |
|---|---|---|---|
| F1 | HuBERT representation | searched over 13 single layers, concat(1, 9) and learned weighted(all). Selected multi-layer inputs in 4/5 CREMA-D folds, single layers in 6/6 RAVDESS folds | one layer per fold (rule from E2 phase A) |
| F2a | per-frame normalisation | none. Pooled vectors are standardised per dimension with training statistics | per-frame LayerNorm after the projection |
| F2b | dimension before pooling | raw 768 → pooled 1,536 (mean+std) | 768 → **128** projection → GELU layer → pooled 256 |
| F3 | gradient clipping | none | global norm 1.0 (active on 95–100% of CREMA-D steps) |
| F4 | search budget and content | 39 candidates per fold: 30 representation × pooling, 8 head (h ∈ {256, 512} × p × lr ∈ {1e-3, 3e-4}), 1 loss variant | 8 candidates: p ∈ {0.2, 0.4} × lr ∈ {1e-3, 3e-4, 1e-4, 3e-5}; head 256 fixed |
| F5 | class-weighted loss *(additional confound found in the files)* | selected **unweighted** CE in 3/5 CREMA-D and 1/6 RAVDESS folds | always weighted |
| — | maximum epochs | 50; reached by 1 of 462 E2 runs, so effectively non-binding | 150 |
| — | batching | random batches of pooled vectors | length-bucketed frame batches, masked pooling |

**[O] Existing validation-only evidence on F1** (E2 phase A, default head):
- On RAVDESS, the best single layer beats weighted(all) in **6/6 folds** (by 0.007–0.036).
- On CREMA-D, weighted(all) is **≥ the best single layer in 4/5 folds** (by 0.007–0.023 in folds 1–4;
  −0.001 in fold 0).
- [H] Prediction: F1 can explain part of the CREMA-D gap but **none** of the RAVDESS gap.
- This is consistent with §7.4 a: on RAVDESS, E2 already used the same single layer as the frame model.

### 8.2 Design principles
1. **Ladder from E2 to the frame model.** Each arm differs from its comparison arm in exactly one
   factor. Two existing endpoints are used read-only, not re-run:
   - **E2**: its own 39-candidate search;
   - **Z** = the E4 HuBERT-only frame model `concat_H`.
2. **Everything else fixed** (as in E4):
   - the same speaker-independent folds;
   - the same per-fold HuBERT layer L (unless the arm's factor is F1);
   - AdamW with weight decay 1e-4, batch size 32;
   - at most 150 epochs, patience 10, early stopping on validation macro-F1;
   - final seeds 0–2, selection seed 0;
   - class-weighted CE, unless the arm's factor is F4/F5.
3. **Matched budgets.**
   - Every ladder arm gets the same **G8** grid: p ∈ {0.2, 0.4} × lr ∈ {1e-3, 3e-4, 1e-4, 3e-5}, head
     width 256.
   - The budget factor is tested separately with **B32** = G8 × head {256, 512} × loss {weighted,
     unweighted}. These are exactly the non-representation dimensions E2 searched, so B32 also covers
     F5 and head width.
4. **Test discipline, stricter than E4.**
   - Phase S (validation only) runs for **all arms and all folds first**.
   - A frozen selection manifest (SHA-256 of every `selection.json`) is then written.
   - Only after that does phase D run (train → checkpoint → reload → test once).
   - No test prediction exists while any selection can still change.
5. **Implementation checks before any test prediction** (§8.6). Exact reproduction of saved E2 and E4
   runs under exactly matched settings is a **gate**. The *comparison* of P0 with the original E2 is a
   **scientific result**. The two are kept separate.
6. **Per-dataset reporting.** Every contrast is reported separately for RAVDESS and CREMA-D (§8.7).

### 8.3 Architectures (HuBERT only; x_t = standardised 768-d frame t; masked mean+std pooling)

| ID | Model | Trainable params |
|---|---|---|
| **P0** | E2 code, single layer L, mean+std pooling: MLP(1536 → 256 → 6) on the standardised pooled vector | 395,014 |
| **B0** | *bridge*: frame pipeline (frame standardisation, bucketing, masking) with **no per-frame layers**. u = [mean_t x_t ; std_t x_t] → standardised with training-recording statistics → the same MLP as P0 | 395,014 |
| **Z** (existing) | h_t = Drop(LN(W_p x_t)), W_p ∈ ℝ^{128×768}; z_t = Drop(GELU(W_f h_t)), W_f ∈ ℝ^{128×128}; [mean ; std] (256) → MLP(256 → 256 → 6) | 182,534 |
| Z-noLN | Z without the LayerNorm | 182,278 |
| Z-768 | Z with d = 768 (no bottleneck); pooled 1,536, the same width as P0's input | 1,577,734 |
| Z-768-noLN | d = 768, no LayerNorm | 1,576,198 |
| Z-UM / P0-UM | Z / P0 on the **uniform 13-layer average** x̄_t = (1/13) Σ_l x_t^l instead of layer L | as Z / P0 |
| P0-W | E2 code with learned weighted(all), mean+std, pooled (as E2's weighted(all) representation) | 395,027 |
| **Z-W** | **frame-level learned layer weighting**, trained jointly with Z: x̃_t^l = (x_t^l − m_l) / s_l (per layer and dimension, training-frame statistics); x̂_t = Σ_l softmax(a)_l x̃_t^l with a ∈ ℝ^13 initialised to 0; x̂_t → Z | 182,547 |

**Uniform 13-layer average (UM): a fixed, label-free approximation.**
- One sequential, chunked pass over the existing `hubert.npy`: 16.4 GB (CREMA-D), 1.9 GB (RAVDESS).
  It produces a new 768-d stream (1.35 / 0.16 GB) and its pooled mean / std. These are new files; the
  caches are read-only.
- **UM is not equivalent to learned layer weighting.**
  - It fixes equal weights on *raw* layers.
  - Learned weighting (P0-W, Z-W) standardises each layer and learns non-uniform weights on training
    data.
  - The two can differ whenever the useful information is concentrated in a few layers.
- UM is kept as a cheap approximation, and reported separately from the learned arms.

**Z-W: frame-level learned weighting, restored for CREMA-D.**
- **Why it is needed.** On CREMA-D, E2's validation results favour weighted(all) over the best single
  layer in 4/5 folds (§8.1).
- **Data path.** All 13 layers do not fit in RAM or VRAM on CREMA-D, so batches are **streamed** from
  the existing `hubert.npy`:
  - each recording's (T, 13, 768) fp16 block is contiguous on disk and is read with buffered file
    reads (no memory map);
  - a background prefetch thread loads the next batch;
  - reading per epoch (training + validation recordings, every layer) is about **14 GB** on CREMA-D and
    about 1.6 GB on RAVDESS.
- **Standardisation.** Per-layer statistics come from per-recording, per-layer frame sums and sums of
  squares, pre-computed in the same pass as UM (float64; about 1.2 GB on CREMA-D, 0.2 GB on RAVDESS).
  Training-fold statistics are then exact sums over training recordings.
- **Datasets: CREMA-D only** (team decision 2026-11-07).
  - **Justification from validation only.** In E2's phase A, weighted(all) was ≥ the best single layer
    in 4/5 CREMA-D folds, and below it in 6/6 RAVDESS folds (§8.1).
  - On RAVDESS, F1 is represented by P0-W (pooled, learned) and by the uniform average (P0-UM, Z-UM).
  - Z-W results therefore apply to CREMA-D only, and are reported as such.
- **Relation to Z.** With the weights fixed one-hot on layer L, Z-W reduces to Z on a per-layer
  standardised input. A unit test checks this nesting (§8.6).
- **Cost.** Z-W is I/O-bound. Its runtime is the largest uncertainty of the study and is measured in the
  pilot (§8.5, §8.8).

**Parameter counts as a confound.** The Z-768 arms have about 4× P0's parameters. The 2×2 design
below lets this be read off: d = 128 vs d = 768.

### 8.4 Experiment matrix (13 new arms + 2 existing references)

| ID | Pipeline | Per-frame layers (d, LN) | Clip | Representation | Grid | Isolates (comparison) |
|---|---|---|---|---|---|---|
| E2 | existing | — | no | searched | E2-39 | reference |
| Z | existing (E4) | 128, LN | yes | L | G8 | reference |
| P0 | E2 code | none | no | L | G8 | **anchor**. Its *comparison* with the original E2 is a scientific result, not a reproduction (§8.6 b) |
| B0 | frame | none | no | L | G8 | implementation bridge (B0 vs P0; expected ≈ 0) |
| P0c | E2 code | none | **yes** | L | G8 | F3, pooled side (P0c vs P0) |
| Z-noclip | frame | 128, LN | **no** | L | G8 | F3, frame side (Z-noclip vs Z) |
| Z-noLN | frame | 128, **no LN** | yes | L | G8 | F2a at d = 128 (vs Z) |
| Z-768 | frame | **768**, LN | yes | L | G8 | F2b with LN (vs Z) |
| Z-768-noLN | frame | **768, no LN** | yes | L | G8 | F2a × F2b (vs Z, Z-noLN, Z-768); then vs B0 = the per-frame layers themselves |
| P0-W | E2 code | none | no | **weighted(all)** | G8 | F1 pooled, learned (vs P0) |
| P0-UM | E2 code | none | no | **UM** | G8 | F1 pooled, uniform (vs P0) |
| Z-UM | frame | 128, LN | yes | **UM** | G8 | F1 frame, uniform approximation (vs Z) |
| **Z-W** | frame, streamed | 128, LN | yes | **learned weighted(all)** | G8 | F1 frame, learned (vs Z); primary on CREMA-D |
| P0-B32 | E2 code | none | no | L | **B32** | F4 + F5 pooled (vs P0) |
| Z-B32 | frame | 128, LN | yes | L | **B32** | F4 + F5 frame (vs Z) |

**Gap decomposition** (read along the ladder):

| Step | Comparison | Measures |
|---|---|---|
| 1 | E2 → P0 | E2's representation search and its extra search dimensions |
| 2 | P0 → B0 | implementation of the data path (expected ≈ 0) |
| 3 | B0 → Z-768-noLN | per-frame layers: projection + GELU, no bottleneck |
| 4 | Z-768-noLN → Z | bottleneck + LayerNorm, decomposed by the 2×2 |
| 5 | clip on / off | measured at both ends |

**[H] Pre-registered expectations** (to be confirmed or refuted, per dataset):
- B0 ≈ P0.
- F1 (P0-W, Z-W; UM separately) has an effect only on CREMA-D.
- If Z-768-noLN ≈ B0 ≈ P0, the bottleneck and LayerNorm explain the gap.
- If the gap persists at matched budget (P0-B32 − Z-B32 ≈ P0 − Z), budget is not the cause.

### 8.5 Runs and estimated runtime
- **Per fold:** each G8 arm = 8 S + 3 D runs; B32 arms = 32 S + 3 D, of which 8 S are shared with
  P0 / Z.
  - New runs per fold: 71 on the pooled side, 79 frame runs at d = 128 (incl. the 8 re-run Z cells for
    the guard), 22 frame runs at d = 768.
- **Time per run.** Pooled and d = 128 times are the measured means of comparable E2 and E4 runs:
  - CREMA-D: pooled ≈ 10 s; frame d = 128 ≈ 29 s (E4 HuBERT-only S-phase mean).
  - RAVDESS: pooled ≈ 1.5 s; d = 128 ≈ 4.3 s.
  - d = 768: about 2× d = 128 is **assumed** (more per-frame compute), to be measured in the pilot.
  - **Z-W (assumption, to be measured):** I/O-bound. At an effective 1–2 GB/s, 14 GB per epoch gives
    about 8–15 s per epoch on CREMA-D. With 25–35 epochs that is 3.5–9 min per run, and 11 runs per fold.

| Item | CREMA-D (5 folds) | RAVDESS (6 folds) |
|---|---|---|
| Pooled arms (P0, P0c, P0-W, P0-UM, P0-B32) | ≈ 1.0 h | ≈ 0.2 h |
| Frame d = 128 (B0, Z guard, Z-noclip, Z-noLN, Z-UM, Z-B32) | ≈ 3.2 h | ≈ 0.6 h |
| Frame d = 768 (Z-768, Z-768-noLN) | ≈ 1.8 h | ≈ 0.3 h |
| **Z-W** (streamed; CREMA-D only) | **≈ 3–8 h** (dominant uncertainty) | — (not run) |
| Pre-computation (UM + per-layer frame sums; one pass) | ≈ 5–15 min | < 1 min |
| Permutation controls (B0, Z-768-noLN), audit | ≈ 0.4 h | ≈ 0.1 h |
| **Total** (all folds, including the fold-0 search) | **≈ 10–15 h** | **≈ 1.2 h** |

- **Execution order.** All non-streamed arms run first. Z-W runs last, in its own processes, so its
  cost cannot block the other arms.
- **Disk.** New files total less than 10 GB:
  - UM streams 1.5 GB;
  - per-layer sums about 1.4 GB;
  - results and about 430 checkpoints, under 2 GB.
  The free space is 755 GB.

**Memory.**
- **GPU.** The non-streamed frame arms hold one 768-d stream (layer L *or* UM) on the GPU, as in E3/E4.
  - Measured peak 1.6–2.1 GB at d = 128; expected about 2–2.6 GB at d = 768 (pilot).
  - Z-W keeps no stream on the GPU. A batch of 32 × 250 × 13 × 768 fp16 is about 160 MB.
- **RAM.** The process uses about 3.1–3.5 GB (measured in E2–E4), plus about 0.5 GB of prefetch
  buffers for Z-W.
- **Guard.** Runs execute one after another under the existing RAM guard: 1 s sampling during loading,
  no automatic restart.
- **Guard metric (approved 2026-11-07).** Z-W reads 14 GB per epoch through the OS file cache, which
  fills the *standby* list. The previous measure, "free" memory, excludes reclaimable standby pages and
  would stop Z-W without real memory pressure.
  - The **1 GB threshold** now applies to **available** memory: free + standby. It is read with the
    Windows `GlobalMemoryStatusEx` call (ullAvailPhys). This is fast and locale-independent; the WMI
    performance-counter class took 4–9 s per query in the self-test.
  - A second stop applies if the worker's working set exceeds 6 GB.
  - Free RAM, available RAM and worker memory are logged separately.
  - The guard never restarts anything.

### 8.6 Implementation checks (gates) vs comparison with E2 (result)

**(a) Exact reproduction under exactly matched settings — a gate.**
- **What must match.** A G-study run whose settings coincide exactly with a saved run must reproduce it
  exactly. Coinciding means the same representation, pooling, head width, dropout, learning rate, loss
  weighting, seed and fold. Five things are compared:
  1. validation macro-F1;
  2. best epoch;
  3. epochs run;
  4. the full per-epoch validation-F1 and training-loss histories;
  5. the validation confusion matrix.
- **Epoch cap.** E2 used a 50-epoch cap and the G-study uses 150. None of the required E2 runs reached
  50 epochs (largest: 42), so the larger cap cannot change them.
- **Required cells.** These were identified from the saved files: **58 E2 cells + 88 E4 cells**.

| Gate | G-study cell | Saved run it must equal | Count |
|---|---|---|---|
| R1 | P0 (p 0.2, lr 1e-3, h 256, weighted) | E2 `A_layer{L}_mean_std` (default head; every fold) | 6 R + 5 C |
| R2 | P0-W (p 0.2, lr 1e-3, h 256, weighted) | E2 `A_weighted_all_mean_std` (every fold) | 6 R + 5 C |
| R3 | P0 (h 256; p ∈ {0.2, 0.4}; lr ∈ {1e-3, 3e-4}; weighted) | E2 `B_h256_*` in the folds where E2 selected layer L + mean+std (RAVDESS folds 0, 2, 3, 5) | 16 R |
| R4 | P0-B32 (h 512; same p and lr; weighted) | E2 `B_h512_*`, same folds | 16 R |
| R5 | P0-B32 (unweighted cell equal to E2's phase-C configuration) | E2 `C_unweighted`, same folds | 4 R |
| R6 | Z (G8 cells; also the G8 part of Z-B32) | E4 `concat_H` S runs | 8 × 11 folds |

  (R = RAVDESS, C = CREMA-D.)

- **Consistency of E2's own files.** On RAVDESS folds 0, 2, 3 and 5, E2's `A_layer{L}_mean_std` and
  `B_h256_d0.2_lr0.001` are the same configuration. They already agree in their saved epochs run, and
  the gate checks them against each other as well.
- **No equivalent on CREMA-D.** No CREMA-D B or C cell coincides, because E2 selected mean pooling or
  multi-layer inputs there. CREMA-D is therefore gated by R1, R2 and R6 only.
- **Any mismatch stops the study before any further run.** This applies in the pilot (fold-0 cells)
  and in the full run (all folds, before phase D).
- **After test predictions (audit).** Wherever a G-study arm's *frozen selection* coincides exactly with
  E2's final configuration (or with Z's), its D-phase test predictions must equal the saved E2 (or E4)
  test predictions.

**(b) Comparing P0 with the original E2 — a scientific result, not a check.**
- E2's final configuration differs from P0 fold by fold:

  | Difference from P0 | RAVDESS | CREMA-D |
  |---|---|---|
  | Representation | — | multi-layer in folds 1–4 |
  | Pooling | mean in folds 1, 4 | mean in folds 0, 2 |
  | Head 512 | 4/6 folds | 3/5 folds |
  | Unweighted loss | 1/6 folds | 3/5 folds |

- E2 also searched a different learning-rate grid ({1e-3, 3e-4}) in a staged A → B → C procedure.
- E2 − P0 therefore measures the **combined** value of E2's selection freedom. It is reported per
  dataset, with a per-fold list of what E2 chose differently.
- **Partial decomposition:**
  - P0-B32 searches head width and loss weighting, so E2 − P0-B32 leaves representation, pooling and
    search procedure.
  - P0-W isolates learned layer weighting on the pooled side.
- **Pooling (mean vs mean+std) is *not* varied:** every G arm uses mean+std, like the frame models.
  E2's phase-A validation comparison of the two poolings is reported descriptively.

**(c) Other implementation checks (unit tests or pilot; hard gates).**
- **B0 inputs.** B0's pooled vectors, computed from frames, equal E2's pooled cache for layer L within
  1e-3 (max abs, after standardisation).
  - B0 is *not* expected to reproduce P0's training exactly (different batching and numerics). Its
    agreement with P0 is tested empirically (§8.7).
- **UM.** UM frames equal the recomputed 13-layer mean on ≥ 100 random recordings (exact after fp16
  rounding), and the pooled UM statistics are recomputed (relative error ≤ 1e-6).
- **Z-W loader and nesting.**
  - Streamed batches equal an in-memory reference exactly (RAVDESS, all 13 layers loaded).
  - Per-layer training statistics from the pre-computed sums equal direct computation (relative error
    ≤ 1e-9).
  - Z-W with one-hot weights equals Z on the same standardised input (synthetic unit test).
- **Default behaviour unchanged.** New options in the E2 trainer (clipping) and in the frame model (d,
  LN) are off or equal to the current values by default. All 147 existing tests pass and R1–R6 confirm
  unchanged behaviour.
- **Integrity.** A SHA-256 manifest covers every existing file under `results/`, the checkpoints,
  `configs/` and the HuBERT caches used. It is checked before and after the pilot and after every
  full-run stage.

### 8.7 Statistical analysis and decision rule (fixed before any run)

**Primary analysis: per dataset, paired over folds.**
- **Unit.** Δ test macro-F1 per fold, after averaging the 3 seeds within the fold, for each contrast
  and **separately for RAVDESS (6 folds) and CREMA-D (5 folds)**.
- **Reported for each contrast and dataset:**
  - the per-fold Δ values;
  - mean Δ, SD of Δ across folds (n − 1);
  - fold wins (Δ > 0) / n, and min / max;
  - the exact two-sided sign-flip p, with Holm adjustment within the dataset over the primary
    contrasts.
- **p-values are descriptive only.** The smallest attainable p is 0.031 (RAVDESS) and 0.062 (CREMA-D);
  after Holm over many contrasts, none can fall below 0.05.
- **Secondary measures:** UA, per-class F1 (neutral on RAVDESS and happy on CREMA-D specifically,
  §7.3), and the validation → test gap of each arm.

**Primary contrasts** (each evaluated per dataset):

| # | Contrast | Factor |
|---|---|---|
| C1 | B0 − P0 | implementation (equivalence) |
| C2 | P0c − P0 | F3, pooled side |
| C3 | Z-noclip − Z | F3, frame side |
| C4 | Z-noLN − Z | F2a |
| C5 | Z-768 − Z | F2b |
| C6 | Z-768-noLN − Z | F2a + F2b |
| C7 | Z-W − Z | F1, frame, learned |
| C8 | Z-UM − Z | F1, frame, uniform approximation |
| C9 | P0-W − P0 | F1, pooled, learned |
| C10 | P0-UM − P0 | F1, pooled, uniform |
| C11 | Z-B32 − Z | F4 + F5, frame |
| C12 | P0-B32 − P0 | F4 + F5, pooled |
| C13 | P0 − Z | the gap under matched representation, budget and loss |

- **Secondary contrasts:**
  - the F2a × F2b interaction (Z, Z-noLN, Z-768, Z-768-noLN);
  - B0 → Z-768-noLN (the per-frame layers);
  - E2 − P0 and E2 − P0-B32 (§8.6 b).

**Classifying each contrast.** Decisions use consistency and effect size, not p-values.
1. **Consistent effect.** All of the following hold:
   - mean Δ has the same sign on both datasets;
   - Δ is in that direction in ≥ 5/6 RAVDESS **and** ≥ 4/5 CREMA-D folds;
   - |mean Δ| is at least the larger within-fold seed std of the two arms, on each dataset.
2. **Dataset-specific effect.** The criteria hold on exactly one dataset. F1 is *predicted* to be
   CREMA-D-specific (§8.1); any such effect is reported as specific to that dataset.
3. **No consistent effect.** Neither of the above.

**"Explains part of the gap"** for a factor requires all of:
- class 1 or 2;
- the change moves the frame model towards P0 (or makes P0 worse, for pooled-side factors);
- the per-dataset **gap recovered** = (arm − Z) / (P0 − Z), which is reported. "Explains most of the
  gap" requires ≥ 0.5 on that dataset.

No claim is based on a p-value alone, and none on pooled data.

**Bridge equivalence** (C1, per dataset).
- Accepted if |mean Δ| ≤ 0.010 **and** the wins in one direction are below the consistency threshold
  (< 5/6, < 4/5).
- If it fails on a dataset, the data path is investigated before any frame-side contrast on that
  dataset is interpreted.

**Exploratory only: pooled 11-fold analysis.**
- An exact sign-flip test over all 11 folds may be reported, in a separate table, labelled
  **EXPLORATORY**.
- It is not used for any classification or claim.

**Limitations of treating folds as observations.**
1. **Folds are not independent.**
   - Each test speaker appears in one test fold only, but the *training* sets overlap heavily between
     folds: RAVDESS trains on 16 of 24 actors per fold.
   - Fold-level Δ values are therefore positively correlated. Their SD understates the true
     uncertainty of a cross-validated mean, and no unbiased estimator of that variance exists for
     k-fold CV (Bengio & Grandvalet, 2004; see also Nadeau & Bengio, 2003).
2. **Sign-flip assumptions.** Sign-flip tests assume independent, exchangeable pairs. With correlated
   folds, the p-values are approximate and likely optimistic.
3. **Very few units.** 6 and 5 folds mean low power for effects of 0.01–0.02 and coarse attainable
   p-values.
4. **Heterogeneous datasets.** The corpora differ in speakers, recording conditions and class balance.
   Pooling assumes a common effect, which heterogeneity can cancel or exaggerate. This is why the
   analysis is per dataset.
5. **Seeds are not replicates.** They are averaged within folds. Seed variability is reported but
   does not add independent observations.
6. **Selection noise differs by budget.** Each configuration is chosen on a small validation set (4 or
   about 15 speakers). B32 arms choose among 4× more candidates than G8 arms, which inflates their
   validation optimism. The validation → test gaps are therefore reported for every arm.
7. **Many contrasts.** 13 primary contrasts × 2 datasets. Holm adjustment within each dataset is
   reported, but it remains descriptive.

**Controls and audit, as in E4:**
- label permutation for the two new model types (B0, Z-768-noLN);
- a full audit:
  - isolation;
  - frozen-manifest ordering (every selection older than every test artifact);
  - exact checkpoint re-prediction;
  - standardisers fitted on training data (including Z-W's per-layer statistics);
  - gates R1–R6 and the post-test coincidence checks;
  - E1–E4 snapshots unchanged.

### 8.8 Pilot (fold 0 of both datasets; validation only; not yet launched)

**Scope.**
- Phase S (validation-only search) on fold 0 of both datasets for all 13 arms: G8 cells, B32 cells and
  Z-W.
- Plus the one-pass pre-computation (UM streams and per-layer frame sums). This writes **new** files
  under `C:\SER_data\features\<DS>\derived_G\` and never modifies the existing caches.

**Validation-only guarantees.**
- Pilot mode has no phase-D code path: the runner refuses phase D when `--pilot` is set.
- Every search run receives `without_test(fold)`; accessing the test partition raises an error.
- The NaN-poisoning unit test (as in E3/E4) shows that test frames never enter any computation,
  including the Z-W streaming loader.
- No checkpoints are written in the pilot.
- After the pilot, a scan confirms that no file under `results/G/` contains a test metric or test
  prediction.

**Measured in the pilot:**
- per arm: seconds per epoch, epochs run, seconds per run;
- GPU peak;
- process RAM peak (working set);
- minimum available and minimum free system RAM;
- disk use of the derived files and of `results/G/`;
- Z-W loader throughput (GB/s) and Z-W seconds per epoch;
- a projected full-run time per arm and in total;
- gates R1, R2, R3, R4, R5 and R6 on fold 0:
  - RAVDESS fold 0: 1 + 1 + 4 + 4 + 1 + 8 cells;
  - CREMA-D fold 0: 1 + 1 + 0 + 0 + 0 + 8 cells;
- the §8.6 c checks;
- B0 − P0 on fold-0 validation (descriptive).

**Hard stopping criteria.** If any of these occurs, the pilot stops and the stage is reported. Nothing
is restarted and no setting is changed.

| # | Criterion |
|---|---|
| S1 | Any failure of R1–R6 or of the §8.6 c checks, or any unit-test failure |
| S2 | A non-finite loss or gradient, any stderr output from a worker, or a guard intervention |
| S3 | Any change in the SHA-256 manifest of existing files (results, checkpoints, configs, caches) |
| S4 | Any test metric or test prediction under `results/G/`, or any checkpoint written during the pilot |
| S5 | Available RAM < 1 GB (guard stop), or worker working set > 6 GB |
| S6 | GPU peak > 3.5 GiB (of 4 GiB) |
| S7 | New disk use > 10 GB |
| S8 | Projected full-run time > 16 h in total, or > 8 h for Z-W on CREMA-D. The pilot stops and reports options; the scope is never reduced automatically |

**Flagged for review (not a stop by itself).** A fold-0 validation difference between B0 and P0 of more
than 0.05 on either dataset. This is a single fold, and the two are not expected to be bit-identical.

**Pilot runtime estimate.**
- CREMA-D: pooled about 10 min; d = 128 about 31 min; d = 768 about 15 min; Z-W 0.5–1.2 h;
  pre-computation 5–15 min.
- RAVDESS: about 15 min.
- **Total about 2–2.5 h.**
- The pilot always ends with a stop for review before the full run.

### 8.9 Expected scientific contribution
1. **A quantified decomposition** of the pooled-vs-frame gap into representation, normalisation and
   bottleneck, optimiser, search budget and implementation. Currently §7.4 a can only list candidate
   causes.
2. **A decision basis for E5.**
   - If a corrected frame model matches P0/E2, frame-level designs (needed for any temporal or
     cross-stream attention) can be built on it.
   - If no factor closes the gap, E5 should build on pooled representations, and the frame-level
     fusion results of E3/E4 are partly explained by a weaker backbone.
3. **Methodological value.**
   - Matched-budget comparisons.
   - An implementation-equivalence bridge.
   - Exact reproduction gates against saved E2 and E4 runs.
   - Per-dataset, consistency-based decisions that do not depend on fold-independence assumptions.
4. **Negative results are informative.** For example, if clipping or budget matter little, those
   explanations can be ruled out for E3/E4.

**Limitations of the design.**
- **Learned layer weighting.** At frame level it is tested only in Z-W. That requires streaming 14 GB
  per epoch on CREMA-D, so its runtime is uncertain until the pilot. UM is a separate *approximation*,
  not a substitute for learned weighting.
- **Interactions.** The ladder measures one-factor effects around two anchors, not all interactions.
  Only F2a × F2b is fully crossed.
- **Pooling.** Mean vs mean+std is not varied; E2 chose mean pooling in 4 of 11 folds.
- **Inference.** Fold-level inference is limited (§8.7). With 6 and 5 folds, small effects may remain
  undecidable.
- **Scope.** Results are for frozen HuBERT-base features on two acted corpora.
- **Grid edges.** Some arms may again select learning rates at the edges of the grid. Following the E4
  rule, the grid is not extended; this would be reported as a limitation.
