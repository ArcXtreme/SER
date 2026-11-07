# Research paper outline (IEEE conference, two-column, target 10–12 pages)

Status: outline only, with no paper text yet.

**Sources.**
- Primary: `reports/stage5_models.md` §0–§7. §8, the G-study, was **not run**: it was cancelled by scope
  decision and must not be presented as an experiment.
- Numbers were checked against `results/final_analysis/final_analysis.json`. Its verification recomputes
  every test macro-F1 from stored metrics, confusion matrices and predictions; all agree with
  `results/E*/…/summary.json` and with the report to 3 decimals.
- Dataset counts were checked against `data/metadata/index.csv`.

---

## 1. Final recommended title
**Revisiting Multi-Feature Fusion for Speech Emotion Recognition: A Controlled Evaluation of HuBERT and
Cross-Attention**

Alternative, if a more result-forward title is preferred: *"Does Cross-Attention Fusion Help Frozen HuBERT?
A Controlled Speaker-Independent Study of Speech Emotion Recognition"*.

## 2. Central research question
Does multi-feature fusion, in particular bidirectional cross-attention between HuBERT, MFCC and prosody
streams, give a meaningful improvement over pooled pre-trained HuBERT representations for
speaker-independent speech emotion recognition on acted English speech?

**Sub-questions** (map one-to-one onto the experiments):
- **RQ1 (E1 vs E2):** How much does a frozen self-supervised representation add over classical MFCC
  statistics?
- **RQ2 (E3):** Does frame-level concatenation of MFCC and prosody with HuBERT help, and what does each
  stream contribute?
- **RQ3 (E4):** Does cross-attention fusion beat concatenation, a parameter-matched self-attention
  control, and a HuBERT-only frame model under the same search?
- **RQ4 (all):** Why is the pooled model (E2) stronger than every frame-level model, and at what
  computational cost?
  - Only what was measured may be stated here.
  - The causes remain hypotheses, because the G-study was not run.

## 3. Contributions (as claims the evidence supports)

**C1. A controlled, leakage-audited, speaker-independent benchmark.**
- Four model families evaluated on identical folds:
  - RAVDESS: 6 folds, 24 speakers;
  - CREMA-D: 5 folds, 91 speakers.
- Validation-only selection, three seeds per fold, and test predictions made once from reloaded
  checkpoints.
- Supporting evidence:
  - automated audits: E2 28/28, E3 36/36, E4 41/41 checks passed;
  - label-permutation controls at chance;
  - exact re-prediction of every checkpoint.

**C2. A negative but consistent result on fusion.**
- Pooled frozen HuBERT with a one-hidden-layer MLP (E2) is the strongest model:
  - 0.762 / 0.713 macro-F1 (RAVDESS / CREMA-D);
  - best of the seven main models in 9 of 11 folds.
- Neither concatenation (E3) nor cross-attention fusion (E4) improves on a HuBERT-only frame model. The
  pre-registered verdict for all three E4 key comparisons is "no consistent improvement".

**C3. Capacity vs interaction.**
- A parameter-matched self-attention control (751,750 vs 769,286 parameters) shows that extra attention
  capacity does not help either.
- Cross-attention is below concatenation in **all 5** CREMA-D folds (−0.020).

**C4. Stream attribution.**
- Leave-one-out, add-one and Shapley analyses show that HuBERT carries nearly all usable information.
- MFCC and prosody are informative alone but redundant given HuBERT.
- DisVoice-103 adds nothing consistent.

**C5. Per-class and error analysis.**
- The RAVDESS deficit of frame models is concentrated on *neutral*. It is shared by the HuBERT-only
  frame model, so it is a frame-pipeline effect, not a fusion effect.
- *Happy* on CREMA-D is the class most hurt by cross-attention.
- The sad / fearful / disgust confusion cluster is common to all models.

**C6. Accuracy–cost trade-off.**
- The most accurate model (E2) is also the cheapest HuBERT model.
- The most sophisticated model (E4) is among the least accurate and most seed-unstable, at about
  14–22× E2's median training time per run (classifier stage only).

**Do NOT claim:**
- reproduction of HuMP-CAT;
- cross-lingual or cross-corpus generalisation;
- statistical significance after correction;
- a causal explanation of why E2 beats frame models.

---

## 4. Section-by-section outline

### Abstract (≈ 200 words)
- **Problem.** Fusion of hand-crafted acoustic features with self-supervised representations is often
  reported to help SER. Controlled, speaker-independent tests are rare.
- **What we do.**
  - Four controlled experiments (E1–E4) on RAVDESS and CREMA-D.
  - Six emotions; speaker-independent cross-validation with 6 and 5 folds.
  - Frozen HuBERT-base, MFCC and frame-level prosody.
  - Concatenation vs bidirectional cross-attention, plus a parameter-matched self-attention control.
- **Key numbers.**
  - E2 (pooled HuBERT + MLP) 0.762 ± 0.054 / 0.713 ± 0.030 macro-F1.
  - E4 (cross-attention) 0.723 ± 0.061 / 0.674 ± 0.023.
  - No fusion variant consistently beats HuBERT alone.
  - Cross-attention is below concatenation in all 5 CREMA-D folds.
- **Take-away.** On these corpora, representation quality dominates fusion architecture. Sophistication
  is not accuracy.
- Mention the limitations in one clause: two acted English corpora, few folds.

### I. Introduction (≈ 1.0 page together with the abstract)
1. **SER motivation.** Applications are named only generically; cite surveys (to be verified).
2. **Traditional acoustic features.** MFCC and prosody: interpretable, cheap, speaker-dependent.
3. **Pre-trained speech representations.** HuBERT and wav2vec 2.0 give strong frozen features; layer
   choice matters (cite SUPERB).
4. **Motivation for fusion.** Prior work, including the reference paper HuMP-CAT (Zhao et al., arXiv
   2501.10408), fuses HuBERT with MFCC and prosody via cross-attention.
   - **State exactly what they report without numbers unless verified from the paper itself.**
   - Their setting (datasets, cross-lingual evaluation) differs from ours.
5. **Gap.** Fusion gains are rarely tested against:
   - a strong pooled-HuBERT baseline;
   - a HuBERT-only frame model;
   - a capacity-matched attention control;
   - equal hyper-parameter search;
   - strict speaker-independent folds.
6. **Objectives and contributions:** C1–C6, condensed into 3–4 bullets.
7. **Paper organisation** (one sentence).

### II. Related Work (≈ 0.8 page)
- **A. Traditional SER:** MFCC / functional statistics with SVM or RF; eGeMAPS-style features (verify the
  citation if used).
- **B. Deep learning SER:** CNN / RNN on spectrograms. Keep it brief, with 2–3 citations.
- **C. Self-supervised speech representations:**
  - wav2vec 2.0, HuBERT;
  - layer-wise probing and weighted layer sums (SUPERB);
  - frozen vs fine-tuned use for SER (verify the citations).
- **D. Feature fusion and attention:**
  - early, late and frame-level fusion;
  - Transformer attention (Vaswani et al.);
  - cross-modal and cross-stream attention.
- **E. HuMP-CAT (reference paper):**
  - It combines HuBERT, MFCC and prosody with cross-attention.
  - **Our E4 adopts the idea, not the full architecture or evaluation.** There is no cross-lingual
    evaluation, no reproduction claim, and their numbers are never mixed with ours (report header,
    line 3).
  - Verify the architecture description from the paper PDF before writing.

### III. Datasets and Experimental Protocol (≈ 1.0 page)
- **A. Corpora** (Table I).
  - **RAVDESS.**
    - 24 actors; speech subset; 6 of 8 emotions (calm and surprised excluded).
    - 1,056 recordings: neutral 96, the other five classes 192 each (`data/metadata/index.csv`).
    - 2 lexically fixed sentences.
  - **CREMA-D.**
    - 91 actors; 6 emotions.
    - 7,442 indexed recordings (neutral 1,087, others 1,271 each); 1 file has status "silent".
    - **Verify whether 7,441 or 7,442 entered the splits** (checklist §8).
  - Both corpora are acted English speech. Say this explicitly.
- **B. Classes and imbalance.**
  - Fixed label order (neutral … disgust).
  - RAVDESS neutral is half-size.
  - Inverse-frequency class weights are computed from the training fold (report §1).
- **C. Speaker-independent splits** (report §0, §1).
  - RAVDESS: 6 folds × 704 / 176 / 176 recordings (16 / 4 / 4 speakers).
  - CREMA-D: 5 folds, about 4,740 / 1,190 / 1,490 (58 / 15 / 18 speakers).
  - The split files are frozen (SHA-256 checked) and never regenerated.
  - Train speakers fit everything learnable. Validation speakers are used only for selection and early
    stopping. Test speakers are predicted once.
  - Figure option: `plots/splits_RAVDESS.png`, `plots/splits_CREMA-D.png` (appendix or omit).
- **D. Pre-processing and features** (Table II; `reports/feature_extraction.md`).
  - Audio: mono → 16 kHz (soxr_hq) → leading/trailing silence trimmed (30 dB below peak, 50 ms margin)
    → DC removal → peak normalisation.
  - **Shared 20 ms frame grid** (25 ms windows, 320-sample hop), aligned across all streams.
  - **HuBERT-base** (`facebook/hubert-base-ls960`, frozen; 13 hidden states × 768; input z-normalised).
    - Cached fp16 frames.
    - Pooled cache: [mean ‖ std] per layer in fp32.
  - **MFCC:** 13 coefficients + Δ + ΔΔ = 39 per frame.
  - **Frame-level prosody (4):** F0 in semitones, voiced flag, periodicity, energy (dB). Praat
    autocorrelation, 60–800 Hz, octave-jump cost 0.7.
    - The tracker was switched from pYIN because pYIN failed to detect voicing in 10.3% of CREMA-D
      recordings, in a class-dependent way (feature_extraction.md §5.4).
    - **This is a methodological point worth one sentence.**
  - **DisVoice-103:** utterance-level prosody statistics; ablation only.
  - Preprocessing fitted on training speakers only (report §2): frame standardiser; DisVoice
    median-impute, signed-log, z-score, clip.

### IV. Methods (≈ 1.6 pages; Table III, Fig. 1)
- **A. E1 — MFCC statistics + SVM / RF** (report §3).
  - Aggregations: mean (39), mean+std (78), functionals (195). The aggregation is selected on
    validation.
  - SVM: RBF kernel, C ∈ {0.1, 1, 10, 100}, γ = m/d with m ∈ {0.1, 1, 10}; 36 candidates including
    aggregation.
  - RF: 300 trees; 12 candidates; seeds 0–2.
  - Class-balanced weights.
- **B. E2 — frozen HuBERT, pooled** (report §4.3–4.4).
  - Representation searched over 13 single layers, concat(1, 9) and learned weighted(all), × mean /
    mean+std pooling.
  - Head: z = Dropout(GELU(W₁x)), logits = W₂z, with h ∈ {256, 512}.
  - Staged validation search: phase A (30) → phase B (8) → phase C, loss weighting (1).
  - Parameters 198,406–1,576,454 per fold; HuBERT itself is never updated (94.4 M frozen).
- **C. E3 — frame-level concatenation** (report §5.2–5.3).
  - Per-stream projection Linear → LN → Dropout to 128.
  - Concatenation, then Linear + GELU.
  - Masked mean+std pooling, then MLP head (256).
  - One HuBERT layer per fold, chosen from that fold's E2 phase-A validation:
    - RAVDESS layers 6, 10, 5, 5, 5, 5;
    - CREMA-D layers 7, 6, 12, 11, 5.
  - 221,574 parameters.
- **D. E4 — bidirectional cross-attention** (report §6.2).
  - Stage 1: MFCC ↔ prosody, then fuse.
  - Stage 2: HuBERT ↔ fused acoustic stream, then fuse + LN.
  - Pooling and head as in E3.
  - Pre-LN blocks: 4 heads, d_k 32, FFN 256, key masking.
  - 769,286 parameters.
  - **Positional-encoding limitation (must be stated).**
    - The attention weights are content-based. With no positional encoding, a query frame cannot prefer
      the time-aligned or neighbouring frames of the other stream.
    - The shared frame grid is used only by the frame-wise fuse layers.
    - The model is invariant to a joint permutation of the time frames (unit-tested).
    - Results therefore speak to content-based cross-attention only.
- **E. Controls and comparators.**
  - Parameter-matched self-attention control: 4 self-attention blocks after concatenation; 751,750
    parameters (−2.3% vs E4).
  - HuBERT-only frame model: 182,534 parameters.
  - Concatenation re-searched on E4's grid.
  - E4 ablations, using E4's hyper-parameters:
    - H↔M (470,022 parameters);
    - H↔P (465,542);
    - H↔concat(M, P) (503,814).
- **F. Architecture comparison table** (Table III). Columns:
  - input;
  - temporal treatment (pooled before vs after the model);
  - fusion;
  - parameters;
  - what the model can and cannot represent.

### V. Experimental Setup (≈ 0.7 page; Table IV)
- **A. Training.**
  - AdamW (weight decay 1e-4), batch 32, early stopping on validation macro-F1 (patience 10).
  - Maximum epochs: 50 (E2, E3), 150 (E4 grid).
  - Gradient clipping at 1.0 for the sequence models only. It was active on 95–100% of CREMA-D steps;
    E2 has no clipping. State this as an optimiser difference.
  - Deterministic training: seeds set before model construction.
- **B. Hyper-parameter selection.**
  - Validation only; `selection.json` written before any test prediction.
  - Grids:
    - E3: p {0.2, 0.4} × lr {1e-3, 3e-4};
    - E4 and its comparators: lr {1e-3, 3e-4, 1e-4, 3e-5}.
  - The grid extension followed the E3 control selecting the lower edge in 11/11 folds; it is
    documented as a protocol revision decided on validation evidence.
  - Final seeds 0, 1, 2; checkpoint → reload → predict test once.
- **C. Metrics.**
  - Macro-F1 is primary. Also UA (= macro recall), accuracy, macro precision, per-class P/R/F1 and
    confusion matrices.
- **D. Statistical reporting.**
  - Seeds averaged within a fold; mean ± sample std over folds.
  - Paired per-fold Δ with fold win counts.
  - Exact two-sided sign-flip p. The minimum attainable p is 0.031 (6 folds) and 0.062 (5 folds).
  - Holm correction over the three key E4 comparisons cannot reach significance.
  - Pre-registered E4 decision rule (report §6.6):
    - mean Δ > 0 on both datasets;
    - ≥ 5/6 and ≥ 4/5 folds better;
    - Δ larger than the seed std.
  - Pooled 11-fold sign-flip values (report §7.2) are **exploratory, post hoc**.
  - Fold non-independence caveat: overlapping training sets (cite Bengio & Grandvalet 2004; verify).
- **E. Leakage and integrity controls.**
  - Label permutation (Table V footnote).
  - Audits.
  - NaN-poisoning isolation tests.
  - Exact checkpoint re-prediction.

### VI. Results (≈ 2.2 pages; Table V, Table VI, Figs. 2–4)
- **A. Overall performance** (Table V; Fig. 3).
  - Ranking: E2 best; then the frame models within about 0.04; E1 far below.
  - E2 is best in 4/6 RAVDESS and 5/5 CREMA-D folds among the 7 main models.
  - Mean rank: E2 1.33 (RAVDESS), 1.0 (CREMA-D).
- **B. E1 vs E2 (RQ1).**
  - E2 − E1 SVM = +0.225 (6/6, p = 0.031) and +0.210 (5/5).
  - Per class +0.10 to +0.36, better in every fold for every class except RAVDESS angry (5/6).
  - Source: report §4.8, `final_analysis.json`.
- **C. Pooled vs frame-level HuBERT.**
  - HuBERT-only frame model (E4 grid) − E2 = −0.025 on both datasets (0/6, 0/5).
  - On RAVDESS, E2 used the **same single layer** as the frame model in all 6 folds; the frame model
    is still lower in 6/6 folds on test and 5/6 on validation.
  - **Confounds, to be listed explicitly:**
    - representation search: E2 used multi-layer inputs in 4/5 CREMA-D folds;
    - search breadth: 39 vs 8 candidates;
    - gradient clipping;
    - the 128-d per-frame bottleneck with per-frame LayerNorm;
    - loss weighting: E2 chose unweighted CE in 3/5 CREMA-D folds.
  - These are hypotheses, not tested (the G-study was not run).
- **D. Concatenation vs cross-attention (RQ2, RQ3)** (Table VI; Fig. 4).
  - E4 − concat: −0.003 (3/6) / −0.020 (0/5).
  - E4 − control: −0.007 (1/6) / −0.012 (1/5).
  - E4 − HuBERT-only: −0.013 (1/6) / −0.014 (1/5).
  - All three verdicts: "no consistent improvement".
  - Control − concat: +0.004 (4/6) / −0.008 (1/5).
  - The wider search helped only the control on CREMA-D (+0.016, 4/5). It did not help concatenation
    or HuBERT-only.
  - Also report E3 as run, on the E3 grid: E3 − E2 = −0.033 (0/6, p = 0.031) / −0.018 (0/5).
- **E. Dataset-wise comparison.**
  - RAVDESS: differences sit inside the seed noise. E4's seed std is 0.034; RAVDESS has 4 test actors
    per fold.
  - CREMA-D: smaller std, and the effects are consistent in direction.
  - Validation → test gap: 0.023–0.052 (RAVDESS) vs 0.014–0.034 (CREMA-D).
  - Interpretation: speaker variability dominates on RAVDESS.

### VII. Ablation and Error Analysis (≈ 1.8 pages; Tables VII–VIII, Figs. 5–7)
- **A. Stream contributions in E3** (Table VII; Fig. 5).
  - HuBERT given M+P: +0.180 / +0.117, in all folds.
  - M+P given H: −0.018 (0/6, p = 0.031) / +0.010 (4/5).
  - MFCC given H+P: −0.020 / +0.003.
  - Prosody given H+M: −0.014 / +0.005.
  - Alone: M+P 0.549 / 0.577 (above E1 SVM); M 0.482 / 0.546; P 0.349 / 0.411.
  - DisVoice +0.005 / −0.004; missingness indicators −0.013 / −0.008.
- **B. Leave-one-out and add-one.** Same table. Text: MFCC and prosody are complementary without
  HuBERT, redundant with it.
- **C. Shapley values** (v(∅) = 1/6):

  | | HuBERT | MFCC | prosody |
  |---|---|---|---|
  | RAVDESS | 0.364 | 0.131 | 0.068 |
  | CREMA-D | 0.283 | 0.156 | 0.090 |

  - **Caveat:** Shapley values average over coalitions *without* HuBERT. They are not "gain added to
    E3".
- **D. E4 ablations.**
  - Prosody given E4: −0.011 / −0.006 (0/5).
  - MFCC given E4: +0.002 / −0.000.
  - Stage-1 attention: −0.007 / +0.002.
  - Cross-attention vs concatenation with the same inputs: +0.004 / −0.021 (0/5).
  - The ablations reuse E4's hyper-parameters and have fewer parameters, so state that limitation.
- **E. Per-class performance** (Table VIII; Fig. 6).
  - RAVDESS neutral: E4 − E2 = −0.100 (0/6); recall 0.851 → 0.684.
    - HuBERT-only − E2 = −0.096 and E3 − E2 = −0.086, so this is a frame-pipeline effect.
  - CREMA-D: every class lower; happy −0.066, and 0/5 folds against each of E2, HuBERT-only and
    concat.
  - RAVDESS sad / fearful trade-off against HuBERT-only:
    - sad precision +0.042 (6/6) but sad recall −0.050;
    - fearful recall +0.035 (5/6).
- **F. Confusion matrices** (Fig. 7).
  - Shared cluster:

    | | Main confusions |
    |---|---|
    | RAVDESS E2 | fearful → sad 15.8%, neutral → sad 11.8% |
    | RAVDESS E4 | neutral → sad 17.0%, sad → fearful 14.1% |
    | CREMA-D E2 | fearful → sad 17.5%, disgust → sad 12.0% |
    | CREMA-D E4 | fearful → sad 19.0%, sad → fearful 13.1% |

  - E1 shows the same structure, with neutral ↔ sad about 21%.
  - Interpretation: the structure belongs to the task and data, and fusion does not change it.
  - **Only as a hypothesis:** shared low-arousal / negative-valence cues.
- **G. Error overlap** (Fig. 8, optional).

  | | Only E2 correct | Only E4 correct | Both wrong | …same wrong label | Prediction agreement |
  |---|---|---|---|---|---|
  | RAVDESS | 9.9% | 6.8% | 16.7% | 12.7% | 79% |
  | CREMA-D | 11.6% | 7.8% | 21.0% | 15.6% | 75% |

  - Do **not** evaluate or claim ensembles: that would be test-set model selection.
- **H. Supplementary:** the feature-swap diagnostic (`plots/E4_feature_swap.png`).
  - The trained models are sensitive to MFCC / prosody: 10–27% of predictions change.
  - Label it explicitly as **supplementary, not evidence of contribution**. Appendix or one sentence.

### VIII. Computational Complexity and Efficiency (≈ 0.5 page; Table IX; Fig. 9)
- **Scope.** Classifier stage only. **HuBERT feature-extraction cost was not logged.** It is shared by
  E2–E4 and must not be mixed in.
- **Median training seconds per final run** (RAVDESS / CREMA-D):

  | Model | Seconds |
  |---|---|
  | E1 SVM | 0.04 / 1.7 |
  | E2 | 0.70 / 6.4 |
  | E3 | 4.3 / 26.2 |
  | HuBERT-only | 2.8 / 22.9 |
  | Concat | 5.3 / 27.0 |
  | Control | 18.5 / 129.5 |
  | E4 | 15.1 / 91.0 |

- **Inference ms per recording.** E1 0.07 / 0.45 (CPU); E2 0.04 / 0.07; non-attention frame models
  0.05–0.07; attention models 0.25–0.27.
- **GPU peak** (MiB):

  | Model | RAVDESS | CREMA-D |
  |---|---|---|
  | E2 | 85 | 759 |
  | Non-attention frame models | 316 | 1,614 |
  | E4 | 715 | 2,098 |

  CREMA-D frame-model peaks include the 1.35 GB HuBERT layer held on the GPU.
- **Total logged training time** (all search and final runs): E2 0.36 h vs E4 6.7 h.
- **Message:** E2 dominates the accuracy–cost Pareto front among the HuBERT models.

### IX. Discussion (≈ 0.9 page)
- **Strongest vs most sophisticated.** The most accurate model is the simplest HuBERT model. The most
  complex is among the least accurate and least stable. Distinguish the two explicitly.
- **Why fusion did not help.**
  - *Measured:* redundancy of MFCC and prosody given HuBERT.
  - *Measured:* no gain from the self-attention control.
  - *Measured:* cross-attention below concatenation on CREMA-D.
  - *Interpretation:* the HuBERT representation already encodes the spectral and prosodic cues relevant
    to these six acted emotions.
  - *Hypothesis:* extra input dimensions add variance on small data (RAVDESS 704 training recordings
    per fold).
- **Why E2 beats frame models.** Not explained. List the candidate factors (§VI-C) as hypotheses and
  name the controlled ablation needed (future work). Do not present the cancelled G-study as evidence.
  - The pooled-vs-frame comparison is between two **order-free** summaries, not "temporal vs pooled".
- **Relation to HuMP-CAT.**
  - Our setting differs: datasets, no cross-lingual test, frozen features, our own architecture details.
  - Our results neither confirm nor refute their reported gains. They show that, under a strict
    protocol with strong baselines, fusion gains did not appear on these corpora.
- **Methodological lessons.**
  - Strong single-representation baselines.
  - Capacity-matched controls.
  - Matched search budgets.
  - Validation-only selection.
  - Per-dataset reporting.

### X. Limitations and Future Work (≈ 0.5 page)
- **Limitations.**
  - Two acted English corpora; RAVDESS has only 2 sentences.
  - Six classes.
  - 6 + 5 folds: low power, and folds are not independent.
  - One frozen HuBERT-base model; one layer per fold for the frame models.
  - Search spaces differ between E2 and the frame models.
  - Learning-rate edge selections:
    - the E3 control at its lower edge in 11/11 folds;
    - on the E4 grid, the control at the lower edge in 3/11 folds and HuBERT-only at the upper edge in
      5/6 RAVDESS folds.
  - No positional encoding in the attention models.
  - Ablations not tuned separately.
  - Feature-extraction cost not measured.
  - No cross-corpus or cross-lingual test.
- **Future work** (proposals only; report §7.9, §8):
  - Controlled one-factor study of the pooled vs frame gap: multi-layer input, LayerNorm and bottleneck,
    clipping, matched budget.
  - Utterance-level fusion on top of E2.
  - Position-aware or windowed cross-attention.
  - Cross-corpus evaluation.
  - Repeated cross-validation for statistical power.
  - Class-focused work on neutral and on sad / fearful / disgust.

### XI. Conclusion (≈ 0.25 page)
- Answer the RQ directly:
  - no meaningful improvement from fusion or cross-attention over pooled HuBERT on these two acted
    English corpora under speaker-independent evaluation;
  - HuBERT dominates;
  - simplicity wins.
- Restate the scope.

### References (≈ 0.75 page, about 20–25 entries)
Candidates. **Every entry must be verified** for authors, title, venue, year and DOI before use.
- Zhao et al., HuMP-CAT, arXiv:2501.10408.
- Hsu et al., HuBERT.
- Baevski et al., wav2vec 2.0.
- Yang et al., SUPERB.
- Livingstone & Russo, RAVDESS.
- Cao et al., CREMA-D.
- Vaswani et al., Transformer.
- Davis & Mermelstein, MFCC.
- Cortes & Vapnik, SVM; Breiman, Random Forests.
- Boersma & Weenink, Praat; Jadoul et al., Parselmouth.
- DisVoice (Vásquez-Correa et al.; verify the exact citation).
- Shapley (1953).
- Bengio & Grandvalet (2004); Nadeau & Bengio (2003).
- Loshchilov & Hutter, AdamW; Ba et al., LayerNorm; Hendrycks & Gimpel, GELU.
- One or two SER surveys and SSL-for-SER papers (e.g. Pepino et al.; Wagner et al.).
  **Select and verify; do not cite from memory.**

---

## 5. Figures and tables (all from original project outputs; no reference-paper images; no new experimental plots)

**Figures** (main text 7–8; the rest go to the appendix or are omitted)

| # | Caption (draft) | Section | Source | Supports |
|---|---|---|---|---|
| Fig. 1 | Overview of E1–E4 and the controls: inputs, temporal treatment, fusion | IV | **Not available: a schematic must be drawn** (architecture diagram, not an experimental plot) | Methods |
| Fig. 2 | Layer-wise validation macro-F1 of pooled HuBERT (E2 phase A) | IV-B / VI-C | `plots/E2_layerwise_validation.png` | Layer choice; multi-layer evidence (CREMA-D) |
| Fig. 3 | Test macro-F1 per fold for E1, E2, E3 and the E4 comparators and ablations (dots = folds; ◆ = mean ± std) | VI-A | `plots/E4_models_test.png` | RQ1–RQ3, overall |
| Fig. 4 | Paired per-fold Δ test macro-F1: key comparisons, ablations, and effect of the wider search | VI-D | `plots/E4_comparisons_test.png` | RQ3 |
| Fig. 5 | Stream contributions in E3: leave-one-out, add-one, control vs concat, DisVoice; Shapley values | VII-A–C | `plots/E3_stream_contributions.png` | RQ2 |
| Fig. 6 | Per-class precision, recall and F1, E2 vs E4, with per-fold ΔF1 | VII-E | `plots/FINAL_per_class_E2_vs_E4.png` | Per-class |
| Fig. 7 | Test confusion matrices (row %), E2 vs E4, and their difference | VII-F | `plots/FINAL_confusion_E2_E4.png` | Error structure |
| Fig. 8 | Error overlap on identical test recordings | VII-G | `plots/FINAL_error_overlap.png` | Complementarity (optional) |
| Fig. 9 | Test macro-F1 vs median training time per run (classifier stage only) | VIII | `plots/FINAL_accuracy_vs_cost.png` | Cost (could be merged into Table IX if space is short) |
| App. | Validation search vs learning rate; feature-swap diagnostic (supplementary); E1 aggregation; E1 confusions; splits; class distribution | Appendix | `plots/E4_search_validation.png`, `plots/E4_feature_swap.png`, `plots/E1_aggregation_comparison.png`, `plots/E1_confusion.png`, `plots/splits_*.png`, `plots/data_class_distribution.png` | Supporting |

Alternatives available: `plots/E1_E2_E3_folds_test.png`, `plots/E3_variants_test.png`,
`plots/E2_vs_E3_per_class_f1.png`, `plots/E3_confusion_test.png`, `plots/E4_confusion_test.png`,
`plots/E4_per_class_f1.png`, `plots/E2_pooling_validation.png`, `plots/E2_layer_weights.png`.

**Notes on reuse.**
- The existing PNGs have long figure-level titles. For IEEE, the in-figure titles should be shortened and
  the text moved to captions. This means a **re-render from the same saved data** with the same scripts,
  not new experiments.
- `FINAL_accuracy_vs_cost.png` has its legend above the title. Fix the layout when re-rendering.

**Tables**

| # | Content | Section | Source |
|---|---|---|---|
| Table I | Dataset statistics: corpus, speakers, recordings per class, folds, train / val / test recordings and speakers per fold, duration (median frames 91 / 118 → about 1.8 / 2.4 s after trimming) | III-A | `data/metadata/index.csv`; report §0; `data/splits/*_speaker_folds.json` |
| Table II | Feature streams: HuBERT (13 × 768, frozen), MFCC (39), prosody (4), DisVoice-103; frame rate; extraction settings | III-D | `reports/feature_extraction.md` §1–§5 |
| Table III | Architecture comparison: E1, E2, E3, self-attention control, HuBERT-only, E4, E4 ablations; parameters (E2: 198,406–1,576,454; E3 221,574; control 751,750; HuBERT-only 182,534; E4 769,286; H↔M 470,022; H↔P 465,542; H↔concat 503,814) | IV | report §4.3, §5.3, §6.2; `results/E4/<DS>/summary.json` (n_params) |
| Table IV | Training and evaluation configuration: optimiser, batch, epochs, patience, clipping, grids, candidates per fold (E1 36 / 12, E2 39, E3 4, E4 8), seeds, selection metric | V | report §3.3–3.4, §4.4, §5.4, §6.4; `configs/e1_mfcc_classical.json`, `configs/e2_hubert_head.json`, `configs/e3_concat_fusion.json`, `configs/e4_cross_attention.json` |
| Table V | Overall test results (mean ± std over folds): macro-F1, UA, accuracy, macro precision for E1 SVM, E1 RF, E2, E3, E3 control, HuBERT-only, concat (E4 grid), control (E4 grid), E4; permutation controls in a footnote (E2 0.187 / 0.186; E3 0.158 / 0.174; E4 0.140 / 0.173) | VI-A | `results/E*/<DS>/summary.json`, `results/E1/<DS>/runs.csv`, `results/final_analysis/final_analysis.json`; report §3.5, §4.6, §5.8, §6.12, §7.2 |
| Table VI | Paired comparisons and the pre-registered decision rule: Δ, folds better, sign-flip p, Holm p per dataset | VI-D | `results/E4/decision.json`; `results/E4/<DS>/summary.json` (paired_test_macro_f1, paired_test_UA); report §6.12 |
| Table VII | Feature ablations: E3 stream subsets (F1 / UA), leave-one-out, add-one, Shapley, DisVoice; E4 ablations | VII-A–D | `results/E3/<DS>/summary.json` (variants, paired, shapley_test_macro_f1); `results/E4/<DS>/summary.json`; report §5.9, §6.12 |
| Table VIII | Per-class F1 (P / R in a supplement): E1, E2, E3, E4, concat, control, HuBERT-only; ΔF1 E4 − E2 with folds better | VII-E | `results/final_analysis/final_analysis.json` (per_class); report §7.3 |
| Table IX | Computational cost: parameters, median training seconds per run, total logged training time, inference ms per recording, GPU peak, process RAM peak (classifier stage only) | VIII | `final_analysis.json` (efficiency); `results/E1/<DS>/runs.csv`; report §7.5 |

The main values for every table are already in report §7.2–7.6, with verified sources.

---

## 6. Page allocation (IEEE two-column, references included)

| Section | Pages | Contents |
|---|---|---|
| Title, abstract, I. Introduction | 1.0 | — |
| II. Related Work | 0.8 | — |
| III. Datasets and Protocol | 1.0 | Table I, Table II |
| IV. Methods | 1.6 | Fig. 1, Table III |
| V. Experimental Setup | 0.7 | Table IV |
| VI. Results | 2.2 | Table V, Table VI, Figs. 2–4 |
| VII. Ablation and Error Analysis | 1.8 | Table VII, Table VIII, Figs. 5–7 (Fig. 8 optional) |
| VIII. Complexity | 0.5 | Table IX (Fig. 9 optional) |
| IX. Discussion | 0.9 | — |
| X. Limitations and Future Work | 0.5 | — |
| XI. Conclusion | 0.25 | — |
| References | 0.75 | — |
| **Total** | **≈ 12.0** | — |

**Getting to about 11 pages:**
- drop Fig. 8 and Fig. 9 (keep their numbers in the text or tables);
- merge Table II into Table I;
- move Table VII's E4 ablation rows into Table VI.

---

## 7. Narrative guard-rails (from the task requirements)

**Our results vs Zhao et al.**
- Never place their numbers in our tables.
- Describe HuMP-CAT only from the paper itself.
- E4 is "inspired by", not "a reproduction of", HuMP-CAT.

**Wording.**
- "Consistent across folds" is acceptable. "Significant" is not, except for the uncorrected RAVDESS
  0/6 or 6/6 comparisons (p = 0.031), which must carry the correction caveat.
- Label any pooled 11-fold p-value **exploratory**.

**E2 vs frame models.** Always attach the confound list:
- representation search;
- search breadth;
- clipping;
- loss weighting and head width;
- the bottleneck and LayerNorm.

**G-study.** Mention it only as future work or a planned controlled study. **Never as results.** The B0
storage-precision discrepancy concerns the unused G pipeline only and is not part of the paper's
evidence.

**Timing.** Classifier-stage times only, with a footnote that HuBERT extraction is excluded and was not
logged.

**Pre-registration and documented corrections.**
- State that the E4 decision rule and the grid extension were fixed before E4's test predictions.
- State the documented correction of the E4 alignment description (report §6.2).

---

## 8. Checklist to verify before drafting

**Numbers** (re-read from source files when writing each table):
- [ ] Table V values: `results/E*/<DS>/summary.json`, plus E1 from `runs.csv` (SVM headline rows).
  `final_analysis.json` → verification has fold means for 9 configurations.
- [ ] Paired Δ, fold counts, p and Holm p: `results/E4/decision.json`, `results/E4/<DS>/summary.json`,
  `results/E3/<DS>/summary.json`.
- [ ] Shapley values and ablations: `results/E3/<DS>/summary.json`.
- [ ] Per-class P / R / F1, confusion percentages, error overlap, efficiency:
  `results/final_analysis/final_analysis.json`.
- [ ] Permutation controls: `results/E2|E3|E4/<DS>/control_label_permutation.json`.
- [ ] Audit counts: `results/E2/audit.json` (28), `results/E3/audit.json` (36), `results/E4/audit.json` (41).
- [ ] Parameter counts (Table III): `n_params` in the summaries; the E2 per-fold range from
  `results/E2/<DS>/summary.json` → folds → n_params.
- [ ] HuBERT layers per fold: `results/E3/<DS>/summary.json` → layer_per_fold. E2 selections:
  `results/E2/<DS>/summary.json` → selection_counts and folds.
- [ ] Seed std and validation → test gaps: `final_analysis.json` → statistics. **Define the gap
  consistently:** report §5.12 uses the *selected* validation F1, §7.6 uses the final-run validation F1.
- [ ] Dataset counts: `data/metadata/index.csv` (RAVDESS 1,056 included / 1,440; CREMA-D 7,442
  indexed).
  - Fold sizes and speakers: `data/splits/RAVDESS_speaker_folds.json`, `CREMA-D_speaker_folds.json`.
- [ ] Durations after trimming: Stage 2 / 4 validation outputs (`results/stage4_validation_*.json`).

**Methods details:**
- [ ] HuBERT checkpoint name and pre-processing: `reports/feature_extraction.md` §2–3.
- [ ] MFCC and prosody settings: `reports/feature_extraction.md` §4–5.
- [ ] Grids and configs: `configs/*.json`.
- [ ] E4 equations, blocks and masking: `src/models/e4_cross_attention.py`, `src/models/attention.py`.
- [ ] E3 / control: `src/models/e3_fusion.py`.
- [ ] E2: `src/models/e2_hubert.py`.
- [ ] Training loops: `src/training/trainer.py` (E2), `src/training/seq_trainer.py` (E3/E4).
  - The `grad_clip` field added later to `trainer.py` defaults to off and was not used by E2.

**Citations:**
- [ ] Every reference in the list above must be verified against the original publication.
- [ ] The HuMP-CAT architecture, datasets and reported results must be taken from the paper PDF.
  **The PDF is not in the project folder.**

---

## 9. Missing material and unresolved inconsistencies
1. **CREMA-D recording count: 7,441 or 7,442.**
   - `index.csv` lists 7,442 included rows, with one status "silent".
   - `feature_extraction.md` §5.4 refers to "7,441 recordings".
   - Confirm from the split files which number entered the folds, and state it once.
2. **The HuMP-CAT paper is not in the project.** Its architecture details, datasets and numbers must be
   read from the original before writing Related Work and Discussion.
3. **No architecture schematic (Fig. 1) exists.** It must be drawn. It is not an experimental plot.
4. **HuBERT feature-extraction time was not logged.** Only classifier-stage costs can be reported.
5. **Two definitions of the validation → test gap** (report §5.12 vs §7.6). Choose one for the paper.
6. **E3 vs E4 grids.** E3's published numbers use the 4-candidate grid. The E4-grid re-search of concat
   and the control gives slightly different numbers (e.g. control 0.669 vs 0.686 on CREMA-D). Report both
   explicitly, labelled by grid.
7. **Learning-rate edge selections remain for some models** (report §6.14). This is a limitation, not a
   fixable inconsistency.
8. **RAVDESS E4 permutation control** (0.140) is below chance. It is explained as the expected behaviour
   of a collapsed classifier, not leakage; mention it in a footnote.
9. **Figure styling.** The existing figures carry long in-figure titles, and one has a legend-above-title
   layout. They must be re-rendered from the saved data for IEEE formatting.
10. **The G-study (report §8) was cancelled.** Its gate results and the B0 discrepancy are not paper
    evidence.
