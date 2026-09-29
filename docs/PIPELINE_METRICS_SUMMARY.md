# VIscaNEr: consolidated pipeline metrics

> Historical comparison report. The deployment decision below was superseded
> on 2026-09-29: the selected current path is the Stage-2C residual five-stream
> Transformer in `five_stream_transformer/infer.py`. On the strict 130-image
> `store_shelves_web` holdout it reached R@1 60.77% versus 48.46% for the
> Stage-2C bottle branch. See `MODEL_INFERENCE_HANDOFF.md` for current files.

**Evidence cutoff:** 2026-09-27
**Purpose:** one place to compare the evaluated detector, retrieval, cascade, fusion, OCR and Transformer variants.

> In the closed-set retrieval tables, `Accuracy` and `Recall@1` are the same metric. Results from different splits or pipeline generations must not be compared as if they came from one controlled experiment.

## Executive summary

- **Best result on the identity-disjoint synthetic `test_unseen` split:** standalone **bottle DINOv3-B**, `Recall@1 = 82.03%`, `Recall@2 = 95.70%` (`N=5,694`). Stage-2C Fusion MLP reached `72.23%`; the candidate Transformer reached `70.64%`.
- **Best result on the held-out manual split:** **Stage-2C Fusion MLP**, `Recall@1 = 62.96%`, `Recall@2 = 81.48%` (`N=27`). The Transformer reached `59.26%` and harmed one previously correct Top-1 prediction without fixing another.
- **Manual-211 full catalog subset:** Transformer was ahead by only one image: `64/99 = 64.65%` versus MLP `63/99 = 63.64%`. This advantage came from the 72 manual images previously exposed during Stage 2C; it did not transfer to the 27 held-out images.
- **Old conditional cascade:** a learned DINO-S bottle resolver improved label DINO-B on its `val_unseen` split from `67.81%` to `73.22%`, but it remained below the later standalone bottle DINO-B result and belonged to an older model/data generation.
- **OCR helped the label pipeline on the separate Real-100 evaluation:** on its held-out report half, label DINO-B improved from `55.17%` to `58.62%` Top-1. This is promising but based on only 29 known-catalog report images.
- **Current decision:** retain **Fusion MLP** instead of the Transformer as the combined ranker. Keep standalone bottle DINO-B as an important benchmark because it remains much stronger on the large identity-disjoint split.

## 1. Evaluation datasets and what they mean

| Split | Rows | Role | Important limitation |
|---|---:|---|---|
| `hard_train` | 10,800 | Stage-II training on difficult pairs | Selected from the old train split |
| `hard_val` | 1,200 | Stage-II model selection / early stopping | Difficult rows from old `val_seen` |
| `test_seen_prior_exposure` | 21,429 | Descriptive seen test | Identities/images originate from prior training exposure; optimistic |
| `test_seen_model_selection` | 2,404 | Former model-selection rows | Seen identities; not a clean unseen estimate |
| `test_unseen` | 5,694 | Identity-disjoint test | Best large closed-set generalization estimate currently available |
| Manual-211, all photos | 211 | Real manually photographed images | 109 `notcat`, 2 `unsure`, 1 unresolved cannot receive closed-set accuracy |
| `manual_all_catalog` | 99 | All Manual-211 rows with catalog identity | Includes 72 Stage-2C training rows |
| `manual_train_prior_exposure` | 72 | Stage-2C adaptation rows | Training-exposed; descriptive only |
| `manual_val` | 27 | Held-out Stage-2C/manual comparison | Honest manual holdout, but statistically small |
| Real-100 known catalog | 58 | Earlier real-photo retrieval evaluation | Separate dataset and older checkpoint generation |
| Real-100 `notcat` | 40 | Open-set rejection evaluation | Separate open-set task |

The hard-set leakage checks were clean: no source overlap between hard train and tests, no identity overlap with `test_unseen`, and all hard validation rows came from the original non-training `val_seen` partition.

## 2. Stage-I hard fine-tuning of the two DINOv3-B branches

Both branches use DINOv3-B/16. “Baseline” is the checkpoint before Stage-I hard fine-tuning; “Final” is the resulting Stage-I checkpoint.

### Bottle branch

| Split | Version | N | R@1 / Acc | R@2 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| `val_seen` | Baseline | 3,329 | 86.33% | 95.58% | 99.58% | 99.85% | 0.9218 |
| `val_seen` | Final | 3,329 | **87.62%** | **96.03%** | 99.58% | 99.88% | **0.9293** |
| `val_unseen` | Baseline | 1,921 | **84.33%** | 95.99% | 99.53% | 99.74% | 0.9131 |
| `val_unseen` | Final | 1,921 | 83.97% | **96.88%** | **99.69%** | **99.79%** | **0.9134** |
| `val_hard` | Final | 6,832 | 76.96% | 89.87% | 97.83% | 99.08% | 0.8594 |

Interpretation: hard fine-tuning improved seen Top-1 by `+1.29 pp` and unseen Top-2 by `+0.89 pp`, but unseen Top-1 decreased by `0.36 pp`.

### Label branch

| Split | Version | N | R@1 / Acc | R@2 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| `val_seen` | Baseline | 3,604 | 82.69% | 94.45% | **99.39%** | **99.72%** | 0.9010 |
| `val_seen` | Final | 3,604 | **85.85%** | **95.17%** | 99.20% | 99.69% | **0.9181** |
| `val_unseen` | Baseline | 5,694 | **69.49%** | 91.18% | 98.93% | **99.51%** | **0.8276** |
| `val_unseen` | Final | 5,694 | 68.39% | **92.31%** | **99.03%** | 99.44% | 0.8252 |

Interpretation: label hard fine-tuning improved seen Top-1 by `+3.16 pp` and unseen Top-2 by `+1.12 pp`, but unseen Top-1 decreased by `1.11 pp`.

## 3. Historical conditional label-to-bottle cascade

Architecture: label DINOv3-B produces candidates; a bottle DINOv3-S resolver is invoked selectively. This is an older generation than the later two-DINO-B fusion system.

### Final locally saved learned-gate run

| Scope | N | Label DINO-B R@1 | Cascade R@1 | Delta | R@2 ceiling |
|---|---:|---:|---:|---:|---:|
| All eligible | 41,527 | 80.09% | **82.91%** | +2.82 pp | 94.46% |
| `val_seen` | 3,604 | 81.49% | **83.99%** | +2.50 pp | 94.40% |
| `val_unseen` | 5,694 | 67.81% | **73.22%** | +5.41 pp | 90.81% |
| Resolver-invoked `val_unseen` | 1,643 | 41.45% | **60.19%** | +18.75 pp | 84.66% |

Learned gate details on `val_unseen`:

- resolver invoked for `1,643 / 5,694 = 28.85%` of queries;
- fixed 458 Top-1 errors and harmed 150 correct predictions;
- net correction: `+308` images;
- learned logistic expected-utility gate, decision threshold `0.04612`, maximum label margin `0.03`.

An earlier cascade snapshot reported `75.15%` on a different `val_unseen` model/data snapshot (`69.53%` base). It should be treated as historical and not merged numerically with the final learned-gate run above.

## 4. Stage 2A: frozen two-branch Fusion MLP

Both Stage-I DINOv3-B branches are frozen. The union of label Top-10 and bottle Top-10 candidates is reranked by an MLP.

| Split | Model | N | R@1 / Acc | R@2 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| `hard_val` | Fusion MLP | 1,200 | **69.33%** | **93.08%** | **99.92%** | **100.00%** | **0.8332** |
|  | Label DINO-B | 1,200 | 62.17% | 86.17% | 97.58% | 99.08% | 0.7786 |
|  | Bottle DINO-B | 1,200 | 52.58% | 80.50% | 96.33% | 98.75% | 0.7159 |
| `test_seen_prior_exposure` | Fusion MLP | 21,429 | 95.55% | **99.60%** | 100.00% | 100.00% | 0.9771 |
|  | Label DINO-B | 21,429 | 92.73% | 97.96% | 99.78% | 99.91% | 0.9593 |
|  | Bottle DINO-B | 21,429 | **96.15%** | 99.36% | 99.98% | 100.00% | **0.9795** |
| `test_seen_model_selection` | Fusion MLP | 2,404 | **99.33%** | **100.00%** | 100.00% | 100.00% | **0.9967** |
|  | Label DINO-B | 2,404 | 97.63% | 99.71% | 100.00% | 100.00% | 0.9876 |
|  | Bottle DINO-B | 2,404 | 98.42% | 99.79% | 100.00% | 100.00% | 0.9917 |
| `test_unseen` | Fusion MLP | 5,694 | 77.43% | **96.47%** | **99.74%** | **99.98%** | 0.8800 |
|  | Label DINO-B | 5,694 | 68.32% | 92.36% | 99.03% | 99.44% | 0.8249 |
|  | Bottle DINO-B | 5,694 | **80.59%** | 93.31% | 98.70% | 99.53% | **0.8872** |

Stage 2A improved difficult and label-only results, but on identity-disjoint Top-1 the standalone bottle branch remained `3.16 pp` ahead of Fusion MLP.

## 5. Stage 2B and Stage 2C

Stage 2B starts from Stage 2A, unfreezes the final two blocks and retrieval heads in both DINOv3-B models, and trains the joint objective. A complete isolated Stage-2B final JSON was not retained in the evidence used for this report. The Stage-2B checkpoint is therefore represented only by the baseline measured immediately before Stage 2C.

Stage 2C adapts the joint system on 72 manually reviewed catalog images, selects on 27 held-out manual images, and uses `hard_val` as a forgetting guardrail.

### Stage-2C change versus its Stage-2B input checkpoint

| Split | Version | N | R@1 / Acc | R@2 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| `manual_val` | Before Stage 2C | 27 | 51.85% | 77.78% | 88.89% | 100.00% | 0.6931 |
| `manual_val` | **After Stage 2C** | 27 | **62.96%** | **81.48%** | **100.00%** | 100.00% | **0.7809** |
| `hard_val` | Before Stage 2C | 1,200 | 82.75% | **98.00%** | **99.83%** | 99.92% | 0.9092 |
| `hard_val` | **After Stage 2C** | 1,200 | **83.17%** | 97.83% | 99.75% | 99.92% | **0.9111** |

Stage 2C gained `+11.11 pp` Top-1 on held-out manual images while preserving hard-set Top-1 (`+0.42 pp`). The manual result is only `17/27` correct, so uncertainty remains large.

### Stage-2C branch comparison on `manual_val`

| Model | R@1 / Acc | R@2 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| **Fusion MLP** | **62.96%** | **81.48%** | **100.00%** | 100.00% | **0.7809** |
| Label DINO-B | 59.26% | 81.48% | 96.30% | 100.00% | 0.7488 |
| Bottle DINO-B | 51.85% | 77.78% | 92.59% | 100.00% | 0.7034 |

## 6. Candidate Transformer versus Stage-2C Fusion MLP

The Transformer freezes the completed Stage-2C visual branches and replaces only the MLP candidate scorer. It has 181,300 parameters and its best checkpoint was epoch 5.

| Split | Model | N | R@1 / Acc | R@2 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| `hard_val` | Transformer | 1,200 | **83.83%** | 97.58% | 99.67% | 99.75% | **0.9135** |
|  | Fusion MLP | 1,200 | 83.17% | **97.83%** | **99.75%** | **99.92%** | 0.9111 |
|  | Label DINO-B | 1,200 | 74.92% | 92.08% | 97.83% | 98.75% | 0.8536 |
|  | Bottle DINO-B | 1,200 | 64.67% | 88.92% | 98.67% | 99.67% | 0.7991 |
| `test_seen_prior_exposure` | Transformer | 21,429 | **97.91%** | **99.95%** | 100.00% | 100.00% | **0.9895** |
|  | Fusion MLP | 21,429 | 97.88% | 99.94% | 100.00% | 100.00% | 0.9893 |
| `test_seen_model_selection` | Transformer | 2,404 | 99.63% | 100.00% | 100.00% | 100.00% | 0.9981 |
|  | Fusion MLP | 2,404 | **99.71%** | 100.00% | 100.00% | 100.00% | **0.9985** |
| `test_unseen` | Transformer | 5,694 | 70.64% | **95.63%** | 99.84% | 99.96% | 0.8447 |
|  | Fusion MLP | 5,694 | 72.23% | 95.57% | **99.91%** | 99.96% | 0.8527 |
|  | Label DINO-B | 5,694 | 59.22% | 89.53% | 98.37% | 99.07% | 0.7715 |
|  | **Bottle DINO-B** | 5,694 | **82.03%** | **95.70%** | 99.67% | 99.86% | **0.9012** |

Transformer versus MLP Top-1 changes:

| Split | Fixed | Harmed | Net |
|---|---:|---:|---:|
| `hard_val` | 42 | 34 | +8 |
| `test_seen_prior_exposure` | 137 | 131 | +6 |
| `test_seen_model_selection` | 0 | 2 | -2 |
| `test_unseen` | 122 | 213 | **-91** |

Conclusion: the Transformer fit `hard_val` slightly better but generalized worse to unseen identities. It should not replace the MLP.

## 7. Manual-211 final evaluation

Predictions were generated for all 211 images. Closed-set retrieval metrics are valid only for the 99 rows with accepted catalog identities.

### All 99 catalog images

| Model | Correct | R@1 / Acc | R@2 | R@5 | R@10 | MRR | Mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | **64/99** | **64.65%** | 87.88% | 96.97% | 98.99% | **0.7927** | 1.697 |
| Fusion MLP | 63/99 | 63.64% | 87.88% | **97.98%** | **100.00%** | 0.7897 | **1.636** |
| Label DINO-B | 61/99 | 61.62% | 80.81% | 95.96% | 97.98% | 0.7596 | 2.010 |
| Bottle DINO-B | 52/99 | 52.53% | 76.77% | 89.90% | 97.98% | 0.6985 | 2.404 |

Transformer fixed two MLP errors and harmed one MLP-correct prediction: net `+1` image.

### Stage-2C-exposed 72 images versus held-out 27 images

| Split | Model | Correct | R@1 / Acc | R@2 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| Prior exposure (72) | Transformer | **48/72** | **66.67%** | 90.28% | 95.83% | 98.61% | **0.8041** |
|  | Fusion MLP | 46/72 | 63.89% | 90.28% | **97.22%** | **100.00%** | 0.7930 |
| Held-out `manual_val` (27) | Transformer | 16/27 | 59.26% | 81.48% | 100.00% | 100.00% | 0.7623 |
|  | **Fusion MLP** | **17/27** | **62.96%** | 81.48% | 100.00% | 100.00% | **0.7809** |
|  | Label DINO-B | 16/27 | 59.26% | 81.48% | 96.30% | 100.00% | 0.7488 |
|  | Bottle DINO-B | 14/27 | 51.85% | 77.78% | 92.59% | 100.00% | 0.7034 |

The Transformer’s apparent advantage on all 99 images is entirely attributable to prior-exposure rows. On the held-out rows it fixed 0 MLP errors and harmed 1 correct MLP prediction.

### Manual source groups, Top-1

| Source group | N | Transformer | Fusion MLP | Label DINO-B | Bottle DINO-B |
|---|---:|---:|---:|---:|---:|
| `source1_1` | 26 | **65.38%** | **65.38%** | 61.54% | 42.31% |
| `source2_2` | 41 | **65.85%** | 60.98% | 60.98% | 48.78% |
| `source3_raz_met` | 32 | 62.50% | **65.63%** | 62.50% | **65.63%** |

The source-group reversal confirms domain sensitivity: no ranker dominates all three capture groups.

## 8. Earlier Real-100 evaluation

This is a different manually photographed dataset: 58 known-catalog images, 40 `notcat`, and 2 `unsure`. It used older branch checkpoints, so it is useful for direction but not a direct score for the current Stage-2C pipeline.

### Closed-set retrieval on the 58 known wines

| Pipeline | Scope | N | R@1 | R@2 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| Whole-bottle DINO | All | 58 | 56.90% | 70.69% | 75.86% | 77.59% | 0.6590 |
| Whole-bottle DINO | Held-out report | 29 | 44.83% | 58.62% | 65.52% | 65.52% | 0.5449 |
| Label DINO | All | 58 | 58.62% | 72.41% | 84.48% | 87.93% | 0.6998 |
| Label DINO | Held-out report | 29 | 55.17% | 68.97% | 82.76% | 89.66% | 0.6745 |
| Label DINO + OCR | All | 58 | **67.24%** | **75.86%** | 84.48% | — | 0.7428 at 5 |
| Label DINO + OCR | Held-out report | 29 | **58.62%** | **72.41%** | 82.76% | — | 0.6897 at 5 |

OCR reranking changed eight Top-1 predictions over all known rows, fixed five and harmed none. On the held-out half it improved Top-1 by `+3.45 pp` (`16/29` to `17/29`). The sample is too small to establish stable production lift.

### Open-set rejection on the held-out report half

| Pipeline | Overall accuracy | Balanced accuracy | Known resolution accuracy | Unknown rejection | False accept | Answer precision | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Whole-bottle thresholds | 44.90% | 51.12% | 17.24% | 85.00% | 15.00% | 50.00% | 20.41% |
| Label thresholds | **53.06%** | **58.79%** | **27.59%** | **90.00%** | **10.00%** | **72.73%** | **22.45%** |

The thresholded open-set systems are conservative and answer only about one fifth of photos. These thresholds were selected on a separate calibration half and are not universal constants.

## 9. YOLO detector evidence

### Current joint bottle + label YOLO11n

| Metric | Overall | Bottle | Wine label |
|---|---:|---:|---:|
| Precision | 98.78% | 99.55% | 98.01% |
| Recall | 90.99% | 92.40% | 89.58% |
| mAP@50 | 92.81% | 94.30% | 91.33% |
| mAP@50:95 | 86.65% | 87.92% | 85.38% |

Critical limitation: all 211 manually annotated images were present in both train and validation; only 12 old images were validation-only. These detector metrics measure deep adaptation, not unbiased generalization.

On the built Manual-211 crops, using manual GT only to select the matching predicted box:

- mean label confidence: `0.9265`;
- mean bottle confidence: `0.9294`;
- mean label IoU with GT: `0.9315`;
- mean bottle IoU with GT: `0.9196`;
- 211/211 label crops and 211/211 bottle inputs were produced.

Runtime policy: use the paired joint-YOLO detections; when bottle confidence is below `0.75`, send the original image to the bottle branch instead of trusting a poor crop.

### Older dedicated bottle detector

| Precision | Recall | F1 | mAP@50 | mAP@50:95 |
|---:|---:|---:|---:|---:|
| 99.05% | 98.50% | 98.78% | 99.42% | 91.66% |

This detector’s validation protocol differs from the current joint detector, so the rows are not an A/B comparison.

## 10. Overall ranking of the approaches

| Question | Best supported answer | Evidence |
|---|---|---|
| Best large identity-disjoint Top-1 | **Standalone bottle DINOv3-B** | 82.03% on `test_unseen`, N=5,694 |
| Best combined ranker on held-out manual photos | **Stage-2C Fusion MLP** | 62.96% on `manual_val`, N=27 |
| Best hard-set checkpoint-selection score | Candidate Transformer | 83.83% on `hard_val`, but it lost on unseen/manual |
| Best old selective resolver | Learned-gate DINO-B → DINO-S cascade | 73.22% on its older `val_unseen` |
| Best Real-100 label variant | Label DINO + OCR | 58.62% held-out Top-1, N=29 |
| Rejected replacement | Candidate Transformer | -1.60 pp versus MLP on `test_unseen`; -3.70 pp on `manual_val` |

## 11. Decision and remaining uncertainty

1. **Do not replace Fusion MLP with the Transformer.** The Transformer’s small hard-set gain did not transfer to identity-disjoint or held-out manual data.
2. **Keep bottle DINOv3-B as a first-class branch and benchmark.** It is the strongest model on the large unseen-identity split, even though it is weaker on the tiny Manual-211 holdout.
3. **Use Stage-2C Fusion MLP as the current combined experimental pipeline.** It is the best evaluated combined method on held-out manual photos.
4. **Do not use seen splits as production claims.** Scores of 95–99% on prior-exposure/model-selection rows are diagnostic, not realistic generalization estimates.
5. **The next decisive evaluation needs a larger untouched manual set.** With only 27 held-out catalog photos, one image changes accuracy by `3.70 pp`; conclusions between 59% and 63% are unstable.
6. **Open-set behavior remains a separate unresolved problem.** Closed-set retrieval does not score the 109 `notcat`, 2 `unsure`, and 1 unresolved Manual-211 rows.

## Provenance

Local machine-readable sources:

- `runs/dino_cascade/summary.json`
- `runs/dino_cascade/gate_experiments/gate_summary.json`
- `runs/evaluation/real_photos_extra/summary.json`
- `runs/evaluation/real_photos_extra_labels/summary.json`
- `runs/evaluation/real_photos_extra_labels_ocr/summary.json`
- `models/joint_yolo/metrics.json`
- `bottle_reranker/outputs/yolo_bottle_detector/official_validation_metrics.json`
- `datasets/manual_211/build_summary.json`
- `datasets/fusion_hardset_v1/build_summary.json`

Kaggle final log blocks supplied in the project discussion:

- Stage-I bottle and label hard-fine-tuning results;
- Stage 2A frozen Fusion MLP results;
- Stage 2C Manual-211 adaptation results;
- candidate Transformer experiment results;
- final Manual-211 Transformer versus Fusion MLP evaluation.

No missing metric was imputed. Where the complete Stage-2B final block was unavailable, the report explicitly uses the Stage-2C pre-adaptation baseline instead.
