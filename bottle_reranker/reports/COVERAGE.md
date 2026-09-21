# Bottle reranker dataset - coverage report

## Status

**Not a finished dataset.** 4 required inputs are absent, so the stages that depend on them did not run. They are listed below with the exact path that was checked.

## 1. Inputs

- **inputs present**: 9/15
- **blocking gaps**: 4
- **blocked steps**: step3, step6

- **photographic images found**: 918
- **rendered images found**: 45000

Missing, blocking:
  - `labels.crops_metadata_csv` -> `datasets/dinov3_target_crops/crops_metadata.csv` (blocks step3)
  - `segmentation.model` -> `yolo11x-seg.pt` (blocks step3)
  - `dino.checkpoint` -> `models/dinov3_retrieval/dinov3_vitb16_wine_retrieval.pt` (blocks step6)
  - `dino.index_csv` -> `datasets/dinov3_retrieval/index.csv` (blocks step6)

## 2. Identities and correspondences

- **identities**: 2103
- **identities with a reference**: 2103
- **distinct reference photographs**: 2075
- **identities sharing one photograph**: 55 in 27 groups
- **near-duplicate label series**: 554 identities in 228 groups
- **clean correspondences**: 45906
- **queued for review**: 12
- **identities with a real photograph**: 531 (25.25%)

## 3. Bottle segmentation and target selection

_Did not run._

## 4a. Catalog reference views

- **references rendered**: 2103
- **references usable for bottle comparison**: 2081
- **references flagged percent**: 1.05
- **quality flags**: {'top_part_estimated_no_neck_narrowing': 29, 'unusual_aspect': 22, 'top_part_estimated_shoulder_too_close_to_top': 9, 'orientation_uncertain': 8, 'top_part_poorly_covered': 4, 'deskew_skipped_elongation_1.06_below_1.8': 3, 'deskew_skipped_elongation_1.07_below_1.8': 3, 'silhouette_occluded': 2, 'deskew_skipped_elongation_1.37_below_1.8': 1, 'deskew_skipped_elongation_1.39_below_1.8': 1}

## 4b. Query bottle and top crops

_Did not run._

## 5. Leakage-free splits

- **split counts**: {'holdout_unseen_identity': 6392, 'train': 31050, 'dev': 7701, 'test': 763}
- **identities per split**: {'train': 1442, 'dev': 362, 'test': 456, 'holdout_unseen_identity': 323}
- **real photographs per split**: {'train': 0, 'dev': 0, 'test': 763, 'holdout_unseen_identity': 143}
- **provenance groups**: 2892
- **reserved identities leaking into training**: 0
- **seen identities pulled into the holdout by grouping**: 24
- **held-out photographic test**: 763 images, 456 identities

> The test pool is held out from reranker training and from every selection decision. Two limits keep it from being a general verdict: it covers only the identities that happen to have been photographed, and it is the project-wide scanner check set, so each further use erodes its independence. Fresh sessions collected per the step 8 plan are the durable answer.

## 6a. DINO candidates

_Did not run._

## 6b. Training pairs

_Did not run._

## 7. Pilot review

_Did not run._

## 7b. Pilot verdicts

_Did not run._

## 8. Collection plan

- **confusion pairs ranked**: 506
- **decidable from the bottle**: 46
- **needing label text or OCR**: 15
- **no possible visual difference**: 29
- **coverage**: {'identities_in_catalog': 2103, 'identities_never_photographed': 1572, 'identities_never_photographed_percent': 74.75, 'identities_with_one_session_only': 299, 'identities_with_three_or_more_sessions': 114}
- **evidence source**: catalog_structure_only

## 9. Package

- **manifest rows**: 2103
- **by role**: {'reference': 2103}
- **by split**: {'gallery': 2103}
- **usable for training**: 2081 (98.95%)
- **kaggle dataset**: f1amex/viscaner-bottle-reranker-data

## What this report does not claim

- No improvement is predicted. Nothing has been trained yet, and a gain claimed before training and an independent test would be a guess.
- Cosine similarities are not probabilities of being correct.
- A figure computed on the DINO validation split is internal validation. The split was already used to pick the checkpoint.
