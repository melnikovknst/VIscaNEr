# VIscaNEr model inference handoff

All paths below are relative to the repository root. After cloning, run
`git lfs pull`; otherwise large archives and checkpoints may remain LFS pointer
files.

## 0. Recommended shared detector (one forward)

For new integration work, use the joint detector when both visual regions are
needed. A single YOLO11n forward predicts `bottle` (class 0) and `wine_label`
(class 1), then the inference helper pairs nested boxes and applies the same
crosshair/ambiguity policy used by the product experiments.

| Role | Repository path |
|---|---|
| One-forward inference and pair selection | `joint_yolo/infer.py` |
| Dataset construction with provenance | `joint_yolo/build_dataset.py` |
| Reproducible training entry point | `joint_yolo/train.py` |
| Deployable checkpoint | `models/joint_yolo/best.pt` |
| Portable two-class dataset | `datasets/wine_bottle_label_joint.zip` |
| Full method and commands | `joint_yolo/README.md` |

Run:

```bash
.venv/bin/python -m joint_yolo.infer /path/to/photo.jpg
```

The helper ranks labels as `confidence + 0.20 × axis_proximity`. Proximity uses
only the horizontal distance from the bbox centre to the vertical target axis;
Y position and crossing the crosshair add no bonus. It returns one bottle/label
pair normally and two only when both combined scores and axis distances are
close. Bottle confidence below `0.75` triggers the
complete-photo fallback for the bottle-side classifier input.

## 1. Whole-bottle pipeline (current main path)

Flow: **whole-bottle YOLO -> DINOv3-B/16 retrieval -> catalogue Top-K**.

| Role | Repository path |
|---|---|
| Ready-to-run CLI and current selection policy | `infer_wine.py` |
| DINO architecture, transforms and checkpoint loader | `dinov3_retrieval.py` |
| Local loaders for the other researched DINO backbones | `deeptune_backbones.py` |
| Joint bottle + label detector | `models/joint_yolo/best.pt` |
| Original DINOv3-B/16 backbone | `models/dinov3/model.safetensors` |
| Fine-tuned full-bottle DINOv3-B/16 | `models/trained_checkpoints/dinov3_vitb16_bottles_best_full.pt` |
| Reference gallery and training-data archive | `datasets/bottle_classifier_crops.zip` |

Important policy implemented in `infer_wine.py`:

- the target score is `YOLO confidence + 0.20 × vertical-axis proximity`;
  vertical position and crossing the crosshair do not affect ranking;
- when two distinct central boxes are genuinely ambiguous, both are passed to
  DINO and each catalogue identity receives the better of the two similarities;
- if the selected bottle YOLO confidence is below `0.75`, the **complete source
  photo** is passed to DINO instead of an unreliable bottle crop;
- a confident bottle box receives `6%` padding;
- DINO input is the project transform at `224x224`; ranking uses cosine
  similarity against 2,103 reference identities.

Run:

```bash
.venv/bin/python -m pip install -r requirements-inference.txt
.venv/bin/python infer_wine.py /path/to/photo.jpg --top-k 5
```

## 2. Label pipeline

Flow: **label YOLO -> DINOv3-B/16 trained on label crops -> catalogue Top-K**.

| Role | Repository path |
|---|---|
| Ready-to-run label CLI | `infer_wine_labels.py` |
| DINO architecture, transforms and checkpoint loader | `dinov3_retrieval.py` |
| Local loaders for the other researched DINO backbones | `deeptune_backbones.py` |
| Joint bottle + label detector | `models/joint_yolo/best.pt` |
| Original DINOv3-B/16 backbone | `models/dinov3/model.safetensors` |
| Fine-tuned label DINOv3-B/16 | `models/trained_checkpoints/dinov3_vitb16_labels_best_full.pt` |
| Catalogue and RGB reference gallery archive | `datasets/wine-scanner_code-catalog.zip` |

`infer_wine_labels.py` uses the same confidence + vertical-axis selector, adds
`10%` padding, saves the exact DINO/OCR input under
`runs/inference/label_crops/`, and emits Top-K JSON. If no label is detected,
it uses the complete photo rather than crashing.

Run:

```bash
.venv/bin/python infer_wine_labels.py /path/to/photo.jpg \
  --top-k 5 --output runs/inference/labels_top5.json
```

## 3. Optional PaddleOCR-VL reranker for label Top-5

| Role | Repository path |
|---|---|
| Portable OCR model archive | `models/paddleocr_vl/PaddleOCR-VL-1.6.zip` |
| OCR execution and JSON output | `ocr_reranker/run_label_ocr.py` |
| Pure Top-5 text-scoring/reranking logic | `ocr_reranker/reranker.py` |
| Environment and commands | `ocr_reranker/README.md` |

This OCR stage is **not a separately fine-tuned project model**. PaddleOCR-VL
1.6 reads Cyrillic/Latin from the saved label input. Project code then compares
that text only with the five DINO candidate slugs. Current experimental values
are `alpha=0.90` and `evidence_gate=0.10`; they were selected on a calibration
split of the 100 manually labelled real-photo audit and are not a final
production threshold.

There is therefore no custom OCR annotation dataset to transfer. The required
runtime inputs are the pretrained OCR archive, the DINO Top-5 JSON, the saved
label crops referenced by that JSON, and the catalogue reference slugs.

## 4. Which checkpoint is which

- `dinov3_vitb16_bottles_best_full.pt`: DINOv3-B/16, fully fine-tuned on
  full-bottle inputs. Use with `infer_wine.py`.
- `dinov3_vitb16_labels_best_full.pt`: DINOv3-B/16, fully fine-tuned on label
  crops. Use with `infer_wine_labels.py` and optionally OCR.
- `dinov3_vits16_bottles_best_full.pt`: older DINOv3-S/16 bottle resolver from
  the abandoned cascade experiment; do not substitute it for either B model.
- `model.safetensors`: original DINOv3-B backbone needed by the project loader;
  it is not the trained retrieval checkpoint.

## 5. Integrity checks

Canonical hashes live in:

- `models/weights_sha256.json` for YOLO, DINO backbones and trained models;
- `models/trained_checkpoints/checkpoints_sha256.json` for retrieval checkpoints;
- `models/paddleocr_vl/weights_sha256.json` for the OCR archive and its primary
  safetensors file.
