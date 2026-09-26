# Whole-bottle ambiguity classifiers

This pipeline is separate from the production label-retrieval path. The label
DINO runs first. A whole-bottle classifier is consulted only when the label
ranking is ambiguous and a complete target bottle is visible.

## Build the crop dataset

The builder uses the two-class joint detector at `models/joint_yolo/best.pt`
(class 0 is `bottle`, class 1 is `wine_label`). It does not
select the highest-confidence bottle blindly: a candidate must own the
geometrically validated target-label centre. If the selected YOLO detection has
confidence below `0.75`, its bounding box is not trusted and the original image
is saved as the classifier input. At `0.75` and above, the padded YOLO crop is
saved. Ambiguous target ownership and high-confidence partial bottles remain
audit-only and are excluded from training.

```bash
cd /Users/konstantinmelnikov/Desktop/work/VIscaNEr
.venv/bin/python build_bottle_classifier_dataset.py \
  --overwrite \
  --batch-size 4 \
  --crop-confidence-threshold 0.75
```

The same destructive rebuild is available as a local notebook:
`bottle_reranker/Build-bottle-classifier-dataset.ipynb`. Run all four cells
from top to bottom; it always invokes the project's `.venv` interpreter.

Outputs:

- `datasets/bottle_classifier_crops/` — crops, references and audit metadata;
- `datasets/bottle_classifier_crops/training_metadata.csv` — only identities
  with at least four unambiguous inputs across trusted crops and original-image
  fallbacks;
- `datasets/bottle_classifier_crops.zip` — self-contained Git-LFS archive.

`crops_metadata.csv` records `image_mode=yolo_crop` or
`image_mode=original_image` for every saved input. Low-confidence original
frames are stored under `low_confidence/` and are included in
`training_metadata.csv`.

The current fallback build audited 41,527 source images. It saved 30,702 YOLO
crops and 8,191 original-image fallbacks; 2,634 images had no usable detection.
The training metadata contains 20,776 real inputs across 2,072 trainable
identities, plus 31 gallery-only placeholders so that all 2,103 reference wines
remain represented.

An interrupted build can continue with `--resume` only when its metadata was
created by the same code version. For a threshold change, use `--overwrite`.
Never use `partial` or `ambiguous` rows as positive classifier training
examples.

## Models

- DINOv3 ViT-S/16: `configs/bottle_classifier_vits16.yaml`;
- DINOv3 ViT-S+/16: `configs/bottle_classifier_vits16plus.yaml`.

Both optimize SupCon plus normalized cross-entropy and report accuracy,
Recall@1/2/5/10, MRR and rank statistics. Their schedule is 5 head-only epochs,
10 epochs with the final four blocks unfrozen, then up to 100 fully unfrozen
epochs. The final phase uses warm-up, cosine decay, checkpoint continuation and
early stopping on `val_unseen Recall@1`.

## Prepare Kaggle datasets

```bash
.venv/bin/python prepare_kaggle_bottle_classifier_bundle.py \
  --kaggle-username konstantinmelnikof
```

This creates three private-dataset upload directories:

- `kaggle_upload/viscaner-bottle-classifier-data`;
- `kaggle_upload/viscaner-bottle-classifier-code`;
- `kaggle_upload/viscaner-bottle-classifier-weights`.

Create all three private Kaggle datasets once:

```bash
kaggle datasets create \
  -p kaggle_upload/viscaner-bottle-classifier-data -r zip
kaggle datasets create \
  -p kaggle_upload/viscaner-bottle-classifier-code -r zip
kaggle datasets create \
  -p kaggle_upload/viscaner-bottle-classifier-weights -r zip
```

For later code or data refreshes, publish a new version instead:

```bash
kaggle datasets version \
  -p kaggle_upload/viscaner-bottle-classifier-data \
  -m "Refresh whole-bottle classifier crops" -r zip
kaggle datasets version \
  -p kaggle_upload/viscaner-bottle-classifier-code \
  -m "Update DINOv3 bottle classifier code" -r zip
kaggle datasets version \
  -p kaggle_upload/viscaner-bottle-classifier-weights \
  -m "Update local DINOv3 S and S+ weights" -r zip
```

The model weights are versioned in Git LFS and live at:

- `models/bottle_classifier_backbones/model-s.safetensors`;
- `models/bottle_classifier_backbones/model-s_plus.safetensors`.

The bundle script copies and checksums both files into the private Kaggle
weights dataset. Their repository-wide checksums are also recorded in
`models/weights_sha256.json`. The training notebooks need neither Internet
access nor API secrets.

## Kaggle notebooks

- `kaggle_notebooks/BottleClassifier-DINOv3-ViTS16.ipynb`;
- `kaggle_notebooks/BottleClassifier-DINOv3-ViTS16Plus.ipynb`.

Each notebook performs a three-stage CUDA smoke test, starts a fresh run unless
`RESUME_CHECKPOINT` is explicitly set, streams one non-duplicated training log,
and displays training curves, worst retrieval failures and Top-2 errors.
