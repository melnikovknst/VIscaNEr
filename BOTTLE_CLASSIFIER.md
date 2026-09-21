# Whole-bottle ambiguity classifiers

This pipeline is separate from the production label-retrieval path. The label
DINO runs first. A whole-bottle classifier is consulted only when the label
ranking is ambiguous and a complete target bottle is visible.

## Build the crop dataset

The builder uses `models/bottle_reranker/best_bottle_detector.pt`. It does not
select the highest-confidence bottle blindly: a candidate must own the
geometrically validated target-label centre. Ambiguous, partial and failed
cases are recorded but excluded from training.

```bash
cd /Users/konstantinmelnikov/Desktop/work/VIscaNEr
.venv/bin/python build_bottle_classifier_dataset.py --overwrite --batch-size 4
```

Outputs:

- `datasets/bottle_classifier_crops/` — crops, references and audit metadata;
- `datasets/bottle_classifier_crops/training_metadata.csv` — only identities
  with at least four complete, unambiguous crops;
- `datasets/bottle_classifier_crops.zip` — self-contained Git-LFS archive.

The current complete build audited 41,527 source images and produced 14,657
high-confidence complete crops, 554 low-confidence hard cases, 1,659 ambiguous
cases, 22,023 partial bottles and 2,634 failures. The train/evaluation metadata
contains 14,505 real crops and a complete 2,103-identity reference gallery.

An interrupted build can continue with `--resume`. Never use `partial` or
`ambiguous` rows as positive classifier training examples.

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

This creates two private-dataset upload directories:

- `kaggle_upload/viscaner-bottle-classifier-data`;
- `kaggle_upload/viscaner-bottle-classifier-code`.

Create both private Kaggle datasets once:

```bash
kaggle datasets create \
  -p kaggle_upload/viscaner-bottle-classifier-data -r zip
kaggle datasets create \
  -p kaggle_upload/viscaner-bottle-classifier-code -r zip
```

For later code or data refreshes, publish a new version instead:

```bash
kaggle datasets version \
  -p kaggle_upload/viscaner-bottle-classifier-data \
  -m "Refresh whole-bottle classifier crops" -r zip
kaggle datasets version \
  -p kaggle_upload/viscaner-bottle-classifier-code \
  -m "Update DINOv3 bottle classifier code" -r zip
```

The model weights are not stored in Git. Accept the gated model terms on both
Hugging Face model pages, enable Kaggle Internet and create a private Kaggle
Secret named `HF_TOKEN`. Each notebook downloads only its own official
snapshot. An attached offline snapshot is also supported.

## Kaggle notebooks

- `kaggle_notebooks/BottleClassifier-DINOv3-ViTS16.ipynb`;
- `kaggle_notebooks/BottleClassifier-DINOv3-ViTS16Plus.ipynb`.

Each notebook performs a three-stage CUDA smoke test, starts a fresh run unless
`RESUME_CHECKPOINT` is explicitly set, streams one non-duplicated training log,
and displays training curves, worst retrieval failures and Top-2 errors.
