# Stage-2C five-stream residual Transformer

This experiment trains a new candidate Transformer from random initialization
over two frozen DINOv3-B branches taken **only** from the complete Stage-2C
checkpoint. It never loads Stage-I branch checkpoints.

Inputs are the persisted label and whole-bottle crops from the 45k source and
Manual-211 datasets. No image detection step is part of this training pipeline.

For every query/candidate pair the ranker receives:

1. whole-bottle DINO-B pre-projection feature (`1536`);
2. label-crop DINO-B pre-projection feature (`1536`);
3. whole-bottle retrieval embedding (`256`);
4. label-crop retrieval embedding (`256`);
5. OCR character-ngram embedding (`512`).

The final candidate score is:

```text
frozen bottle-DINO score / temperature + bounded Transformer residual
```

The residual head is zero-initialized, so epoch 0 exactly reproduces the
Stage-2C bottle ranking. The loss combines listwise cross-entropy, a pairwise
positive-vs-hard-negative margin, KL protection on rows where the bottle branch
is already correct, and an L2 penalty on residual corrections for those rows.

Model selection uses `val_hard + val_manual`, with an explicit `val_hard`
floor relative to the frozen bottle baseline. `test_unseen` is evaluated only
after loading the selected checkpoint and is never used for model selection or
refitting.

The Kaggle entrypoint is:

`kaggle_notebooks/FiveStream-Stage2C-Residual.ipynb`
