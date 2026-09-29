# Trained cascade checkpoints

These are complete fine-tuned retrieval checkpoints, not the original DINOv3
pretraining weights. Git tracks the `.pt` files through Git LFS.

- `dinov3_vitb16_labels_best_full.pt`: primary DINOv3-B/16 model trained on
  wine-label crops. It searches the full 2,103-reference gallery.
- `dinov3_vitb16_bottles_best_full.pt`: current DINOv3-B/16 model fully
  fine-tuned on whole-bottle inputs. It is the main checkpoint used by
  `infer_wine.py`.
- `dinov3_vits16_bottles_best_full.pt`: DINOv3-S/16 model trained on whole-bottle
  crops. It belongs to the older cascade experiment and is not the current
  primary full-bottle model.

Checksums and exact byte sizes are recorded in `checkpoints_sha256.json`.

## Hard-negative fine-tune candidates

These checkpoints are kept separate from the current inference weights because
`val_unseen` Recall@1 did not improve in the standalone retrieval evaluation.
Both were selected by `val_seen` Recall@1 at epoch 21, with the entire DINOv3-B/16
backbone and both heads trainable.

| Input | Checkpoint | Kaggle run |
| --- | --- | --- |
| Bottle | `dinov3_vitb16_bottles_hard_best_full.pt` | [run](https://www.kaggle.com/code/f1amex/viscaner-dinov3-b-bottles-hard-fine-tune?scriptVersionId=352923332) |
| Label | `dinov3_vitb16_labels_hard_best_full.pt` | [run](https://www.kaggle.com/code/f1amex/viscaner-dinov3-b-labels-hard-fine-tune?scriptVersionId=352930049) |

Training code is in `hard_finetune.py`. The label Kaggle run skipped the
repeated per-file crop inventory; use `--skip-file-validation` with the trusted
attached dataset to reproduce that setup.
