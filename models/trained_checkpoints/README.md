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
