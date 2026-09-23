# Trained cascade checkpoints

These are complete fine-tuned retrieval checkpoints, not the original DINOv3
pretraining weights. Git tracks the `.pt` files through Git LFS.

- `dinov3_vitb16_labels_best_full.pt`: primary DINOv3-B/16 model trained on
  wine-label crops. It searches the full 2,103-reference gallery.
- `dinov3_vits16_bottles_best_full.pt`: DINOv3-S/16 model trained on whole-bottle
  crops. The cascade invokes it only for an ambiguous DINO-B Top-1/Top-2 pair.

Checksums and exact byte sizes are recorded in `checkpoints_sha256.json`.
