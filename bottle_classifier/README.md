# Whole-bottle DINOv3 classifiers

This package is an auxiliary ambiguity resolver. It does not replace the
existing label-crop DINO pipeline. The label model produces its ordinary
ranking first; a bottle classifier is consulted only when that ranking is
ambiguous.

Two independent project-local backbones are supported:

- `vits16`: `models/bottle_classifier_backbones/model-s.safetensors`
  (about 21M parameters);
- `vits16plus`: `models/bottle_classifier_backbones/model-s_plus.safetensors`
  (about 29M parameters).

Both use the same leakage-safe identity splits, SupCon + normalized
cross-entropy objective, Recall@1/2/5/10 reporting and visual retrieval audit.
Each run has separate checkpoints and output directories.

The staged schedule is:

1. retrieval projection and classifier heads;
2. heads plus the final four transformer blocks;
3. the entire backbone, with warm-up, cosine decay, resumable checkpoints and
   early stopping on `val_unseen Recall@1`.

The Kaggle notebooks read both files from the private
`viscaner-bottle-classifier-weights` dataset. Internet access and API tokens
are not required.
