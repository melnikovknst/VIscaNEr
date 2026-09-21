# Kaggle notebooks

- `DINOv3-finetune.ipynb` — two-stage retrieval fine-tuning: heads, then the
  last four backbone blocks.
- `DINOv3-deeptune.ipynb` — three-stage deep fine-tuning: heads, last four
  blocks, then the fully unfrozen DINOv3 backbone with resumable checkpoints.
Both notebooks use only attached Kaggle datasets, copy code into
`/kaggle/working/VIscaNEr`, reject stale code bundles before training, stream
subprocess output into the notebook and save a single non-appending local log.

Required private inputs:

- `viscaner-dinov3-code`;
- `viscaner-dinov3-data`;
- `viscaner-dinov3-vitb16-weights`.

The YOLO whole-bottle confidence audit is local and lives at
`bottle_reranker/YOLO-bottle-confidence-audit.ipynb`.

Whole-bottle ambiguity classifiers:

- `BottleClassifier-DINOv3-ViTS16.ipynb` — DINOv3 ViT-S/16;
- `BottleClassifier-DINOv3-ViTS16Plus.ipynb` — DINOv3 ViT-S+/16.

Both use the private datasets `viscaner-bottle-classifier-data` and
`viscaner-bottle-classifier-code`. The official model repositories are gated:
accept the Meta DINOv3 license on Hugging Face, enable Kaggle Internet and add
an `HF_TOKEN` Kaggle Secret. Alternatively attach a private Kaggle dataset
containing the complete Hugging Face snapshot and set
`OPTIONAL_WEIGHTS_DATASET_SLUG` in the notebook.

The notebooks are source artifacts. Local validation checks their JSON and
Python syntax; real execution requires Kaggle CUDA and the attached datasets.
