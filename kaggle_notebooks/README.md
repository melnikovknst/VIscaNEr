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

The notebooks are source artifacts. Local validation checks their JSON and
Python syntax; real execution requires Kaggle CUDA and the attached datasets.
