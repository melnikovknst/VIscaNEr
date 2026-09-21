# Kaggle notebooks

- `DINOv3-finetune.ipynb` — two-stage retrieval fine-tuning: heads, then the
  last four backbone blocks.
- `DINOv3-deeptune.ipynb` — three-stage deep fine-tuning: heads, last four
  blocks, then the fully unfrozen DINOv3 backbone with resumable checkpoints.
- `YOLO-bottle-confidence-audit.ipynb` — Kaggle-only YOLO11x-seg pass over the
  bottle reranker dataset. It creates one whole-bottle crop per image and
  exports the 30 lowest-confidence and 10 highest-confidence valid crops.

Both notebooks use only attached Kaggle datasets, copy code into
`/kaggle/working/VIscaNEr`, reject stale code bundles before training, stream
subprocess output into the notebook and save a single non-appending local log.

Required private inputs:

- `viscaner-dinov3-code`;
- `viscaner-dinov3-data`;
- `viscaner-dinov3-vitb16-weights`.

The YOLO confidence audit instead expects an attached dataset named
`bottle-reranker-dataset` or `viscaner-bottle-reranker-data`. Its input root can
also be set explicitly in the notebook.

The notebooks are source artifacts. Local validation checks their JSON and
Python syntax; real execution requires Kaggle CUDA and the attached datasets.
