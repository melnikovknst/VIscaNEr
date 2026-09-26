# Kaggle notebooks

- `DINOv3-finetune.ipynb` — two-stage retrieval fine-tuning: heads, then the
  last four backbone blocks.
- `DINOv3-deeptune.ipynb` — parameterized three-stage deep fine-tuning. The
  first cell selects either DINOv3 ViT-B/16 or DINOv3 ConvNeXt-B and either
  target-aligned wine-label crops or whole-bottle crops. Training uses heads,
  then the last four backbone blocks, then the fully unfrozen backbone with
  resumable checkpoints.
Both notebooks use only attached Kaggle datasets, copy code into
`/kaggle/working/VIscaNEr`, reject stale code bundles before training, stream
subprocess output into the notebook and save a single non-appending local log.

Required private inputs:

- `viscaner-dinov3-code`;
- `viscaner-dinov3-data`;
- `viscaner-bottle-classifier-data`;
- `viscaner-dinov3-vitb16-weights` — contains both `model.safetensors`
  (ViT-B/16) and `dinov3-convnext-b.safetensors` (ConvNeXt-B).

`DINOv3-deeptune.ipynb` reads only the data and weight dataset selected in its
first cell, but attaching all four datasets makes all four combinations
available without editing Kaggle inputs. Both B checkpoints are stored in the
same private weights dataset, validated by SHA256, and loaded without querying
a model hub.

The YOLO whole-bottle confidence audit is local and lives at
`bottle_reranker/YOLO-bottle-confidence-audit.ipynb`.

Whole-bottle ambiguity classifiers:

- `BottleClassifier-DINOv3-ViTS16.ipynb` — DINOv3 ViT-S/16;
- `BottleClassifier-DINOv3-ViTS16Plus.ipynb` — DINOv3 ViT-S+/16.

Both use three private datasets: `viscaner-bottle-classifier-data`,
`viscaner-bottle-classifier-code`, and
`viscaner-bottle-classifier-weights`. The last dataset contains the two local
`safetensors` files. Internet access and API secrets are not required.

The notebooks are source artifacts. Local validation checks their JSON and
Python syntax; real execution requires Kaggle CUDA and the attached datasets.

## Stage II fusion

- `Fusion-Stage2A-Frozen.ipynb` — freezes the hard-fine-tuned label and bottle
  DINOv3-B models, forms a Top-10 + Top-10 candidate union, trains Fusion MLP,
  and fits CatBoostRanker as a control.
- `Fusion-Stage2B-Joint.ipynb` — starts from Stage 2A, unfreezes the final two
  blocks of **both DINOv3-B models**, and jointly trains them with the final
  ranking loss plus two branch retrieval losses.

Additional private inputs:

- `viscaner-fusion-stage2-code`;
- `viscaner-fusion-hardset-v1`;
- completed outputs of both Stage-I hard-fine-tune notebooks;
- the existing label and whole-bottle data datasets.

Stage 2B additionally requires the saved output of Stage 2A. The notebooks do
not clone GitHub and do not download model weights from a model hub.
