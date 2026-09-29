# VIscaNEr cascade resolver — Codex handoff

This document describes the current implemented state of the local wine
retrieval cascade. Treat the repository code and tracked model artifacts as the
source of truth. Generated experiment outputs under `runs/` are intentionally
Git-ignored.

## Goal and current architecture

The resolver recognizes one of 2,103 wine identities from a user photo while
avoiding a second expensive model call for every request.

1. YOLO provides the target-aligned label crop and, when available, a whole-
   bottle crop.
2. Fine-tuned DINOv3-B/16 embeds the label crop and searches the complete label
   reference gallery. It produces Top-10 cosine similarities.
3. A small learned gate decides whether running DINOv3-S/16 is expected to
   improve the Top-1 decision. The gate uses only information available before
   DINO-S execution.
4. If selected and a bottle crop exists, fine-tuned DINOv3-S/16 embeds the whole
   bottle and compares it only against DINO-B's Top-1 and Top-2 candidates.
5. DINO-S may reorder those two candidates. It never performs another global
   retrieval and cannot introduce a third identity.

The DINO scores are cosine similarities, not calibrated probabilities.

## Tracked trained artifacts

- `models/trained_checkpoints/dinov3_vitb16_labels_best_full.pt`
  - primary label-crop DINOv3-B/16 retrieval model;
  - full fine-tuned checkpoint, stage 3, epoch 20;
  - SHA-256: `92388ea75c4648584187ac57c4d9d511b99df545f61c4e62df1454045a4cd560`.
- `models/trained_checkpoints/dinov3_vits16_bottles_best_full.pt`
  - conditional whole-bottle DINOv3-S/16 resolver;
  - full fine-tuned checkpoint, stage 3, epoch 56;
  - SHA-256: `8ff9ba049f996802a203ed3a59120f355229cb4f3d8eee8647a31aea6080ca04`.
- `models/cascade_gate/dino_s_gate.joblib`
  - selected lightweight learned gate;
  - multinomial logistic regression with `C=0.1`;
  - decision score threshold: approximately `0.046121`;
  - allowed candidate region: DINO-B Top-1/Top-2 gap `<= 0.03`.
- `models/cascade_gate/dino_s_gate.json`
  - gate training protocol, features, selection result, and metrics.

The `.pt` files are tracked through Git LFS. The gate is only about 6 KB and is
tracked as a normal Git file.

## Dataset coverage

- Source inventory: 45,000 synthetic source images over 2,103 identities.
- Valid label crops usable by DINO-B: 41,527.
- Sources without a valid label crop: 3,473; they remain in the coverage audit
  but cannot enter DINO retrieval.
- Valid/usable whole-bottle crops among label-crop queries: 38,893.
- Splits for the current target-aligned dataset:
  - `train`: 32,229 queries;
  - `val_seen`: 3,604 queries;
  - `val_unseen`: 5,694 queries.

`all_eligible` and `train_descriptive` contain training examples and must not be
used as unbiased production-quality claims. `val_unseen` is the primary
generalization metric available in this synthetic dataset.

## Critical ranking correction

The old rank calculation used `count(similarity > true_similarity) + 1`. With
exactly tied reference similarities, this could report rank 1 even when the
actual Top-1 identity was another tied reference. This slightly inflated old
Recall@1 values and made metrics disagree with emitted predictions.

The code now applies deterministic tie-breaking: lower gallery index wins an
exact float32 tie. Both `cascade_resolver.evaluation.rank_primary()` and
`dinov3_retrieval.retrieval_metrics()` use the same rule. After this correction,
the DINO-B `val_unseen` baseline is `0.678082`, not the previously printed
`0.6953`.

Do not compare new corrected metrics directly against old logs without noting
this change. The checkpoints and embeddings were not corrupted; only ranking
accounting changed.

## Fixed-margin experiment

`tune_cascade_threshold.py` sweeps the DINO-B Top-1/Top-2 cosine gap without
rerunning either neural network. The broad supervision table must first contain
DINO-S scores for every bottle-available candidate up to the requested maximum
margin.

Selection protocol:

- range: `0.00000 ... 0.03000`;
- step: `0.00025`;
- select threshold on `val_seen` by maximum Recall@1;
- break ties by fewer DINO-S calls, then smaller threshold;
- evaluate the chosen threshold on untouched `val_unseen`.

The validation-selected fixed threshold was `0.01525`.

## Learned gate experiment

A scalar least-squares threshold is not the correct statistical formulation.
Instead, `train_cascade_gate.py` learns expected resolver utility:

- `+1`: DINO-S fixes a DINO-B Top-1 error;
- `0`: DINO-S does not change Top-1 correctness;
- `-1`: DINO-S breaks a correct DINO-B Top-1 answer.

Candidate models:

- Ridge least-squares utility regression with alpha values
  `0.01, 0.1, 1, 10, 100`;
- multinomial logistic regression with C values `0.01, 0.1, 1, 10`;
- logistic gate score is `P(fix) - P(harm)`.

Training and selection protocol:

- fit candidate models on `train` only;
- select model, hyperparameter, and gate-score threshold on `val_seen` only;
- evaluate the selected artifact once on `val_unseen`;
- never use wine identity or the ground-truth label as a runtime feature.

Runtime features:

- DINO-B Top-1 through Top-10 cosine similarities;
- Top-1/Top-k and Top-2/Top-3 margins;
- label-detector confidence;
- bottle-detector confidence;
- bottle-crop status as a categorical feature.

The selected model is logistic regression with `C=0.1`. Ridge was evaluated but
was not selected because the logistic model produced the best `val_seen`
Recall@1 with a lower invocation rate among the top candidates.

The resolver only reorders DINO-B's first two candidates, so its effect is
limited to Top-1 versus Top-2: it cannot fix cases where the truth is outside
DINO-B Top-2.

## Reproduction commands

Activate the local environment:

```bash
cd /Users/konstantinmelnikov/Desktop/work/VIscaNEr
source .venv/bin/activate
pip install -r requirements.txt
```

Run the current production-style learned cascade:

```bash
python run_dino_cascade.py
```

Recreate broad DINO-S supervision, retrain the gate, then evaluate it:

```bash
python run_dino_cascade.py \
  --gating-strategy margin \
  --ambiguity-margin 0.03

python train_cascade_gate.py
python run_dino_cascade.py
```

Reproduce the fixed-threshold sweep:

```bash
python run_dino_cascade.py \
  --gating-strategy margin \
  --ambiguity-margin 0.03

python tune_cascade_threshold.py \
  --max-threshold 0.03 \
  --step 0.00025
```

The training script intentionally refuses to train if some candidate rows up to
`max_margin` lack DINO-S supervision. This prevents selection bias from training
only on requests already accepted by an older gate.

## Code map

- `run_dino_cascade.py`: end-to-end local evaluation and operational cascade.
- `train_cascade_gate.py`: Ridge/logistic gate training, model selection, and
  artifact serialization.
- `tune_cascade_threshold.py`: fixed cosine-gap sweep.
- `cascade_resolver/config.py`: resolved local paths and gating configuration.
- `cascade_resolver/data.py`: source inventory, portable crop paths, and splits.
- `cascade_resolver/modeling.py`: checkpoint loading and embedding caches.
- `cascade_resolver/evaluation.py`: deterministic retrieval, resolver metrics,
  and qualitative visualizations.
- `cascade_resolver/gate.py`: shared learned-gate features and inference.
- `configs/dino_cascade.yaml`: defaults; currently `gating_strategy: learned`.
- `CASCADE_RESOLVER.md`: user-facing run guide.

Generated outputs:

- `runs/dino_cascade/predictions.csv`;
- `runs/dino_cascade/stage_metrics.csv`;
- `runs/dino_cascade/summary.json`;
- `runs/dino_cascade/visualizations/`;
- `runs/dino_cascade/threshold_experiments/`;
- `runs/dino_cascade/gate_experiments/`.

## Guardrails for future changes

1. Never select a threshold or gate model on `val_unseen` and then report the
   same `val_unseen` number as unbiased.
2. Keep DINO-S conditional. Do not turn this into a second global retrieval or
   an unconditional second pass.
3. Keep the DINO-S candidate set equal to DINO-B Top-1/Top-2.
4. Report both Recall@1 and invocation rate; quality without compute cost is an
   incomplete comparison.
5. Preserve deterministic tie-breaking in training and cascade evaluation.
6. A real-user-photo test set is still required before making production
   accuracy claims; current 45k data are synthetic.
