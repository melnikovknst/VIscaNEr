# Conditional DINOv3 wine resolver

This local evaluation pipeline combines the two trained retrieval models without
retraining either model:

1. DINOv3-B/16 embeds the label crop and searches all 2,103 label references.
2. A lightweight learned gate estimates whether DINO-S is likely to improve the
   Top-1 decision. It uses DINO-B similarities/margins and YOLO crop metadata,
   but never the target identity or a DINO-S embedding.
3. Only when the gate selects the request and a bottle crop exists, DINOv3-S/16
   embeds the whole bottle once and compares it only with the same two candidate
   identities.
4. DINO-S may reorder those two candidates. It cannot introduce a third wine.

The DINO scores are cosine similarities, not calibrated probabilities. The gate
is trained on `train`, and both its model variant and decision threshold are
selected on `val_seen`; `val_unseen` remains evaluation-only. A fixed-margin
fallback remains available with `--gating-strategy margin`.

## Run locally

```bash
cd /Users/konstantinmelnikov/Desktop/work/VIscaNEr
source .venv/bin/activate
pip install -r requirements.txt

# Fast data/checkpoint audit without inference
python run_dino_cascade.py --audit-only

# Small end-to-end smoke run
python run_dino_cascade.py --limit 64

# Full evaluation of every image that has a valid label crop
python run_dino_cascade.py
```

Apple Silicon uses MPS automatically; CUDA is selected automatically on an
NVIDIA machine, with CPU as fallback. Override examples:

```bash
python run_dino_cascade.py --device mps --gating-strategy margin --ambiguity-margin 0.02
python run_dino_cascade.py --device cpu --limit 16
```

## Train the learned gate

The gate training run compares Ridge least-squares regression against logistic
expected-utility models. First create DINO-S supervision throughout the allowed
margin, then train and select the gate:

```bash
python run_dino_cascade.py --gating-strategy margin --ambiguity-margin 0.03
python train_cascade_gate.py
```

The resulting model is `models/cascade_gate/dino_s_gate.joblib`. The normal
`python run_dino_cascade.py` command uses it automatically.

Outputs are written to `runs/dino_cascade/`:

- `source_inventory.csv`: coverage of all 45,000 source images;
- `predictions.csv`: per-query DINO-B candidates, ambiguity decision, DINO-S
  scores, and final prediction;
- `stage_metrics.csv`: accuracy, Recall@1/2/5/10, MRR, and ranks before and after
  the resolver, including `val_unseen`;
- `summary.json`: configuration, checksums, timings, coverage, and diagnostics;
- `visualizations/`: worst final errors, cases outside B Top-2, fixes, harms, and
  cases where the truth was B Top-2.

Embedding caches are reused when only the ambiguity margin changes. Add `--force`
to recompute them.

## Tune the ambiguity threshold

First create DINO-S scores across the full experimental range, then sweep that
range without running either neural network again:

```bash
python run_dino_cascade.py --gating-strategy margin --ambiguity-margin 0.03
python tune_cascade_threshold.py
```

The selector maximizes Recall@1 on `val_seen` and reports the chosen operating
point once on untouched `val_unseen`. This avoids selecting and evaluating the
threshold on the same split. Outputs are written to
`runs/dino_cascade/threshold_experiments/`.

The sweep can only use thresholds covered by the saved DINO-S scores. To inspect
a wider interval, first produce those scores and then rerun the sweep:

```bash
python run_dino_cascade.py --gating-strategy margin --ambiguity-margin 0.05
python tune_cascade_threshold.py --max-threshold 0.05
```

Because the resolver only reorders DINO-B Top-1 and Top-2, it can improve
Accuracy/Recall@1 and MRR, while Recall@2/5/10 should remain unchanged. If the
true wine is outside DINO-B Top-2, this resolver intentionally cannot recover it.

## Interpretation boundary

The 45k set is synthetic and contains training examples. `all_eligible` and
`train_descriptive` are useful for pipeline diagnostics, not unbiased quality
claims. `val_unseen` is the least optimistic split available here because its
wine identities were not used to train either retrieval model. A separate test
on real user photos is still required before claiming production accuracy.
