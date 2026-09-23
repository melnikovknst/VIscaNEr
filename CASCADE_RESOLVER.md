# Conditional DINOv3 wine resolver

This local evaluation pipeline combines the two trained retrieval models without
retraining either model:

1. DINOv3-B/16 embeds the label crop and searches all 2,103 label references.
2. If the cosine-similarity gap between Top-1 and Top-2 is greater than the
   configured ambiguity margin, DINO-B Top-1 is returned immediately.
3. Only when that gap is small and a bottle crop exists, DINOv3-S/16 embeds the
   whole bottle once and compares it only with the same two candidate identities.
4. DINO-S may reorder those two candidates. It cannot introduce a third wine.

The scores are cosine similarities, not calibrated probabilities. The default
ambiguity margin (`0.03`) is a starting operating point and should be selected on
`val_unseen`, balancing accuracy against the percentage of requests that invoke
the second model.

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
python run_dino_cascade.py --device mps --ambiguity-margin 0.02
python run_dino_cascade.py --device cpu --limit 16
```

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

Because the resolver only reorders DINO-B Top-1 and Top-2, it can improve
Accuracy/Recall@1 and MRR, while Recall@2/5/10 should remain unchanged. If the
true wine is outside DINO-B Top-2, this resolver intentionally cannot recover it.

## Interpretation boundary

The 45k set is synthetic and contains training examples. `all_eligible` and
`train_descriptive` are useful for pipeline diagnostics, not unbiased quality
claims. `val_unseen` is the least optimistic split available here because its
wine identities were not used to train either retrieval model. A separate test
on real user photos is still required before claiming production accuracy.
