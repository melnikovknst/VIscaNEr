# Stage II: label + bottle DINOv3-B fusion

This experiment consumes the two **completed Stage-I hard-fine-tuned
DINOv3-B checkpoints**. Both branches use ViT-B/16:

- target-aligned label crop → label DINOv3-B;
- whole-bottle crop → bottle DINOv3-B;
- union of Top-10 candidates from each branch → Fusion MLP;
- CatBoostRanker is trained as a frozen-feature control in Stage 2A.

The previous diagram that named DINOv3-S for the bottle branch is obsolete.

## Data policy

`build_hard_dataset.py` ranks paired label/bottle crops using both old DINO-B
retrieval ranks, Top-1/Top-2 margins, branch disagreement and detector/crop
quality. It selects 12,000 difficult rows without contaminating checkpoint
selection:

- `hard_train`: 10,800 rows from the old training split;
- `hard_val`: 1,200 difficult rows from the old non-training `val_seen`, used
  for Stage-II checkpoint selection and early stopping;
- `test_seen_prior_exposure`: remaining old train rows;
- `test_seen_model_selection`: the former `val_seen` rows;
- `test_unseen`: the former identity-disjoint `val_unseen` rows.

The test partitions are reported before Stage II and once after restoring the
best `hard_val` checkpoint. They are not evaluated each epoch.

If bottle YOLO produced no usable box, the whole original photo is used for
the bottle branch with status `failed_original_fallback`. These difficult rows
are retained instead of being silently removed; this is the same policy as the
existing `< 0.75 confidence → original photo` fallback.

Build locally:

```bash
cd /Users/konstantinmelnikov/Desktop/work/VIscaNEr
.venv/bin/python -u -m fusion_stage2.build_hard_dataset --force
```

Outputs:

- `datasets/fusion_hardset_v1/manifest.csv`;
- `datasets/fusion_hardset_v1/build_summary.json`;
- `datasets/fusion_hardset_v1.zip`.

## Training stages

1. Stage 2A keeps both DINOv3-B models frozen, computes the candidate union,
   trains the Fusion MLP and compares it with CatBoostRanker.
2. Stage 2B starts from the best Fusion MLP, unfreezes the last two blocks plus
   retrieval heads of both DINOv3-B models and optimizes:

   `fusion listwise CE + 0.25 × label retrieval CE + 0.25 × bottle retrieval CE`.

   Candidate unions are re-mined every three epochs. A final fresh gallery and
   candidate pass is run before the untouched test metrics are written.

The manifest currently has no verified OCR transcriptions. The MLP therefore
receives `ocr_available=0` and `ocr_lexical_score=0`; this is an explicit
missing branch, not synthetic OCR. Real precomputed OCR text can later be added
to the `ocr_text` column without changing the feature schema.
Alternatively pass a separate file with
`--ocr-csv path/to/file.csv`; it must contain unique `source_relative_path`
and `ocr_text` columns.

Kaggle notebooks:

- `kaggle_notebooks/Fusion-Stage2A-Frozen.ipynb`;
- `kaggle_notebooks/Fusion-Stage2B-Joint.ipynb`.

Run Stage 2B only after Stage 2A has completed and its saved output has been
attached as an input.

`prepare_kaggle_fusion_bundle.py` also creates two Kaggle CLI kernel folders
with `kernel-metadata.json`. Pushing a kernel creates/updates a Kaggle notebook
version and may start its execution, so do that only when the required upstream
outputs are complete.

## Kaggle publication commands

Prepare local staging folders:

```bash
cd /Users/konstantinmelnikov/Desktop/work/VIscaNEr
.venv/bin/python prepare_kaggle_bottle_classifier_bundle.py \
  --kaggle-username konstantinmelnikof
.venv/bin/python prepare_kaggle_fusion_bundle.py \
  --kaggle-username konstantinmelnikof
```

The existing bottle dataset needs a new version because Stage II also needs
`partial`, `ambiguous`, and original-photo fallback inputs:

```bash
.venv/bin/kaggle datasets version \
  -p kaggle_upload/viscaner-bottle-classifier-data \
  --dir-mode zip \
  -m "Add difficult bottle crops and original fallbacks for Stage II"
```

Create the two new private datasets once:

```bash
.venv/bin/kaggle datasets create -p kaggle_upload/viscaner-fusion-hardset-v1 --dir-mode zip
.venv/bin/kaggle datasets create -p kaggle_upload/viscaner-fusion-stage2-code --dir-mode zip
```

For later code/data revisions, replace `datasets create` with:

```bash
.venv/bin/kaggle datasets version -p kaggle_upload/viscaner-fusion-hardset-v1 --dir-mode zip -m "Upload Stage II hard-set images"
.venv/bin/kaggle datasets version -p kaggle_upload/viscaner-fusion-stage2-code --dir-mode zip -m "Fix Stage II Kaggle dataset extraction"
```

Only after both Stage-I notebooks show `COMPLETE`, upload/run Stage 2A:

```bash
.venv/bin/kaggle kernels push -p kaggle_upload/viscaner-fusion-stage2a-kernel
```

After Stage 2A is `COMPLETE`, upload/run Stage 2B:

```bash
.venv/bin/kaggle kernels push -p kaggle_upload/viscaner-fusion-stage2b-kernel
```
