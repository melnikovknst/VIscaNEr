# Whole-bottle DINOv3 dataset curation

This workflow turns retrieval errors into an inspectable, reversible dataset
curation process. It never deletes source images. Every exclusion is written to
`curation_decisions.csv` and exported to a quarantine manifest.

## What the reviewer sees

Each item shows four images side by side:

1. original user photo;
2. selected whole-bottle crop;
3. ground-truth catalogue bottle and its slug;
4. DINOv3 Top-1 catalogue bottle and its slug.

Decisions:

- `keep_hard`: crop and label are correct; preserve this useful hard example;
- `reject_wrong_crop`: YOLO selected the wrong bottle or produced a broken crop;
- `reject_wrong_label`: the source label does not describe the pictured bottle;
- `quarantine_visual_ambiguity`: a human cannot distinguish the two catalogue
  identities from the available visual evidence;
- `skip`: leave unresolved.

Repeated true/predicted confusion pairs can be reviewed together and assigned a
single batch decision. Batch actions still require an explicit confirmation.

## Commands

From the project root, using the project environment:

```bash
.venv/bin/python -m bottle_dataset_curation.prepare
.venv/bin/streamlit run bottle_dataset_curation/review_app.py
```

The first command builds an initial queue from the latest Kaggle validation
audit and ingests `~/Downloads/Вино.zip`. To evaluate every train/validation
crop plus the fresh archive with the latest full-bottle DINO-B checkpoint and
refresh the queue in one command:

```bash
.venv/bin/python -m bottle_dataset_curation.run_full_local
```

After review, export the hard-fine-tuning subset:

```bash
.venv/bin/python -m bottle_dataset_curation.export \
  --fresh-limit 1500 \
  --replay-ratio 0.30 \
  --materialize
```

Generated data stays under `datasets/bottle_dino_curation/` and is ignored by
Git. The final subset is `hard_finetune_manifest.csv`; excluded rows are in
`quarantine_manifest.csv` and unresolved rows in `unresolved_manifest.csv`.
`curated_master_manifest.csv` is the non-destructive cleaned view of the full
dataset. With `--materialize`, a portable `hard_finetune_dataset.zip` is also
created. Only reviewed errors from the original `train` split are admitted to
that fine-tuning subset; reviewed validation errors remain audit evidence and
never leak into training.

Do not evaluate the fine-tuned model on the same reviewed errors used for
training. Keep `val_unseen` (or a new untouched real-photo holdout) frozen for
the final comparison.
