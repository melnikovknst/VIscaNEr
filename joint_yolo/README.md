# Joint YOLO: bottle + wine_label

One YOLO11n forward predicts both nested objects:

- class `0`: `bottle`;
- class `1`: `wine_label`.

The exact base is `datasets/wine_labels_yolo_500.zip`: 500 reviewed synthetic
frames with both classes. `datasets/wine_bottles_yolo_1000.zip` contributes
extra bottle diversity only after the existing label detector supplies a
confident label box geometrically owned by a manual bottle box. Pseudo-labelled
images are train-only. The 100-image exact joint validation split is never
pseudo-labelled.

The generated revision contains 812 train images (2,336 bottle boxes and
2,466 label boxes) plus 100 exact validation images (213 bottle boxes and 217
label boxes). The archive is versioned as
`datasets/wine_bottle_label_joint.zip` through Git LFS.

## 1. Build/rebuild the dataset

```bash
.venv/bin/python -m joint_yolo.build_dataset --force
```

Outputs:

- unpacked local dataset: `datasets/wine_bottle_label_joint/`;
- portable Git-LFS archive: `datasets/wine_bottle_label_joint.zip`;
- provenance: `datasets/wine_bottle_label_joint/manifest.csv`;
- build statistics: `datasets/wine_bottle_label_joint/build_summary.json`.

## 2. Start training

Default configuration is YOLO11n initialized from the fine-tuned bottle
detector, `imgsz=768`, at most 100 epochs, patience 20, AdamW, MPS/CUDA/CPU
auto-selection and no horizontal/vertical flips:

```bash
.venv/bin/python -m joint_yolo.train
```

On the current Apple Silicon machine, batch 16 was verified to fit. It can be
selected explicitly with `--batch 16`; the conservative portable default is 8.

Quick smoke run:

```bash
.venv/bin/python -m joint_yolo.train --epochs 1 --name smoke
```

Outputs:

- training run: `runs/joint_yolo/yolo11n_bottle_label_joint/`;
- deployable best model: `models/joint_yolo/best.pt`;
- last checkpoint: `models/joint_yolo/last.pt`;
- final validation metrics: `models/joint_yolo/metrics.json`.

The committed checkpoint stopped after 29 epochs (patience 20); epoch 9 was
best. On the untouched 100-image exact synthetic validation split it measured
Precision `0.9118`, Recall `0.9320`, mAP50 `0.9548`, and mAP50-95 `0.5839`.
Per-class values are stored in `models/joint_yolo/metrics.json`. These figures
measure the supplied synthetic validation set, not end-to-end real-photo
product accuracy.

Re-run validation without training:

```bash
.venv/bin/python -m joint_yolo.evaluate
```

## 3. One-forward inference and paired crops

```bash
.venv/bin/python -m joint_yolo.infer /path/to/photo.jpg
```

The target label is selected relative to the crosshair. Its owning bottle is
selected geometrically. Normally one `label_crop` + `bottle_crop` pair is
returned. A second pair is emitted only when the crosshair is genuinely
ambiguous. If bottle confidence is below `0.75`, the bottle-side image is the
complete source photo while the label crop remains available.

## 4. Visual crop audit

Generate a deterministic 20-case audit from the complete validation split. It
mixes difficult and low-confidence cases with random examples:

```bash
.venv/bin/python -m joint_yolo.visualize --n 20
.venv/bin/python -m http.server 8771 --directory runs/joint_yolo/crop_audit
```

Open `http://127.0.0.1:8771`. Each row shows the original with ground-truth and
predicted boxes, the exact bottle-side input, the label crop, confidence,
coordinates, selection mode and complete JSON metadata.
