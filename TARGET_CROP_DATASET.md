# Target-aligned DINO crop dataset

`build_target_aligned_crops.py` creates the clean offline training crops used by
DINO retrieval. It does not replace or modify the production inference selector
in `run_bulk_crop.py`.

The builder re-renders a scene for every row of the 45k generator manifest,
tracks the exact silhouette of the requested target bottle, runs YOLO once on
the whole generated scene, and accepts a label box only when its centre and at
least 60% of its area lie inside that target silhouette. Labels from neighbours
cannot receive the target wine identity.

The generated scene is used only to produce the crop; full scenes are not saved.
Existing bottle-scene images are used as optional photo backgrounds to retain
multi-bottle and shelf complexity. The source manifests, YOLO checkpoint and
production inference code are read-only inputs.

## Full build

From the project root:

```bash
source .venv/bin/activate
python build_target_aligned_crops.py --overwrite
```

If the process is interrupted, continue without deleting completed work:

```bash
source .venv/bin/activate
python build_target_aligned_crops.py --resume
```

For a deterministic random smoke test:

```bash
python build_target_aligned_crops.py \
  --output-root /tmp/viscaner-target-crop-smoke \
  --sample 256 --overwrite
```

## Outputs

```text
datasets/dinov3_target_crops/
├── successful/                 accepted target-aligned label crops
├── rejected_debug/             bounded sample of rejected scenes
├── audit/                      first accepted scenes with target mask + bbox
├── crops_metadata.csv          all accepted and rejected source rows
└── build_summary.json
```

Before uploading, inspect `audit/`, `rejected_debug/` and
`build_summary.json`. A row is rejected instead of guessed when the target is
insufficiently visible or no YOLO label box passes the target-mask checks.

## Build and upload Kaggle datasets

Create fresh staging directories after the full crop build finishes:

```bash
source .venv/bin/activate
python prepare_kaggle_dinov3_bundle.py \
  --kaggle-username konstantinmelnikof
```

Update the existing private data Dataset:

```bash
kaggle datasets version \
  -p kaggle_upload/viscaner-dinov3-data \
  -m "Replace crops with generator-target-aligned DINO dataset" \
  -r zip
```

The crop builder and new path configuration are code changes, so update the
private code Dataset as well:

```bash
kaggle datasets version \
  -p kaggle_upload/viscaner-dinov3-code \
  -m "Add generator-target-aligned crop builder and DINO paths" \
  -r zip
```

The original DINOv3 weights Dataset is unchanged and does not need a new
version.
