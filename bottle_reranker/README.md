# bottle_reranker - data preparation for a bottle-level reranker

Builds the dataset for a second-stage model that re-ranks the two closest
candidates from the main DINOv3 label retriever by comparing the **whole bottle**
and its **top part** (capsule, neck, shoulders) against each candidate's catalog
photograph.

**This package prepares data only. It trains nothing.** The existing DINOv3
pipeline (`dinov3_retrieval.py`, `full_finetune.py`,
`build_target_aligned_crops.py`) and its datasets are read-only inputs and are
not modified.

> **Read [`reports/FINDINGS.md`](reports/FINDINGS.md) first.** It documents a
> finding that decides how this data can be used: the 45,000 "source photographs"
> are synthetic renders built from the catalog references themselves, so on that
> pool the bottle comparison is the identity function. 918 real photographs
> exist, covering 25% of the catalog.

---

## Install

Stages 1, 2, 5, 8 and the report need only the standard library plus PyYAML.
Stages 3, 4, 6 and 7 additionally need the imaging and model stack.

```bash
python -m pip install -r bottle_reranker/requirements.txt
```

The pins match the project `requirements.txt`, so one virtualenv serves both.

## Local YOLO confidence audit

[`YOLO-bottle-confidence-audit.ipynb`](YOLO-bottle-confidence-audit.ipynb)
checks YOLO11x-seg whole-bottle detections on the local real-photo pool. Start
Jupyter with the project environment and run it top to bottom:

```bash
.venv/bin/jupyter lab bottle_reranker/YOLO-bottle-confidence-audit.ipynb
```

It automatically uses CUDA, Apple MPS, or CPU and writes resumable audit
artifacts under `bottle_reranker/outputs/` (ignored by Git). This confidence
audit deliberately picks the highest-confidence bottle; production target
selection remains label-box anchored and is implemented by the `segment`
stage below.

## Run

```bash
python -m bottle_reranker.cli --config bottle_reranker/configs/bottle_reranker.yaml <stage>
```

with `PYTHONPATH=bottle_reranker` (or after `pip install -e bottle_reranker`).

| stage | what it does | needs |
|---|---|---|
| `inputs` | check every input, name what is missing | - |
| `identities` | audit identities and image-to-wine correspondences | catalog, manifests |
| `segment` | segment bottles, pick the target by the label box | seg. weights, label boxes |
| `crops` | render bottle / mask / normalized / top views | numpy, OpenCV |
| `splits` | leakage-free train / dev / test / holdout | step 2 |
| `candidates` | DINO top-k, margins, rank of the truth | DINO checkpoint + index |
| `pairs` | training pairs and triplets | step 6a |
| `pilot` | 400-case HTML review sheet | steps 3, 4, 6a |
| `plan` | rank confusion pairs, say what to photograph | catalog (+ step 6a) |
| `package` | manifest and Kaggle staging folder | steps 4, 5 |
| `report` | roll everything into `reports/COVERAGE.md` | - |
| `audit` | run 1, 2, 5, 8 and the report - no model needed | - |

Start with `inputs`: it prints exactly which files are missing and which stages
they block, so nothing fails halfway through a long run.

`splits --phash` additionally hashes every frame to catch near-identical
photographs (a few minutes over 45k images). Without the flag only exact hashes
and declared provenance groups are used.

Every expensive stage is resumable. An interrupted run continues where it
stopped - the ledger under `datasets/bottle_reranker/audit/` records finished
work items:

```bash
python -m bottle_reranker.cli ... segment      # interrupted
python -m bottle_reranker.cli ... segment      # continues, does not redo
```

## Tests

```bash
python -m pytest bottle_reranker/tests -q
```

23 tests over the decision logic that a checkout without the checkpoints cannot
exercise end to end: box format conversion and frame mismatch detection, mask
measures, axis orientation (including the reflection case that once rendered two
references upside down), shoulder detection, target selection, and bit-exact
agreement with the DINO identity holdout hash.

## Outputs

```
datasets/bottle_reranker/
├── bottle_reranker_manifest.csv   one row per prepared bottle or reference
├── queries/{bottle,mask,normalized,top}/
├── references/{bottle,mask,normalized,top}/
├── masks_npz/                     target masks, for re-rendering without the model
├── pairs/organic_pairs.csv        candidate pairs as seen at inference
├── pairs/injected_triplets.csv    training triplets with the truth injected
├── plan/confusion_pairs_to_collect.csv
└── audit/
    ├── correspondences.csv        every frame -> file, slug, reference
    ├── review_queue.csv           doubtful rows, with reasons, kept out of training
    ├── segmentation.csv           per-frame mask selection and why
    ├── query_crops.csv            per-crop geometry, transforms and quality flags
    ├── reference_crops.csv
    ├── splits.csv
    └── dino_candidates.csv        top-k, cosine similarities, rank of the truth

bottle_reranker/reports/
├── FINDINGS.md                    what the data supports, with numbers
├── COVERAGE.md                    rolled-up coverage report
├── step*.json                     per-stage machine-readable records
└── pilot_review.html              the manual review sheet
```

Manifest paths are relative to `datasets/bottle_reranker/`, so the package works
unchanged in the repository and under `/kaggle/input/<slug>/`.

---

## Design decisions worth knowing

**One coordinate frame.** Everything lives in *source pixels*: the image after
`ImageOps.exif_transpose`, at full resolution. EXIF rotation is applied once at
decode. Segmentation runs on a resized copy and masks are mapped straight back.
A label box declared against a different size is rescaled explicitly and
**rejected** if it does not fit - never clamped into a frame it does not belong
to.

**Target selection never rewards the bigger bottle.** Masks that contain the
label box centre, then the largest intersection with the box, which must cover
at least 60% of it. A runner-up within 15% of the winner means the frame is
flagged `ambiguous_target`, not guessed. Detector confidence is recorded and
never decisive.

**What the crops deliberately do not do.** No stretching to a fixed aspect. No
colour normalisation - glass tint and capsule colour are the signal. No
upscaling. Crops are saved near native resolution so the encoder input size stays
a training-time choice. A missing or occluded neck is flagged, never painted in.

**Tilt correction is conservative.** The mask principal axis is used only when
the silhouette is clearly elongated. The 180-degree ambiguity is resolved by two
signals that must agree - end width and area balance - because end width alone
turns a bottle photographed on a reflective surface upside down (this happened;
see FINDINGS §4). When they disagree the crop keeps its orientation and is
flagged.

**The identity holdout is inherited exactly.** `sha256("{seed}:{slug}")` matches
`dinov3_retrieval._stable_unit_interval` bit-for-bit, so an identity reserved for
generalisation there stays reserved here - as anchor, positive *and* negative.
Pairs that would use one are dropped and counted, not relabelled.

**Provenance groups travel together.** Exact duplicates, near-identical frames
and frames from one shooting session land in the same split. For real photos the
session is the Telegram post; for renders every frame of a wine descends from
that wine's single reference, so the identity is the group.

**Similarities are not probabilities.** Every score column is named
`*_cosine_similarity` and is a raw cosine between L2-normalised embeddings. The
candidate table also records `dino_exposure` - whether the prediction was made on
an image the checkpoint trained on. Thresholds tuned on memorised training
predictions look far better than they are.

**Injected triplets and organic pairs stay separate.** A triplet built by adding
the true reference next to a rival is a training aid. An organic (top-1, top-2)
pair is what the model meets in production. One metric over both would flatter
the model.

**`neither_correct` is a label.** When the truth is in neither slot the example
says so; one of the two is never promoted.

---

## Configuration

Everything lives in [`configs/bottle_reranker.yaml`](configs/bottle_reranker.yaml),
commented inline. Paths are relative to `project_root`, which defaults to the
repository root. Nothing is hard-coded - including Kaggle ownership:

```yaml
kaggle:
  source_owner: konstantinmelnikof    # read-only upstream datasets
  owner: f1amex                       # the working copy published to
  dataset_slug: viscaner-bottle-reranker-data
  private: true
```

`package` writes the staging folder and prints the upload command. **It never
uploads.** Publishing is a deliberate act:

```bash
kaggle datasets create -p kaggle_upload/viscaner-bottle-reranker-data -r zip
```

## Reproducibility

Every stage writes a JSON record with the git revision, Python and package
versions, the config keys that determined the result, and - where a model is
involved - the checkpoint sha256. Re-running with different settings is visible
in the record rather than silently mixed into the previous output.

## Next stage

This package stops at prepared data. The encoder is trained separately, and the
first thing to do with this output is the step 7 pilot review: it answers whether
a usable bottle-level signal exists before any GPU time is spent.
