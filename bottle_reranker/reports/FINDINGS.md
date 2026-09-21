# Bottle-level reranker: what the data actually supports

Every number below was produced by the scripts in this package against the data
in this repository. The commands that produce each one are named. Nothing here
is an estimate.

---

## 1. The headline finding

**The 45,000 "source photographs" are not photographs.** They are synthetic
renders, and each one is built from the catalog reference of the very wine it
depicts.

`datasets/wine-scanner/scripts/build_trainset.py` renders every frame as:

```python
ref = load_rgba(w.ref_rgba)          # data/refs/rgba/<slug>.webp
img, params = _aug(ref, seed)        # rotate, relight, paste onto a background
```

`scanner/augment.rotate_cylinder` even states it outright: *the silhouette of the
bottle does not change*. The augmenter warps the label texture around a cylinder,
adds highlights, shadow, a price tag and JPEG noise, and composites the result
onto a shelf or photo background next to neighbouring bottles - which are
themselves other catalog cut-outs.

So in every one of the 45,000 frames, the bottle's shape, shoulders, neck,
capsule and glass colour are *the same pixels* as the catalog reference the
reranker would be asked to match them against, up to an affine transform and a
relight.

### Why this blocks the stated goal

The task is to train an encoder that compares a photographed bottle with a
candidate's catalog photograph using shape, shoulders, neck, capsule and glass
colour. On this data that comparison is the identity function. A model trained
on it would:

- score close to perfect on any validation split drawn from the same 45k, because
  the task it actually learned is copy detection, not bottle recognition;
- learn nothing about how a real bottle's silhouette varies with perspective,
  lens, distance or lighting, because that variation is not present;
- carry none of it over to a real photograph, where the query bottle and the
  catalog bottle are two different physical photographs of the same product.

This is not a reason to stop. It is a reason to be precise about what the
resulting dataset is: **a correctly built pipeline over data that cannot, on its
own, teach bottle-level discrimination.** The pipeline is complete and runs; the
photographic material it needs does not exist yet in the required volume.

### The only real photographs

| | frames | identities | independent sessions |
|---|---|---|---|
| `synthetic_45k` (renders) | 45,000 | 2,103 (100%) | 2,103 (one reference each) |
| `real_photos_v4` (photographs) | 918 | 531 (25.2%) | 796 Telegram posts |

Real photographs are **2.0%** of the available frames. Of the 531 photographed
identities, 286 have exactly one photograph. And `real_photos_v4` is the
project's evaluation set - spending it on training would destroy the only
honest measurement available.

*Produced by:* `cli inputs`, `cli identities` →
`reports/step1_inputs.json`, `reports/step2_identities.json`

---

## 2. Identities: what the labelling separates, and what it cannot

The catalog holds **2,103 identities** and **2,075 distinct reference
photographs**, all 2,103 with a reference file present.

The labelling **does** distinguish vintages, volumes and packaging - it does not
merge them, and neither does this pipeline:

| distinction | identities carrying the token |
|---|---|
| vintage (a 19xx/20xx year) | 101 |
| volume | 6 |
| packaging variant (bag-in-box, magnum, …) | 25 |
| colour | 1,693 |
| sweetness / sparkling style | 1,697 |

But three structural limits cap what any image model can do:

**a. 55 identities in 27 groups share one catalog photograph byte-for-byte.**
The gallery holds the same image under both slugs, so no encoder - bottle,
label, or otherwise - can separate them. Examples: `aligote-barrel-2024` vs
`aligote-barrel-2025`; the three `soyuz-vino … beg-in-boks` variants; two
unrelated `skalistyy-bereg` wines.

**b. 104 identities in 51 groups differ only by vintage, volume or the trailing
alcohol figure.** Those differences live in the label text, not the silhouette.
A whole-bottle encoder is the wrong tool for them; an OCR or label-zone check is
the right one, and that is a different component.

**c. 554 identities in 228 groups are near-duplicate label series** - the hard
cases the reranker is meant to fix. 27 of those 228 groups have two or more
members with a real photograph. The other 201 groups cannot currently be
evaluated on real photographs at all.

*Produced by:* `cli identities` → `reports/step2_identities.json`

---

## 3. The split, and what a number on it is worth

The DINO identity holdout is reproduced bit-for-bit
(`sha256("42:{slug}")[:8] / 2**64 < 0.15`), so an identity reserved there stays
reserved here: **299 identities (14.2%)**.

| split | images | identities | real photographs |
|---|---|---|---|
| train | 31,050 | 1,442 | 0 |
| dev | 7,701 | 362 | 0 |
| test | 763 | 456 | 763 |
| holdout (reserved identities) | 6,392 | 323 | 143 |

45,906 images fall into 2,892 provenance groups. No reserved identity reaches
training, and no provenance group is split across parts. 24 seen identities had
some images held out because those images shared a group with a reserved
identity - the intended cost of grouping, not a leak.

**Note the disjoint train/dev identity sets.** Every frame of a synthetic wine is
rendered from that wine's single reference, so all of its frames form one
provenance group and the whole identity moves as a unit. dev therefore measures
generalisation to *unseen identities*, not "same wine, new photograph" - the
synthetic pool contains no independent within-identity variation to measure the
latter with. This is a direct structural consequence of section 1, and it is
stated in the config beside the grouping rule rather than left to be discovered.

**Two honest caveats:**

- **60 of 228 near-duplicate groups straddle the holdout boundary**, and 5 of the
  27 identical-photo groups do. For those, the natural hard negative is a
  reserved identity. `step6_pairs` drops such pairs rather than leaking them, and
  reports how many were dropped. That is a real, quantified loss of hard
  negatives.
- **Only one pool can serve as a test, and it is thin.** The DINO synthetic
  validation split already selected the current checkpoint, so any figure
  computed on it is *internal validation*, not a test. `real_photos_v4` is
  genuinely held out from reranker training and from every selection decision -
  763 images over 456 identities - but it covers only the wines that happen to
  have been photographed (a quarter of the catalog, median one photograph each),
  and it is also the project-wide scanner check set, so each further use erodes
  its independence. Fresh sessions collected per section 6 are the durable
  answer.

*Produced by:* `cli splits` → `reports/step5_splits.json`

---

## 4. Reference views: built, and measured

All **2,103** catalog references were rendered into the four views the encoder
needs - whole bottle, mask, normalized bottle, and capsule+neck+shoulders.
**2,081 (98.9%)** are usable for bottle-level comparison. 2,045 carry no flag at
all.

| flag | references | what it means |
|---|---|---|
| `top_part_estimated_no_neck_narrowing` | 29 | the silhouette never narrows: neck out of frame, or not a bottle |
| `unusual_aspect` | 22 | silhouette outside 1.6-5.0 h/w: dessert bottles, bag-in-box cartons |
| `top_part_estimated_shoulder_too_close_to_top` | 9 | shoulder line implausibly high; top crop fell back to a fixed fraction |
| `orientation_uncertain` | 8 | flip signals disagreed; original orientation kept |
| `top_part_poorly_covered` | 4 | genuinely deficient neck crop |
| `deskew_skipped_elongation_*` | 8 | not elongated enough for a trustworthy axis (all bag-in-box) |
| `silhouette_occluded` | 2 | holes bitten out of the silhouette |

38 of 2,103 top crops fall back to a fixed fraction of the height instead of a
measured shoulder line, and every one of them is flagged.

### Three calibration decisions worth recording

**The top-part fill threshold was wrong and was fixed against measurement.** The
initial floor of 0.45 flagged 1,952 of 2,103 references. Measuring the actual
distribution showed a healthy capsule+neck+shoulder crop fills 0.36-0.46 of its
bounding box (p1 0.359, median 0.419) - a narrow neck inside a rectangle simply
does not fill more. The floor is now 0.28, below the p1, and flags 4.

**The 180-degree flip test needed two signals, not one.** The first
implementation resolved the principal axis ambiguity by comparing silhouette
width at both ends. It turned two references upside down: both are catalog
photographs taken on a reflective surface, so the mask contains the bottle *and
its mirror image* and is narrow at both ends. The test now requires the
end-width signal and an area-balance signal to agree, and keeps the original
orientation when they disagree. Zero references are flipped now, and the two
formerly broken crops render upright.

`tests/test_geometry_and_selection.py` pins that case.

**Near-duplicate detection had to be confined to one identity.** Perceptual
hashing across all 45,906 frames chained 953 images spanning dozens of
identities into a single provenance group and distorted every split (test grew
from 763 images to 946). The cause is specific to this data: a 64-bit hash of a
synthetic shelf scene mostly describes the shared background, not the bottle.
Linking is now confined to frames of the same wine. That finds 8 real links -
reposted photographs of the same wine in different Telegram posts - and leaves
2,887 provenance groups. Genuinely shared shots of two *different* wines are
already joined by their source post, so nothing is lost.

*Produced by:* `cli crops --only references`, `cli splits --phash` →
`reports/step4_reference_crops.json`, `reports/step5_splits.json`

---

## 5. What could not run, and exactly why

| stage | blocked on | path checked |
|---|---|---|
| 3. segmentation + target selection | chosen label boxes | `datasets/dinov3_target_crops/crops_metadata.csv` |
| 3. segmentation + target selection | segmentation weights | `yolo11x-seg.pt` (not cached locally) |
| 4b. query crops | depends on stage 3 | - |
| 6a. DINO candidates | trained retrieval checkpoint | `models/dinov3_retrieval/dinov3_vitb16_wine_retrieval.pt` |
| 6a. DINO candidates | gallery index | `datasets/dinov3_retrieval/index.csv` |
| 6b. pairs, 7. pilot | depend on stage 6a | - |

All five are generated artefacts that `.gitignore` keeps out of the repository
(`models/**/*.pt`, `datasets/**/*`) and that live on the Kaggle account
`konstantinmelnikof`. The `kaggle` CLI is not installed here and there is no
`~/.kaggle/kaggle.json`, so they could not be fetched.

Also absent, but not blocking: `datasets/bottle_images_45k/bottle_images_manifest.csv`
(a flat symlink mirror of the same 45k renders - `prepare_bottle_images.py`
rebuilds it) and `datasets/dinov3_target_crops/` (label crops; useful for
traceability, useless as a source of bottle shape, exactly as the task warned).

**Nothing was substituted.** No synthetic stand-in was generated for any missing
input, and no stage reports a result it did not compute.

*Produced by:* `cli inputs` → `reports/step1_inputs.json`

---

## 6. What to collect next

With no DINO candidate table, `step8` ranks the **structurally** risky pairs -
identities the catalog itself marks as near-duplicate series or identical
photographs. It found **506 pairs**:

- **46** decidable from the bottle alone (shape, neck, capsule, glass)
- **15** needing legible label text or a vintage reader
- **29** with no possible visual difference - these need a catalog correction,
  not a camera

Coverage is the real constraint:

- **1,572 of 2,103 identities (74.8%) have never been photographed.**
- 299 have exactly one shooting session - no independent angle, light or
  background.
- Only 109 have three or more.

The per-pair rows in `plan/confusion_pairs_to_collect.csv` name what is missing
for each side. One caveat on the `distinguishing_evidence_required` column: it is
a heuristic prior from the catalog fields. A "Reserve vs non-Reserve" pair is
marked *bottle* because the slugs genuinely differ, but in practice many such
pairs differ only in a word on the label. The step 7 pilot is what settles each
case; the column is a starting point for the shot list, not a verdict.

*Produced by:* `cli plan` → `reports/step8_gap_plan.json`

---

## 7. Recommendation

The pipeline is finished and reproducible. Before spending GPU time on the
encoder, the cheap and decisive experiment is:

1. Fetch the DINO checkpoint and index, and the label boxes, from Kaggle.
2. Run `cli segment`, `cli crops`, `cli candidates`, `cli pilot`.
3. Review the 400-case pilot sheet by hand.

That review answers the only question that matters: **of the cases where the
correct answer sits at rank 2, how many show a difference in the bottle that a
person can actually see?** If that share is small, a whole-bottle encoder is the
wrong investment and the effort belongs in an OCR or label-zone check instead.

Separately, and regardless of the outcome: the 45k renders cannot teach
bottle-level discrimination, so the photographic collection in section 6 is on
the critical path either way.

**No improvement is predicted here.** Claiming one before training and an
independent test would be a guess.
