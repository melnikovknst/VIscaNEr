#!/usr/bin/env python
"""Build a YOLO dataset of WHOLE bottles - base to capsule - with exact boxes.

The boxes come from the generator, not from a detector: the scene renderer is
instrumented so the silhouette of every pasted bottle is known exactly, then
occlusion between bottles is resolved in paste order. No box is ever guessed.

Why the images are re-rendered instead of annotated in place
-----------------------------------------------------------
The shipped 45k trainset frames can no longer be reproduced bit-for-bit in this
environment - only about 6% of a sampled 80 match, because the JPEG encoder and
imaging libraries have moved on since that build. Annotating a shipped JPEG with
a box measured on a *re-rendered* frame would silently misplace boxes, so this
builder ships the frames it actually measured. `build_target_aligned_crops.py`
takes the same approach for the same reason.

What counts as a whole bottle
-----------------------------
An instance is counted towards `--target-instances` only when it is
  * not cut by the frame edge (the full silhouette, base to capsule, is inside),
  * not mostly hidden behind another bottle (visibility >= --min-visibility).

Every other bottle in a kept frame is still annotated. Leaving a visible bottle
unboxed would teach a detector that it is background.

Usage
-----
    python build_bottle_yolo_dataset.py --target-instances 1000 --padding 0.10
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
import time
import zipfile
import zlib
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SCANNER_ROOT = PROJECT_ROOT / "datasets" / "wine-scanner"

CLASS_ID = 0
CLASS_NAME = "bottle"

MANIFEST_FIELDS = [
    "image", "split", "instance_index", "role", "wine_slug",
    "render_mode", "render_background", "render_seed",
    "tight_x1", "tight_y1", "tight_x2", "tight_y2",
    "padded_x1", "padded_y1", "padded_x2", "padded_y2",
    "yolo_cx", "yolo_cy", "yolo_w", "yolo_h",
    "image_width", "image_height",
    "visibility", "truncated_by_frame", "whole_bottle",
]


# --------------------------------------------------------------- tracking
@dataclass
class TrackedBottle:
    """One pasted bottle: its own silhouette and where it came from."""

    mask: np.ndarray                    # own alpha, clipped to the canvas
    own_pixels: int                     # alpha pixels of the full object
    order: int                          # paste order; later pastes occlude earlier
    slug: str = ""
    role: str = "neighbour"
    extra: dict[str, Any] = field(default_factory=dict)


class SceneTracker(AbstractContextManager["SceneTracker"]):
    """Instrument the generator without editing a line of its source.

    `paste` and `perspective` are module-level functions in `scanner.augment`,
    so replacing the module attributes intercepts every bottle the scene
    composer draws. `FieldAugmenter._ref` is wrapped as well: each neighbour
    loads its reference immediately before being pasted, so the call order
    identifies which wine each neighbour is.
    """

    def __init__(self, augment_module: Any, augmenter_class: Any) -> None:
        self.module = augment_module
        self.augmenter_class = augmenter_class
        self._paste_original = augment_module.paste
        self._perspective_original = augment_module.perspective
        self._ref_original = augmenter_class._ref
        self.bottles: list[TrackedBottle] = []
        self.ref_calls: list[str] = []

    def __enter__(self) -> "SceneTracker":
        self.module.paste = self._paste
        self.module.perspective = self._perspective
        self.augmenter_class._ref = self._ref
        return self

    def __exit__(self, *exc: Any) -> None:
        self.module.paste = self._paste_original
        self.module.perspective = self._perspective_original
        self.augmenter_class._ref = self._ref_original

    def reset(self) -> None:
        self.bottles = []
        self.ref_calls = []

    def _ref(self, path: Path) -> np.ndarray:
        self.ref_calls.append(Path(path).stem)
        return self._ref_original(path)

    def _paste(self, canvas: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
        self._paste_original(canvas, rgba, x, y)
        height, width = canvas.shape[:2]
        object_height, object_width = rgba.shape[:2]
        alpha = rgba[..., 3] > 32
        mask = np.zeros((height, width), np.uint8)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + object_width, width), min(y + object_height, height)
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = alpha[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.uint8) * 255
        self.bottles.append(TrackedBottle(
            mask=mask,
            own_pixels=int(alpha.sum()),
            order=len(self.bottles),
            extra={"intended_box": (x, y, x + object_width, y + object_height)},
        ))

    def _perspective(self, image: np.ndarray, rng: Any, max_shift: float) -> np.ndarray:
        """The generator transform, replayed on every tracked mask.

        Reimplemented rather than called, because the masks have to go through
        exactly the same matrix - and the matrix is built from `rng` draws that
        can only be consumed once.
        """
        height, width = image.shape[:2]
        source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        jitter = [[rng.uniform(-max_shift, max_shift) * width,
                   rng.uniform(-max_shift, max_shift) * height] for _ in range(4)]
        destination = source + np.float32(jitter)
        angle = rng.uniform(-10, 10)
        rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        destination = cv2.transform(destination[None], rotation)[0]
        matrix = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
        for bottle in self.bottles:
            bottle.mask = cv2.warpPerspective(
                bottle.mask, matrix, (width, height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            )
        return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)

    def render(self, augmenter: Any, reference: np.ndarray, seed: int, target_slug: str):
        self.reset()
        image, params = augmenter(reference, seed)
        if not self.bottles:
            raise RuntimeError("the generator pasted no bottles")
        # Every implemented mode pastes neighbours first and the requested
        # bottle last; the neighbours line up with the reference loads in order.
        self.bottles[-1].role = "target"
        self.bottles[-1].slug = target_slug
        for bottle, slug in zip(self.bottles[:-1], self.ref_calls):
            bottle.slug = slug
        return image, params, self.bottles


# ------------------------------------------------------- non-bottle filter
# The catalog is not only bottles: bag-in-box cartons, kegs and gift packs are
# separate identities with their own reference cut-outs. Boxing a carton as a
# `bottle` is simply a wrong label, so those references are kept out of the
# scene entirely - as targets AND as neighbours, since neighbours are drawn at
# random from the reference pool.
PACKAGING_TOKENS = ("beg-in-boks", "bag-in-box", "tetra", "banka", "keg", "kega")


def reference_aspect(path: Path, alpha_threshold: int = 16) -> float:
    """Height / width of the reference silhouette. A wine bottle is tall."""
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        rgba = np.asarray(ImageOps.exif_transpose(handle).convert("RGBA"))
    alpha = rgba[..., 3]
    if (alpha < 250).mean() < 0.01:                 # stored without transparency
        mask = (np.abs(rgba[..., :3].astype(np.int16) - 255).max(axis=2) > 12)
    else:
        mask = alpha > alpha_threshold
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0
    return float((ys.max() - ys.min() + 1) / max(1, xs.max() - xs.min() + 1))


def bottle_shaped_references(wines: list[Any], min_aspect: float, cache_path: Path):
    """Split the catalog into bottle-shaped references and everything else."""
    cache: dict[str, float] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}

    keep, rejected = [], []
    dirty = False
    for wine in wines:
        token = next((t for t in PACKAGING_TOKENS if t in wine.slug), None)
        aspect = cache.get(wine.slug)
        if aspect is None:
            aspect = reference_aspect(wine.ref_rgba)
            cache[wine.slug] = aspect
            dirty = True
        if token is not None:
            rejected.append((wine.slug, f"packaging_token:{token}", round(aspect, 3)))
        elif aspect < min_aspect:
            rejected.append((wine.slug, "silhouette_too_squat", round(aspect, 3)))
        else:
            keep.append(wine)
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return keep, rejected


# --------------------------------------------------------------- geometry
def resolve_occlusion(bottles: list[TrackedBottle]) -> list[np.ndarray]:
    """Visible silhouette per bottle: its own mask minus everything drawn later."""
    visible: list[np.ndarray] = []
    for index, bottle in enumerate(bottles):
        own = bottle.mask > 0
        for later in bottles[index + 1:]:
            own = own & ~(later.mask > 0)
        visible.append(own)
    return visible


def tight_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_box(box: tuple[int, int, int, int], width: int, height: int, padding: float):
    """Expand by `padding` of the box size on each side, clipped to the image.

    Matches `padded_box` in build_target_aligned_crops.py, so a 10% padding
    means the same thing in both datasets.
    """
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    return (
        max(0, int(np.floor(x1 - box_width * padding))),
        max(0, int(np.floor(y1 - box_height * padding))),
        min(width, int(np.ceil(x2 + box_width * padding))),
        min(height, int(np.ceil(y2 + box_height * padding))),
    )


def is_truncated(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    intended_box: tuple[int, int, int, int],
    *,
    edge_tolerance: int,
) -> bool:
    """Is any part of this bottle missing from the frame?

    Two independent signals, because neither alone is sufficient:

    * The paste box. `paste` clips to the canvas, so a bottle placed partly
      outside loses those pixels permanently - true even if a later perspective
      warp pulls the remainder back inside.
    * The silhouette reaching a frame edge. This catches truncation introduced
      by the warp itself.

    The edge test needs a tolerance rather than an exact touch: `warpPerspective`
    writes with BORDER_CONSTANT and resamples, so a genuinely cut bottle often
    lands a pixel or two short of the border. An earlier version tested only
    column/row 0 and passed 37 cut bottles as whole.
    """
    height, width = mask.shape[:2]
    ix1, iy1, ix2, iy2 = intended_box
    if ix1 < 0 or iy1 < 0 or ix2 > width or iy2 > height:
        return True
    tolerance = max(1, edge_tolerance)
    x1, y1, x2, y2 = box
    return bool(x1 <= tolerance or y1 <= tolerance
                or x2 >= width - tolerance or y2 >= height - tolerance)


def to_yolo(box: tuple[int, int, int, int], width: int, height: int):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


# ------------------------------------------------------------------ build
def build(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(SCANNER_ROOT))
    from scanner import augment as augment_module          # noqa: E402
    from scanner import config as scanner_config           # noqa: E402
    from scanner.augment import FieldAugmenter, list_images  # noqa: E402
    from scanner.catalog import load_catalog               # noqa: E402
    from scanner.normalize import load_rgba                # noqa: E402

    trainset = SCANNER_ROOT / "data" / "trainset"
    manifest_path = trainset / "manifest.csv"
    for required in (manifest_path, scanner_config.CATALOG):
        if not required.exists():
            raise FileNotFoundError(f"required input missing: {required}")

    all_wines = load_catalog()
    wines, rejected_refs = bottle_shaped_references(
        all_wines, args.min_reference_aspect,
        Path(args.output).parent / ".reference_aspect_cache.json",
    )
    print(f"references: {len(wines)} bottle-shaped, {len(rejected_refs)} excluded "
          f"(bag-in-box, kegs and other non-bottle packaging)")
    for slug, reason, aspect in rejected_refs[:8]:
        print(f"    excluded {slug[:56]:58s} {reason} aspect={aspect}")
    if len(rejected_refs) > 8:
        print(f"    ... and {len(rejected_refs) - 8} more (listed in build.json)")
    by_slug = {w.slug: w for w in wines}
    backgrounds = list_images(scanner_config.BACKGROUNDS) if scanner_config.BACKGROUNDS.is_dir() else []
    if not backgrounds:
        # Without the upload folder the generator falls back to synthetic
        # backdrops. That changes the look of the scenes, so it is recorded
        # rather than silently accepted.
        print("WARNING: background folder not found; scenes will use synthetic backdrops only")
    augmenter = FieldAugmenter([w.ref_rgba for w in wines], backgrounds, args.long_side)

    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    random.Random(args.seed).shuffle(rows)

    output = Path(args.output)
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_rng = random.Random(args.seed + 1)
    instances: list[dict[str, Any]] = []
    whole_count = 0
    frames_kept = 0
    frames_tried = 0
    skipped: dict[str, int] = {}
    started = time.monotonic()

    with SceneTracker(augment_module, FieldAugmenter) as tracker:
        for row in rows:
            if whole_count >= args.target_instances or frames_tried >= args.max_frames:
                break
            slug, split_name, view = row["slug"], row["split"], int(row["view"])
            if slug not in by_slug:
                skipped["not_a_bottle"] = skipped.get("not_a_bottle", 0) + 1
                continue
            frames_tried += 1
            seed = zlib.crc32(f"{slug}:{split_name}:{view}".encode())

            try:
                image, params, bottles = tracker.render(
                    augmenter, load_rgba(by_slug[slug].ref_rgba), seed, slug
                )
            except Exception as error:  # noqa: BLE001 - one bad scene must not stop the run
                skipped["render_error"] = skipped.get("render_error", 0) + 1
                if args.verbose:
                    print(f"  skip {slug}/{view}: {type(error).__name__}: {error}")
                continue

            height, width = image.shape[:2]
            visible_masks = resolve_occlusion(bottles)
            frame_rows: list[dict[str, Any]] = []
            frame_whole = 0

            for bottle, visible in zip(bottles, visible_masks):
                box = tight_box(visible)
                if box is None:
                    continue
                visible_pixels = int(visible.sum())
                own_visible = int((bottle.mask > 0).sum())
                if own_visible == 0:
                    continue
                # Visibility is measured against the bottle as drawn on this
                # canvas, so a bottle half off-frame is not punished twice.
                visibility = visible_pixels / own_visible
                truncated = is_truncated(
                    bottle.mask, box, bottle.extra["intended_box"],
                    edge_tolerance=args.edge_tolerance,
                )
                if (box[2] - box[0]) < args.min_box_px or (box[3] - box[1]) < args.min_box_px:
                    continue
                whole = (not truncated) and visibility >= args.min_visibility
                padded = pad_box(box, width, height, args.padding)
                cx, cy, bw, bh = to_yolo(padded, width, height)
                frame_rows.append({
                    "instance_index": bottle.order,
                    "role": bottle.role,
                    "wine_slug": bottle.slug,
                    "render_mode": params.get("mode", ""),
                    "render_background": params.get("background", ""),
                    "render_seed": seed,
                    "tight_x1": box[0], "tight_y1": box[1], "tight_x2": box[2], "tight_y2": box[3],
                    "padded_x1": padded[0], "padded_y1": padded[1],
                    "padded_x2": padded[2], "padded_y2": padded[3],
                    "yolo_cx": round(cx, 6), "yolo_cy": round(cy, 6),
                    "yolo_w": round(bw, 6), "yolo_h": round(bh, 6),
                    "image_width": width, "image_height": height,
                    "visibility": round(visibility, 4),
                    "truncated_by_frame": truncated,
                    "whole_bottle": whole,
                })
                frame_whole += int(whole)

            if frame_whole == 0:
                # A frame with no whole bottle teaches nothing about what a
                # whole bottle looks like; it is not worth its disk space here.
                skipped["no_whole_bottle"] = skipped.get("no_whole_bottle", 0) + 1
                continue

            name = f"{frames_kept:05d}_{slug[:48]}_{view}"
            split = "train" if split_rng.random() < args.train_fraction else "val"
            Image.fromarray(image).convert("RGB").save(
                output / "images" / split / f"{name}.jpg", quality=args.quality
            )
            label_lines = [
                f"{CLASS_ID} {r['yolo_cx']:.6f} {r['yolo_cy']:.6f} {r['yolo_w']:.6f} {r['yolo_h']:.6f}"
                for r in frame_rows
            ]
            (output / "labels" / split / f"{name}.txt").write_text(
                "\n".join(label_lines) + "\n", encoding="utf-8"
            )
            for record in frame_rows:
                record["image"] = f"images/{split}/{name}.jpg"
                record["split"] = split
                instances.append(record)

            frames_kept += 1
            whole_count += frame_whole
            if frames_kept % 25 == 0:
                elapsed = time.monotonic() - started
                print(f"  {frames_kept} frames, {whole_count}/{args.target_instances} whole bottles "
                      f"({frames_tried} rendered, {elapsed:.0f}s)", flush=True)

    with (output / "bottles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(instances)

    (output / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: bottle\n",
        encoding="utf-8",
    )

    counts = {
        "frames_rendered": frames_tried,
        "frames_kept": frames_kept,
        "frames_train": sum(1 for r in instances if r["split"] == "train") and len(
            {r["image"] for r in instances if r["split"] == "train"}),
        "frames_val": len({r["image"] for r in instances if r["split"] == "val"}),
        "boxes_total": len(instances),
        "whole_bottles": sum(1 for r in instances if r["whole_bottle"]),
        "truncated_by_frame": sum(1 for r in instances if r["truncated_by_frame"]),
        "partially_occluded": sum(1 for r in instances if r["visibility"] < 0.999),
        "identities": len({r["wine_slug"] for r in instances if r["wine_slug"]}),
        "skipped": skipped,
        "by_render_mode": {},
    }
    for record in instances:
        key = record["render_mode"] or "?"
        counts["by_render_mode"][key] = counts["by_render_mode"].get(key, 0) + 1

    build_record = {
        "generator": "scanner.augment.FieldAugmenter (instrumented, not modified)",
        "boxes": "exact generator silhouettes, occlusion resolved in paste order",
        "images": "re-rendered by this build; the shipped 45k JPEGs are NOT reproducible here",
        "padding": args.padding,
        "long_side": args.long_side,
        "jpeg_quality": args.quality,
        "min_visibility": args.min_visibility,
        "edge_tolerance_px": args.edge_tolerance,
        "min_box_px": args.min_box_px,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "backgrounds_available": len(backgrounds),
        "references_used": len(wines),
        "references_excluded_as_non_bottle": [
            {"slug": slug, "reason": reason, "silhouette_aspect": aspect}
            for slug, reason, aspect in rejected_refs
        ],
        "min_reference_aspect": args.min_reference_aspect,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "counts": counts,
    }
    (output / "build.json").write_text(
        json.dumps(build_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(_readme(args, counts), encoding="utf-8")
    return build_record


def _readme(args: argparse.Namespace, counts: dict[str, Any]) -> str:
    return f"""# wine_bottles_yolo_1000

YOLO dataset of **whole bottles - base to capsule** - for training a bottle
detector/segmenter.

- `{counts['whole_bottles']}` whole-bottle instances across `{counts['frames_kept']}` frames
- `{counts['boxes_total']}` boxes in total (every bottle in a kept frame is annotated)
- one class: `0: bottle`
- boxes carry **{args.padding:.0%} padding** on each side, clipped to the frame

## Layout

```
data.yaml                 ultralytics config
images/train, images/val  frames
labels/train, labels/val  YOLO boxes: class cx cy w h, normalised
bottles.csv               per-instance record: tight box, padded box,
                          visibility, truncation, wine_slug, render mode
build.json                exact build settings and library versions
```

## Where the boxes come from

The scene generator (`scanner.augment`) is instrumented so the silhouette of
every pasted bottle is captured exactly, then occlusion between bottles is
resolved in paste order. Boxes are measured, never predicted, so there is no
detector error in this data. The generator source is not modified.

## What "whole bottle" means

`whole_bottle` is true when the silhouette is fully inside the frame and at
least {args.min_visibility:.0%} of it is unoccluded by another bottle. Only those
count towards the headline number. Other bottles in the same frame are still
boxed - leaving a visible bottle unlabelled would train the detector to treat
bottles as background.

## Two caveats worth knowing

- **The frames are re-rendered, not taken from the shipped 45k trainset.** Those
  JPEGs are no longer reproducible bit-for-bit in this environment (about 6% of
  a sample of 80 matched), so annotating them with boxes measured on a
  re-render would misplace the boxes. The frames measured here are the frames
  shipped here.
- **The shop shelf rail is drawn over bottle bottoms after they are pasted.** A
  box therefore runs to the true base of the bottle even where a few pixels of
  it sit behind the rail. That is intentional for "base to capsule"; a detector
  trained for strictly visible extents would want this recomputed.

## Provenance

Bottles are the catalog cut-outs, so every instance carries its `wine_slug`
(`{counts['identities']}` distinct identities here). These are rendered scenes,
not photographs - see `bottle_reranker/reports/FINDINGS.md`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-instances", type=int, default=1000,
                        help="how many WHOLE bottles to collect (default: 1000)")
    parser.add_argument("--padding", type=float, default=0.10,
                        help="box padding per side, as a fraction of box size")
    parser.add_argument("--min-visibility", type=float, default=0.80,
                        help="unoccluded fraction required to count as whole")
    parser.add_argument("--edge-tolerance", type=int, default=3,
                        help="a silhouette this close to a frame edge counts as cut")
    parser.add_argument("--min-box-px", type=int, default=12,
                        help="drop boxes thinner than this in either dimension")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--max-frames", type=int, default=6000,
                        help="safety cap on frames rendered")
    parser.add_argument("--min-reference-aspect", type=float, default=1.8,
                        help="minimum silhouette height/width for a reference to "
                             "count as a bottle; excludes bag-in-box and kegs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="datasets/wine_bottles_yolo_1000")
    parser.add_argument("--archive", default="datasets/wine_bottles_yolo_1000.zip",
                        help="zip the result here; empty string to skip")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build(args)
    counts = record["counts"]
    print(json.dumps(counts, ensure_ascii=False, indent=2))

    if args.archive:
        output = Path(args.output)
        archive = Path(args.archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    handle.write(path, Path(output.name) / path.relative_to(output))
        print(f"archive: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
