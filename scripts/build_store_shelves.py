"""Turn store-shelf photos into one "customer shot" per wine, plus labels.

The shelf photos show 3-30 bottles each, so a whole photo has no single answer.
Each wine with a readable label was marked by hand with a point on its label and
its approximate width (datasets/store_shelves_v1/points.csv, percent of the photo).
Around every point this script cuts a 3:4 frame the way a person would shoot it
with the phone: the label at the centre - under the scanner's crosshair - and
the neighbouring bottles visible at the sides. The frame is centred on the point
without shrinking near the photo edge, so the crosshair always stays on the
marked bottle. Areas outside the source photo get a neutral grey background;
padding cannot restore parts of a bottle missing from the original photo.

Outputs (images are shared through Git LFS):
  datasets/store_shelves_v1/queries/s-NNN.jpg
  datasets/store_shelves_v1/labels.csv

    python -m scripts.build_store_shelves
    python infer_wine.py datasets/store_shelves_v1/queries --top-k 5 \
        --output runs/inference/store_shelves_v1_bottle_pipeline.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/store_shelves_v1/source"
QUERIES = ROOT / "datasets/store_shelves_v1/queries"
POINTS = ROOT / "datasets/store_shelves_v1/points.csv"
LABELS = ROOT / "datasets/store_shelves_v1/labels.csv"
REVIEW = ROOT / "datasets/store_shelves_v1/review.json"

# The marked bottle fills about this share of the frame width, as in a phone shot
# of one bottle on a shelf.
BOTTLE_SHARE = 0.45
PADDING_COLOR = (127, 127, 127)


def frame(image: Image.Image, x: float, y: float, bottle_w: float) -> Image.Image:
    if not (0 <= x <= 100 and 0 <= y <= 100 and 0 < bottle_w <= 100):
        raise ValueError("Expected x/y in [0, 100] and bottle width in (0, 100]")
    W, H = image.size
    cx, cy = x / 100 * W, y / 100 * H
    unit = max(1, round(min(W, bottle_w / 100 * W / BOTTLE_SHARE) / 3))
    width, height = 3 * unit, 4 * unit
    left, top = round(cx - width / 2), round(cy - height / 2)
    # Paste only real pixels. No reflection or repeated pixels that might create
    # a second label; keep the marked point within half a pixel of the centre.
    box = (max(0, left), max(0, top), min(W, left + width), min(H, top + height))
    shot = Image.new("RGB", (width, height), PADDING_COLOR)
    shot.paste(image.crop(box), (box[0] - left, box[1] - top))
    return shot


def main() -> None:
    QUERIES.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(POINTS.open(encoding="utf-8")))
    reviews = json.loads(REVIEW.read_text(encoding="utf-8"))
    query_ids = {f"s-{n:03d}" for n in range(1, len(rows) + 1)}
    if set(reviews) - query_ids:
        raise ValueError("Review notes reference unknown query IDs")
    cache: dict[str, Image.Image] = {}
    out = []
    for n, row in enumerate(rows, 1):
        src = row["source"]
        if src not in cache:
            with Image.open(SOURCE / src) as im:
                cache[src] = ImageOps.exif_transpose(im).convert("RGB")
        shot = frame(cache[src], float(row["x"]), float(row["y"]), float(row["w"]))
        name = f"s-{n:03d}.jpg"
        shot.save(QUERIES / name, quality=92)
        status = row["status"]
        needs_review = f"s-{n:03d}" in reviews
        training_use = ("exclude" if status not in {"ok", "notcat"} else "review" if needs_review
                        else "catalog" if status == "ok" else "out_of_catalog")
        out.append({"query_id": f"s-{n:03d}", "image_path": name, "status": status,
                    "scored": status in {"ok", "notcat"},
                    "in_catalog": {"ok": True, "notcat": False}.get(status, ""),
                    "accepted_slugs": row["accepted_slugs"], "note": row["note"],
                    "source": src, "x": row["x"], "y": row["y"],
                    "needs_review": needs_review, "training_use": training_use})
    with LABELS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"{len(out)} shots -> {QUERIES}\n{LABELS}")


if __name__ == "__main__":
    main()
