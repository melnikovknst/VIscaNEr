"""Relabel real_photos_v4: review sheets, catalog search, and an auditable decision log.

Every decision is appended to evaluation/relabel/decisions.jsonl with who made
it (the human reviewer or Claude) and why; the latest decision per photo wins.
`export` merges them into evaluation/real_photos_v4_relabeled.csv - the
original labels file is never edited.

Decision types
  keep            original label is right
  fix             the wine is a different catalog entry        (accepted = [slug])
  multi           several bottles; each listed one is a valid answer (accepted = [...])
  not_in_catalog  the photographed wine is not in the catalog  (accepted = [])
  wrong_unknown   the label is wrong and the true wine was not identified, but
                  the model's answer is known to be wrong: scored as a miss
  remove          no recognisable target: people, glasses, grapes, crowds, junk
  unsure          cannot be decided from the photo; excluded from scoring

    python -m scripts.relabel_tools sheet q-0008 q-0016 ...
    python -m scripts.relabel_tools search "совиньон блан" [--winery Mantra]
    python -m scripts.relabel_tools export
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "datasets/real_photos_v4/labels.csv"
IMAGES = ROOT / "datasets/real_photos_v4/queries"
REFS = ROOT / "datasets/wine-scanner/data/refs/rgb"
CATALOG = ROOT / "datasets/wine-scanner/data/catalog.csv"
RELABEL = ROOT / "evaluation/relabel"
DECISIONS = RELABEL / "decisions.jsonl"
PREDICTIONS = RELABEL / "predictions.json"      # model top-5 + label box for every photo
EXPORT = ROOT / "evaluation/real_photos_v4_relabeled.csv"
SHEETS = ROOT / "runs/relabel_sheets"
BOTTLE_RUN = ROOT / "runs/inference/real_photos_v4_bottle_pipeline.json"


def _bottle_top() -> dict[str, list[tuple[str, float]]]:
    if not BOTTLE_RUN.is_file():
        return {}
    data = json.loads(BOTTLE_RUN.read_text(encoding="utf-8"))
    return {Path(r["source"]).stem: [(p["wine_slug"], p["similarity"]) for p in r["predictions"]]
            for r in data["results"]}


BOTTLE_TOP = _bottle_top()

DECISION_TYPES = {"keep", "fix", "multi", "not_in_catalog", "wrong_unknown", "remove", "unsure"}


def labels() -> dict[str, dict]:
    return {r["query_id"]: r for r in csv.DictReader(LABELS.open(encoding="utf-8-sig"))}


def catalog() -> dict[str, dict]:
    return {r["slug"]: r for r in csv.DictReader(CATALOG.open(encoding="utf-8"))}


# ------------------------------------------------------------------ decisions
def record(query_id: str, decision: str, accepted: list[str] | None = None, *,
           reviewer: str, note: str = "") -> None:
    if decision not in DECISION_TYPES:
        raise ValueError(f"unknown decision {decision!r}")
    accepted = accepted or []
    known = catalog()
    for slug in accepted:
        if slug not in known:
            raise ValueError(f"{query_id}: {slug!r} is not a catalog slug")
    if decision in {"fix", "multi"} and not accepted:
        raise ValueError(f"{query_id}: {decision} needs at least one accepted slug")
    RELABEL.mkdir(parents=True, exist_ok=True)
    with DECISIONS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "query_id": query_id, "decision": decision, "accepted": accepted,
            "reviewer": reviewer, "note": note,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, ensure_ascii=False) + "\n")


def decisions() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if DECISIONS.is_file():
        for line in DECISIONS.open(encoding="utf-8"):
            if line.strip():
                item = json.loads(line)
                latest[item["query_id"]] = item
    return latest


def accepted_answers(query_id: str, row: dict, decision: dict | None) -> tuple[list[str] | None, bool | None]:
    """(slugs that count as correct, is the wine in the catalog).

    (None, None) means the photo is excluded from scoring. wrong_unknown is an
    in-catalog photo with no acceptable answer known, so any answer misses.
    """
    kind = "keep" if decision is None else decision["decision"]
    if kind == "keep":
        return ([row["slug"]], True) if row["in_catalog"] == "yes" else ([], False)
    if kind in {"fix", "multi"}:
        return decision["accepted"], True
    if kind == "wrong_unknown":
        return [], True
    if kind == "not_in_catalog":
        return [], False
    return None, None                            # remove / unsure


def export() -> Path:
    rows, latest = labels(), decisions()
    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    with EXPORT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, ["query_id", "image_path", "original_slug", "original_in_catalog",
                                         "status", "scored", "in_catalog", "accepted_slugs", "reviewer", "note"])
        writer.writeheader()
        for qid, row in rows.items():
            decision = latest.get(qid)
            answers, in_catalog = accepted_answers(qid, row, decision)
            status = ("unreviewed" if decision is None else decision["decision"])
            writer.writerow({
                "query_id": qid, "image_path": row["image_path"],
                "original_slug": row["slug"], "original_in_catalog": row["in_catalog"],
                "status": status, "scored": answers is not None,
                "in_catalog": "" if in_catalog is None else in_catalog,
                "accepted_slugs": "" if answers is None else ";".join(answers),
                "reviewer": decision["reviewer"] if decision else "",
                "note": decision["note"] if decision else "",
            })
    return EXPORT


# -------------------------------------------------------------------- search
def normalise(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold().replace("ё", "е")).strip()


def search(query: str, winery: str = "", limit: int = 25) -> list[dict]:
    words = normalise(query).split()
    winery_words = normalise(winery).split()
    hits = []
    for slug, row in catalog().items():
        haystack = normalise(f"{row['name']} {row['winery']} {slug}")
        if all(w in haystack for w in words) and all(w in normalise(row["winery"] + " " + slug) for w in winery_words):
            hits.append({"slug": slug, "name": row["name"], "winery": row["winery"], "color": row["category"]})
    return hits[:limit]


# -------------------------------------------------------------------- sheets
def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines[:4]


def sheet(query_ids: list[str], out: Path, extra: dict[str, list[str]] | None = None) -> Path:
    """One row per photo: photo, label crop, labeled truth, model top-5 (+ extra slugs)."""
    rows, cat = labels(), catalog()
    preds = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    f_big, f_small = _font(20), _font(15)
    tile, photo_h = 190, 430
    row_h = photo_h + 20
    width = 20 + 360 + 20 + 330 + 20 + 7 * (tile + 12)
    canvas = Image.new("RGB", (width, row_h * len(query_ids) + 10), "white")
    draw = ImageDraw.Draw(canvas)
    for index, qid in enumerate(query_ids):
        y = index * row_h + 10
        row, pred = rows[qid], preds[qid]
        with Image.open(IMAGES / row["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((2048, 2048))
        full = image.copy()
        if pred.get("label_box"):
            ImageDraw.Draw(image).rectangle(pred["label_box"], outline=(255, 40, 40), width=max(3, image.width // 150))
        image.thumbnail((360, photo_h))
        canvas.paste(image, (20, y))
        draw.text((20, y + image.height + 2), qid, fill=(0, 0, 0), font=f_big)
        # Label region at high resolution, so small print is legible.
        x = 400
        if pred.get("label_box"):
            x1, y1, x2, y2 = pred["label_box"]
            pad_x, pad_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
            crop = full.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(full.width, x2 + pad_x), min(full.height, y2 + pad_y)))
            crop.thumbnail((330, photo_h))
            canvas.paste(crop, (x, y))
        else:
            draw.text((x, y + 10), "этикетка не найдена", fill=(160, 0, 0), font=f_big)
        x = 400 + 350
        truth = row["slug"] if row["in_catalog"] == "yes" else None
        # Hints from both models: label-crop DINO ("Э") and the colleagues'
        # whole-bottle pipeline ("Б"), interleaved and de-duplicated.
        bottle = BOTTLE_TOP.get(qid, [])
        hints, seen = [], {truth}
        for pair in zip([("Э", s["slug"], s["score"]) for s in pred["top5"]], [("Б", s, v) for s, v in bottle]):
            for tag, slug, score in pair:
                if slug not in seen:
                    seen.add(slug)
                    hints.append((f"{tag} {score:.2f}", slug))
        slugs = [("РАЗМЕТКА", truth)] + [("доп.", s) for s in (extra or {}).get(qid, [])] + hints
        for caption, slug in slugs[:7]:
            if slug:
                ref = Image.open(REFS / f"{slug}.jpg").convert("RGB")
                ref.thumbnail((tile, tile))
                canvas.paste(ref, (x, y))
                colour = (0, 90, 200) if caption == "РАЗМЕТКА" else (0, 0, 0)
                draw.text((x, y + tile + 2), caption, fill=colour, font=f_big)
                text = f"{cat[slug]['name']} | {cat[slug]['winery']}"
                for n, line in enumerate(_wrap(draw, text, f_small, tile)):
                    draw.text((x, y + tile + 28 + n * 18), line, fill=(40, 40, 40), font=f_small)
            else:
                draw.text((x, y + 10), "РАЗМЕТКА: нет в каталоге", fill=(0, 90, 200), font=f_small)
            x += tile + 12
        draw.line((0, y + row_h - 6, width, y + row_h - 6), fill=(200, 200, 200), width=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=88)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sheet"); s.add_argument("ids", nargs="+"); s.add_argument("--out", default=str(SHEETS / "sheet.jpg"))
    q = sub.add_parser("search"); q.add_argument("query"); q.add_argument("--winery", default="")
    sub.add_parser("export")
    args = parser.parse_args()
    if args.command == "sheet":
        print(sheet(args.ids, Path(args.out)))
    elif args.command == "search":
        for hit in search(args.query, args.winery):
            print(f"{hit['slug']}  |  {hit['name']}  |  {hit['winery']}  |  {hit['color']}")
    else:
        print(export())


if __name__ == "__main__":
    sys.exit(main())
