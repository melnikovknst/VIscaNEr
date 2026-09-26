#!/usr/bin/env python3
"""Generate a local HTML audit of joint YOLO bottle and label crops."""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from .infer import (
    DEFAULT_MODEL,
    choose_device,
    padded_crop,
    pair_bottle,
    predict_detections,
    select_label_candidates,
)
from .train import DEFAULT_DATASET, PROJECT_ROOT


DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "joint_yolo" / "crop_audit"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
COLORS = {
    "gt_bottle": "#18a558",
    "gt_label": "#16a3d8",
    "pred_bottle": "#ff9f1c",
    "pred_label": "#ef476f",
    "crosshair": "#ffd60a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--bottle-crop-threshold", type=float, default=0.75)
    parser.add_argument("--ambiguity-margin", type=float, default=0.06)
    parser.add_argument(
        "--include-id",
        action="append",
        default=[],
        help="Always include this image stem in the HTML audit (repeatable).",
    )
    return parser.parse_args()


def read_ground_truth(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        class_id, cx, cy, box_width, box_height = raw.split()
        cx, cy, box_width, box_height = map(float, (cx, cy, box_width, box_height))
        rows.append(
            {
                "class_id": int(class_id),
                "box": (
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ),
            }
        )
    return rows


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: str,
    label: str,
    width: int,
) -> None:
    coordinates = tuple(round(value) for value in box)
    draw.rectangle(coordinates, outline=color, width=width)
    text_font = font(max(17, width * 4))
    bounds = draw.textbbox((0, 0), label, font=text_font, stroke_width=1)
    text_width = bounds[2] - bounds[0] + 10
    text_height = bounds[3] - bounds[1] + 8
    top = max(0, coordinates[1] - text_height)
    draw.rectangle(
        (coordinates[0], top, coordinates[0] + text_width, top + text_height),
        fill=color,
    )
    draw.text((coordinates[0] + 5, top + 3), label, fill="black", font=text_font)


def save_web_image(image: Image.Image, path: Path, maximum: int) -> None:
    result = image.convert("RGB")
    result.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path, quality=88, optimize=True)


def selected_confidences(record: dict[str, Any]) -> tuple[float, float]:
    if not record["selected"]:
        return -1.0, -1.0
    first = record["selected"][0]
    bottle_confidence = first["bottle_confidence"]
    return first["label_confidence"], -1.0 if bottle_confidence is None else bottle_confidence


def select_mixed(
    records: list[dict[str, Any]],
    count: int,
    seed: int,
    include_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    count = min(count, len(records))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(items: list[dict[str, Any]], limit: int) -> None:
        for item in items:
            if len(selected) >= count or limit == 0:
                break
            if item["id"] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item["id"])
            limit -= 1

    requested = set(include_ids or [])
    missing = sorted(requested.difference(item["id"] for item in records))
    if missing:
        raise ValueError(f"Requested audit IDs are absent from the split: {missing}")
    add([item for item in records if item["id"] in requested], len(requested))

    difficult = sorted(
        records,
        key=lambda item: (
            0 if item["selection_mode"] != "single_pair" else 1,
            min(selected_confidences(item)),
        ),
    )
    add(difficult, max(6, count // 2))
    add(sorted(records, key=lambda item: selected_confidences(item)[0]), max(4, count // 4))
    remainder = [item for item in records if item["id"] not in selected_ids]
    random.Random(seed).shuffle(remainder)
    add(remainder, count - len(selected))
    return selected


def infer_record(
    model: YOLO,
    image_path: Path,
    label_path: Path,
    device: str | int,
    confidence: float,
    bottle_crop_threshold: float,
    ambiguity_margin: float,
) -> dict[str, Any]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    detections = predict_detections(model, image, device, confidence)
    bottles = [item for item in detections if item["class_id"] == 0]
    labels = [item for item in detections if item["class_id"] == 1]
    crosshair = (image.width / 2, image.height / 2)
    selected_labels, ambiguous = select_label_candidates(
        labels,
        crosshair,
        math.hypot(image.width, image.height),
        ambiguity_margin,
        image_width=image.width,
    )
    selected = []
    for label in selected_labels:
        bottle = pair_bottle(label, bottles)
        if bottle is None:
            mode = "original_no_bottle"
        elif bottle["confidence"] < bottle_crop_threshold:
            mode = "original_low_confidence"
        else:
            mode = "yolo_bottle_crop"
        selected.append(
            {
                "label": label,
                "bottle": bottle,
                "label_confidence": label["confidence"],
                "bottle_confidence": bottle["confidence"] if bottle else None,
                "bottle_image_mode": mode,
            }
        )
    if not selected:
        selection_mode = "no_label_detection"
    elif ambiguous:
        selection_mode = "ambiguous_two_pairs"
    else:
        selection_mode = "single_pair"
    return {
        "id": image_path.stem,
        "image_path": image_path,
        "width": image.width,
        "height": image.height,
        "ground_truth": read_ground_truth(label_path, image.width, image.height),
        "detections": detections,
        "bottle_count": len(bottles),
        "label_count": len(labels),
        "selection_mode": selection_mode,
        "selected": selected,
    }


def render_record(
    record: dict[str, Any],
    output: Path,
    index: int,
    bottle_crop_threshold: float,
) -> dict[str, Any]:
    with Image.open(record["image_path"]) as source:
        image = source.convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    line_width = max(3, round(min(image.size) / 280))
    for item in record["ground_truth"]:
        name = "GT bottle" if item["class_id"] == 0 else "GT label"
        color = COLORS["gt_bottle"] if item["class_id"] == 0 else COLORS["gt_label"]
        draw_box(draw, item["box"], color, name, line_width)
    for item in record["detections"]:
        name = "P bottle" if item["class_id"] == 0 else "P label"
        color = COLORS["pred_bottle"] if item["class_id"] == 0 else COLORS["pred_label"]
        draw_box(draw, item["box"], color, f"{name} {item['confidence']:.3f}", line_width)
    center_x, center_y = image.width // 2, image.height // 2
    arm = max(30, min(image.size) // 30)
    draw.line((center_x - arm, center_y, center_x + arm, center_y), fill=COLORS["crosshair"], width=line_width)
    draw.line((center_x, center_y - arm, center_x, center_y + arm), fill=COLORS["crosshair"], width=line_width)

    prefix = f"{index:02d}_{record['id']}"
    overlay_name = f"assets/{prefix}_overlay.jpg"
    save_web_image(overlay, output / overlay_name, 1200)
    candidates = []
    for candidate_index, item in enumerate(record["selected"], start=1):
        label_crop, label_box = padded_crop(image, item["label"]["box"], 0.10)
        label_name = f"assets/{prefix}_c{candidate_index}_label.jpg"
        save_web_image(label_crop, output / label_name, 900)
        bottle = item["bottle"]
        if bottle and bottle["confidence"] >= bottle_crop_threshold:
            bottle_crop, bottle_box = padded_crop(image, bottle["box"], 0.06)
        else:
            bottle_crop = image.copy()
            bottle_box = list(bottle["box"]) if bottle else None
        bottle_name = f"assets/{prefix}_c{candidate_index}_bottle.jpg"
        save_web_image(bottle_crop, output / bottle_name, 900)
        candidates.append(
            {
                "candidate": candidate_index,
                "label_image": label_name,
                "bottle_image": bottle_name,
                "label_confidence": item["label_confidence"],
                "bottle_confidence": item["bottle_confidence"],
                "bottle_image_mode": item["bottle_image_mode"],
                "label_crop_box": list(label_box),
                "bottle_box": [round(value, 2) for value in bottle_box] if bottle_box else None,
            }
        )
    return {
        "id": record["id"],
        "source": str(record["image_path"].relative_to(PROJECT_ROOT)),
        "dimensions": [record["width"], record["height"]],
        "selection_mode": record["selection_mode"],
        "predicted_bottles": record["bottle_count"],
        "predicted_labels": record["label_count"],
        "ground_truth_bottles": sum(item["class_id"] == 0 for item in record["ground_truth"]),
        "ground_truth_labels": sum(item["class_id"] == 1 for item in record["ground_truth"]),
        "overlay_image": overlay_name,
        "candidates": candidates,
    }


def build_html(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        candidate_panels = []
        for candidate in row["candidates"]:
            label_confidence = f"{candidate['label_confidence']:.3f}"
            bottle_confidence = candidate["bottle_confidence"]
            bottle_text = "none" if bottle_confidence is None else f"{bottle_confidence:.3f}"
            candidate_panels.append(
                f'<div class="candidate"><figure><img src="{candidate["bottle_image"]}" loading="lazy"><figcaption>Bottle input · confidence {bottle_text}<br>{html.escape(candidate["bottle_image_mode"])}</figcaption></figure>'
                f'<figure><img src="{candidate["label_image"]}" loading="lazy"><figcaption>Label crop · confidence {label_confidence}</figcaption></figure></div>'
            )
        if not candidate_panels:
            candidate_panels.append('<div class="empty">No selected label candidate</div>')
        public_metadata = {key: value for key, value in row.items() if key not in {"overlay_image", "candidates"}}
        metadata = html.escape(json.dumps(public_metadata, ensure_ascii=False, indent=2))
        candidates_json = html.escape(json.dumps(row["candidates"], ensure_ascii=False, indent=2))
        label_sort = row["candidates"][0]["label_confidence"] if row["candidates"] else -1
        first_bottle = row["candidates"][0]["bottle_confidence"] if row["candidates"] else None
        bottle_sort = -1 if first_bottle is None else first_bottle
        cards.append(
            f'<article class="case" data-mode="{row["selection_mode"]}" data-label-conf="{label_sort}" data-bottle-conf="{bottle_sort}">'
            f'<header><h2>{html.escape(row["id"])}</h2><span class="mode">{row["selection_mode"]}</span></header>'
            f'<div class="facts">{row["dimensions"][0]}×{row["dimensions"][1]} · GT B/L {row["ground_truth_bottles"]}/{row["ground_truth_labels"]} · predicted B/L {row["predicted_bottles"]}/{row["predicted_labels"]}</div>'
            f'<div class="visuals"><figure class="original"><img src="{row["overlay_image"]}" loading="lazy"><figcaption>Original + GT + predictions + crosshair</figcaption></figure><div class="candidate-list">{"".join(candidate_panels)}</div></div>'
            f'<details><summary>All metadata</summary><pre>{metadata}\n\nCANDIDATES\n{candidates_json}</pre></details></article>'
        )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Joint YOLO crop audit</title>
<style>
:root{{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif}}body{{margin:0;background:#111315;color:#f3f4f6}}.shell{{max-width:1680px;margin:auto;padding:28px}}h1{{margin:0 0 8px;font-size:30px}}.subtitle{{color:#aab1ba;margin-bottom:20px}}a{{color:#7dd3fc}}.toolbar{{position:sticky;top:0;z-index:5;display:flex;gap:8px;flex-wrap:wrap;padding:12px 0;background:#111315ee;backdrop-filter:blur(8px)}}button,select{{border:1px solid #3a414a;background:#1c2025;color:#fff;border-radius:8px;padding:9px 12px;cursor:pointer}}button.active{{border-color:#ffd60a;color:#ffd60a}}.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:#c9ced5;margin:8px 0 22px}}.swatch{{width:12px;height:12px;display:inline-block;margin-right:5px;border-radius:2px;vertical-align:-1px}}.case{{background:#191d21;border:1px solid #30363d;border-radius:14px;padding:18px;margin:0 0 22px}}.case header{{display:flex;justify-content:space-between;align-items:center;gap:16px}}h2{{font-size:19px;margin:0;overflow-wrap:anywhere}}.mode{{color:#ffd60a;font:600 12px ui-monospace,monospace}}.facts{{color:#aab1ba;margin:7px 0 14px;font-size:13px}}.visuals{{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(340px,1fr);gap:16px;align-items:start}}figure{{margin:0;background:#0c0e10;border-radius:10px;overflow:hidden}}figure img{{width:100%;max-height:640px;object-fit:contain;display:block;background:#08090a}}figcaption{{padding:9px 11px;color:#cbd0d6;font-size:12px;line-height:1.45}}.candidate-list{{display:grid;gap:14px}}.candidate{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.empty{{min-height:180px;display:grid;place-items:center;color:#ef476f;border:1px dashed #ef476f;border-radius:10px}}details{{margin-top:14px}}summary{{cursor:pointer;color:#cbd0d6}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#b8c0ca;background:#0d0f11;padding:12px;border-radius:8px;font-size:11px}}@media(max-width:900px){{.visuals{{grid-template-columns:1fr}}.candidate{{grid-template-columns:1fr 1fr}}.shell{{padding:14px}}}}
</style></head><body><main class="shell"><h1>Joint YOLO crop audit · {len(rows)} cases</h1><div class="subtitle">Original with ground truth and predictions → bottle input → label crop. <a href="audit_metadata.json">Full JSON metadata</a></div><div class="toolbar"><button class="active" data-filter="all">All</button><button data-filter="single_pair">Single</button><button data-filter="ambiguous_two_pairs">Ambiguous</button><button data-filter="no_label_detection">No label</button><select id="sort"><option value="default">Audit order</option><option value="label">Lowest label confidence</option><option value="bottle">Lowest bottle confidence</option></select></div><div class="legend"><span><i class="swatch" style="background:{COLORS["gt_bottle"]}"></i>GT bottle</span><span><i class="swatch" style="background:{COLORS["gt_label"]}"></i>GT label</span><span><i class="swatch" style="background:{COLORS["pred_bottle"]}"></i>Pred bottle</span><span><i class="swatch" style="background:{COLORS["pred_label"]}"></i>Pred label</span><span><i class="swatch" style="background:{COLORS["crosshair"]}"></i>Crosshair</span></div><section id="cases">{"".join(cards)}</section></main>
<script>const root=document.getElementById('cases');const initial=[...root.children];document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{{document.querySelectorAll('[data-filter]').forEach(item=>item.classList.remove('active'));button.classList.add('active');const filter=button.dataset.filter;initial.forEach(card=>card.hidden=filter!=='all'&&card.dataset.mode!==filter)}}));document.getElementById('sort').addEventListener('change',event=>{{const cards=[...initial];const key=event.target.value;if(key==='label')cards.sort((a,b)=>+a.dataset.labelConf-+b.dataset.labelConf);else if(key==='bottle')cards.sort((a,b)=>+a.dataset.bottleConf-+b.dataset.bottleConf);else cards.sort((a,b)=>initial.indexOf(a)-initial.indexOf(b));cards.forEach(card=>root.appendChild(card))}});</script></body></html>'''


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    image_root = dataset / "images" / args.split
    label_root = dataset / "labels" / args.split
    image_paths = sorted(path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise ValueError(f"No images in {image_root}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    device = choose_device(args.device)
    model = YOLO(str(model_path))
    records = []
    for index, image_path in enumerate(image_paths, start=1):
        records.append(
            infer_record(
                model,
                image_path,
                label_root / f"{image_path.stem}.txt",
                device,
                args.confidence,
                args.bottle_crop_threshold,
                args.ambiguity_margin,
            )
        )
        if index % 10 == 0 or index == len(image_paths):
            print(f"AUDIT INFERENCE | {index}/{len(image_paths)}", flush=True)
    chosen = select_mixed(records, args.n, args.seed, args.include_id)
    rows = [
        render_record(record, output, index, args.bottle_crop_threshold)
        for index, record in enumerate(chosen, start=1)
    ]
    metadata = {
        "model": str(model_path.relative_to(PROJECT_ROOT)),
        "dataset": str(dataset.relative_to(PROJECT_ROOT)),
        "split": args.split,
        "evaluated_images": len(records),
        "displayed_images": len(rows),
        "confidence_threshold": args.confidence,
        "bottle_crop_threshold": args.bottle_crop_threshold,
        "ambiguity_margin": args.ambiguity_margin,
        "rows": rows,
    }
    (output / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(build_html(rows), encoding="utf-8")
    print(f"AUDIT HTML | {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
