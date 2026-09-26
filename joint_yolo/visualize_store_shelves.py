#!/usr/bin/env python3
"""Audit the joint YOLO on the hand-labelled store-shelf catalogue set.

The store-shelf data has identity and target-point annotations, not YOLO
bounding boxes.  Every generated query is centred on the manually marked wine.
This viewer therefore draws the true target point, every predicted bottle and
label box, every resulting crop, and the exact crop pair selected by production
ranking.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from yolo_target_selection import confidence_axis_score

from .infer import (
    DEFAULT_MODEL,
    choose_device,
    padded_crop,
    pair_bottle,
    predict_detections,
    select_label_candidates,
)
from .train import PROJECT_ROOT


DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "store_shelves_catalog"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "joint_yolo" / "crop_audit"
COLORS = {
    "bottle": "#ff9f1c",
    "label": "#ef476f",
    "selected": "#ffffff",
    "axis": "#ffd60a",
    "truth": "#20d67b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--bottle-crop-threshold", type=float, default=0.75)
    parser.add_argument("--ambiguity-margin", type=float, default=0.06)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N rows; zero means the complete dataset.",
    )
    return parser.parse_args()


def load_rows(dataset: Path) -> list[dict[str, str]]:
    labels = dataset / "labels.csv"
    queries = dataset / "queries"
    if not labels.is_file() or not queries.is_dir():
        raise FileNotFoundError(f"Expected labels.csv and queries/ under {dataset}")
    with labels.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_id", "image_path", "accepted_slugs", "note", "source", "x", "y"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Missing columns in {labels}: {sorted(missing)}")
    for row in rows:
        image_path = queries / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        row["resolved_image_path"] = str(image_path)
    return rows


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
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
    text_font = font(max(15, width * 4))
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
    result.save(path, format="WEBP", quality=80, method=4)


def contains_point(box: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def object_index(target: dict[str, Any] | None, items: list[dict[str, Any]]) -> int | None:
    if target is None:
        return None
    for index, item in enumerate(items, start=1):
        if item is target:
            return index
    return None


def point_proxy(
    items: list[dict[str, Any]], point: tuple[float, float]
) -> tuple[int | None, dict[str, Any] | None]:
    """Choose a point-supervised proxy box without pretending it is manual GT."""
    containing = [item for item in items if contains_point(item["box"], point)]
    if not containing:
        return None, None
    proxy = max(containing, key=lambda item: item["confidence"])
    return object_index(proxy, items), proxy


def infer_record(
    model: YOLO,
    truth: dict[str, str],
    device: str | int,
    confidence: float,
    ambiguity_margin: float,
) -> dict[str, Any]:
    image_path = Path(truth["resolved_image_path"])
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    detections = predict_detections(model, image, device, confidence)
    bottles = [item for item in detections if item["class_id"] == 0]
    labels = [item for item in detections if item["class_id"] == 1]
    target = (image.width / 2, image.height / 2)
    selected_labels, ambiguous = select_label_candidates(
        labels,
        target,
        math.hypot(image.width, image.height),
        ambiguity_margin,
        image_width=image.width,
    )
    selected_ids = {id(item) for item in selected_labels}
    return {
        "id": truth["query_id"],
        "image_path": image_path,
        "width": image.width,
        "height": image.height,
        "truth": {key: value for key, value in truth.items() if key != "resolved_image_path"},
        "target": target,
        "bottles": bottles,
        "labels": labels,
        "selected_ids": selected_ids,
        "ambiguous": ambiguous,
    }


def render_record(
    record: dict[str, Any],
    output: Path,
    index: int,
    bottle_crop_threshold: float,
) -> dict[str, Any]:
    with Image.open(record["image_path"]) as source:
        image = source.convert("RGB")
    bottles = record["bottles"]
    labels = record["labels"]
    target = record["target"]
    prefix = f"{index:03d}_{record['id']}"

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    line_width = max(3, round(min(image.size) / 260))
    axis_x = round(target[0])
    draw.line((axis_x, 0, axis_x, image.height), fill=COLORS["axis"], width=line_width)

    selected_label_indices: list[int] = []
    proxy_bottle_index, proxy_bottle = point_proxy(bottles, target)
    proxy_label_index, proxy_label = point_proxy(labels, target)
    for bottle_index, bottle in enumerate(bottles, start=1):
        hit = contains_point(bottle["box"], target)
        draw_box(
            draw,
            bottle["box"],
            COLORS["bottle"],
            f"B{bottle_index} {bottle['confidence']:.3f}{' TRUE-POINT' if hit else ''}",
            line_width,
        )
    for label_index, label in enumerate(labels, start=1):
        selected = id(label) in record["selected_ids"]
        hit = contains_point(label["box"], target)
        draw_box(
            draw,
            label["box"],
            COLORS["selected"] if selected else COLORS["label"],
            f"L{label_index} {label['confidence']:.3f}{' SELECTED' if selected else ''}{' TRUE-POINT' if hit else ''}",
            line_width * (2 if selected else 1),
        )
        if selected:
            selected_label_indices.append(label_index)

    # The shelf set provides a manual point, but no manually drawn bbox.  Draw
    # the strongest predicted box containing that point as an explicitly named
    # weak-GT proxy, never as a true manual bounding box.
    if proxy_bottle is not None:
        draw_box(
            draw,
            proxy_bottle["box"],
            COLORS["truth"],
            f"WEAK GT B{proxy_bottle_index} (point-derived)",
            line_width * 2,
        )
    if proxy_label is not None:
        draw_box(
            draw,
            proxy_label["box"],
            COLORS["truth"],
            f"WEAK GT L{proxy_label_index} (point-derived)",
            line_width * 2,
        )

    radius = max(12, min(image.size) // 45)
    draw.ellipse(
        (target[0] - radius, target[1] - radius, target[0] + radius, target[1] + radius),
        outline=COLORS["truth"],
        width=line_width * 2,
    )
    draw.line((target[0] - radius * 2, target[1], target[0] + radius * 2, target[1]), fill=COLORS["truth"], width=line_width)
    draw.line((target[0], target[1] - radius * 2, target[0], target[1] + radius * 2), fill=COLORS["truth"], width=line_width)

    overlay_name = f"assets/{prefix}_overlay.webp"
    save_web_image(overlay, output / overlay_name, 1100)

    bottle_rows: list[dict[str, Any]] = []
    for bottle_index, bottle in enumerate(bottles, start=1):
        crop, crop_box = padded_crop(image, bottle["box"], 0.06)
        crop_name = f"assets/{prefix}_b{bottle_index}.webp"
        save_web_image(crop, output / crop_name, 480)
        bottle_rows.append(
            {
                "index": bottle_index,
                "image": crop_name,
                "confidence": bottle["confidence"],
                "box": [round(value, 2) for value in bottle["box"]],
                "crop_box": crop_box,
                "contains_true_target": contains_point(bottle["box"], target),
                "weak_gt_proxy": bottle_index == proxy_bottle_index,
            }
        )

    label_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for label_index, label in enumerate(labels, start=1):
        score, axis_distance, axis_proximity = confidence_axis_score(
            label["confidence"], label["box"], image.width
        )
        label_crop, label_crop_box = padded_crop(image, label["box"], 0.10)
        label_name = f"assets/{prefix}_l{label_index}.webp"
        save_web_image(label_crop, output / label_name, 480)
        bottle = pair_bottle(label, bottles)
        bottle_index = object_index(bottle, bottles)
        selected = id(label) in record["selected_ids"]
        label_row = {
            "index": label_index,
            "image": label_name,
            "confidence": label["confidence"],
            "selection_score": score,
            "axis_distance": axis_distance,
            "axis_proximity": axis_proximity,
            "box": [round(value, 2) for value in label["box"]],
            "crop_box": label_crop_box,
            "contains_true_target": contains_point(label["box"], target),
            "weak_gt_proxy": label_index == proxy_label_index,
            "selected": selected,
            "paired_bottle": bottle_index,
        }
        label_rows.append(label_row)
        if not selected:
            continue
        if bottle is not None and bottle["confidence"] >= bottle_crop_threshold:
            bottle_input = bottle_rows[bottle_index - 1]["image"]
            bottle_mode = "yolo_bottle_crop"
        else:
            bottle_input = f"assets/{prefix}_source.webp"
            bottle_mode = "original_low_confidence" if bottle else "original_no_bottle"
            if not (output / bottle_input).is_file():
                save_web_image(image, output / bottle_input, 700)
        selected_rows.append(
            {
                "label_index": label_index,
                "label_image": label_name,
                "label_confidence": label["confidence"],
                "selection_score": score,
                "axis_distance": axis_distance,
                "bottle_index": bottle_index,
                "bottle_image": bottle_input,
                "bottle_confidence": bottle["confidence"] if bottle else None,
                "bottle_image_mode": bottle_mode,
                "contains_true_target": contains_point(label["box"], target),
            }
        )

    top_selected = selected_rows[0] if selected_rows else None
    status = (
        "no_label_detection"
        if not selected_rows
        else "selected_hits_true_point"
        if top_selected["contains_true_target"]
        else "selected_misses_true_point"
    )
    if record["ambiguous"] and selected_rows:
        status = "ambiguous_two_labels"
    proxy_reference = {
        "label_index": proxy_label_index,
        "label_image": (
            label_rows[proxy_label_index - 1]["image"] if proxy_label_index is not None else None
        ),
        "label_confidence": proxy_label["confidence"] if proxy_label is not None else None,
        "bottle_index": proxy_bottle_index,
        "bottle_image": (
            bottle_rows[proxy_bottle_index - 1]["image"] if proxy_bottle_index is not None else None
        ),
        "bottle_confidence": proxy_bottle["confidence"] if proxy_bottle is not None else None,
        "selected_matches_label_proxy": (
            bool(selected_label_indices) and selected_label_indices[0] == proxy_label_index
        ),
    }
    return {
        "id": record["id"],
        "source": str(record["image_path"].relative_to(PROJECT_ROOT)),
        "dimensions": [record["width"], record["height"]],
        "truth": record["truth"],
        "true_target_in_query": [record["width"] / 2, record["height"] / 2],
        "status": status,
        "ambiguous": record["ambiguous"],
        "overlay_image": overlay_name,
        "selected_label_indices": selected_label_indices,
        "weak_gt_proxy": proxy_reference,
        "selected": selected_rows,
        "all_labels": label_rows,
        "all_bottles": bottle_rows,
    }


def crop_panel(item: dict[str, Any], kind: str) -> str:
    selected = item.get("selected", False)
    hit = item["contains_true_target"]
    classes = "crop selected" if selected else "crop"
    badges = []
    if selected:
        badges.append('<b class="badge selected-badge">SELECTED</b>')
    if hit:
        badges.append('<b class="badge truth-badge">TRUE POINT</b>')
    if item.get("weak_gt_proxy"):
        badges.append('<b class="badge weak-gt-badge">WEAK GT</b>')
    if kind == "label":
        details = (
            f"conf {item['confidence']:.3f} · score {item['selection_score']:.3f} · "
            f"axis Δ {item['axis_distance']:.3f} · paired B{item['paired_bottle'] or '—'}"
        )
        title = f"L{item['index']} · label"
    else:
        details = f"conf {item['confidence']:.3f}"
        title = f"B{item['index']} · bottle"
    return (
        f'<figure class="{classes}"><img src="{item["image"]}" loading="lazy">'
        f'<figcaption><strong>{title}</strong><span class="badges">{"".join(badges)}</span>'
        f'<small>{details}</small></figcaption></figure>'
    )


def build_html(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    cards: list[str] = []
    for row in rows:
        truth = row["truth"]
        truth_slugs = html.escape(truth["accepted_slugs"])
        truth_note = html.escape(truth["note"])
        all_labels = "".join(crop_panel(item, "label") for item in row["all_labels"])
        all_bottles = "".join(crop_panel(item, "bottle") for item in row["all_bottles"])
        if not all_labels:
            all_labels = '<div class="empty">No label detections</div>'
        if not all_bottles:
            all_bottles = '<div class="empty">No bottle detections</div>'
        selected_panels: list[str] = []
        for selected in row["selected"]:
            bottle_conf = "none" if selected["bottle_confidence"] is None else f"{selected['bottle_confidence']:.3f}"
            selected_panels.append(
                f'<div class="selected-pair"><figure><img src="{selected["label_image"]}" loading="lazy"><figcaption>'
                f'<strong>SELECTED L{selected["label_index"]}</strong><small>conf {selected["label_confidence"]:.3f} · '
                f'score {selected["selection_score"]:.3f} · axis Δ {selected["axis_distance"]:.3f}</small></figcaption></figure>'
                f'<figure><img src="{selected["bottle_image"]}" loading="lazy"><figcaption><strong>MODEL BOTTLE INPUT · '
                f'B{selected["bottle_index"] or "—"}</strong><small>conf {bottle_conf} · {html.escape(selected["bottle_image_mode"])}</small>'
                f'</figcaption></figure></div>'
            )
        if not selected_panels:
            selected_panels.append('<div class="empty danger">No crop selected</div>')
        proxy = row["weak_gt_proxy"]
        proxy_panels: list[str] = []
        if proxy["label_image"]:
            proxy_panels.append(
                f'<figure class="proxy"><img src="{proxy["label_image"]}" loading="lazy"><figcaption>'
                f'<strong>WEAK GT LABEL · L{proxy["label_index"]}</strong><small>point-derived proxy · '
                f'conf {proxy["label_confidence"]:.3f}</small></figcaption></figure>'
            )
        if proxy["bottle_image"]:
            proxy_panels.append(
                f'<figure class="proxy"><img src="{proxy["bottle_image"]}" loading="lazy"><figcaption>'
                f'<strong>WEAK GT BOTTLE · B{proxy["bottle_index"]}</strong><small>point-derived proxy · '
                f'conf {proxy["bottle_confidence"]:.3f}</small></figcaption></figure>'
            )
        if not proxy_panels:
            proxy_panels.append('<div class="empty danger">No predicted bbox contains the true target point</div>')
        raw = html.escape(json.dumps(row, ensure_ascii=False, indent=2))
        cards.append(
            f'<article class="case" data-status="{row["status"]}" data-search="{html.escape((row["id"] + " " + truth["accepted_slugs"]).lower())}">'
            f'<header><div><h2>{html.escape(row["id"])}</h2><div class="slug">TRUE · {truth_slugs}</div></div>'
            f'<span class="status {row["status"]}">{row["status"]}</span></header>'
            f'<p class="note">{truth_note}</p><div class="facts">source {html.escape(truth["source"])} · original point '
            f'{html.escape(truth["x"])}/{html.escape(truth["y"])}% · predictions B/L '
            f'{len(row["all_bottles"])}/{len(row["all_labels"])} · selected {html.escape(str(row["selected_label_indices"]))}</div>'
            f'<div class="main-grid"><figure class="overlay"><img src="{row["overlay_image"]}" loading="lazy"><figcaption>'
            f'<strong>ORIGINAL + TRUE TARGET + ALL BOXES</strong><small>green point = manual annotation; green bbox = point-derived WEAK GT; '
            f'white = selected label; yellow = vertical target axis</small></figcaption></figure><section><h3>Selected model input</h3>{"".join(selected_panels)}'
            f'<h3>Point-derived target reference</h3><div class="selected-pair proxy-pair">{"".join(proxy_panels)}</div></section></div>'
            f'<details open><summary>All label crops ({len(row["all_labels"])})</summary><div class="crop-grid">{all_labels}</div></details>'
            f'<details><summary>All bottle crops ({len(row["all_bottles"])})</summary><div class="crop-grid">{all_bottles}</div></details>'
            f'<details><summary>Complete metadata</summary><pre>{raw}</pre></details></article>'
        )

    counts = metadata["status_counts"]
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Joint YOLO · store shelves audit</title>
<style>
:root{{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif}}*{{box-sizing:border-box}}body{{margin:0;background:#101214;color:#f4f5f6}}.shell{{max-width:1800px;margin:auto;padding:24px}}h1{{margin:0 0 6px;font-size:30px}}.subtitle,.facts{{color:#aeb5bd}}.summary{{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}}.pill,.status,.badge{{border-radius:999px;padding:5px 9px;font-size:12px;font-weight:700}}.pill{{background:#252a30}}.toolbar{{position:sticky;top:0;z-index:8;display:flex;gap:8px;flex-wrap:wrap;padding:12px 0;background:#101214ed;backdrop-filter:blur(8px)}}button,input{{border:1px solid #39414a;background:#1a1e22;color:#fff;border-radius:9px;padding:10px 12px}}button{{cursor:pointer}}button.active{{border-color:#ffd60a;color:#ffd60a}}input{{min-width:290px}}.case{{background:#181c20;border:1px solid #30363d;border-radius:16px;padding:18px;margin:0 0 24px}}.case header{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}}h2{{margin:0;font-size:22px}}h3{{margin:14px 0 10px;font-size:16px}}.slug{{color:#20d67b;font-weight:700;margin-top:5px;overflow-wrap:anywhere}}.note{{margin:10px 0 5px;color:#d4d8dc}}.facts{{font-size:12px;margin-bottom:14px}}.status{{white-space:nowrap;background:#293038}}.selected_hits_true_point{{color:#20d67b}}.selected_misses_true_point,.no_label_detection{{color:#ff667d}}.ambiguous_two_labels{{color:#ffd60a}}.main-grid{{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(420px,.75fr);gap:16px;align-items:start}}figure{{margin:0;background:#0b0d0f;border:1px solid #2d333a;border-radius:10px;overflow:hidden}}figure img{{width:100%;height:100%;max-height:720px;object-fit:contain;display:block;background:#08090a}}figcaption{{padding:9px 11px;display:grid;gap:5px}}figcaption small{{color:#b8c0c8;line-height:1.45}}.selected-pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}}.selected-pair figure{{border:2px solid #fff}}.proxy-pair figure,.proxy{{border-color:#20d67b}}details{{margin-top:14px}}summary{{cursor:pointer;font-weight:750;color:#dfe3e6;padding:8px 0}}.crop-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}}.crop img{{height:230px;object-fit:contain}}.crop.selected{{border:3px solid #fff}}.badges{{display:flex;gap:5px;flex-wrap:wrap}}.selected-badge{{background:#fff;color:#111}}.truth-badge{{background:#20d67b;color:#07140d}}.weak-gt-badge{{background:#0b5d38;color:#b9ffdc}}.empty{{padding:36px;border:1px dashed #56606b;border-radius:10px;color:#aeb5bd}}.danger{{border-color:#ff667d;color:#ff667d}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b0d0f;padding:12px;border-radius:8px;color:#bac2ca;font-size:11px}}a{{color:#7dd3fc}}@media(max-width:1000px){{.main-grid{{grid-template-columns:1fr}}.shell{{padding:12px}}}}@media(max-width:600px){{.selected-pair{{grid-template-columns:1fr}}input{{min-width:100%}}}}
</style></head><body><main class="shell"><h1>Joint YOLO · store shelves catalogue</h1><div class="subtitle">209 hand-labelled customer-style shelf crops from the latest <code>touitsu</code> dataset. This dataset has a true identity and target point, not GT bounding boxes. <a href="audit_metadata.json">JSON metadata</a></div>
<div class="summary"><span class="pill">evaluated {metadata['evaluated_images']}</span><span class="pill">target hit {counts.get('selected_hits_true_point',0)}</span><span class="pill">target miss {counts.get('selected_misses_true_point',0)}</span><span class="pill">ambiguous {counts.get('ambiguous_two_labels',0)}</span><span class="pill">no label {counts.get('no_label_detection',0)}</span></div>
<div class="toolbar"><button class="active" data-filter="all">All</button><button data-filter="selected_misses_true_point">Target misses</button><button data-filter="ambiguous_two_labels">Ambiguous</button><button data-filter="no_label_detection">No label</button><button data-filter="selected_hits_true_point">Target hits</button><input id="search" placeholder="query id or true slug"></div><section id="cases">{"".join(cards)}</section></main>
<script>const cards=[...document.querySelectorAll('.case')];let filter='all';let query='';function apply(){{cards.forEach(c=>c.hidden=!((filter==='all'||c.dataset.status===filter)&&c.dataset.search.includes(query)))}}document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');filter=b.dataset.filter;apply()}});document.getElementById('search').oninput=e=>{{query=e.target.value.trim().toLowerCase();apply()}};</script></body></html>'''


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    truths = load_rows(dataset)
    if args.limit > 0:
        truths = truths[: args.limit]
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    device = choose_device(args.device)
    model = YOLO(str(model_path))
    records: list[dict[str, Any]] = []
    for index, truth in enumerate(truths, start=1):
        records.append(infer_record(model, truth, device, args.confidence, args.ambiguity_margin))
        if index % 10 == 0 or index == len(truths):
            print(f"AUDIT INFERENCE | {index}/{len(truths)}", flush=True)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rows.append(render_record(record, output, index, args.bottle_crop_threshold))
        if index % 20 == 0 or index == len(records):
            print(f"AUDIT RENDER | {index}/{len(records)}", flush=True)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    metadata = {
        "model": str(model_path.relative_to(PROJECT_ROOT)),
        "dataset": str(dataset.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "evaluated_images": len(rows),
        "confidence_threshold": args.confidence,
        "bottle_crop_threshold": args.bottle_crop_threshold,
        "ambiguity_margin": args.ambiguity_margin,
        "status_counts": status_counts,
        "rows": rows,
    }
    (output / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "index.html").write_text(build_html(rows, metadata), encoding="utf-8")
    print(f"AUDIT SUMMARY | {json.dumps(status_counts, ensure_ascii=False)}", flush=True)
    print(f"AUDIT HTML | {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
