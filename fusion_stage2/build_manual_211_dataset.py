#!/usr/bin/env python3
"""Build the 211-image manual adaptation dataset with the joint YOLO.

The manual LabelMe rectangles identify the intended bottle and label.  YOLO is
run once per image; detections are matched to those rectangles by IoU so a
neighbouring shelf bottle can never silently inherit the target identity.
Catalog identities come from the reviewed store_shelves_v2 annotations.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from joint_yolo.infer import (
    box_iou,
    choose_device,
    padded_crop,
    predict_detections,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "joint_yolo_manual_finetune" / "manifest.csv"
DEFAULT_LABELS = PROJECT_ROOT / "datasets" / "store_shelves_v2" / "labels.csv"
DEFAULT_POINTS = PROJECT_ROOT / "datasets" / "store_shelves_v2" / "points.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "joint_yolo" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "manual_211"
DEFAULT_ARCHIVE = PROJECT_ROOT / "datasets" / "manual_211.zip"


def rectangle(payload: dict[str, Any], label_names: set[str]) -> tuple[float, float, float, float]:
    matches = [
        shape for shape in payload.get("shapes", [])
        if str(shape.get("label", "")).strip().lower() in label_names
        and str(shape.get("shape_type", "rectangle")) == "rectangle"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one rectangle for {sorted(label_names)}, found {len(matches)}")
    points = matches[0]["points"]
    xs, ys = [float(point[0]) for point in points], [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def best_detection(
    detections: list[dict[str, Any]],
    class_id: int,
    target: tuple[float, float, float, float],
) -> tuple[dict[str, Any] | None, float]:
    candidates = [item for item in detections if int(item["class_id"]) == class_id]
    if not candidates:
        return None, 0.0
    ranked = sorted(
        ((box_iou(tuple(item["box"]), target), item) for item in candidates),
        key=lambda pair: (pair[0], float(pair[1]["confidence"])),
        reverse=True,
    )
    overlap, selected = ranked[0]
    return selected, float(overlap)


def reviewed_identity(
    row: Any,
    payload: dict[str, Any],
    labels: pd.DataFrame,
    points: pd.DataFrame,
    bottle_box: tuple[float, float, float, float],
) -> dict[str, Any]:
    image_name = Path(row.source_image).name
    if str(row.source_directory) in {"source2_2", "source3_raz_met"}:
        matches = labels[labels["query_id"].eq(Path(image_name).stem)]
        if len(matches) != 1:
            return {"status": "unresolved", "accepted_slugs": "", "mapping_mode": "missing_query", "mapping_dx": math.nan}
        match = matches.iloc[0]
        return {
            "status": str(match["status"]),
            "accepted_slugs": str(match.get("accepted_slugs", "") or ""),
            "mapping_mode": "reviewed_query_id",
            "mapping_dx": 0.0,
            "note": str(match.get("note", "") or ""),
        }

    candidates = points[points["source"].eq(image_name)].copy()
    if candidates.empty:
        return {"status": "unresolved", "accepted_slugs": "", "mapping_mode": "missing_source_point", "mapping_dx": math.nan}
    image_width = float(payload["imageWidth"])
    centre_x_percent = 50.0 * (bottle_box[0] + bottle_box[2]) / image_width
    candidates["mapping_dx"] = (pd.to_numeric(candidates["x"]) - centre_x_percent).abs()
    match = candidates.sort_values(["mapping_dx", "x"]).iloc[0]
    distance = float(match["mapping_dx"])
    if distance > 8.0:
        return {
            "status": "unresolved",
            "accepted_slugs": "",
            "mapping_mode": "source_point_too_far",
            "mapping_dx": distance,
            "note": "No reviewed target point is close to the manual bottle centre.",
        }
    return {
        "status": str(match["status"]),
        "accepted_slugs": str(match.get("accepted_slugs", "") or ""),
        "mapping_mode": "reviewed_source_x",
        "mapping_dx": distance,
        "note": str(match.get("note", "") or ""),
    }


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=95, optimize=True)


def json_box(box: tuple[float, float, float, float] | list[float] | None) -> str:
    return "" if box is None else json.dumps([round(float(value), 3) for value in box])


def build(args: argparse.Namespace) -> dict[str, Any]:
    manual = pd.read_csv(args.manual_manifest)
    manual = manual[manual["origin"].eq("manual_new")].copy()
    if len(manual) != 211:
        raise ValueError(f"Expected 211 manual images, found {len(manual)}")
    labels = pd.read_csv(args.reviewed_labels).fillna("")
    points = pd.read_csv(args.reviewed_points).fillna("")
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    for name in ("originals", "label_crops", "bottle_crops"):
        (args.output_root / name).mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = YOLO(str(args.model.resolve()))
    records: list[dict[str, Any]] = []
    for row in tqdm(list(manual.itertuples(index=False)), desc="Manual-211 YOLO", unit="image"):
        source = Path(row.source_image)
        annotation = Path(row.source_annotation)
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        gt_bottle = rectangle(payload, {"bottle"})
        gt_label = rectangle(payload, {"label", "wine_label"})
        identity = reviewed_identity(row, payload, labels, points, gt_bottle)
        accepted = [part.strip() for part in str(identity["accepted_slugs"]).split(";") if part.strip()]

        with Image.open(source) as opened:
            image = opened.convert("RGB")
        detections = predict_detections(model, image, device, args.confidence)
        predicted_bottle, bottle_iou = best_detection(detections, 0, gt_bottle)
        predicted_label, label_iou = best_detection(detections, 1, gt_label)

        label_source = "yolo"
        if predicted_label is not None and label_iou >= args.minimum_iou:
            label_crop, effective_label_box = padded_crop(image, tuple(predicted_label["box"]), args.label_padding)
        else:
            label_crop, effective_label_box = padded_crop(image, gt_label, args.label_padding)
            label_source = "manual_gt_fallback"

        bottle_confidence = float(predicted_bottle["confidence"]) if predicted_bottle else 0.0
        if (
            predicted_bottle is not None
            and bottle_iou >= args.minimum_iou
            and bottle_confidence >= args.bottle_crop_threshold
        ):
            bottle_crop, effective_bottle_box = padded_crop(image, tuple(predicted_bottle["box"]), args.bottle_padding)
            bottle_mode = "yolo_bottle_crop"
        else:
            bottle_crop, effective_bottle_box = image.copy(), [0, 0, image.width, image.height]
            bottle_mode = "original_low_confidence_or_miss"

        sample_id = str(row.dataset_stem)
        original_rel = Path("originals") / f"{sample_id}.jpg"
        label_rel = Path("label_crops") / f"{sample_id}.jpg"
        bottle_rel = Path("bottle_crops") / f"{sample_id}.jpg"
        save_jpeg(image, args.output_root / original_rel)
        save_jpeg(label_crop, args.output_root / label_rel)
        save_jpeg(bottle_crop, args.output_root / bottle_rel)

        records.append({
            "sample_id": sample_id,
            "source_group": str(row.source_directory),
            "source_filename": source.name,
            "original_path": original_rel.as_posix(),
            "label_path": label_rel.as_posix(),
            "bottle_path": bottle_rel.as_posix(),
            "identity_status": identity["status"],
            "accepted_slugs": ";".join(accepted),
            "trainable_catalog": bool(identity["status"] == "ok" and accepted),
            "mapping_mode": identity["mapping_mode"],
            "mapping_dx": identity["mapping_dx"],
            "note": identity.get("note", ""),
            "label_crop_source": label_source,
            "bottle_image_mode": bottle_mode,
            "label_confidence": float(predicted_label["confidence"]) if predicted_label else 0.0,
            "bottle_confidence": bottle_confidence,
            "label_iou_with_gt": label_iou,
            "bottle_iou_with_gt": bottle_iou,
            "gt_label_box": json_box(gt_label),
            "gt_bottle_box": json_box(gt_bottle),
            "predicted_label_box": json_box(predicted_label["box"] if predicted_label else None),
            "predicted_bottle_box": json_box(predicted_bottle["box"] if predicted_bottle else None),
            "effective_label_box": json_box(effective_label_box),
            "effective_bottle_box": json_box(effective_bottle_box),
            "num_label_detections": sum(int(item["class_id"]) == 1 for item in detections),
            "num_bottle_detections": sum(int(item["class_id"]) == 0 for item in detections),
        })

    frame = pd.DataFrame(records).sort_values("sample_id").reset_index(drop=True)
    frame.to_csv(args.output_root / "manifest.csv", index=False)
    summary = {
        "complete": True,
        "rows": int(len(frame)),
        "trainable_catalog_rows": int(frame["trainable_catalog"].sum()),
        "identity_status_counts": frame["identity_status"].value_counts(dropna=False).to_dict(),
        "unique_catalog_identities": int(
            len({slug for value in frame.loc[frame["trainable_catalog"], "accepted_slugs"] for slug in str(value).split(";") if slug})
        ),
        "label_crop_source_counts": frame["label_crop_source"].value_counts().to_dict(),
        "bottle_image_mode_counts": frame["bottle_image_mode"].value_counts().to_dict(),
        "mean_label_confidence": float(frame["label_confidence"].mean()),
        "mean_bottle_confidence": float(frame["bottle_confidence"].mean()),
        "mean_label_iou_with_gt": float(frame["label_iou_with_gt"].mean()),
        "mean_bottle_iou_with_gt": float(frame["bottle_iou_with_gt"].mean()),
        "model": str(args.model.resolve()),
        "device": str(device),
        "policy": {
            "target_alignment": "highest IoU with manual GT rectangle",
            "minimum_iou": args.minimum_iou,
            "bottle_crop_threshold": args.bottle_crop_threshold,
            "low_confidence_bottle": "full original image",
        },
    }
    (args.output_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.archive.exists():
        args.archive.unlink()
    archive_base = str(args.archive.with_suffix(""))
    shutil.make_archive(archive_base, "zip", root_dir=args.output_root.parent, base_dir=args.output_root.name)
    return {**summary, "archive": str(args.archive.resolve()), "archive_size_bytes": args.archive.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reviewed-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--reviewed-points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--minimum-iou", type=float, default=0.10)
    parser.add_argument("--bottle-crop-threshold", type=float, default=0.75)
    parser.add_argument("--label-padding", type=float, default=0.10)
    parser.add_argument("--bottle-padding", type=float, default=0.06)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.manual_manifest, args.reviewed_labels, args.reviewed_points, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = build(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
