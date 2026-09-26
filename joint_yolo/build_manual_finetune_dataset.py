#!/usr/bin/env python3
"""Build a focused joint-YOLO dataset from local AnyLabeling annotations."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps

from .train import DEFAULT_DATASET, PROJECT_ROOT


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_IDS = {"bottle": 0, "label": 1, "wine_label": 1}
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "joint_yolo_manual_finetune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=None,
        help="AnyLabeling directory; repeat for multiple folders.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--old-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--old-val-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def image_by_stem(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem in {root}: {path.stem}")
            result[path.stem] = path
    return result


def yolo_rows(annotation: Path, image_path: Path) -> tuple[list[str], dict[str, Any]]:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        width, height = image.size
    annotated_width = int(payload.get("imageWidth") or width)
    annotated_height = int(payload.get("imageHeight") or height)
    if (annotated_width, annotated_height) != (width, height):
        raise ValueError(
            f"Dimension mismatch for {image_path}: image={width}x{height}, "
            f"annotation={annotated_width}x{annotated_height}"
        )

    rows: list[str] = []
    counts = {"bottle": 0, "wine_label": 0}
    for shape in payload.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        if label not in CLASS_IDS:
            raise ValueError(f"Unknown class {label!r} in {annotation}")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"Only rectangle shapes are accepted: {annotation}")
        points = shape.get("points") or []
        if len(points) != 2 or any(len(point) != 2 for point in points):
            raise ValueError(f"Invalid rectangle in {annotation}")
        x1, x2 = sorted((float(points[0][0]), float(points[1][0])))
        y1, y2 = sorted((float(points[0][1]), float(points[1][1])))
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Empty rectangle in {annotation}")
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        rows.append(
            f"{CLASS_IDS[label]} {cx:.8f} {cy:.8f} {box_width:.8f} {box_height:.8f}"
        )
        counts["bottle" if CLASS_IDS[label] == 0 else "wine_label"] += 1
    if counts != {"bottle": 1, "wine_label": 1}:
        raise ValueError(f"Expected exactly one bottle and label in {annotation}; got {counts}")
    return rows, {"width": width, "height": height, **counts}


def copy_new_pair(
    source_name: str,
    image_path: Path,
    annotation: Path,
    output: Path,
    manifest: list[dict[str, Any]],
) -> None:
    labels, stats = yolo_rows(annotation, image_path)
    dataset_stem = f"{source_name}__{image_path.stem}"
    for split in ("train", "val"):
        image_target = output / "images" / split / f"{dataset_stem}{image_path.suffix.lower()}"
        label_target = output / "labels" / split / f"{dataset_stem}.txt"
        shutil.copy2(image_path, image_target)
        label_target.write_text("\n".join(labels) + "\n", encoding="utf-8")
    manifest.append(
        {
            "dataset_stem": dataset_stem,
            "origin": "manual_new",
            "source_directory": source_name,
            "source_image": str(image_path),
            "source_annotation": str(annotation),
            "included_train": True,
            "included_val": True,
            **stats,
        }
    )


def copy_old_validation(
    old_dataset: Path,
    output: Path,
    count: int,
    seed: int,
    manifest: list[dict[str, Any]],
) -> list[str]:
    image_root = old_dataset / "images" / "val"
    label_root = old_dataset / "labels" / "val"
    candidates = sorted(path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if count > len(candidates):
        raise ValueError(f"Requested {count} old validation examples, only {len(candidates)} exist")
    forced = next((path for path in candidates if path.stem == "exact_0065"), None)
    remainder = [path for path in candidates if path != forced]
    random.Random(seed).shuffle(remainder)
    selected = ([forced] if forced else []) + remainder[: count - (1 if forced else 0)]
    for image_path in sorted(selected):
        source_label = label_root / f"{image_path.stem}.txt"
        if not source_label.is_file():
            raise FileNotFoundError(source_label)
        dataset_stem = f"old_val__{image_path.stem}"
        shutil.copy2(image_path, output / "images" / "val" / f"{dataset_stem}{image_path.suffix.lower()}")
        shutil.copy2(source_label, output / "labels" / "val" / f"{dataset_stem}.txt")
        manifest.append(
            {
                "dataset_stem": dataset_stem,
                "origin": "old_validation_only",
                "source_directory": str(old_dataset),
                "source_image": str(image_path),
                "source_annotation": str(source_label),
                "included_train": False,
                "included_val": True,
                "width": "",
                "height": "",
                "bottle": "",
                "wine_label": "",
            }
        )
    return [path.stem for path in sorted(selected)]


def main() -> None:
    args = parse_args()
    sources = args.source or [
        Path.home() / "Downloads" / "1",
        Path.home() / "Downloads" / "2",
        Path.home() / "Downloads" / "raz met",
    ]
    sources = [path.resolve() for path in sources]
    output = args.output.resolve()
    old_dataset = args.old_dataset.resolve()
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(source)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} exists; pass --force to rebuild")
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    skipped_unannotated: list[str] = []
    source_counts: dict[str, int] = {}
    for source_index, source in enumerate(sources, start=1):
        images = image_by_stem(source)
        annotations = {path.stem: path for path in source.glob("*.json")}
        paired = sorted(images.keys() & annotations.keys())
        skipped_unannotated.extend(str(images[stem]) for stem in sorted(images.keys() - annotations.keys()))
        source_name = f"source{source_index}_{source.name.replace(' ', '_')}"
        source_counts[source_name] = len(paired)
        for stem in paired:
            copy_new_pair(source_name, images[stem], annotations[stem], output, manifest)

    old_ids = copy_old_validation(
        old_dataset, output, args.old_val_count, args.seed, manifest
    )
    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output),
                "train": "images/train",
                "val": "images/val",
                "nc": 2,
                "names": {0: "bottle", 1: "wine_label"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    fieldnames = list(manifest[0])
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "sources": {str(path): source_counts[f"source{index}_{path.name.replace(' ', '_')}"] for index, path in enumerate(sources, 1)},
        "new_manual_images": sum(source_counts.values()),
        "train_images": sum(source_counts.values()),
        "validation_images": sum(source_counts.values()) + len(old_ids),
        "old_validation_images": old_ids,
        "skipped_unannotated_images": skipped_unannotated,
        "validation_note": "All new manual images occur in both train and val by explicit user request; validation measures adaptation, not generalization.",
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
