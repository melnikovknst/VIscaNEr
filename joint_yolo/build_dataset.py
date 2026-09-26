#!/usr/bin/env python3
"""Build a complete two-class YOLO dataset: bottle=0, wine_label=1.

The 500-image source already has exact annotations for both classes. The
bottle-only source contributes extra training images only after the existing
label detector supplies a geometrically valid label pseudo-box inside a
manually annotated bottle. Validation always remains the exact 100-image split.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOINT_ARCHIVE = PROJECT_ROOT / "datasets" / "wine_labels_yolo_500.zip"
DEFAULT_BOTTLE_ARCHIVE = PROJECT_ROOT / "datasets" / "wine_bottles_yolo_1000.zip"
DEFAULT_LABEL_MODEL = PROJECT_ROOT / "models" / "yolo_label_detector" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "wine_bottle_label_joint"
DEFAULT_ARCHIVE = PROJECT_ROOT / "datasets" / "wine_bottle_label_joint.zip"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def choose_device(requested: str) -> str | int:
    if requested != "auto":
        return 0 if requested == "cuda" else requested
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_once(archive: Path, destination: Path, expected_root: str) -> Path:
    root = destination / expected_root
    if root.is_dir():
        return root
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        source.extractall(destination)
    if not root.is_dir():
        raise RuntimeError(f"Archive {archive} did not contain {expected_root}/")
    return root


def parse_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        class_id = int(parts[0])
        values = tuple(float(value) for value in parts[1:])
        if not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"{path}:{line_number}: invalid normalized box")
        rows.append((class_id, *values))
    return rows


def format_yolo(rows: list[tuple[int, float, float, float, float]]) -> str:
    return "".join(
        f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n"
        for class_id, cx, cy, width, height in rows
    )


def normalized_to_xyxy(
    row: tuple[int, float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    _, cx, cy, box_width, box_height = row
    return (
        (cx - box_width / 2) * width,
        (cy - box_height / 2) * height,
        (cx + box_width / 2) * width,
        (cy + box_height / 2) * height,
    )


def xyxy_to_normalized(
    class_id: int, box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        class_id,
        max(0.0, min(1.0, (x1 + x2) / (2 * width))),
        max(0.0, min(1.0, (y1 + y2) / (2 * height))),
        max(0.0, min(1.0, (x2 - x1) / width)),
        max(0.0, min(1.0, (y2 - y1) / height)),
    )


def intersection_over_label(
    label: tuple[float, float, float, float], bottle: tuple[float, float, float, float]
) -> float:
    x1 = max(label[0], bottle[0])
    y1 = max(label[1], bottle[1])
    x2 = min(label[2], bottle[2])
    y2 = min(label[3], bottle[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    label_area = max((label[2] - label[0]) * (label[3] - label[1]), 1e-9)
    return intersection / label_area


def owning_bottle(
    label: tuple[float, float, float, float],
    bottles: list[tuple[float, float, float, float]],
    minimum_coverage: float,
) -> int | None:
    cx = (label[0] + label[2]) * 0.5
    cy = (label[1] + label[3]) * 0.5
    candidates = [
        (index, intersection_over_label(label, bottle))
        for index, bottle in enumerate(bottles)
        if bottle[0] <= cx <= bottle[2] and bottle[1] <= cy <= bottle[3]
    ]
    if not candidates:
        return None
    index, coverage = max(candidates, key=lambda item: item[1])
    return index if coverage >= minimum_coverage else None


def copy_exact_joint(source_root: Path, output: Path, manifest: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"images": 0, "bottles": 0, "labels": 0, "unpaired_labels": 0}
    for split in ("train", "val"):
        for image_path in sorted((source_root / "images" / split).iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = source_root / "labels" / split / f"{image_path.stem}.txt"
            rows = parse_yolo(label_path)
            # Source classes: 0=label, 1=bottle. Final classes: 0=bottle, 1=wine_label.
            remapped = [(1 if row[0] == 0 else 0, *row[1:]) for row in rows]
            target_stem = f"exact_{image_path.stem}"
            target_image = output / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
            target_label = output / "labels" / split / f"{target_stem}.txt"
            shutil.copy2(image_path, target_image)
            target_label.write_text(format_yolo(remapped), encoding="utf-8")
            with Image.open(image_path) as image:
                width, height = image.size
            bottles = [normalized_to_xyxy(row, width, height) for row in remapped if row[0] == 0]
            label_boxes = [normalized_to_xyxy(row, width, height) for row in remapped if row[0] == 1]
            unpaired = sum(owning_bottle(box, bottles, 0.50) is None for box in label_boxes)
            counts["images"] += 1
            counts["bottles"] += len(bottles)
            counts["labels"] += len(label_boxes)
            counts["unpaired_labels"] += unpaired
            manifest.append(
                {
                    "image": str(target_image.relative_to(output)),
                    "split": split,
                    "source": "exact_joint_500",
                    "bottle_boxes": len(bottles),
                    "label_boxes": len(label_boxes),
                    "unpaired_labels": unpaired,
                    "mean_pseudo_confidence": "",
                }
            )
    return counts


def add_bottle_images_with_pseudo_labels(
    source_root: Path,
    output: Path,
    detector: YOLO,
    device: str | int,
    pseudo_confidence: float,
    minimum_coverage: float,
    manifest: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {
        "source_images": 0,
        "kept_images": 0,
        "skipped_no_label": 0,
        "bottles": 0,
        "pseudo_labels": 0,
        "rejected_unpaired": 0,
    }
    image_paths = sorted(
        path for split in ("train", "val")
        for path in (source_root / "images" / split).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for index, image_path in enumerate(tqdm(image_paths, desc="Pseudo-label bottle frames", unit="image"), start=1):
        counts["source_images"] += 1
        source_split = image_path.parent.name
        manual_path = source_root / "labels" / source_split / f"{image_path.stem}.txt"
        manual_rows = parse_yolo(manual_path)
        with Image.open(image_path) as image:
            width, height = image.size
        bottle_boxes = [normalized_to_xyxy(row, width, height) for row in manual_rows]
        result = detector.predict(
            source=str(image_path),
            imgsz=768,
            conf=pseudo_confidence,
            iou=0.65,
            max_det=30,
            device=device,
            verbose=False,
        )[0]
        pseudo_rows: list[tuple[int, float, float, float, float]] = []
        pseudo_confidences: list[float] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().float().cpu().tolist()
            confidences = result.boxes.conf.detach().float().cpu().tolist()
            classes = result.boxes.cls.detach().long().cpu().tolist()
            for raw_box, confidence, class_id in zip(boxes, confidences, classes, strict=True):
                if int(class_id) != 0:
                    continue
                box = tuple(float(value) for value in raw_box)
                if owning_bottle(box, bottle_boxes, minimum_coverage) is None:
                    counts["rejected_unpaired"] += 1
                    continue
                row = xyxy_to_normalized(1, box, width, height)
                if row[3] > 0 and row[4] > 0:
                    pseudo_rows.append(row)
                    pseudo_confidences.append(float(confidence))
        if not pseudo_rows:
            counts["skipped_no_label"] += 1
            continue
        # Manual bottle class 0 is already the final class id.
        final_rows = [(0, *row[1:]) for row in manual_rows] + pseudo_rows
        target_stem = f"pseudo_{source_split}_{image_path.stem}"
        target_image = output / "images" / "train" / f"{target_stem}{image_path.suffix.lower()}"
        target_label = output / "labels" / "train" / f"{target_stem}.txt"
        shutil.copy2(image_path, target_image)
        target_label.write_text(format_yolo(final_rows), encoding="utf-8")
        counts["kept_images"] += 1
        counts["bottles"] += len(manual_rows)
        counts["pseudo_labels"] += len(pseudo_rows)
        manifest.append(
            {
                "image": str(target_image.relative_to(output)),
                "split": "train",
                "source": "bottle_1000_plus_label_pseudo",
                "bottle_boxes": len(manual_rows),
                "label_boxes": len(pseudo_rows),
                "unpaired_labels": 0,
                "mean_pseudo_confidence": sum(pseudo_confidences) / len(pseudo_confidences),
            }
        )
        if index % 50 == 0:
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == 0:
                torch.cuda.empty_cache()
    return counts


def validate_dataset(output: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}}
    for split in ("train", "val"):
        images = sorted(path for path in (output / "images" / split).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted((output / "labels" / split).glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise RuntimeError(f"Image/label mismatch in {split}")
        class_counts = {0: 0, 1: 0}
        empty = 0
        for path in labels:
            rows = parse_yolo(path)
            empty += int(not rows)
            for row in rows:
                if row[0] not in class_counts:
                    raise ValueError(f"Unexpected class {row[0]} in {path}")
                class_counts[row[0]] += 1
        if empty:
            raise RuntimeError(f"Found {empty} empty labels in {split}")
        summary["splits"][split] = {
            "images": len(images),
            "bottle_boxes": class_counts[0],
            "wine_label_boxes": class_counts[1],
        }
    return summary


def create_archive(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in tqdm(sorted(source.rglob("*")), desc="Archive joint dataset", unit="file"):
            if path.is_file():
                archive.write(path, Path(source.name) / path.relative_to(source))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-archive", type=Path, default=DEFAULT_JOINT_ARCHIVE)
    parser.add_argument("--bottle-archive", type=Path, default=DEFAULT_BOTTLE_ARCHIVE)
    parser.add_argument("--label-model", type=Path, default=DEFAULT_LABEL_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pseudo-confidence", type=float, default=0.25)
    parser.add_argument("--minimum-label-coverage", type=float, default=0.60)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.joint_archive, args.bottle_archive, args.label_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    staging = PROJECT_ROOT / "runs" / "joint_yolo" / "staging"
    exact_root = extract_once(args.joint_archive.resolve(), staging, "wine_labels_yolo_500")
    bottle_root = extract_once(args.bottle_archive.resolve(), staging, "wine_bottles_yolo_1000")
    manifest: list[dict[str, Any]] = []
    started = time.time()
    exact_counts = copy_exact_joint(exact_root, output, manifest)
    device = choose_device(args.device)
    detector = YOLO(str(args.label_model.resolve()))
    pseudo_counts = add_bottle_images_with_pseudo_labels(
        bottle_root,
        output,
        detector,
        device,
        args.pseudo_confidence,
        args.minimum_label_coverage,
        manifest,
    )
    data_yaml = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 2\n"
        "names:\n"
        "  0: bottle\n"
        "  1: wine_label\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    validation = validate_dataset(output)
    summary = {
        "classes": {"0": "bottle", "1": "wine_label"},
        "exact_source": exact_counts,
        "pseudo_source": pseudo_counts,
        "dataset": validation,
        "pseudo_confidence": args.pseudo_confidence,
        "minimum_label_coverage": args.minimum_label_coverage,
        "validation_policy": "exact joint val only; pseudo-labels train only",
        "elapsed_seconds": time.time() - started,
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    create_archive(output, args.archive.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"DATASET | {output}", flush=True)
    print(f"ARCHIVE | {args.archive.resolve()}", flush=True)


if __name__ == "__main__":
    main()
