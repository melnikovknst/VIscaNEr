"""Batch-crop wine labels from the unified 45k bottle-image directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "datasets" / "bottle_images_45k"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "yolo_label_detector" / "crops"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "datasets" / "yolo_label_detector" / "predictions_debug"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo_label_detector" / "best.pt"
DEFAULT_MANIFEST_PATH = DEFAULT_INPUT_DIR / "bottle_images_manifest.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
METADATA_COLUMNS = [
    "source_path",
    "source_filename",
    "crop_path",
    "status",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "padding",
    "image_width",
    "image_height",
]


def select_device() -> tuple[Any, str]:
    if torch.cuda.is_available():
        return 0, "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "mps"
    return "cpu", "cpu"


def synchronize(device_name: str) -> None:
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        torch.mps.synchronize()


def list_images(input_dir: Path, limit: int | None = None) -> list[Path]:
    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:limit] if limit is not None else images


def load_source_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    required = {"merged_filename", "source_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return {
        row["merged_filename"]: row.to_dict()
        for _, row in manifest.iterrows()
    }


def unique_output_path(directory: Path, input_path: Path, extension: str) -> Path:
    return directory / f"{input_path.stem}{extension}"


def extract_best_detection(result: Any) -> tuple[float | None, tuple[float, float, float, float] | None]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    candidates = [
        (float(score), tuple(float(value) for value in coordinates))
        for coordinates, score, class_id in zip(xyxy, confidence, classes)
        if class_id == 0
    ]
    return max(candidates, key=lambda item: item[0]) if candidates else (None, None)


def padded_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    raw_x1, raw_y1, raw_x2, raw_y2 = box
    box_width = max(0.0, raw_x2 - raw_x1)
    box_height = max(0.0, raw_y2 - raw_y1)
    x1 = max(0, int(math.floor(raw_x1 - box_width * padding)))
    y1 = max(0, int(math.floor(raw_y1 - box_height * padding)))
    x2 = min(image_width, int(math.ceil(raw_x2 + box_width * padding)))
    y2 = min(image_height, int(math.ceil(raw_y2 + box_height * padding)))
    return x1, y1, x2, y2


def write_debug(
    image: np.ndarray,
    destination: Path,
    status: str,
    confidence: float | None,
    box: tuple[int, int, int, int] | None,
) -> None:
    debug = image.copy()
    color = (45, 170, 45) if status == "successful" else (30, 100, 240)
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 3)
        text = f"wine_label {confidence:.3f}" if confidence is not None else "wine_label"
        cv2.putText(debug, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(debug, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(str(destination), debug, [cv2.IMWRITE_JPEG_QUALITY, 90])


def existing_metadata_rows(metadata_path: Path, partial_path: Path) -> list[dict[str, str]]:
    source = partial_path if partial_path.exists() else metadata_path
    if not source.exists():
        return []
    with source.open(newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def run_crop(
    input_dir: Path,
    output_root: Path,
    debug_dir: Path,
    model_path: Path,
    manifest_path: Path,
    confidence_threshold: float,
    low_confidence_threshold: float,
    padding: float,
    batch_size: int,
    imgsz: int,
    save_debug_successful: bool,
    limit: int | None,
) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")
    if not (0 <= low_confidence_threshold <= confidence_threshold <= 1):
        raise ValueError("Require 0 <= low_confidence_threshold <= confidence_threshold <= 1")

    successful_dir = output_root / "successful"
    low_confidence_dir = output_root / "low_confidence"
    failed_dir = output_root / "failed"
    for directory in (successful_dir, low_confidence_dir, failed_dir, debug_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metadata_path = output_root / "crops_metadata.csv"
    partial_path = output_root / "crops_metadata.partial.csv"
    summary_path = output_root / "crop_summary.json"
    error_log_path = output_root / "crop_errors.log"
    existing_rows = existing_metadata_rows(metadata_path, partial_path)
    processed_source_paths = {row["source_path"] for row in existing_rows}
    if existing_rows and not partial_path.exists():
        shutil.copy2(metadata_path, partial_path)

    manifest = load_source_manifest(manifest_path)
    all_images = list_images(input_dir, limit=limit)
    pending_images = []
    for input_path in all_images:
        source_path = manifest.get(input_path.name, {}).get("source_path", str(input_path.resolve()))
        if source_path not in processed_source_paths:
            pending_images.append(input_path)

    device, device_name = select_device()
    detector = YOLO(str(model_path))
    print(f"Device: {device_name}")
    print(f"Model: {model_path}")
    print(f"Input images: {len(all_images):,}; already processed: {len(all_images) - len(pending_images):,}")
    print(f"Pending: {len(pending_images):,}; batch size: {batch_size}")

    write_header = not partial_path.exists() or partial_path.stat().st_size == 0
    metadata_handle = partial_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(metadata_handle, fieldnames=METADATA_COLUMNS)
    if write_header:
        writer.writeheader()

    new_rows = 0
    new_errors = 0
    started_at = time.perf_counter()

    def persist_result(input_path: Path, result: Any) -> None:
        nonlocal new_rows
        manifest_row = manifest.get(input_path.name, {})
        original_source_path = Path(manifest_row.get("source_path", str(input_path.resolve())))
        original_source_filename = original_source_path.name
        image = result.orig_img
        image_height, image_width = image.shape[:2]
        confidence, raw_box = extract_best_detection(result)
        crop_box = padded_box(raw_box, image_width, image_height, padding) if raw_box is not None else None

        if confidence is not None and confidence >= confidence_threshold:
            status = "successful"
        elif confidence is not None and confidence >= low_confidence_threshold:
            status = "low_confidence"
        else:
            status = "failed"

        crop_path: Path
        if status in {"successful", "low_confidence"} and crop_box is not None:
            x1, y1, x2, y2 = crop_box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError("Empty crop after boundary clamping")
            destination_dir = successful_dir if status == "successful" else low_confidence_dir
            crop_path = unique_output_path(destination_dir, input_path, ".jpg")
            if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise IOError(f"Could not write crop: {crop_path}")
        else:
            crop_path = unique_output_path(failed_dir, input_path, input_path.suffix.lower())
            shutil.copy2(input_path, crop_path)

        if save_debug_successful or status in {"low_confidence", "failed"}:
            write_debug(
                image,
                unique_output_path(debug_dir, input_path, ".jpg"),
                status,
                confidence,
                crop_box,
            )

        x1, y1, x2, y2 = crop_box if crop_box is not None else (None, None, None, None)
        writer.writerow(
            {
                "source_path": str(original_source_path),
                "source_filename": original_source_filename,
                "crop_path": str(crop_path.resolve()),
                "status": status,
                "confidence": "" if confidence is None else f"{confidence:.8f}",
                "x1": "" if x1 is None else x1,
                "y1": "" if y1 is None else y1,
                "x2": "" if x2 is None else x2,
                "y2": "" if y2 is None else y2,
                "padding": padding,
                "image_width": image_width,
                "image_height": image_height,
            }
        )
        metadata_handle.flush()
        new_rows += 1

    def persist_failure(input_path: Path, error: Exception) -> None:
        """Record an unreadable/unprocessable image without stopping the run."""
        nonlocal new_rows, new_errors
        manifest_row = manifest.get(input_path.name, {})
        original_source_path = Path(manifest_row.get("source_path", str(input_path.resolve())))
        failed_path = unique_output_path(failed_dir, input_path, input_path.suffix.lower())
        shutil.copy2(input_path, failed_path)
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        image_width = image.shape[1] if image is not None else ""
        image_height = image.shape[0] if image is not None else ""
        if image is not None:
            write_debug(
                image,
                unique_output_path(debug_dir, input_path, ".jpg"),
                "failed",
                None,
                None,
            )
        writer.writerow(
            {
                "source_path": str(original_source_path),
                "source_filename": original_source_path.name,
                "crop_path": str(failed_path.resolve()),
                "status": "failed",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "padding": padding,
                "image_width": image_width,
                "image_height": image_height,
            }
        )
        metadata_handle.flush()
        new_rows += 1
        new_errors += 1
        with error_log_path.open("a", encoding="utf-8") as error_handle:
            error_handle.write(f"{input_path}\t{type(error).__name__}: {error}\n")

    progress = tqdm(total=len(pending_images), desc="Cropping 45k bottle images", unit="image")
    try:
        for batch_start in range(0, len(pending_images), batch_size):
            batch_paths = pending_images[batch_start:batch_start + batch_size]
            try:
                results = detector.predict(
                    source=[str(path) for path in batch_paths],
                    imgsz=imgsz,
                    conf=0.001,
                    device=device,
                    verbose=False,
                )
                synchronize(device_name)
                for input_path, result in zip(batch_paths, results):
                    try:
                        persist_result(input_path, result)
                    except Exception as exc:
                        persist_failure(input_path, exc)
                    finally:
                        progress.update(1)
            except Exception as batch_exc:
                with error_log_path.open("a", encoding="utf-8") as error_handle:
                    error_handle.write(f"BATCH {batch_start}\t{type(batch_exc).__name__}: {batch_exc}\n")
                for input_path in batch_paths:
                    try:
                        result = detector.predict(
                            source=str(input_path),
                            imgsz=imgsz,
                            conf=0.001,
                            device=device,
                            verbose=False,
                        )[0]
                        synchronize(device_name)
                        persist_result(input_path, result)
                    except Exception as exc:
                        persist_failure(input_path, exc)
                    finally:
                        progress.update(1)
    finally:
        progress.close()
        metadata_handle.close()

    metadata = pd.read_csv(partial_path)
    if len(metadata) == len(all_images):
        partial_path.replace(metadata_path)
    status_counts = metadata["status"].value_counts().to_dict()
    confidences = pd.to_numeric(metadata["confidence"], errors="coerce").dropna()
    elapsed_seconds = time.perf_counter() - started_at
    summary = {
        "input_dir": str(input_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "device": device_name,
        "total": int(len(metadata)),
        "successful": int(status_counts.get("successful", 0)),
        "low_confidence": int(status_counts.get("low_confidence", 0)),
        "failed": int(status_counts.get("failed", 0)),
        "average_confidence": float(confidences.mean()) if not confidences.empty else None,
        "new_rows": new_rows,
        "new_errors": new_errors,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": (new_rows / elapsed_seconds) if elapsed_seconds > 0 else None,
        "metadata_csv": str((metadata_path if metadata_path.exists() else partial_path).resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--low-confidence", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-debug-successful", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run_crop(
            input_dir=arguments.input_dir,
            output_root=arguments.output_root,
            debug_dir=arguments.debug_dir,
            model_path=arguments.model,
            manifest_path=arguments.manifest,
            confidence_threshold=arguments.confidence,
            low_confidence_threshold=arguments.low_confidence,
            padding=arguments.padding,
            batch_size=arguments.batch_size,
            imgsz=arguments.imgsz,
            save_debug_successful=arguments.save_debug_successful,
            limit=arguments.limit,
        )
    except KeyboardInterrupt:
        print("Interrupted. Progress is preserved in crops_metadata.partial.csv and the next run will resume.")
        sys.exit(130)
