#!/usr/bin/env python3
"""Build an identity-safe whole-bottle crop dataset with the fine-tuned YOLO.

The source render manifest provides the wine identity.  A previously validated
target-label box anchors the selection: a bottle prediction is eligible only
when it geometrically owns that label.  Confidence alone is never used to pick
between bottles.  Ambiguous selections are saved for audit but excluded from
classifier training by the retrieval index builder.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "joint_yolo" / "best.pt"
DEFAULT_LABEL_METADATA = PROJECT_ROOT / "datasets" / "dinov3_target_crops" / "crops_metadata.csv"
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "datasets" / "bottle_images_45k" / "bottle_images_manifest.csv"
DEFAULT_REFS_ROOT = PROJECT_ROOT / "datasets" / "wine-scanner" / "data" / "refs" / "rgb"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "bottle_classifier_crops"
DEFAULT_ARCHIVE = PROJECT_ROOT / "datasets" / "bottle_classifier_crops.zip"

METADATA_COLUMNS = [
    "source_path",
    "source_filename",
    "source_relative_path",
    "wine_slug",
    "crop_path",
    "status",
    "image_mode",
    "confidence",
    "bottle_x1",
    "bottle_y1",
    "bottle_x2",
    "bottle_y2",
    "label_x1",
    "label_y1",
    "label_x2",
    "label_y2",
    "label_coverage",
    "centre_margin",
    "candidate_score",
    "runner_up_score",
    "score_margin",
    "num_detections",
    "num_eligible_candidates",
    "touches_frame",
    "vertically_truncated",
    "padding",
    "image_width",
    "image_height",
    "reject_reason",
]


def choose_output_policy(
    confidence: float,
    *,
    ambiguous: bool,
    vertically_truncated: bool,
    crop_confidence_threshold: float,
) -> tuple[str, str]:
    """Choose whether to save a trusted YOLO crop or the complete source image."""
    image_mode = (
        "original_image"
        if confidence < crop_confidence_threshold
        else "yolo_crop"
    )
    if ambiguous:
        return "ambiguous", image_mode
    if image_mode == "original_image":
        return "low_confidence", image_mode
    if vertically_truncated:
        return "partial", image_mode
    return "successful", image_mode


def choose_device(requested: str) -> tuple[str | int, str]:
    if requested != "auto":
        return (0, "cuda") if requested == "cuda" else (requested, requested)
    if torch.cuda.is_available():
        return 0, "cuda"
    if torch.backends.mps.is_available():
        return "mps", "mps"
    return "cpu", "cpu"


def synchronize(device_name: str) -> None:
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        torch.mps.synchronize()


def empty_accelerator_cache(device_name: str) -> None:
    gc.collect()
    if device_name == "cuda":
        torch.cuda.empty_cache()
    elif device_name == "mps":
        torch.mps.empty_cache()


def intersection_over_box(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> float:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    width = max(0.0, min(ox2, ix2) - max(ox1, ix1))
    height = max(0.0, min(oy2, iy2) - max(oy1, iy1))
    inner_area = max((ix2 - ix1) * (iy2 - iy1), 1e-9)
    return width * height / inner_area


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def centre_margin(
    bottle: tuple[float, float, float, float],
    label: tuple[float, float, float, float],
) -> float:
    bx1, by1, bx2, by2 = bottle
    cx = (label[0] + label[2]) * 0.5
    cy = (label[1] + label[3]) * 0.5
    if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
        return -1.0
    nearest_edge = min(cx - bx1, bx2 - cx, cy - by1, by2 - cy)
    scale = max(math.sqrt(max((bx2 - bx1) * (by2 - by1), 1.0)), 1.0)
    return float(np.clip(nearest_edge / scale, 0.0, 0.5) * 2.0)


def select_target_bottle(
    result: Any,
    label_box: tuple[float, float, float, float],
    minimum_confidence: float,
    minimum_label_coverage: float,
    ambiguity_margin: float,
    duplicate_iou: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, {"detections": 0, "eligible": 0, "ambiguous": False}

    coordinates = boxes.xyxy.detach().float().cpu().numpy()
    confidences = boxes.conf.detach().float().cpu().numpy()
    classes = boxes.cls.detach().long().cpu().numpy()
    candidates: list[dict[str, Any]] = []
    detections = 0
    for raw_box, confidence, class_id in zip(coordinates, confidences, classes, strict=True):
        if int(class_id) != 0:
            continue
        detections += 1
        confidence_value = float(confidence)
        bottle_box = tuple(float(value) for value in raw_box)
        coverage = intersection_over_box(bottle_box, label_box)
        margin = centre_margin(bottle_box, label_box)
        if confidence_value < minimum_confidence or margin < 0.0:
            continue
        if coverage < minimum_label_coverage:
            continue
        # The validated label box can be very large in close-up renders, so its
        # centre is the ownership anchor and area coverage is only supporting
        # evidence. Confidence alone never decides the target bottle.
        normalized_coverage = min(coverage / 0.50, 1.0)
        score = 0.50 * confidence_value + 0.35 * margin + 0.15 * normalized_coverage
        candidates.append(
            {
                "box": bottle_box,
                "confidence": confidence_value,
                "label_coverage": coverage,
                "centre_margin": margin,
                "score": score,
            }
        )

    candidates.sort(
        key=lambda item: (item["score"], item["label_coverage"], item["confidence"]),
        reverse=True,
    )
    if not candidates:
        return None, {"detections": detections, "eligible": 0, "ambiguous": False}

    winner = candidates[0]
    distinct_runners = [
        candidate
        for candidate in candidates[1:]
        if box_iou(winner["box"], candidate["box"]) < duplicate_iou
    ]
    runner = distinct_runners[0] if distinct_runners else None
    score_margin = winner["score"] - runner["score"] if runner else math.inf
    winner["runner_up_score"] = runner["score"] if runner else math.nan
    winner["score_margin"] = score_margin
    context = {
        "detections": detections,
        "eligible": len(candidates),
        "ambiguous": runner is not None and score_margin < ambiguity_margin,
    }
    return winner, context


def padded_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    pad_x = box_width * padding
    pad_y = box_height * padding
    return (
        max(0, int(math.floor(x1 - pad_x))),
        max(0, int(math.floor(y1 - pad_y))),
        min(width, int(math.ceil(x2 + pad_x))),
        min(height, int(math.ceil(y2 + pad_y))),
    )


def scale_label_box(row: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    reference_width = float(row.get("image_width") or width)
    reference_height = float(row.get("image_height") or height)
    scale_x = width / max(reference_width, 1.0)
    scale_y = height / max(reference_height, 1.0)
    return (
        float(row["x1"]) * scale_x,
        float(row["y1"]) * scale_y,
        float(row["x2"]) * scale_x,
        float(row["y2"]) * scale_y,
    )


def draw_debug(
    image: np.ndarray,
    label_box: tuple[float, float, float, float],
    selected: dict[str, Any] | None,
    destination: Path,
    title: str,
) -> None:
    canvas = image.copy()
    lx1, ly1, lx2, ly2 = (int(round(value)) for value in label_box)
    cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (255, 80, 0), 2)
    if selected is not None:
        bx1, by1, bx2, by2 = (int(round(value)) for value in selected["box"])
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 210, 0), 3)
    cv2.putText(canvas, title[:120], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
    cv2.putText(canvas, title[:120], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def load_jobs(label_metadata: Path, source_manifest: Path) -> pd.DataFrame:
    labels = pd.read_csv(label_metadata, dtype=str).fillna("")
    manifest = pd.read_csv(source_manifest, dtype=str).fillna("")
    required_labels = {
        "source_path", "source_relative_path", "wine_slug", "status",
        "x1", "y1", "x2", "y2", "image_width", "image_height",
    }
    required_manifest = {
        "source_path", "source_relative_path", "wine_slug", "merged_path", "merged_filename",
    }
    if missing := required_labels.difference(labels.columns):
        raise ValueError(f"Label metadata is missing columns: {sorted(missing)}")
    if missing := required_manifest.difference(manifest.columns):
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
    labels = labels[labels["status"].eq("successful")].copy()
    if labels.empty:
        raise ValueError("No geometrically validated target-label rows found")
    if labels["source_path"].duplicated().any() or manifest["source_path"].duplicated().any():
        raise ValueError("Input metadata contains duplicate source_path rows")
    jobs = labels.merge(
        manifest[["source_path", "source_relative_path", "wine_slug", "merged_path", "merged_filename"]],
        on="source_path",
        how="left",
        suffixes=("_label", "_manifest"),
        validate="one_to_one",
    )
    if jobs["merged_path"].eq("").any():
        raise ValueError("Some validated label rows are absent from the source manifest")
    for field in ("source_relative_path", "wine_slug"):
        left = jobs[f"{field}_label"].astype(str)
        right = jobs[f"{field}_manifest"].astype(str)
        if not left.eq(right).all():
            raise ValueError(f"Identity/provenance mismatch in {field}")
        jobs[field] = right
    return jobs.sort_values("merged_filename").reset_index(drop=True)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def create_archive(dataset_root: Path, archive_path: Path) -> None:
    include_names = {
        "successful", "low_confidence", "ambiguous", "refs",
        "crops_metadata.csv", "training_metadata.csv", "bottle_images_manifest.csv", "build_summary.json",
    }
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()
    files = sorted(
        path for path in dataset_root.rglob("*")
        if path.is_file() and path.relative_to(dataset_root).parts[0] in include_names
    )
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in tqdm(files, desc="Creating dataset ZIP", unit="file"):
            relative = Path(dataset_root.name) / path.relative_to(dataset_root)
            archive.write(path, relative.as_posix())
    temporary.replace(archive_path)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    for required in (args.model, args.label_metadata, args.source_manifest, args.refs_root):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    if args.overwrite and args.archive.exists():
        args.archive.unlink()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for status in ("successful", "low_confidence", "ambiguous", "partial"):
        (args.output_root / status).mkdir(parents=True, exist_ok=True)
    debug_root = args.output_root / "audit"
    failed_debug_root = args.output_root / "failed_debug"
    debug_root.mkdir(exist_ok=True)
    failed_debug_root.mkdir(exist_ok=True)

    jobs = load_jobs(args.label_metadata, args.source_manifest)
    if args.sample is not None:
        jobs = jobs.sample(n=min(args.sample, len(jobs)), random_state=args.seed).sort_values("merged_filename")
    elif args.limit is not None:
        jobs = jobs.head(args.limit)
    expected_rows = len(jobs)

    partial_path = args.output_root / "crops_metadata.partial.csv"
    metadata_path = args.output_root / "crops_metadata.csv"
    existing: list[dict[str, str]] = []
    existing_source = partial_path if partial_path.exists() else metadata_path
    if existing_source.exists():
        with existing_source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != METADATA_COLUMNS:
                raise RuntimeError(
                    "Existing metadata uses an older schema; regenerate with --overwrite"
                )
            existing = list(reader)
        if not args.resume and not args.overwrite:
            raise RuntimeError(
                f"Output already contains {len(existing)} rows; use --resume or --overwrite"
            )
        if existing_source == metadata_path:
            shutil.copy2(metadata_path, partial_path)
    processed = {row["source_path"] for row in existing}
    jobs = jobs[~jobs["source_path"].isin(processed)].reset_index(drop=True)

    write_header = not partial_path.exists() or partial_path.stat().st_size == 0
    handle = partial_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
    if write_header:
        writer.writeheader()

    detector = YOLO(str(args.model))
    device, device_name = choose_device(args.device)
    counters = {
        "successful": 0,
        "low_confidence": 0,
        "ambiguous": 0,
        "partial": 0,
        "failed": 0,
        "errors": 0,
    }
    audit_written = 0
    failed_debug_written = 0
    started = time.perf_counter()

    def persist(row: dict[str, Any]) -> None:
        writer.writerow({column: row.get(column, "") for column in METADATA_COLUMNS})
        handle.flush()

    progress = tqdm(total=len(jobs), desc="Building bottle crops", unit="image")
    try:
        for batch_number, batch_start in enumerate(range(0, len(jobs), args.batch_size)):
            batch = jobs.iloc[batch_start : batch_start + args.batch_size]
            records = batch.to_dict("records")
            image_paths = [Path(record["merged_path"]) for record in records]
            try:
                predict_kwargs: dict[str, Any] = {
                    "source": [str(path) for path in image_paths],
                    "imgsz": args.imgsz,
                    "conf": args.minimum_confidence,
                    "iou": args.nms_iou,
                    "max_det": args.max_det,
                    "batch": args.batch_size,
                    "device": device,
                    "stream": False,
                    "verbose": False,
                }
                if device_name == "cuda" and args.half:
                    predict_kwargs["half"] = True
                results = detector.predict(**predict_kwargs)
                synchronize(device_name)
            except Exception as error:
                for record in records:
                    persist(
                        {
                            "source_path": record["source_path"],
                            "source_filename": Path(record["source_path"]).name,
                            "source_relative_path": record["source_relative_path"],
                            "wine_slug": record["wine_slug"],
                            "status": "failed",
                            "reject_reason": f"batch_inference_error:{type(error).__name__}:{error}",
                        }
                    )
                    counters["failed"] += 1
                    counters["errors"] += 1
                    progress.update(1)
                empty_accelerator_cache(device_name)
                continue

            for record, result in zip(records, results, strict=True):
                image = result.orig_img
                height, width = image.shape[:2]
                label_box = scale_label_box(record, width, height)
                selected, context = select_target_bottle(
                    result,
                    label_box,
                    minimum_confidence=args.minimum_confidence,
                    minimum_label_coverage=args.minimum_label_coverage,
                    ambiguity_margin=args.ambiguity_margin,
                    duplicate_iou=args.duplicate_iou,
                )
                common = {
                    "source_path": record["source_path"],
                    "source_filename": Path(record["source_path"]).name,
                    "source_relative_path": record["source_relative_path"],
                    "wine_slug": record["wine_slug"],
                    "label_x1": f"{label_box[0]:.3f}",
                    "label_y1": f"{label_box[1]:.3f}",
                    "label_x2": f"{label_box[2]:.3f}",
                    "label_y2": f"{label_box[3]:.3f}",
                    "num_detections": context["detections"],
                    "num_eligible_candidates": context["eligible"],
                    "padding": args.padding,
                    "image_width": width,
                    "image_height": height,
                }
                if selected is None:
                    persist({**common, "status": "failed", "reject_reason": "no_bottle_owns_target_label"})
                    counters["failed"] += 1
                    if failed_debug_written < args.max_failed_debug:
                        draw_debug(
                            image, label_box, None,
                            failed_debug_root / Path(record["merged_filename"]).with_suffix(".jpg"),
                            "FAILED: no bottle owns target label",
                        )
                        failed_debug_written += 1
                    progress.update(1)
                    continue

                raw_x1, raw_y1, raw_x2, raw_y2 = selected["box"]
                vertically_truncated = int(raw_y1 <= 1.0 or raw_y2 >= height - 1.0)
                status, image_mode = choose_output_policy(
                    selected["confidence"],
                    ambiguous=context["ambiguous"],
                    vertically_truncated=bool(vertically_truncated),
                    crop_confidence_threshold=args.crop_confidence_threshold,
                )
                crop_box = padded_box(selected["box"], width, height, args.padding)
                x1, y1, x2, y2 = crop_box
                output_image = image if image_mode == "original_image" else image[y1:y2, x1:x2]
                if output_image.size == 0:
                    persist({**common, "status": "failed", "reject_reason": "empty_crop"})
                    counters["failed"] += 1
                    progress.update(1)
                    continue

                crop_filename = Path(record["merged_filename"]).with_suffix(".jpg").name
                crop_path = args.output_root / status / crop_filename
                merged_source = Path(record["merged_path"])
                if image_mode == "original_image" and merged_source.suffix.lower() in {".jpg", ".jpeg"}:
                    shutil.copy2(merged_source, crop_path)
                elif not cv2.imwrite(
                    str(crop_path), output_image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                ):
                    raise IOError(f"Could not write classifier input: {crop_path}")
                touches_frame = int(x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1)
                persist(
                    {
                        **common,
                        "crop_path": str(crop_path.resolve()),
                        "status": status,
                        "image_mode": image_mode,
                        "confidence": f"{selected['confidence']:.8f}",
                        "bottle_x1": x1,
                        "bottle_y1": y1,
                        "bottle_x2": x2,
                        "bottle_y2": y2,
                        "label_coverage": f"{selected['label_coverage']:.8f}",
                        "centre_margin": f"{selected['centre_margin']:.8f}",
                        "candidate_score": f"{selected['score']:.8f}",
                        "runner_up_score": (
                            "" if math.isnan(selected["runner_up_score"])
                            else f"{selected['runner_up_score']:.8f}"
                        ),
                        "score_margin": (
                            "" if math.isinf(selected["score_margin"])
                            else f"{selected['score_margin']:.8f}"
                        ),
                        "touches_frame": touches_frame,
                        "vertically_truncated": vertically_truncated,
                        "reject_reason": (
                            "ambiguous_target_owner"
                            if status == "ambiguous"
                            else "bottle_top_or_bottom_outside_frame"
                            if status == "partial"
                            else "low_detector_confidence_original_fallback"
                            if image_mode == "original_image"
                            else ""
                        ),
                    }
                )
                counters[status] += 1
                if audit_written < args.audit_images or status == "ambiguous":
                    if audit_written < args.audit_images + args.max_ambiguous_debug:
                        draw_debug(
                            image,
                            label_box,
                            selected,
                            debug_root / status / crop_filename,
                            (
                                f"{status}/{image_mode} conf={selected['confidence']:.2f} "
                                f"coverage={selected['label_coverage']:.2f} "
                                f"margin={selected['score_margin']:.2f}"
                            ),
                        )
                        audit_written += 1
                progress.update(1)

            del results
            if device_name == "mps" and (batch_number + 1) % args.cache_clear_interval == 0:
                empty_accelerator_cache(device_name)
    finally:
        progress.close()
        handle.close()

    metadata = pd.read_csv(partial_path, dtype=str).fillna("")
    complete = len(metadata) == expected_rows
    if complete:
        partial_path.replace(metadata_path)
    status_counts = metadata["status"].value_counts().astype(int).to_dict()
    image_mode_counts = metadata["image_mode"].value_counts().astype(int).to_dict()

    refs_destination = args.output_root / "refs"
    refs_destination.mkdir(exist_ok=True)
    for reference in tqdm(sorted(args.refs_root.iterdir()), desc="Copying references", unit="file"):
        if reference.is_file():
            link_or_copy(reference, refs_destination / reference.name)
    shutil.copy2(args.source_manifest, args.output_root / "bottle_images_manifest.csv")

    successful_rows = metadata[metadata["status"].eq("successful")]
    represented = set(successful_rows["wine_slug"])
    trainable_candidates = metadata[
        metadata["status"].isin(["successful", "low_confidence"])
    ]
    trainable_per_identity = trainable_candidates.groupby("wine_slug").size()
    training_identities = set(
        trainable_per_identity[
            trainable_per_identity.ge(args.min_trainable_per_identity)
        ].index.astype(str)
    )
    trainable_rows = metadata[
        metadata["wine_slug"].isin(training_identities)
        & metadata["status"].isin(["successful", "low_confidence"])
    ].copy()
    # Keep one ignored row for every non-trainable identity so the shared index
    # builder still exports a complete 2,103-reference gallery. These rows have
    # no crop and never enter a train/validation loader.
    manifest_frame = pd.read_csv(args.source_manifest, dtype=str).fillna("")
    manifest_identities = set(manifest_frame["wine_slug"].astype(str))
    all_metadata_identities = set(metadata["wine_slug"].astype(str))
    gallery_only_identities = all_metadata_identities - training_identities
    gallery_only = (
        metadata[metadata["wine_slug"].isin(gallery_only_identities)]
        .sort_values(["wine_slug", "source_path"])
        .groupby("wine_slug", as_index=False)
        .head(1)
        .copy()
    )
    gallery_only["status"] = "gallery_only"
    gallery_only["crop_path"] = ""
    gallery_only["confidence"] = ""
    missing_metadata_identities = manifest_identities - all_metadata_identities
    missing_gallery_rows: list[dict[str, Any]] = []
    if missing_metadata_identities:
        manifest_examples = (
            manifest_frame[manifest_frame["wine_slug"].isin(missing_metadata_identities)]
            .sort_values(["wine_slug", "source_path"])
            .groupby("wine_slug", as_index=False)
            .head(1)
        )
        for row in manifest_examples.to_dict("records"):
            missing_gallery_rows.append(
                {
                    "source_path": row["source_path"],
                    "source_filename": Path(row["source_path"]).name,
                    "source_relative_path": row["source_relative_path"],
                    "wine_slug": row["wine_slug"],
                    "status": "gallery_only",
                }
            )
    missing_gallery = pd.DataFrame(missing_gallery_rows, columns=METADATA_COLUMNS)
    training_metadata = pd.concat(
        [trainable_rows, gallery_only, missing_gallery], ignore_index=True
    )
    training_metadata.to_csv(args.output_root / "training_metadata.csv", index=False)
    elapsed = time.perf_counter() - started
    confidences = pd.to_numeric(metadata["confidence"], errors="coerce").dropna()
    coverages = pd.to_numeric(metadata["label_coverage"], errors="coerce").dropna()
    summary = {
        "complete": complete,
        "rows": int(len(metadata)),
        "expected_rows": int(expected_rows),
        "status_counts": status_counts,
        "image_mode_counts": image_mode_counts,
        "successful_rate": float(status_counts.get("successful", 0) / max(len(metadata), 1)),
        "successful_identities": len(represented),
        "training_identities": len(training_identities),
        "training_rows": int(len(training_metadata)),
        "gallery_identities": int(training_metadata["wine_slug"].nunique()),
        "min_trainable_rows_per_identity": args.min_trainable_per_identity,
        "catalog_identities": len(manifest_identities),
        "identities_without_successful_crop": sorted(manifest_identities - represented),
        "identities_without_training_coverage": sorted(manifest_identities - training_identities),
        "mean_confidence": float(confidences.mean()) if len(confidences) else None,
        "mean_label_coverage": float(coverages.mean()) if len(coverages) else None,
        "touches_frame": int(pd.to_numeric(metadata["touches_frame"], errors="coerce").fillna(0).sum()),
        "model": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "device": device_name,
        "elapsed_seconds": elapsed,
        "images_per_second": len(jobs) / elapsed if elapsed > 0 else None,
        "thresholds": {
            "minimum_confidence": args.minimum_confidence,
            "crop_confidence_threshold": args.crop_confidence_threshold,
            "minimum_label_coverage": args.minimum_label_coverage,
            "ambiguity_margin": args.ambiguity_margin,
            "duplicate_iou": args.duplicate_iou,
            "padding": args.padding,
        },
    }
    summary_path = args.output_root / "build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if complete and not args.no_zip:
        create_archive(args.output_root, args.archive)
        summary["archive"] = str(args.archive.resolve())
        summary["archive_size_bytes"] = args.archive.stat().st_size
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--label-metadata", type=Path, default=DEFAULT_LABEL_METADATA)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--refs-root", type=Path, default=DEFAULT_REFS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--minimum-confidence", type=float, default=0.05)
    parser.add_argument(
        "--crop-confidence-threshold",
        "--success-confidence",
        dest="crop_confidence_threshold",
        type=float,
        default=0.75,
        help=(
            "Use the original source image below this detector confidence; "
            "--success-confidence remains as a backwards-compatible alias"
        ),
    )
    parser.add_argument("--minimum-label-coverage", type=float, default=0.01)
    parser.add_argument("--ambiguity-margin", type=float, default=0.12)
    parser.add_argument("--duplicate-iou", type=float, default=0.80)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=60)
    parser.add_argument("--padding", type=float, default=0.06)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--min-trainable-per-identity",
        "--min-successful-per-identity",
        dest="min_trainable_per_identity",
        type=int,
        default=4,
        help=(
            "Minimum combined yolo_crop/original_image rows per identity; "
            "--min-successful-per-identity remains as a compatibility alias"
        ),
    )
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--cache-clear-interval", type=int, default=8)
    parser.add_argument("--audit-images", type=int, default=100)
    parser.add_argument("--max-ambiguous-debug", type=int, default=100)
    parser.add_argument("--max-failed-debug", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.limit is not None and args.sample is not None:
        parser.error("--limit and --sample are mutually exclusive")
    if not 0 <= args.minimum_confidence <= args.crop_confidence_threshold <= 1:
        parser.error(
            "Require 0 <= minimum-confidence <= crop-confidence-threshold <= 1"
        )
    if not 0 <= args.minimum_label_coverage <= 1:
        parser.error("--minimum-label-coverage must be within 0..1")
    if args.batch_size < 1 or args.cache_clear_interval < 1:
        parser.error("Batch size and cache-clear interval must be positive")
    if args.min_trainable_per_identity < 4:
        parser.error("--min-trainable-per-identity must be at least 4 for a 2-val/2-train split")
    return args


if __name__ == "__main__":
    build_dataset(parse_args())
