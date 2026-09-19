#!/usr/bin/env python3
"""Build high-precision DINO crops aligned to the generated target bottle.

The 45k source frames are synthetic and their manifest stores the target wine
identity and render seed.  This builder re-renders a fresh scene from the same
identity/seed while tracking the exact target-bottle silhouette.  YOLO still
localises the label, but a detection is accepted only when its centre and most
of its area lie inside the tracked target bottle.  Labels on neighbouring
bottles therefore cannot inherit the target wine identity.

The existing inference crop selector is intentionally not modified.  This is
an offline training-data builder, not production inference logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
GENERATOR_ROOT = PROJECT_ROOT / "datasets" / "wine-scanner"
SOURCE_MANIFEST = PROJECT_ROOT / "datasets" / "bottle_images_45k" / "bottle_images_manifest.csv"
RENDER_MANIFEST = GENERATOR_ROOT / "data" / "trainset" / "manifest.csv"
DEFAULT_BACKGROUND_DIR = PROJECT_ROOT / "datasets" / "bottle_images_45k"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolo_label_detector" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "dinov3_target_crops"

METADATA_COLUMNS = [
    "source_path",
    "source_filename",
    "source_relative_path",
    "wine_slug",
    "source_split",
    "source_view",
    "source_seed",
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
    "target_visible_fraction",
    "target_mask_pixels",
    "target_overlap",
    "candidate_score",
    "num_detections",
    "num_target_candidates",
    "render_mode",
    "render_background",
    "render_neighbors",
    "reject_reason",
]


@dataclass
class TrackedObject:
    mask: np.ndarray
    full_alpha_pixels: int


class TargetTrackingContext(AbstractContextManager["TargetTrackingContext"]):
    """Instrument the existing generator without changing its source code."""

    def __init__(self, augment_module: Any) -> None:
        self.module = augment_module
        self.original_paste = augment_module.paste
        self.original_perspective = augment_module.perspective
        self.objects: list[TrackedObject] = []

    def __enter__(self) -> "TargetTrackingContext":
        self.module.paste = self._paste
        self.module.perspective = self._perspective
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.module.paste = self.original_paste
        self.module.perspective = self.original_perspective

    def reset(self) -> None:
        self.objects = []

    def _paste(self, canvas: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
        self.original_paste(canvas, rgba, x, y)
        height, width = canvas.shape[:2]
        object_height, object_width = rgba.shape[:2]
        alpha = rgba[..., 3] > 32
        mask = np.zeros((height, width), dtype=np.uint8)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + object_width, width), min(y + object_height, height)
        if x0 < x1 and y0 < y1:
            source_alpha = alpha[y0 - y : y1 - y, x0 - x : x1 - x]
            mask[y0:y1, x0:x1] = source_alpha.astype(np.uint8) * 255
        self.objects.append(
            TrackedObject(mask=mask, full_alpha_pixels=int(alpha.sum()))
        )

    def _perspective(
        self,
        image: np.ndarray,
        rng: Any,
        max_shift: float,
    ) -> np.ndarray:
        """Exact generator perspective transform, also applied to masks."""
        height, width = image.shape[:2]
        source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        jitter = [
            [
                rng.uniform(-max_shift, max_shift) * width,
                rng.uniform(-max_shift, max_shift) * height,
            ]
            for _ in range(4)
        ]
        destination = source + np.float32(jitter)
        angle = rng.uniform(-10, 10)
        rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        destination = cv2.transform(destination[None], rotation)[0]
        matrix = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
        for tracked in self.objects:
            tracked.mask = cv2.warpPerspective(
                tracked.mask,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
        return cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def render_target(
        self,
        augmenter: Any,
        reference: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        self.reset()
        image, parameters = augmenter(reference, seed)
        if not self.objects:
            raise RuntimeError("Generator produced no tracked bottle objects")
        # In every implemented generator mode neighbours are pasted first and
        # the requested/target bottle is pasted last.
        target = self.objects[-1]
        visible_pixels = int((target.mask > 0).sum())
        visible_fraction = visible_pixels / max(target.full_alpha_pixels, 1)
        return image, target.mask, float(visible_fraction), parameters


def choose_device() -> tuple[str | int, str]:
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


def load_jobs(source_manifest: Path, render_manifest: Path) -> pd.DataFrame:
    source = pd.read_csv(source_manifest, dtype=str).fillna("")
    render = pd.read_csv(render_manifest, dtype=str).fillna("")
    required_source = {
        "source_path",
        "source_relative_path",
        "wine_slug",
        "merged_filename",
    }
    required_render = {"path", "slug", "split", "view", "seed"}
    if missing := required_source.difference(source.columns):
        raise ValueError(f"Source manifest missing columns: {sorted(missing)}")
    if missing := required_render.difference(render.columns):
        raise ValueError(f"Render manifest missing columns: {sorted(missing)}")
    if source["source_path"].duplicated().any():
        raise ValueError("Source manifest has duplicate source_path values")
    render["source_relative_path"] = render["path"].str.replace(
        r"^images/", "", regex=True
    )
    jobs = source.merge(
        render[
            ["source_relative_path", "slug", "split", "view", "seed", "params"]
        ],
        on="source_relative_path",
        how="left",
        validate="one_to_one",
    )
    if jobs[["slug", "seed"]].eq("").any().any():
        raise ValueError("Some source images are absent from the render manifest")
    mismatched = jobs[~jobs["wine_slug"].eq(jobs["slug"])]
    if not mismatched.empty:
        raise ValueError(f"Manifest identity mismatch: {mismatched.head(3).to_dict('records')}")
    return jobs.sort_values("merged_filename").reset_index(drop=True)


def padded_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    raw_x1, raw_y1, raw_x2, raw_y2 = box
    width = max(0.0, raw_x2 - raw_x1)
    height = max(0.0, raw_y2 - raw_y1)
    return (
        max(0, int(math.floor(raw_x1 - width * padding))),
        max(0, int(math.floor(raw_y1 - height * padding))),
        min(image_width, int(math.ceil(raw_x2 + width * padding))),
        min(image_height, int(math.ceil(raw_y2 + height * padding))),
    )


def select_target_label(
    result: Any,
    target_mask: np.ndarray,
    minimum_confidence: float,
    minimum_target_overlap: float,
) -> tuple[dict[str, Any] | None, int, int]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, 0, 0
    image_height, image_width = target_mask.shape
    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    candidates: list[dict[str, Any]] = []
    detections = 0

    for box, confidence, class_id in zip(coordinates, confidences, classes):
        if class_id != 0:
            continue
        detections += 1
        confidence_value = float(confidence)
        if confidence_value < minimum_confidence:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        ix1 = max(0, min(image_width - 1, int(math.floor(x1))))
        iy1 = max(0, min(image_height - 1, int(math.floor(y1))))
        ix2 = max(ix1 + 1, min(image_width, int(math.ceil(x2))))
        iy2 = max(iy1 + 1, min(image_height, int(math.ceil(y2))))
        box_mask = target_mask[iy1:iy2, ix1:ix2] > 0
        overlap = float(box_mask.mean()) if box_mask.size else 0.0
        center_x = max(0, min(image_width - 1, int(round((x1 + x2) / 2))))
        center_y = max(0, min(image_height - 1, int(round((y1 + y2) / 2))))
        center_inside = bool(target_mask[center_y, center_x] > 0)
        if not center_inside or overlap < minimum_target_overlap:
            continue
        area_ratio = ((ix2 - ix1) * (iy2 - iy1)) / max(
            int((target_mask > 0).sum()), 1
        )
        if area_ratio < 0.005 or area_ratio > 1.20:
            continue
        score = confidence_value + 0.25 * overlap + 0.03 * min(math.sqrt(area_ratio), 1.0)
        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "confidence": confidence_value,
                "target_overlap": overlap,
                "area_ratio": area_ratio,
                "score": score,
            }
        )

    candidates.sort(key=lambda item: (item["score"], item["confidence"]), reverse=True)
    return (candidates[0] if candidates else None), detections, len(candidates)


def write_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int] | None,
    destination: Path,
    title: str,
) -> None:
    overlay = image.copy()
    contour_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 120, 0), 2)
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        overlay,
        title[:110],
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        title[:110],
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])


def existing_rows(metadata_path: Path, partial_path: Path) -> list[dict[str, str]]:
    candidate = partial_path if partial_path.exists() else metadata_path
    if not candidate.exists():
        return []
    with candidate.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    for required in (
        args.model,
        args.source_manifest,
        args.render_manifest,
        GENERATOR_ROOT,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    successful_dir = args.output_root / "successful"
    rejected_debug_dir = args.output_root / "rejected_debug"
    audit_dir = args.output_root / "audit"
    for directory in (successful_dir, rejected_debug_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metadata_path = args.output_root / "crops_metadata.csv"
    partial_path = args.output_root / "crops_metadata.partial.csv"
    summary_path = args.output_root / "build_summary.json"
    rows = existing_rows(metadata_path, partial_path)
    processed = {row["source_path"] for row in rows}
    if rows and not args.resume and not args.overwrite:
        raise RuntimeError(
            f"Output already contains {len(rows)} rows. Use --resume or --overwrite."
        )
    if rows and not partial_path.exists():
        shutil.copy2(metadata_path, partial_path)

    jobs = load_jobs(args.source_manifest, args.render_manifest)
    if args.sample is not None:
        jobs = jobs.sample(
            n=min(args.sample, len(jobs)),
            random_state=args.sample_seed,
        ).sort_values("merged_filename")
    elif args.limit is not None:
        jobs = jobs.head(args.limit)
    expected_total = len(jobs)
    jobs = jobs[~jobs["source_path"].isin(processed)].reset_index(drop=True)

    sys.path.insert(0, str(GENERATOR_ROOT))
    from scanner import augment as augment_module  # type: ignore
    from scanner.catalog import load_catalog  # type: ignore
    from scanner.normalize import load_rgba  # type: ignore

    wines = load_catalog()
    wine_by_slug = {wine.slug: wine for wine in wines}
    references = [wine.ref_rgba for wine in wines]
    background_paths = (
        augment_module.list_images(args.background_dir)
        if args.background_dir.is_dir()
        else []
    )
    augmenter = augment_module.FieldAugmenter(
        references,
        background_paths,
        long_side=args.long_side,
    )
    detector = YOLO(str(args.model))
    device, device_name = choose_device()

    write_header = not partial_path.exists() or partial_path.stat().st_size == 0
    handle = partial_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
    if write_header:
        writer.writeheader()

    counters = {"successful": 0, "rejected": 0, "errors": 0}
    audit_written = 0
    rejected_debug_written = 0
    started = time.perf_counter()

    def persist_row(row: dict[str, Any]) -> None:
        writer.writerow({key: row.get(key, "") for key in METADATA_COLUMNS})
        handle.flush()

    progress = tqdm(total=len(jobs), desc="Building target-aligned crops", unit="image")
    try:
        with TargetTrackingContext(augment_module) as tracker:
            for batch_start in range(0, len(jobs), args.batch_size):
                batch = jobs.iloc[batch_start : batch_start + args.batch_size]
                rendered: list[dict[str, Any]] = []
                for job in batch.to_dict("records"):
                    base = {
                        "source_path": job["source_path"],
                        "source_filename": Path(job["source_path"]).name,
                        "source_relative_path": job["source_relative_path"],
                        "wine_slug": job["wine_slug"],
                        "source_split": job["split"],
                        "source_view": job["view"],
                        "source_seed": job["seed"],
                        "padding": args.padding,
                    }
                    try:
                        wine = wine_by_slug[job["wine_slug"]]
                        image, mask, visible_fraction, parameters = tracker.render_target(
                            augmenter,
                            load_rgba(wine.ref_rgba),
                            int(job["seed"]),
                        )
                        if visible_fraction < args.minimum_target_visibility:
                            persist_row(
                                {
                                    **base,
                                    "status": "rejected",
                                    "target_visible_fraction": f"{visible_fraction:.8f}",
                                    "target_mask_pixels": int((mask > 0).sum()),
                                    "image_width": image.shape[1],
                                    "image_height": image.shape[0],
                                    "render_mode": parameters.get("mode", ""),
                                    "render_background": parameters.get("background", ""),
                                    "render_neighbors": parameters.get("neighbors", 0),
                                    "reject_reason": "target_visibility_below_threshold",
                                }
                            )
                            counters["rejected"] += 1
                            progress.update(1)
                            continue
                        rendered.append(
                            {
                                "base": base,
                                "image": image,
                                "mask": mask,
                                "visible_fraction": visible_fraction,
                                "parameters": parameters,
                                "filename": Path(job["merged_filename"]).with_suffix(".jpg").name,
                            }
                        )
                    except Exception as error:
                        persist_row(
                            {
                                **base,
                                "status": "rejected",
                                "reject_reason": f"render_error:{type(error).__name__}:{error}",
                            }
                        )
                        counters["rejected"] += 1
                        counters["errors"] += 1
                        progress.update(1)

                if not rendered:
                    continue
                try:
                    results = detector.predict(
                        source=[item["image"] for item in rendered],
                        imgsz=args.imgsz,
                        conf=0.001,
                        iou=0.50,
                        device=device,
                        verbose=False,
                    )
                    synchronize(device_name)
                except Exception as error:
                    for item in rendered:
                        persist_row(
                            {
                                **item["base"],
                                "status": "rejected",
                                "image_width": item["image"].shape[1],
                                "image_height": item["image"].shape[0],
                                "target_visible_fraction": f"{item['visible_fraction']:.8f}",
                                "target_mask_pixels": int((item["mask"] > 0).sum()),
                                "reject_reason": f"batch_inference_error:{type(error).__name__}:{error}",
                            }
                        )
                        counters["rejected"] += 1
                        counters["errors"] += 1
                        progress.update(1)
                    continue

                for item, result in zip(rendered, results, strict=True):
                    image = result.orig_img
                    image_height, image_width = image.shape[:2]
                    selected, detections, target_candidates = select_target_label(
                        result,
                        item["mask"],
                        minimum_confidence=args.minimum_confidence,
                        minimum_target_overlap=args.minimum_target_overlap,
                    )
                    parameters = item["parameters"]
                    common = {
                        **item["base"],
                        "image_width": image_width,
                        "image_height": image_height,
                        "target_visible_fraction": f"{item['visible_fraction']:.8f}",
                        "target_mask_pixels": int((item["mask"] > 0).sum()),
                        "num_detections": detections,
                        "num_target_candidates": target_candidates,
                        "render_mode": parameters.get("mode", ""),
                        "render_background": parameters.get("background", ""),
                        "render_neighbors": parameters.get("neighbors", 0),
                    }
                    if selected is None:
                        persist_row(
                            {
                                **common,
                                "status": "rejected",
                                "reject_reason": "no_yolo_box_aligned_to_target_bottle",
                            }
                        )
                        counters["rejected"] += 1
                        if rejected_debug_written < args.max_rejected_debug:
                            write_overlay(
                                image,
                                item["mask"],
                                None,
                                rejected_debug_dir / item["filename"],
                                "REJECTED: no target-aligned label",
                            )
                            rejected_debug_written += 1
                        progress.update(1)
                        continue

                    crop_box = padded_box(
                        selected["box"], image_width, image_height, args.padding
                    )
                    x1, y1, x2, y2 = crop_box
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        persist_row(
                            {
                                **common,
                                "status": "rejected",
                                "reject_reason": "empty_crop_after_clamping",
                            }
                        )
                        counters["rejected"] += 1
                        progress.update(1)
                        continue
                    crop_path = successful_dir / item["filename"]
                    if not cv2.imwrite(
                        str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]
                    ):
                        raise IOError(f"Could not write crop: {crop_path}")
                    persist_row(
                        {
                            **common,
                            "crop_path": str(crop_path.resolve()),
                            "status": "successful",
                            "confidence": f"{selected['confidence']:.8f}",
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "target_overlap": f"{selected['target_overlap']:.8f}",
                            "candidate_score": f"{selected['score']:.8f}",
                            "reject_reason": "",
                        }
                    )
                    counters["successful"] += 1
                    if audit_written < args.audit_images:
                        write_overlay(
                            image,
                            item["mask"],
                            crop_box,
                            audit_dir / item["filename"],
                            (
                                f"TARGET OK conf={selected['confidence']:.2f} "
                                f"overlap={selected['target_overlap']:.2f}"
                            ),
                        )
                        audit_written += 1
                    progress.update(1)
    finally:
        progress.close()
        handle.close()

    metadata = pd.read_csv(partial_path, dtype=str).fillna("")
    complete = len(metadata) == expected_total
    if complete:
        partial_path.replace(metadata_path)
    confidences = pd.to_numeric(metadata["confidence"], errors="coerce").dropna()
    overlaps = pd.to_numeric(metadata["target_overlap"], errors="coerce").dropna()
    status_counts = metadata["status"].value_counts().astype(int).to_dict()
    elapsed = time.perf_counter() - started
    summary = {
        "complete": complete,
        "rows": int(len(metadata)),
        "expected_rows": int(expected_total),
        "successful": int(status_counts.get("successful", 0)),
        "rejected": int(status_counts.get("rejected", 0)),
        "success_rate": float(status_counts.get("successful", 0) / max(len(metadata), 1)),
        "average_confidence": float(confidences.mean()) if not confidences.empty else None,
        "average_target_overlap": float(overlaps.mean()) if not overlaps.empty else None,
        "device": device_name,
        "background_dir": str(args.background_dir.resolve()),
        "model": str(args.model.resolve()),
        "output_root": str(args.output_root.resolve()),
        "elapsed_seconds": elapsed,
        "images_per_second": len(jobs) / elapsed if elapsed > 0 else None,
        "thresholds": {
            "minimum_confidence": args.minimum_confidence,
            "minimum_target_overlap": args.minimum_target_overlap,
            "minimum_target_visibility": args.minimum_target_visibility,
            "padding": args.padding,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--render-manifest", type=Path, default=RENDER_MANIFEST)
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BACKGROUND_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--minimum-confidence", type=float, default=0.05)
    parser.add_argument("--minimum-target-overlap", type=float, default=0.60)
    parser.add_argument("--minimum-target-visibility", type=float, default=0.35)
    parser.add_argument("--audit-images", type=int, default=100)
    parser.add_argument("--max-rejected-debug", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", type=int, help="Deterministic random smoke-test sample")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.resume and arguments.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if arguments.limit is not None and arguments.sample is not None:
        parser.error("--limit and --sample are mutually exclusive")
    if not 0 <= arguments.minimum_confidence <= 1:
        parser.error("--minimum-confidence must be within 0..1")
    if not 0 <= arguments.minimum_target_overlap <= 1:
        parser.error("--minimum-target-overlap must be within 0..1")
    if not 0 <= arguments.minimum_target_visibility <= 1:
        parser.error("--minimum-target-visibility must be within 0..1")
    return arguments


if __name__ == "__main__":
    build_dataset(parse_args())
