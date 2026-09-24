#!/usr/bin/env python3
"""Final VIscaNEr inference: full-bottle YOLO -> DINOv3-B retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO

from dinov3_retrieval import (
    SUPPORTED_EXTENSIONS,
    build_transforms,
    choose_device,
    load_trained_model,
    open_rgb,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YOLO = PROJECT_ROOT / "models" / "bottle_reranker" / "best_bottle_detector.pt"
DEFAULT_DINO_WEIGHTS = PROJECT_ROOT / "models" / "dinov3" / "model.safetensors"
DEFAULT_DINO_CHECKPOINT = (
    PROJECT_ROOT / "models" / "trained_checkpoints" / "dinov3_vitb16_bottles_best_full.pt"
)
DEFAULT_REFS_ROOT = PROJECT_ROOT / "datasets" / "bottle_classifier_crops" / "refs"
DEFAULT_DATASET_ZIP = PROJECT_ROOT / "datasets" / "bottle_classifier_crops.zip"
DEFAULT_GALLERY_CACHE = (
    PROJECT_ROOT / "runs" / "inference" / "dinov3_vitb16_bottles_gallery.pt"
)
DEFAULT_AMBIGUITY_MARGIN = 0.06
DEFAULT_DUPLICATE_IOU = 0.80


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def select_target_detections(
    result: Any,
    image_width: int,
    image_height: int,
    candidate_confidence: float,
    crosshair_x: float,
    crosshair_y: float,
    ambiguity_confidence: float,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    duplicate_iou: float = DEFAULT_DUPLICATE_IOU,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select one central bottle, or two when the crosshair target is ambiguous."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], [], {"ambiguous": False, "score_gap": None, "pair_iou": None}
    coordinates = boxes.xyxy.detach().float().cpu().numpy()
    confidences = boxes.conf.detach().float().cpu().numpy()
    classes = boxes.cls.detach().long().cpu().numpy()
    candidates: list[dict[str, Any]] = []
    sigma_x, sigma_y = 0.24, 0.34
    crosshair_sigma = 0.16
    for raw_box, raw_confidence, class_id in zip(
        coordinates, confidences, classes, strict=True
    ):
        confidence = float(raw_confidence)
        if int(class_id) != 0 or confidence < candidate_confidence:
            continue
        x1, y1, x2, y2 = (float(value) for value in raw_box)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            continue

        center_x = (x1 + x2) * 0.5 / image_width
        center_y = (y1 + y2) * 0.5 / image_height
        center_score = math.exp(
            -0.5
            * (
                ((center_x - crosshair_x) / sigma_x) ** 2
                + ((center_y - crosshair_y) / sigma_y) ** 2
            )
        )
        nx1, ny1 = x1 / image_width, y1 / image_height
        nx2, ny2 = x2 / image_width, y2 / image_height
        distance_x = max(nx1 - crosshair_x, 0.0, crosshair_x - nx2)
        distance_y = max(ny1 - crosshair_y, 0.0, crosshair_y - ny2)
        crosshair_distance = math.hypot(distance_x, distance_y)
        crosshair_inside = crosshair_distance <= 1e-9
        crosshair_score = math.exp(-0.5 * (crosshair_distance / crosshair_sigma) ** 2)
        target_score = 0.70 * crosshair_score + 0.30 * center_score
        area_ratio = width * height / float(image_width * image_height)
        size_score = min(1.0, math.sqrt(max(area_ratio, 0.0)) / 0.35)
        edge_clearance = min(center_x, 1 - center_x, center_y, 1 - center_y)
        edge_score = min(1.0, max(0.0, edge_clearance / 0.5))
        selection_score = (
            0.58 * crosshair_score
            + 0.20 * center_score
            + 0.08 * size_score
            + 0.08 * confidence
            + 0.06 * edge_score
        )
        candidates.append(
            {
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "selection_score": selection_score,
                "target_score": target_score,
                "crosshair_inside": crosshair_inside,
                "crosshair_distance": crosshair_distance,
            }
        )
    candidates.sort(
        key=lambda item: (
            bool(item["crosshair_inside"]),
            item["target_score"],
            item["selection_score"],
            item["confidence"],
        ),
        reverse=True,
    )
    if not candidates:
        return [], [], {"ambiguous": False, "score_gap": None, "pair_iou": None}

    selected = [candidates[0]]
    score_gap: float | None = None
    pair_iou: float | None = None
    ambiguous = False
    if len(candidates) >= 2:
        first, second = candidates[:2]
        score_gap = abs(float(first["target_score"]) - float(second["target_score"]))
        pair_iou = box_iou(first["box"], second["box"])
        same_crosshair_relation = bool(first["crosshair_inside"]) == bool(
            second["crosshair_inside"]
        )
        both_reliable = min(first["confidence"], second["confidence"]) >= ambiguity_confidence
        ambiguous = (
            same_crosshair_relation
            and both_reliable
            and score_gap <= ambiguity_margin
            and pair_iou < duplicate_iou
        )
        if ambiguous:
            selected.append(second)

    return selected, candidates, {
        "ambiguous": ambiguous,
        "score_gap": score_gap,
        "pair_iou": pair_iou,
        "ambiguity_margin": ambiguity_margin,
        "duplicate_iou": duplicate_iou,
    }


def select_target_detection(
    result: Any,
    image_width: int,
    image_height: int,
    candidate_confidence: float,
    crosshair_x: float,
    crosshair_y: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Backward-compatible single-target selection helper."""
    selected, candidates, _ = select_target_detections(
        result=result,
        image_width=image_width,
        image_height=image_height,
        candidate_confidence=candidate_confidence,
        crosshair_x=crosshair_x,
        crosshair_y=crosshair_y,
        ambiguity_confidence=1.01,
    )
    return (selected[0] if selected else None), candidates


def existing_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"Image input does not exist or is unsupported: {path}")


def collect_inputs(values: Iterable[str]) -> list[Path]:
    images: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        for image in existing_images(Path(value).expanduser().resolve()):
            if image not in seen:
                seen.add(image)
                images.append(image)
    if not images:
        raise ValueError("No supported input images found")
    return images


def ensure_references(refs_root: Path, dataset_zip: Path) -> list[Path]:
    refs_root = refs_root.expanduser().resolve()
    refs = sorted(
        path
        for path in refs_root.glob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if refs:
        return refs
    if not dataset_zip.is_file():
        raise FileNotFoundError(
            f"Reference gallery is absent: {refs_root}; dataset ZIP is absent: {dataset_zip}"
        )

    expected_prefix = "bottle_classifier_crops/refs/"
    refs_root.mkdir(parents=True, exist_ok=True)
    log(f"Reference gallery missing; extracting refs from {dataset_zip}")
    with zipfile.ZipFile(dataset_zip) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith(expected_prefix)
            and not name.endswith("/")
            and Path(name).suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not members:
            raise RuntimeError(f"No reference images found in {dataset_zip}")
        for member in tqdm(members, desc="Extract refs", unit="image", file=sys.stderr):
            destination = refs_root / Path(member).name
            if destination.exists():
                continue
            with archive.open(member) as source, destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)

    refs = sorted(
        path
        for path in refs_root.glob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not refs:
        raise RuntimeError("Reference extraction produced no images")
    return refs


def gallery_signature(
    refs: list[Path], checkpoint: Path, weights: Path, image_size: int
) -> str:
    digest = hashlib.sha256()
    for path in (checkpoint, weights):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    digest.update(f"image_size:{image_size}\n".encode())
    for path in refs:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_or_load_gallery(
    model: torch.nn.Module,
    transform: Any,
    refs: list[Path],
    checkpoint: Path,
    weights: Path,
    cache_path: Path,
    device: torch.device,
    image_size: int,
    batch_size: int,
    rebuild: bool,
) -> tuple[torch.Tensor, list[str]]:
    signature = gallery_signature(refs, checkpoint, weights, image_size)
    if cache_path.is_file() and not rebuild:
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache.get("signature") == signature:
            embeddings = cache["embeddings"].float()
            slugs = list(cache["wine_slugs"])
            if embeddings.ndim == 2 and len(embeddings) == len(slugs) == len(refs):
                log(f"Gallery cache: {cache_path} ({len(slugs)} wines)")
                return F.normalize(embeddings, dim=1).to(device), slugs

    slugs = [path.stem for path in refs]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Reference gallery contains duplicate filename stems")
    outputs: list[torch.Tensor] = []
    log(f"Building gallery embeddings for {len(refs)} wines")
    model.eval()
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(refs), batch_size),
            desc="DINO gallery",
            unit="batch",
            file=sys.stderr,
        ):
            batch_paths = refs[start : start + batch_size]
            pixels = torch.stack([transform(open_rgb(path)) for path in batch_paths]).to(device)
            with autocast_context(device):
                embeddings, _ = model(pixels)
            outputs.append(embeddings.float().cpu())
    gallery = F.normalize(torch.cat(outputs), dim=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"signature": signature, "embeddings": gallery, "wine_slugs": slugs},
        cache_path,
    )
    log(f"Gallery cache saved: {cache_path}")
    return gallery.to(device), slugs


def crop_with_policy(
    image: Image.Image,
    selected: dict[str, Any] | None,
    confidence_threshold: float,
    padding: float,
) -> tuple[Image.Image, str, list[int] | None, float | None]:
    if selected is None:
        return image.copy(), "original_no_detection", None, None
    confidence = float(selected["confidence"])
    raw_box = tuple(float(value) for value in selected["box"])
    if confidence < confidence_threshold:
        return image.copy(), "original_low_confidence", [round(v) for v in raw_box], confidence

    x1, y1, x2, y2 = raw_box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    crop_box = (
        max(0, math.floor(x1 - width * padding)),
        max(0, math.floor(y1 - height * padding)),
        min(image.width, math.ceil(x2 + width * padding)),
        min(image.height, math.ceil(y2 + height * padding)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return image.copy(), "original_invalid_crop", [round(v) for v in raw_box], confidence
    return image.crop(crop_box), "yolo_bottle_crop", list(crop_box), confidence


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def fuse_gallery_similarities(similarities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse one or two bottle views by the best similarity for every wine."""
    if similarities.ndim != 2 or similarities.shape[0] not in {1, 2}:
        raise ValueError("Expected similarities with shape [1|2, num_wines]")
    return similarities.max(dim=0)


def predict_one(
    path: Path,
    detector: YOLO,
    model: torch.nn.Module,
    transform: Any,
    gallery_embeddings: torch.Tensor,
    gallery_slugs: list[str],
    device: torch.device,
    top_k: int,
    confidence_threshold: float,
    candidate_confidence: float,
    padding: float,
    crosshair_x: float,
    crosshair_y: float,
    yolo_imgsz: int,
    ambiguity_margin: float,
    duplicate_iou: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    image = open_rgb(path)
    yolo_device: str | int = (
        (device.index if device.index is not None else 0)
        if device.type == "cuda"
        else device.type
    )
    detection_started = time.perf_counter()
    result = detector.predict(
        source=image,
        imgsz=yolo_imgsz,
        conf=candidate_confidence,
        iou=0.70,
        max_det=60,
        device=yolo_device,
        verbose=False,
    )[0]
    synchronize(device)
    detection_ms = (time.perf_counter() - detection_started) * 1000
    selected, candidates, selection_context = select_target_detections(
        result,
        image_width=image.width,
        image_height=image.height,
        candidate_confidence=candidate_confidence,
        crosshair_x=crosshair_x,
        crosshair_y=crosshair_y,
        ambiguity_confidence=confidence_threshold,
        ambiguity_margin=ambiguity_margin,
        duplicate_iou=duplicate_iou,
    )
    selected_for_inference: list[dict[str, Any] | None] = selected or [None]
    prepared_inputs = [
        crop_with_policy(image, candidate, confidence_threshold, padding)
        for candidate in selected_for_inference
    ]

    dino_started = time.perf_counter()
    pixels = torch.stack([transform(prepared[0]) for prepared in prepared_inputs]).to(device)
    with torch.inference_mode(), autocast_context(device):
        query_embeddings, _ = model(pixels)
    query_embeddings = F.normalize(query_embeddings.float(), dim=1)
    similarities_by_bottle = query_embeddings @ gallery_embeddings.T
    similarities, winning_bottles = fuse_gallery_similarities(similarities_by_bottle)
    effective_top_k = min(top_k, len(gallery_slugs))
    scores, indices = torch.topk(similarities, effective_top_k)
    synchronize(device)
    dino_ms = (time.perf_counter() - dino_started) * 1000
    predictions = [
        {
            "rank": rank,
            "wine_slug": gallery_slugs[index],
            "similarity": float(score),
            "source_bottle": int(winning_bottles[index].item()) + 1,
            "similarities_by_bottle": [
                float(value) for value in similarities_by_bottle[:, index].cpu().tolist()
            ],
        }
        for rank, (score, index) in enumerate(
            zip(scores.cpu().tolist(), indices.cpu().tolist(), strict=True), start=1
        )
    ]
    input_metadata = [
        {
            "bottle": index,
            "image_mode": prepared[1],
            "yolo_box": prepared[2],
            "yolo_confidence": prepared[3],
        }
        for index, prepared in enumerate(prepared_inputs, start=1)
    ]
    primary = input_metadata[0]
    return {
        "source": str(path),
        "selection_mode": (
            "ambiguous_center_pair" if selection_context["ambiguous"] else "single_center_bottle"
        ),
        "image_mode": primary["image_mode"],
        "yolo_confidence": primary["yolo_confidence"],
        "yolo_box": primary["yolo_box"],
        "yolo_candidates": len(candidates),
        "dino_inputs": input_metadata,
        "selection_context": selection_context,
        "crosshair": {"x": crosshair_x, "y": crosshair_y},
        "predictions": predictions,
        "latency_ms": {
            "yolo": detection_ms,
            "dino_and_search": dino_ms,
            "total": (time.perf_counter() - started) * 1000,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Image files or directories")
    parser.add_argument("--yolo", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--dino-weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument("--dino-checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--refs-root", type=Path, default=DEFAULT_REFS_ROOT)
    parser.add_argument("--dataset-zip", type=Path, default=DEFAULT_DATASET_ZIP)
    parser.add_argument("--gallery-cache", type=Path, default=DEFAULT_GALLERY_CACHE)
    parser.add_argument("--rebuild-gallery", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gallery-batch-size", type=int, default=32)
    parser.add_argument("--yolo-imgsz", type=int, default=768)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--candidate-confidence", type=float, default=0.05)
    parser.add_argument("--padding", type=float, default=0.06)
    parser.add_argument("--crosshair-x", type=float, default=0.50)
    parser.add_argument("--crosshair-y", type=float, default=0.50)
    parser.add_argument("--ambiguity-margin", type=float, default=DEFAULT_AMBIGUITY_MARGIN)
    parser.add_argument("--duplicate-iou", type=float, default=DEFAULT_DUPLICATE_IOU)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    if args.top_k < 1 or args.gallery_batch_size < 1:
        parser.error("--top-k and --gallery-batch-size must be positive")
    if not 0 <= args.candidate_confidence <= args.confidence_threshold <= 1:
        parser.error("Require 0 <= candidate-confidence <= confidence-threshold <= 1")
    if not 0 <= args.crosshair_x <= 1 or not 0 <= args.crosshair_y <= 1:
        parser.error("Crosshair coordinates must be within 0..1")
    if args.padding < 0:
        parser.error("--padding cannot be negative")
    if args.ambiguity_margin < 0 or not 0 <= args.duplicate_iou <= 1:
        parser.error("Require ambiguity-margin >= 0 and duplicate-iou within 0..1")
    return args


def main() -> None:
    args = parse_args()
    required = [args.yolo, args.dino_weights, args.dino_checkpoint]
    missing = [path for path in required if not path.expanduser().is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}. Run `git lfs pull` first.")

    inputs = collect_inputs(args.inputs)
    refs = ensure_references(args.refs_root, args.dataset_zip)
    device = choose_device(args.device)
    log(f"Device: {device}; inputs: {len(inputs)}; references: {len(refs)}")
    detector = YOLO(str(args.yolo.expanduser().resolve()))
    class_names = detector.names
    bottle_name = class_names.get(0) if isinstance(class_names, dict) else class_names[0]
    if str(bottle_name).lower() != "bottle":
        raise ValueError(f"YOLO class 0 must be 'bottle', found: {bottle_name!r}")

    model, checkpoint = load_trained_model(
        args.dino_checkpoint.expanduser().resolve(),
        args.dino_weights.expanduser().resolve(),
        device,
    )
    model.eval()
    image_size = int(checkpoint["config"].get("image_size", 224))
    _, transform = build_transforms(image_size)
    gallery_embeddings, gallery_slugs = build_or_load_gallery(
        model=model,
        transform=transform,
        refs=refs,
        checkpoint=args.dino_checkpoint.expanduser().resolve(),
        weights=args.dino_weights.expanduser().resolve(),
        cache_path=args.gallery_cache.expanduser().resolve(),
        device=device,
        image_size=image_size,
        batch_size=args.gallery_batch_size,
        rebuild=args.rebuild_gallery,
    )

    results = [
        predict_one(
            path=path,
            detector=detector,
            model=model,
            transform=transform,
            gallery_embeddings=gallery_embeddings,
            gallery_slugs=gallery_slugs,
            device=device,
            top_k=args.top_k,
            confidence_threshold=args.confidence_threshold,
            candidate_confidence=args.candidate_confidence,
            padding=args.padding,
            crosshair_x=args.crosshair_x,
            crosshair_y=args.crosshair_y,
            yolo_imgsz=args.yolo_imgsz,
            ambiguity_margin=args.ambiguity_margin,
            duplicate_iou=args.duplicate_iou,
        )
        for path in tqdm(inputs, desc="Inference", unit="image", file=sys.stderr)
    ]
    payload = {
        "pipeline": "full_bottle_yolo_to_dinov3_vitb16",
        "confidence_threshold": args.confidence_threshold,
        "device": str(device),
        "num_results": len(results),
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().resolve().write_text(rendered + "\n", encoding="utf-8")
        log(f"Saved: {args.output.expanduser().resolve()}")
    print(rendered)


if __name__ == "__main__":
    main()
