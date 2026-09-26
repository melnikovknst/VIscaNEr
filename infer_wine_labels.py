#!/usr/bin/env python3
"""Wine retrieval from a label: YOLO label crop -> fine-tuned DINOv3-B Top-K."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO

from dinov3_retrieval import build_transforms, choose_device, load_trained_model, open_rgb
from infer_wine import (
    autocast_context,
    build_or_load_gallery,
    collect_inputs,
    synchronize,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YOLO = PROJECT_ROOT / "models" / "yolo_label_detector" / "best.pt"
DEFAULT_DINO_WEIGHTS = PROJECT_ROOT / "models" / "dinov3" / "model.safetensors"
DEFAULT_DINO_CHECKPOINT = (
    PROJECT_ROOT / "models" / "trained_checkpoints" / "dinov3_vitb16_labels_best_full.pt"
)
DEFAULT_REFS_ROOT = PROJECT_ROOT / "datasets" / "wine-scanner" / "data" / "refs" / "rgb"
DEFAULT_REFS_ARCHIVE = PROJECT_ROOT / "datasets" / "wine-scanner_code-catalog.zip"
DEFAULT_GALLERY_CACHE = PROJECT_ROOT / "runs" / "inference" / "dinov3_vitb16_labels_gallery.pt"
DEFAULT_CROPS_DIR = PROJECT_ROOT / "runs" / "inference" / "label_crops"
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def ensure_label_references(refs_root: Path, archive_path: Path) -> list[Path]:
    refs_root = refs_root.expanduser().resolve()
    refs = sorted(path for path in refs_root.glob("*") if path.suffix.lower() in SUPPORTED)
    if refs:
        return refs
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing gallery and archive: {refs_root}, {archive_path}")
    prefix = "wine-scanner/data/refs/rgb/"
    refs_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name for name in archive.namelist()
            if name.startswith(prefix) and not name.endswith("/")
            and Path(name).suffix.lower() in SUPPORTED
        ]
        if not members:
            raise RuntimeError(f"No RGB references under {prefix} in {archive_path}")
        for member in tqdm(members, desc="Extract label gallery", unit="image", file=sys.stderr):
            destination = refs_root / Path(member).name
            if not destination.exists():
                with archive.open(member) as source, destination.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
    return sorted(path for path in refs_root.glob("*") if path.suffix.lower() in SUPPORTED)


def padded_box(box: tuple[float, float, float, float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = max(0.0, x2 - x1) * padding
    pad_y = max(0.0, y2 - y1) * padding
    return (
        max(0, math.floor(x1 - pad_x)),
        max(0, math.floor(y1 - pad_y)),
        min(width, math.ceil(x2 + pad_x)),
        min(height, math.ceil(y2 + pad_y)),
    )


def select_central_label(result: Any, width: int, height: int) -> tuple[dict[str, Any] | None, int]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, 0
    detections: list[dict[str, Any]] = []
    for raw_box, raw_conf, raw_class in zip(
        boxes.xyxy.detach().float().cpu().tolist(),
        boxes.conf.detach().float().cpu().tolist(),
        boxes.cls.detach().long().cpu().tolist(),
        strict=True,
    ):
        if int(raw_class) != 0:
            continue
        box = tuple(float(value) for value in raw_box)
        confidence = float(raw_conf)
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        distance = math.hypot((cx - width * 0.5) / width, (cy - height * 0.5) / height)
        detections.append({"box": box, "confidence": confidence, "cost": distance - 0.25 * confidence})
    return (min(detections, key=lambda item: item["cost"]) if detections else None), len(detections)


def save_model_input(image: Image.Image, source: Path, crops_dir: Path) -> Path:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    destination = crops_dir / f"{source.stem}__{digest}.jpg"
    crops_dir.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination, quality=95)
    return destination.resolve()


def predict_one(
    source: Path,
    detector: YOLO,
    model: torch.nn.Module,
    transform: Any,
    gallery_embeddings: torch.Tensor,
    gallery_slugs: list[str],
    device: torch.device,
    top_k: int,
    detector_confidence: float,
    padding: float,
    yolo_imgsz: int,
    crops_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    image = open_rgb(source)
    yolo_device: str | int = (device.index or 0) if device.type == "cuda" else device.type
    yolo_started = time.perf_counter()
    result = detector.predict(
        source=image,
        imgsz=yolo_imgsz,
        conf=detector_confidence,
        iou=0.70,
        max_det=30,
        device=yolo_device,
        verbose=False,
    )[0]
    synchronize(device)
    yolo_ms = (time.perf_counter() - yolo_started) * 1000
    selected, candidate_count = select_central_label(result, image.width, image.height)
    if selected is None:
        model_input = image.copy()
        image_mode = "original_no_detection"
        crop_box = None
        confidence = None
    else:
        crop_box = padded_box(selected["box"], image.width, image.height, padding)
        model_input = image.crop(crop_box)
        image_mode = "yolo_label_crop"
        confidence = float(selected["confidence"])
    crop_path = save_model_input(model_input, source, crops_dir)

    dino_started = time.perf_counter()
    pixels = transform(model_input).unsqueeze(0).to(device)
    with torch.inference_mode(), autocast_context(device):
        embedding, _ = model(pixels)
    similarities = F.normalize(embedding.float(), dim=1) @ gallery_embeddings.T
    scores, indices = torch.topk(similarities.squeeze(0), min(top_k, len(gallery_slugs)))
    synchronize(device)
    predictions = [
        {"rank": rank, "wine_slug": gallery_slugs[index], "similarity": float(score)}
        for rank, (score, index) in enumerate(
            zip(scores.cpu().tolist(), indices.cpu().tolist(), strict=True), start=1
        )
    ]
    return {
        "source": str(source.resolve()),
        "ocr_input": str(crop_path),
        "image_mode": image_mode,
        "yolo_confidence": confidence,
        "yolo_box": list(crop_box) if crop_box else None,
        "yolo_candidates": candidate_count,
        "predictions": predictions,
        "latency_ms": {
            "yolo": yolo_ms,
            "dino_and_search": (time.perf_counter() - dino_started) * 1000,
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
    parser.add_argument("--refs-archive", type=Path, default=DEFAULT_REFS_ARCHIVE)
    parser.add_argument("--gallery-cache", type=Path, default=DEFAULT_GALLERY_CACHE)
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--rebuild-gallery", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gallery-batch-size", type=int, default=32)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [args.yolo, args.dino_weights, args.dino_checkpoint]
    missing = [path for path in required if not path.expanduser().is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}. Run `git lfs pull` first.")
    if not args.refs_root.expanduser().is_dir() and not args.refs_archive.expanduser().is_file():
        raise FileNotFoundError(
            f"Missing both label gallery and archive: {args.refs_root}, {args.refs_archive}"
        )
    inputs = collect_inputs(args.inputs)
    refs = ensure_label_references(args.refs_root, args.refs_archive)
    device = choose_device(args.device)
    log(f"Device: {device}; inputs: {len(inputs)}; references: {len(refs)}")
    detector = YOLO(str(args.yolo.resolve()))
    class_names = detector.names
    class_zero = class_names.get(0) if isinstance(class_names, dict) else class_names[0]
    # The original annotation export named class 0 ``item`` even though the
    # dataset contains only wine labels.  Class id 0 is the stable contract.
    if str(class_zero).lower() not in {"wine_label", "label", "item"}:
        raise ValueError(f"Unexpected YOLO class 0: {class_zero!r}")
    log(f"Label detector class 0: {class_zero!r}")
    model, checkpoint = load_trained_model(args.dino_checkpoint.resolve(), args.dino_weights.resolve(), device)
    model.eval()
    image_size = int(checkpoint["config"].get("image_size", 224))
    _, transform = build_transforms(image_size)
    gallery_embeddings, gallery_slugs = build_or_load_gallery(
        model=model,
        transform=transform,
        refs=refs,
        checkpoint=args.dino_checkpoint.resolve(),
        weights=args.dino_weights.resolve(),
        cache_path=args.gallery_cache.resolve(),
        device=device,
        image_size=image_size,
        batch_size=args.gallery_batch_size,
        rebuild=args.rebuild_gallery,
    )
    results = [
        predict_one(
            source=path,
            detector=detector,
            model=model,
            transform=transform,
            gallery_embeddings=gallery_embeddings,
            gallery_slugs=gallery_slugs,
            device=device,
            top_k=args.top_k,
            detector_confidence=args.detector_confidence,
            padding=args.padding,
            yolo_imgsz=args.yolo_imgsz,
            crops_dir=args.crops_dir.resolve(),
        )
        for path in tqdm(inputs, desc="Label inference", unit="image", file=sys.stderr)
    ]
    payload = {"pipeline": "label_yolo_to_dinov3_vitb16", "device": str(device), "results": results}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(rendered + "\n", encoding="utf-8")
        log(f"Saved: {args.output.resolve()}")
    print(rendered)


if __name__ == "__main__":
    main()
