#!/usr/bin/env python3
"""One YOLO forward producing paired bottle and wine-label crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "joint_yolo" / "best.pt"
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}
CONFIDENCE_WEIGHT = 0.90
PROXIMITY_WEIGHT = 0.10
AMBIGUITY_DISTANCE_MARGIN = 0.08


def choose_device(requested: str) -> str | int:
    if requested != "auto":
        return 0 if requested == "cuda" else requested
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def point_box_distance(x: float, y: float, box: tuple[float, float, float, float]) -> float:
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def padded_crop(image: Image.Image, box: tuple[float, float, float, float], padding: float) -> tuple[Image.Image, list[int]]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    crop_box = [
        max(0, math.floor(x1 - width * padding)),
        max(0, math.floor(y1 - height * padding)),
        min(image.width, math.ceil(x2 + width * padding)),
        min(image.height, math.ceil(y2 + height * padding)),
    ]
    return image.crop(tuple(crop_box)), crop_box


def target_score(item: dict[str, Any], crosshair: tuple[float, float], diagonal: float) -> float:
    distance = point_box_distance(*crosshair, item["box"]) / max(diagonal, 1.0)
    proximity = math.exp(-0.5 * (distance / 0.12) ** 2)
    return CONFIDENCE_WEIGHT * item["confidence"] + PROXIMITY_WEIGHT * proximity


def pair_bottle(label: dict[str, Any], bottles: list[dict[str, Any]]) -> dict[str, Any] | None:
    cx = (label["box"][0] + label["box"][2]) * 0.5
    cy = (label["box"][1] + label["box"][3]) * 0.5
    containing = [
        bottle for bottle in bottles
        if bottle["box"][0] <= cx <= bottle["box"][2]
        and bottle["box"][1] <= cy <= bottle["box"][3]
    ]
    if containing:
        return max(containing, key=lambda bottle: (box_iou(label["box"], bottle["box"]), bottle["confidence"]))
    return min(
        bottles,
        key=lambda bottle: point_box_distance(cx, cy, bottle["box"]),
        default=None,
    )


def save_crop(image: Image.Image, output: Path, stem: str) -> str:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}.jpg"
    image.convert("RGB").save(path, quality=95)
    return str(path.resolve())


def predict_detections(
    model: YOLO,
    image: Image.Image,
    device: str | int,
    confidence: float,
) -> list[dict[str, Any]]:
    """Run the joint detector once and return device-independent detections."""
    result = model.predict(
        source=image,
        imgsz=768,
        conf=confidence,
        iou=0.65,
        max_det=60,
        device=device,
        verbose=False,
    )[0]
    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box, score, class_id in zip(
            result.boxes.xyxy.detach().float().cpu().tolist(),
            result.boxes.conf.detach().float().cpu().tolist(),
            result.boxes.cls.detach().long().cpu().tolist(),
            strict=True,
        ):
            detections.append(
                {
                    "class_id": int(class_id),
                    "confidence": float(score),
                    "box": tuple(float(value) for value in box),
                }
            )
    return detections


def select_label_candidates(
    labels: list[dict[str, Any]],
    crosshair: tuple[float, float],
    diagonal: float,
    ambiguity_margin: float,
) -> tuple[list[dict[str, Any]], bool]:
    ranked = sorted(labels, key=lambda item: target_score(item, crosshair, diagonal), reverse=True)
    selected = ranked[:1]
    ambiguous = False
    if len(ranked) >= 2:
        first_score = target_score(ranked[0], crosshair, diagonal)
        second_score = target_score(ranked[1], crosshair, diagonal)
        first_distance = point_box_distance(*crosshair, ranked[0]["box"]) / max(diagonal, 1.0)
        second_distance = point_box_distance(*crosshair, ranked[1]["box"]) / max(diagonal, 1.0)
        ambiguous = (
            first_score - second_score <= ambiguity_margin
            and abs(first_distance - second_distance) <= AMBIGUITY_DISTANCE_MARGIN
            and box_iou(ranked[0]["box"], ranked[1]["box"]) < 0.50
        )
        if ambiguous:
            selected.append(ranked[1])
    return selected, ambiguous


def infer_one(
    model: YOLO,
    image_path: Path,
    output: Path,
    device: str | int,
    confidence: float,
    bottle_crop_threshold: float,
    ambiguity_margin: float,
    crosshair_x: float,
    crosshair_y: float,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    detections = predict_detections(model, image, device, confidence)
    bottles = [item for item in detections if item["class_id"] == 0]
    labels = [item for item in detections if item["class_id"] == 1]
    crosshair = (image.width * crosshair_x, image.height * crosshair_y)
    diagonal = math.hypot(image.width, image.height)
    selected, ambiguous = select_label_candidates(labels, crosshair, diagonal, ambiguity_margin)
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    candidates: list[dict[str, Any]] = []
    for index, label in enumerate(selected, start=1):
        bottle = pair_bottle(label, bottles)
        label_crop, label_box = padded_crop(image, label["box"], 0.10)
        label_path = save_crop(label_crop, output, f"{image_path.stem}__{digest}__c{index}_label")
        if bottle is None or bottle["confidence"] < bottle_crop_threshold:
            bottle_image = image.copy()
            bottle_mode = "original_low_confidence" if bottle else "original_no_bottle"
            bottle_box = list(bottle["box"]) if bottle else None
        else:
            bottle_image, bottle_box = padded_crop(image, bottle["box"], 0.06)
            bottle_mode = "yolo_bottle_crop"
        bottle_path = save_crop(bottle_image, output, f"{image_path.stem}__{digest}__c{index}_bottle")
        candidates.append(
            {
                "candidate": index,
                "label_crop": label_path,
                "label_box": label_box,
                "label_confidence": label["confidence"],
                "bottle_crop": bottle_path,
                "bottle_box": bottle_box,
                "bottle_confidence": bottle["confidence"] if bottle else None,
                "bottle_image_mode": bottle_mode,
            }
        )
    return {
        "source": str(image_path.resolve()),
        "selection_mode": (
            "no_label_detection" if not selected
            else "ambiguous_two_pairs" if ambiguous
            else "single_pair"
        ),
        "detections": {"bottles": len(bottles), "wine_labels": len(labels)},
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "joint_yolo" / "inference_crops")
    parser.add_argument("--json", type=Path, default=PROJECT_ROOT / "runs" / "joint_yolo" / "inference.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--bottle-crop-threshold", type=float, default=0.75)
    parser.add_argument("--ambiguity-margin", type=float, default=0.06)
    parser.add_argument("--crosshair-x", type=float, default=0.50)
    parser.add_argument("--crosshair-y", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    paths = [
        path for value in args.inputs
        for path in ([Path(value)] if Path(value).is_file() else sorted(Path(value).rglob("*")))
        if path.is_file() and path.suffix.lower() in SUPPORTED
    ]
    if not paths:
        raise ValueError("No supported input images")
    device = choose_device(args.device)
    model = YOLO(str(args.model.resolve()))
    results = [
        infer_one(
            model, path, args.output_dir.resolve(), device, args.confidence,
            args.bottle_crop_threshold, args.ambiguity_margin,
            args.crosshair_x, args.crosshair_y,
        )
        for path in paths
    ]
    payload = {"model": str(args.model), "device": str(device), "results": results}
    args.json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json.resolve().write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
