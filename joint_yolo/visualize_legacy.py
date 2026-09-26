#!/usr/bin/env python3
"""Rebuild the original joint-YOLO validation viewer on a separate endpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from . import visualize as legacy
from .train import DEFAULT_DATASET, PROJECT_ROOT


DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "joint_yolo" / "crop_audit_legacy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=legacy.DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--bottle-crop-threshold", type=float, default=0.75)
    parser.add_argument("--ambiguity-margin", type=float, default=0.06)
    parser.add_argument("--include-id", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    image_root = dataset / "images" / args.split
    label_root = dataset / "labels" / args.split

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    image_paths = sorted(
        path for path in image_root.iterdir()
        if path.suffix.lower() in legacy.IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No images in {image_root}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    device = legacy.choose_device(args.device)
    model = YOLO(str(model_path))
    records = []
    for index, image_path in enumerate(image_paths, start=1):
        records.append(
            legacy.infer_record(
                model,
                image_path,
                label_root / f"{image_path.stem}.txt",
                device,
                args.confidence,
                args.bottle_crop_threshold,
                args.ambiguity_margin,
            )
        )
        if index % 10 == 0 or index == len(image_paths):
            print(f"LEGACY AUDIT INFERENCE | {index}/{len(image_paths)}", flush=True)

    chosen = legacy.select_mixed(records, args.n, args.seed, args.include_id)
    rows = [
        legacy.render_record(record, output, index, args.bottle_crop_threshold)
        for index, record in enumerate(chosen, start=1)
    ]
    metadata = {
        "viewer": "legacy_joint_yolo_validation",
        "model": str(model_path.relative_to(PROJECT_ROOT)),
        "dataset": str(dataset.relative_to(PROJECT_ROOT)),
        "split": args.split,
        "evaluated_images": len(records),
        "displayed_images": len(rows),
        "confidence_threshold": args.confidence,
        "bottle_crop_threshold": args.bottle_crop_threshold,
        "ambiguity_margin": args.ambiguity_margin,
        "rows": rows,
    }
    (output / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(legacy.build_html(rows), encoding="utf-8")
    print(f"LEGACY AUDIT HTML | {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
