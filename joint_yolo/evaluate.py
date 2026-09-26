#!/usr/bin/env python3
"""Evaluate the joint detector and persist overall and per-class metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

from .train import DEFAULT_DATASET, DEFAULT_MODELS, DEFAULT_RUNS, choose_device, metric_value, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODELS / "best.pt")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    dataset = args.dataset.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not (dataset / "data.yaml").is_file():
        raise FileNotFoundError(dataset / "data.yaml")
    runtime_yaml = DEFAULT_RUNS / "runtime_data.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/train",
                "val": "images/val",
                "nc": 2,
                "names": {0: "bottle", 1: "wine_label"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    metrics = YOLO(str(model_path)).val(
        data=str(runtime_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=choose_device(args.device),
        workers=args.workers,
        plots=True,
        project=str(DEFAULT_RUNS),
        name="joint_best_validation",
        exist_ok=True,
        verbose=False,
    )
    by_class = {}
    for index, class_name in metrics.names.items():
        precision, recall, map50, map50_95 = metrics.box.class_result(index)
        by_class[str(class_name)] = {
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(map50),
            "map50_95": float(map50_95),
        }
    metrics_path = DEFAULT_MODELS / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    payload.update(
        {
            "best_checkpoint": str(model_path.relative_to(DEFAULT_MODELS.parents[1])),
            "best_sha256": sha256(model_path),
            "best_bytes": model_path.stat().st_size,
            "validation": {
                "precision": metric_value(metrics.box.mp),
                "recall": metric_value(metrics.box.mr),
                "map50": metric_value(metrics.box.map50),
                "map50_95": metric_value(metrics.box.map),
                "by_class": by_class,
            },
        }
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["validation"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
