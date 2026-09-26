#!/usr/bin/env python3
"""Train and validate the joint bottle + wine-label YOLO11n detector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "wine_bottle_label_joint"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "bottle_reranker" / "best_bottle_detector.pt"
DEFAULT_RUNS = PROJECT_ROOT / "runs" / "joint_yolo"
DEFAULT_MODELS = PROJECT_ROOT / "models" / "joint_yolo"


def choose_device(requested: str) -> str | int:
    if requested != "auto":
        return 0 if requested == "cuda" else requested
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metric_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def portable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove machine-specific project prefixes before persisting metadata."""
    result = dict(config)
    for key in ("model", "data", "project"):
        value = result.get(key)
        if value is None:
            continue
        path = Path(str(value))
        try:
            result[key] = str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            result[key] = str(path)
    return result


def history_summary(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, None
    score_key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row[score_key]))
    return int(float(rows[-1]["epoch"])), int(float(best["epoch"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--name", default="yolo11n_bottle_label_joint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    base_model = args.model.resolve()
    for required in (dataset / "data.yaml", base_model):
        if not required.exists():
            raise FileNotFoundError(required)
    device = choose_device(args.device)
    runs = args.runs.resolve()
    models_dir = args.models_dir.resolve()
    runs.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = runs / "runtime_data.yaml"
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
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = {
        "model": str(base_model),
        "data": str(runtime_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "patience": args.patience,
        "device": device,
        "workers": args.workers,
        "project": str(runs),
        "name": args.name,
        "exist_ok": True,
        "seed": args.seed,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.01,
        "warmup_epochs": 3.0,
        "cos_lr": True,
        "amp": True,
        "cache": False,
        "save": True,
        "save_period": 10,
        "plots": True,
        # Physically plausible augmentation; text is never mirrored.
        "degrees": 4.0,
        "translate": 0.08,
        "scale": 0.25,
        "shear": 1.0,
        "perspective": 0.0005,
        "hsv_h": 0.01,
        "hsv_s": 0.30,
        "hsv_v": 0.25,
        "fliplr": 0.0,
        "flipud": 0.0,
        "mosaic": 0.20,
        "close_mosaic": 15,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    print(
        "SETUP | joint YOLO11n | classes=bottle,wine_label | "
        f"device={device} | imgsz={args.imgsz} | epochs={args.epochs}",
        flush=True,
    )
    print(json.dumps(config, ensure_ascii=False, indent=2, default=str), flush=True)
    started = time.time()
    model = YOLO(str(base_model))
    model.train(**config)
    run_dir = runs / args.name
    best_source = run_dir / "weights" / "best.pt"
    last_source = run_dir / "weights" / "last.pt"
    if not best_source.is_file():
        raise FileNotFoundError(f"Training did not produce {best_source}")
    best_target = models_dir / "best.pt"
    last_target = models_dir / "last.pt"
    shutil.copy2(best_source, best_target)
    if last_source.is_file():
        shutil.copy2(last_source, last_target)

    best_model = YOLO(str(best_target))
    validation = best_model.val(
        data=str(runtime_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        plots=True,
        project=str(runs),
        name=f"{args.name}_best_validation",
        exist_ok=True,
        verbose=False,
    )
    class_names = validation.names
    per_class = {}
    for index, class_name in class_names.items():
        precision, recall, map50, map50_95 = validation.box.class_result(index)
        per_class[str(class_name)] = {
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(map50),
            "map50_95": float(map50_95),
        }
    epochs_completed, best_epoch = history_summary(run_dir / "results.csv")
    summary = {
        "model": "YOLO11n joint detector",
        "classes": {"0": "bottle", "1": "wine_label"},
        "base_model": str(base_model.relative_to(PROJECT_ROOT)),
        "best_checkpoint": str(best_target.relative_to(PROJECT_ROOT)),
        "last_checkpoint": str(last_target.relative_to(PROJECT_ROOT)) if last_target.is_file() else None,
        "best_sha256": sha256(best_target),
        "best_bytes": best_target.stat().st_size,
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics_version,
        },
        "validation": {
            "precision": metric_value(validation.box.mp),
            "recall": metric_value(validation.box.mr),
            "map50": metric_value(validation.box.map50),
            "map50_95": metric_value(validation.box.map),
            "by_class": per_class,
        },
        "train_config": portable_config(config),
    }
    (models_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("=" * 78, flush=True)
    print("FINAL JOINT YOLO METRICS", flush=True)
    print("=" * 78, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
