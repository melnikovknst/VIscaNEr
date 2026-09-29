#!/usr/bin/env python
"""Reproduce the two YOLO11n detectors the DINO cascade was trained against.

The trained DINO checkpoints in `models/trained_checkpoints/` consume crops:
DINO-B/16 reads label crops (YOLO label detector, 10% padding), DINO-S/16 reads
whole-bottle crops (YOLO bottle detector, 6% padding). Before the detectors
were pushed, the backend ran on reproductions trained by this script from the
in-repo data, with the exact recipes recorded in

  train_yolo_label_detector.ipynb                (label detector)
  bottle_reranker/YOLO-bottle-confidence-audit.ipynb  (bottle detector)

The original weights are now in the repository (models/weights_sha256.json)
and are what the backend serves. This script stays as a reproducibility check
and writes to runs/local_repro_detectors/, never over the tracked originals.

    python train_yolo_detectors.py              # both
    python train_yolo_detectors.py --only bottle
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMMON = {
    "pretrained": True,
    "optimizer": "auto",
    "seed": 42,
    "deterministic": True,
    "cache": False,
    "plots": True,
    "fliplr": 0.0,          # labels carry text; a mirrored label is not a label
    "flipud": 0.0,
    "mosaic": 0.20,
    "close_mosaic": 10,
}

RECIPES = {
    # train_yolo_label_detector.ipynb, TRAIN_CONFIG
    "label": {
        "data": ROOT / "datasets/yolo_label_detector_local/data.yaml",
        "output": ROOT / "runs/local_repro_detectors/label_best.pt",
        "train": {
            **COMMON,
            "imgsz": 640, "epochs": 50, "patience": 10, "single_cls": True,
            "degrees": 5.0, "translate": 0.08, "scale": 0.25, "shear": 1.5,
            "perspective": 0.0005, "hsv_h": 0.015, "hsv_s": 0.40, "hsv_v": 0.25,
            "mixup": 0.0, "copy_paste": 0.0,
        },
    },
    # bottle_reranker/YOLO-bottle-confidence-audit.ipynb, TRAIN_CONFIG
    "bottle": {
        "data": ROOT / "datasets/wine_bottles_yolo_1000/data_local.yaml",
        "output": ROOT / "runs/local_repro_detectors/bottle_best.pt",
        "train": {
            **COMMON,
            "imgsz": 768, "epochs": 80, "patience": 15,
            "degrees": 7.0, "translate": 0.08, "scale": 0.25,
            "perspective": 0.0005, "hsv_h": 0.01, "hsv_s": 0.35, "hsv_v": 0.25,
        },
    },
}


def train(name: str, device: str, batch: int) -> dict:
    from ultralytics import YOLO

    recipe = RECIPES[name]
    data = recipe["data"]
    if not data.is_file():
        raise FileNotFoundError(f"{data} - extract the dataset first (see backend/README)")
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(data), device=device, batch=batch, workers=4,
        project=str(ROOT / "runs/yolo_detectors"), name=name, exist_ok=True,
        amp=device != "cpu", **recipe["train"],
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    recipe["output"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, recipe["output"])

    metrics = YOLO(str(recipe["output"])).val(
        data=str(data), split="val", imgsz=recipe["train"]["imgsz"],
        device=device, batch=batch, verbose=False, plots=False,
        project=str(ROOT / "runs/yolo_detectors"), name=f"{name}_val", exist_ok=True,
    )
    summary = {
        "detector": name,
        "weights": str(recipe["output"].relative_to(ROOT)),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "note": "local reproduction of the colleagues' recipe, not their original weights",
    }
    recipe["output"].with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(RECIPES), action="append")
    parser.add_argument("--device", default="0", help="CUDA index, 'mps' or 'cpu'")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    for name in args.only or sorted(RECIPES):
        print(json.dumps(train(name, args.device, args.batch), indent=2))


if __name__ == "__main__":
    main()
