#!/usr/bin/env python3
"""Stage the web-shelf holdout, evaluation code and Kaggle notebook."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def link_or_copy(source: Path | str, destination: Path | str) -> str:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--username", default="konstantinmelnikof")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = (args.output_root or project / "kaggle_upload").resolve()
    holdout = output / "viscaner-store-shelves-web"
    code = output / "viscaner-store-shelves-web-eval-code"
    kernel = output / "viscaner-store-shelves-web-eval-kernel"
    for directory in (holdout, code, kernel):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)

    source_data = project / "datasets" / "store_shelves_web"
    for name in ("README.md", "labels.csv", "points.csv", "sources.csv"):
        link_or_copy(source_data / name, holdout / name)
    shutil.copytree(source_data / "queries", holdout / "queries", copy_function=link_or_copy)
    (holdout / "dataset-metadata.json").write_text(json.dumps({
        "title": "VIscaNEr Store Shelves Web Holdout",
        "id": f"{args.username}/viscaner-store-shelves-web",
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")

    code_files = (
        "dinov3_retrieval.py", "infer_wine.py", "yolo_target_selection.py",
        "fusion_stage2/__init__.py", "fusion_stage2/kaggle_io.py",
        "joint_yolo/__init__.py", "joint_yolo/infer.py",
        "five_stream_transformer/__init__.py", "five_stream_transformer/dino.py",
        "five_stream_transformer/model.py", "five_stream_transformer/text.py",
        "five_stream_transformer/data.py", "five_stream_transformer/ocr.py",
        "five_stream_transformer/features.py", "five_stream_transformer/stage2c.py",
        "five_stream_transformer/evaluate_store_shelves_web.py",
    )
    for name in code_files:
        link_or_copy(project / name, code / name)
    link_or_copy(project / "models" / "joint_yolo" / "best.pt", code / "best_joint_yolo.pt")
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": "VIscaNEr Store Shelves Web Evaluation Code",
        "id": f"{args.username}/viscaner-store-shelves-web-eval-code",
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")

    notebook = project / "kaggle_notebooks" / "StoreShelvesWeb-Stage2C-Residual-Eval.ipynb"
    link_or_copy(notebook, kernel / notebook.name)
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.username}/viscaner-store-shelves-web-stage2c-eval",
        "title": "VIscaNEr Store Shelves Web Stage2C Eval",
        "code_file": notebook.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [
            f"{args.username}/viscaner-store-shelves-web-eval-code",
            f"{args.username}/viscaner-store-shelves-web",
            f"{args.username}/viscaner-stage2c-checkpoint",
            f"{args.username}/viscaner-easyocr-ru-en-weights",
            f"{args.username}/viscaner-dinov3-data",
            f"{args.username}/viscaner-bottle-classifier-data",
        ],
        "kernel_sources": [f"{args.username}/viscaner-stage-2c-residual-transformer"],
        "competition_sources": [],
    }, indent=2), encoding="utf-8")
    print(json.dumps({"holdout": str(holdout), "code": str(code), "kernel": str(kernel)}, indent=2))


if __name__ == "__main__":
    main()
