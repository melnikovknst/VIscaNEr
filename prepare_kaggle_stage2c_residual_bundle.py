#!/usr/bin/env python3
"""Stage Kaggle datasets and notebook for Stage-2C residual training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--publish-username",
        default="konstantinmelnikof",
        help="Kaggle account that will own the new code/checkpoint/OCR datasets and notebook.",
    )
    parser.add_argument(
        "--data-owner",
        default="konstantinmelnikof",
        help="Kaggle account that owns the existing crop datasets shared with the runner.",
    )
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = (args.output_root or project / "kaggle_upload").resolve()
    code = output / "viscaner-stage2c-residual-code"
    stage2c = output / "viscaner-stage2c-checkpoint"
    ocr_weights = output / "viscaner-easyocr-ru-en-weights"
    kernel = output / "viscaner-stage2c-residual-kernel"
    for directory in (code, stage2c, ocr_weights, kernel):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)

    code_files = (
        Path("fusion_stage2/__init__.py"),
        Path("fusion_stage2/kaggle_io.py"),
        Path("five_stream_transformer/__init__.py"),
        Path("five_stream_transformer/dino.py"),
        Path("five_stream_transformer/model.py"),
        Path("five_stream_transformer/text.py"),
        Path("five_stream_transformer/data.py"),
        Path("five_stream_transformer/ocr.py"),
        Path("five_stream_transformer/features.py"),
        Path("five_stream_transformer/stage2c.py"),
        Path("five_stream_transformer/train.py"),
        Path("five_stream_transformer/README.md"),
        Path("kaggle_notebooks/FiveStream-Stage2C-Residual.ipynb"),
    )
    for relative in code_files:
        source = project / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        link_or_copy(source, code / relative)
    checksums = {relative.as_posix(): sha256(code / relative) for relative in code_files}
    (code / "code_sha256.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8"
    )
    (code / "dataset-metadata.json").write_text(
        json.dumps({
            "title": "VIscaNEr Stage-2C residual Transformer code",
            "id": f"{args.publish_username}/viscaner-stage2c-residual-code",
            "licenses": [{"name": "other"}],
        }, indent=2), encoding="utf-8",
    )

    checkpoint = project / "models" / "stage2c" / "manual_stage2c_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing {checkpoint}. Download the completed Stage-2C export before staging."
        )
    link_or_copy(checkpoint, stage2c / checkpoint.name)
    stage2c_digest = sha256(stage2c / checkpoint.name)
    (stage2c / "stage2c_sha256.json").write_text(
        json.dumps({checkpoint.name: stage2c_digest}, indent=2), encoding="utf-8"
    )
    (stage2c / "dataset-metadata.json").write_text(
        json.dumps({
            "title": "VIscaNEr complete Stage-2C DINO-B checkpoint",
            "id": f"{args.publish_username}/viscaner-stage2c-checkpoint",
            "licenses": [{"name": "other"}],
        }, indent=2), encoding="utf-8",
    )

    easyocr_sources = {
        "craft_mlt_25k.pth": project / "models" / "easyocr_ru_en" / "craft_mlt_25k.pth",
        "cyrillic_g2.pth": project / "models" / "easyocr_ru_en" / "cyrillic_g2.pth",
    }
    for name, source in easyocr_sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        link_or_copy(source, ocr_weights / name)
    (ocr_weights / "weights_sha256.json").write_text(
        json.dumps({name: sha256(ocr_weights / name) for name in easyocr_sources}, indent=2),
        encoding="utf-8",
    )
    (ocr_weights / "dataset-metadata.json").write_text(
        json.dumps({
            "title": "VIscaNEr EasyOCR Russian English weights",
            "id": f"{args.publish_username}/viscaner-easyocr-ru-en-weights",
            "licenses": [{"name": "other"}],
        }, indent=2), encoding="utf-8",
    )

    notebook = project / "kaggle_notebooks" / "FiveStream-Stage2C-Residual.ipynb"
    link_or_copy(notebook, kernel / notebook.name)
    metadata = {
        "id": f"{args.publish_username}/viscaner-stage2c-residual-transformer",
        "title": "VIscaNEr Stage-2C Residual Transformer",
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            f"{args.publish_username}/viscaner-stage2c-residual-code",
            f"{args.publish_username}/viscaner-stage2c-checkpoint",
            f"{args.publish_username}/viscaner-easyocr-ru-en-weights",
            f"{args.data_owner}/viscaner-fusion-hardset-v1",
            f"{args.data_owner}/viscaner-dinov3-data",
            f"{args.data_owner}/viscaner-bottle-classifier-data",
            f"{args.data_owner}/viscaner-manual-211",
        ],
        "kernel_sources": [],
        "competition_sources": [],
    }
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "code": str(code), "stage2c": str(stage2c),
        "stage2c_sha256": stage2c_digest,
        "ocr_weights": str(ocr_weights), "kernel": str(kernel),
    }, indent=2))


if __name__ == "__main__":
    main()
