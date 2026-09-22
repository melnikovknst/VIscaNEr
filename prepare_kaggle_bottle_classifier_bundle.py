#!/usr/bin/env python3
"""Prepare private Kaggle data, code and weights datasets for bottle classifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from tqdm.auto import tqdm


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def metadata(title: str, dataset_id: str) -> dict[str, object]:
    return {"title": title, "id": dataset_id, "licenses": [{"name": "other"}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/Users/konstantinmelnikov/Desktop/work/VIscaNEr"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--kaggle-username", required=True)
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()

    project = args.project_root.expanduser().resolve()
    output = (args.output_root or project / "kaggle_upload").expanduser().resolve()
    data_bundle = output / "viscaner-bottle-classifier-data"
    code_bundle = output / "viscaner-bottle-classifier-code"
    weights_bundle = output / "viscaner-bottle-classifier-weights"
    recreate = (code_bundle,) if args.code_only else (data_bundle, code_bundle, weights_bundle)
    for bundle in recreate:
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True, exist_ok=True)

    if not args.code_only:
        dataset = project / "datasets" / "bottle_classifier_crops"
        summary = dataset / "build_summary.json"
        if not summary.is_file():
            raise FileNotFoundError(
                "Bottle crop dataset is not complete. Run build_bottle_classifier_dataset.py first."
            )
        build = json.loads(summary.read_text(encoding="utf-8"))
        if not build.get("complete"):
            raise RuntimeError("Bottle crop dataset build_summary.json says complete=false")
        for status in ("successful", "low_confidence"):
            source_dir = dataset / status
            if not source_dir.is_dir():
                continue
            for source in tqdm(sorted(source_dir.iterdir()), desc=f"bundle {status}", unit="file"):
                if source.is_file():
                    link_or_copy(source, data_bundle / "crops" / status / source.name)
        for source in tqdm(sorted((dataset / "refs").iterdir()), desc="bundle references", unit="file"):
            if source.is_file():
                link_or_copy(source, data_bundle / "refs" / source.name)
        link_or_copy(dataset / "training_metadata.csv", data_bundle / "crops_metadata.csv")
        link_or_copy(dataset / "crops_metadata.csv", data_bundle / "audit_metadata.csv")
        for filename in ("bottle_images_manifest.csv", "build_summary.json"):
            link_or_copy(dataset / filename, data_bundle / filename)
        (data_bundle / "dataset-metadata.json").write_text(
            json.dumps(
                metadata(
                    "VIscaNEr whole-bottle classifier data",
                    f"{args.kaggle_username}/viscaner-bottle-classifier-data",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    code_files = (
        Path("dinov3_retrieval.py"),
        Path("full_finetune.py"),
        Path("train_bottle_classifier.py"),
        Path("bottle_classifier/__init__.py"),
        Path("bottle_classifier/local_backbone.py"),
        Path("bottle_classifier/requirements.txt"),
        Path("bottle_classifier/README.md"),
        Path("configs/bottle_classifier_vits16.yaml"),
        Path("configs/bottle_classifier_vits16plus.yaml"),
        Path("kaggle_notebooks/BottleClassifier-DINOv3-ViTS16.ipynb"),
        Path("kaggle_notebooks/BottleClassifier-DINOv3-ViTS16Plus.ipynb"),
    )
    for relative in code_files:
        source = project / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        link_or_copy(source, code_bundle / relative)
    checksums = {
        relative.as_posix(): hashlib.sha256((code_bundle / relative).read_bytes()).hexdigest()
        for relative in code_files
    }
    (code_bundle / "code_sha256.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8"
    )
    (code_bundle / "dataset-metadata.json").write_text(
        json.dumps(
            metadata(
                "VIscaNEr whole-bottle classifier code",
                f"{args.kaggle_username}/viscaner-bottle-classifier-code",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.code_only:
        weight_files = (
            project / "models" / "bottle_classifier_backbones" / "model-s.safetensors",
            project / "models" / "bottle_classifier_backbones" / "model-s_plus.safetensors",
        )
        expected_checksums = {
            "model-s.safetensors": "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d",
            "model-s_plus.safetensors": "208146e499dace99e4c9376ddb8a26f77d64c31c46c4dc4b86ff8bc63b0235e2",
        }
        actual_checksums: dict[str, str] = {}
        for source in weight_files:
            if not source.is_file():
                raise FileNotFoundError(source)
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            expected = expected_checksums[source.name]
            if actual != expected:
                raise ValueError(
                    f"Wrong or corrupted {source.name}: expected {expected}, got {actual}"
                )
            link_or_copy(source, weights_bundle / source.name)
            actual_checksums[source.name] = actual
        (weights_bundle / "weights_sha256.json").write_text(
            json.dumps(actual_checksums, indent=2, sort_keys=True), encoding="utf-8"
        )
        (weights_bundle / "dataset-metadata.json").write_text(
            json.dumps(
                metadata(
                    "VIscaNEr DINOv3 S and S+ local weights",
                    f"{args.kaggle_username}/viscaner-bottle-classifier-weights",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    if not args.code_only:
        print(f"Data bundle: {data_bundle}")
        print(f"Weights bundle: {weights_bundle}")
    print(f"Code bundle: {code_bundle}")


if __name__ == "__main__":
    main()
