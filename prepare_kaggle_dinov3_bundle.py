#!/usr/bin/env python3
"""Create Kaggle-upload folders without changing the source datasets.

Files are hard-linked when possible (no duplicate disk usage on the same
volume) and copied only when hard links are unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def kaggle_metadata(title: str, dataset_id: str) -> dict[str, str]:
    return {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "other"}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/Users/konstantinmelnikov/Desktop/work/VIscaNEr"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--kaggle-username", required=True)
    args = parser.parse_args()

    project = args.project_root.expanduser().resolve()
    output = (args.output_root or project / "kaggle_upload").expanduser().resolve()
    data_bundle = output / "viscaner-dinov3-data"
    weights_bundle = output / "viscaner-dinov3-vitb16-weights"
    code_bundle = output / "viscaner-dinov3-code"
    # These are generated staging directories. Recreate them so files removed
    # or moved between crop statuses cannot survive from an older bundle.
    for bundle in (data_bundle, weights_bundle, code_bundle):
        if bundle.exists():
            shutil.rmtree(bundle)
    data_bundle.mkdir(parents=True, exist_ok=True)
    weights_bundle.mkdir(parents=True, exist_ok=True)
    code_bundle.mkdir(parents=True, exist_ok=True)

    crops_source = project / "datasets/yolo_label_detector/crops"
    for status in ("successful", "low_confidence"):
        files = sorted((crops_source / status).glob("*"))
        for source in tqdm(files, desc=f"bundle {status}"):
            if source.is_file():
                link_or_copy(source, data_bundle / "crops" / status / source.name)

    refs_source = project / "datasets/wine-scanner/data/refs/rgb"
    for source in tqdm(sorted(refs_source.glob("*")), desc="bundle references"):
        if source.is_file():
            link_or_copy(source, data_bundle / "refs" / source.name)

    for source, relative in (
        (crops_source / "crops_metadata.csv", Path("crops_metadata.csv")),
        (project / "datasets/bottle_images_45k/bottle_images_manifest.csv", Path("bottle_images_manifest.csv")),
    ):
        link_or_copy(source, data_bundle / relative)

    link_or_copy(project / "models/dinov3/model.safetensors", weights_bundle / "model.safetensors")
    link_or_copy(project / "models/dinov3/config.json", weights_bundle / "config.json")

    for relative in (
        Path("dinov3_retrieval.py"),
        Path("train_dinov3_retrieval.py"),
        Path("train_dinov3_retrieval.ipynb"),
        Path("requirements.txt"),
        Path("configs/dinov3_retrieval.yaml"),
    ):
        link_or_copy(project / relative, code_bundle / relative)

    (data_bundle / "dataset-metadata.json").write_text(
        json.dumps(
            kaggle_metadata("VIscaNEr DINOv3 retrieval data", f"{args.kaggle_username}/viscaner-dinov3-data"),
            indent=2,
        ),
        encoding="utf-8",
    )
    (weights_bundle / "dataset-metadata.json").write_text(
        json.dumps(
            kaggle_metadata(
                "VIscaNEr DINOv3 ViT-B16 weights",
                f"{args.kaggle_username}/viscaner-dinov3-vitb16-weights",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (code_bundle / "dataset-metadata.json").write_text(
        json.dumps(
            kaggle_metadata("VIscaNEr DINOv3 retrieval code", f"{args.kaggle_username}/viscaner-dinov3-code"),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Data bundle: {data_bundle}")
    print(f"Weights bundle: {weights_bundle}")
    print(f"Code bundle: {code_bundle}")
    print("Upload all three folders as private Kaggle Datasets.")


if __name__ == "__main__":
    main()
