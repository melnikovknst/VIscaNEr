#!/usr/bin/env python3
"""Create Kaggle-upload folders without changing the source datasets.

Files are hard-linked when possible (no duplicate disk usage on the same
volume) and copied only when hard links are unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Rebuild only viscaner-dinov3-code for a lightweight code update",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Rebuild only the shared ViT-B and ConvNeXt-B weights dataset",
    )
    args = parser.parse_args()
    if args.code_only and args.weights_only:
        parser.error("--code-only and --weights-only are mutually exclusive")

    project = args.project_root.expanduser().resolve()
    output = (args.output_root or project / "kaggle_upload").expanduser().resolve()
    data_bundle = output / "viscaner-dinov3-data"
    weights_bundle = output / "viscaner-dinov3-vitb16-weights"
    code_bundle = output / "viscaner-dinov3-code"
    # These are generated staging directories. Recreate them so files removed
    # or moved between crop statuses cannot survive from an older bundle.
    if args.code_only:
        bundles_to_recreate = (code_bundle,)
    elif args.weights_only:
        bundles_to_recreate = (weights_bundle,)
    else:
        bundles_to_recreate = (
            data_bundle,
            weights_bundle,
            code_bundle,
        )
    for bundle in bundles_to_recreate:
        if bundle.exists():
            shutil.rmtree(bundle)
    if not args.code_only and not args.weights_only:
        data_bundle.mkdir(parents=True, exist_ok=True)
    if not args.code_only:
        weights_bundle.mkdir(parents=True, exist_ok=True)
    if not args.weights_only:
        code_bundle.mkdir(parents=True, exist_ok=True)

    if not args.code_only and not args.weights_only:
        crops_source = project / "datasets/dinov3_target_crops"
        # The offline builder accepts only YOLO boxes geometrically aligned with
        # the generator-tracked target bottle. Rejected rows remain in metadata;
        # only identity-safe successful crops are needed by DINO.
        for status in ("successful",):
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
            (
                project / "datasets/bottle_images_45k/bottle_images_manifest.csv",
                Path("bottle_images_manifest.csv"),
            ),
        ):
            link_or_copy(source, data_bundle / relative)

    if not args.code_only:
        for source, destination in (
            (
                project / "models/dinov3/model.safetensors",
                weights_bundle / "model.safetensors",
            ),
            (
                project / "models/dinov3/config.json",
                weights_bundle / "config.json",
            ),
            (
                project / "models/dinov3/dinov3-convnext-b.safetensors",
                weights_bundle / "dinov3-convnext-b.safetensors",
            ),
        ):
            if not source.is_file():
                raise FileNotFoundError(source)
            link_or_copy(source, destination)
        weights_checksums = {
            "model.safetensors": hashlib.sha256(
                (weights_bundle / "model.safetensors").read_bytes()
            ).hexdigest(),
            "config.json": hashlib.sha256(
                (weights_bundle / "config.json").read_bytes()
            ).hexdigest(),
            "dinov3-convnext-b.safetensors": hashlib.sha256(
                (weights_bundle / "dinov3-convnext-b.safetensors").read_bytes()
            ).hexdigest()
        }
        (weights_bundle / "weights_sha256.json").write_text(
            json.dumps(weights_checksums, indent=2, sort_keys=True), encoding="utf-8"
        )

    code_files = (
        Path("build_target_aligned_crops.py"),
        Path("deeptune_backbones.py"),
        Path("dinov3_retrieval.py"),
        Path("full_finetune.py"),
        Path("train_dinov3_deeptune.py"),
        Path("train_dinov3_retrieval.py"),
        Path("train_dinov3_retrieval.ipynb"),
        Path("kaggle_notebooks/DINOv3-finetune.ipynb"),
        Path("kaggle_notebooks/DINOv3-deeptune.ipynb"),
        Path("kaggle_notebooks/README.md"),
        Path("validate_dinov3_new_crops.ipynb"),
        Path("README_FULL_TRAINING.md"),
        Path("requirements.txt"),
        Path("configs/dinov3_retrieval.yaml"),
        Path("configs/dinov3_full_finetune.yaml"),
    )
    if not args.weights_only:
        for relative in code_files:
            link_or_copy(project / relative, code_bundle / relative)

        code_checksums = {
            str(relative): hashlib.sha256((code_bundle / relative).read_bytes()).hexdigest()
            for relative in code_files
        }
        (code_bundle / "code_sha256.json").write_text(
            json.dumps(code_checksums, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if not args.code_only and not args.weights_only:
        (data_bundle / "dataset-metadata.json").write_text(
            json.dumps(
                kaggle_metadata(
                    "VIscaNEr DINOv3 retrieval data",
                    f"{args.kaggle_username}/viscaner-dinov3-data",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        (weights_bundle / "dataset-metadata.json").write_text(
            json.dumps(
                kaggle_metadata(
                    "VIscaNEr DINOv3 ViT-B and ConvNeXt-B weights",
                    f"{args.kaggle_username}/viscaner-dinov3-vitb16-weights",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    if not args.code_only:
        # Weight metadata is also written in --weights-only mode.
        if not (weights_bundle / "dataset-metadata.json").is_file():
            (weights_bundle / "dataset-metadata.json").write_text(
                json.dumps(
                    kaggle_metadata(
                        "VIscaNEr DINOv3 ViT-B and ConvNeXt-B weights",
                        f"{args.kaggle_username}/viscaner-dinov3-vitb16-weights",
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
    if not args.weights_only:
        (code_bundle / "dataset-metadata.json").write_text(
            json.dumps(
                kaggle_metadata(
                    "VIscaNEr DINOv3 retrieval code",
                    f"{args.kaggle_username}/viscaner-dinov3-code",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    if not args.code_only and not args.weights_only:
        print(f"Data bundle: {data_bundle}")
    if not args.code_only:
        print(f"Weights bundle: {weights_bundle}")
    if not args.weights_only:
        print(f"Code bundle: {code_bundle}")
    if args.code_only:
        print("Only the code bundle was rebuilt; data and weights were untouched.")
    elif args.weights_only:
        print("Only the shared weights bundle was rebuilt; data and code were untouched.")
    else:
        print("Upload the data, code and shared weights folders as private Kaggle Datasets.")


if __name__ == "__main__":
    main()
