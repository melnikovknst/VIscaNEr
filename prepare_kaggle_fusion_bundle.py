#!/usr/bin/env python3
"""Build private Kaggle staging folders for Stage-II fusion.

The hard-set archive is unpacked into the Kaggle data bundle so notebooks do
not spend their GPU session extracting it. Source datasets and checkpoints are
not modified. Existing staging folders are recreated to prevent stale files.
"""

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


def metadata(title: str, dataset_id: str) -> dict[str, object]:
    return {"title": title, "id": dataset_id, "licenses": [{"name": "other"}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--kaggle-username", required=True)
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    output = (args.output_root or project / "kaggle_upload").expanduser().resolve()
    data_bundle = output / "viscaner-fusion-hardset-v1"
    code_bundle = output / "viscaner-fusion-stage2-code"
    stage2a_kernel = output / "viscaner-fusion-stage2a-kernel"
    stage2b_kernel = output / "viscaner-fusion-stage2b-kernel"
    recreate = (
        (code_bundle, stage2a_kernel, stage2b_kernel)
        if args.code_only
        else (data_bundle, code_bundle, stage2a_kernel, stage2b_kernel)
    )
    for bundle in recreate:
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True)

    if not args.code_only:
        hardset = project / "datasets" / "fusion_hardset_v1"
        summary_path = hardset / "build_summary.json"
        manifest_path = hardset / "manifest.csv"
        if not summary_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("Run python -m fusion_stage2.build_hard_dataset first")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("complete") or int(summary.get("hard_rows", 0)) < 10_000:
            raise RuntimeError(f"Hard-set is incomplete or too small: {summary}")
        for source in sorted(hardset.rglob("*")):
            if source.is_file():
                link_or_copy(source, data_bundle / source.relative_to(hardset))
        (data_bundle / "dataset-metadata.json").write_text(
            json.dumps(
                metadata(
                    "VIscaNEr Stage II fusion hard set",
                    f"{args.kaggle_username}/viscaner-fusion-hardset-v1",
                ), indent=2,
            ), encoding="utf-8",
        )

    code_files = (
        Path("dinov3_retrieval.py"),
        Path("full_finetune.py"),
        Path("cascade_resolver/__init__.py"),
        Path("cascade_resolver/config.py"),
        Path("cascade_resolver/modeling.py"),
        Path("bottle_classifier/__init__.py"),
        Path("bottle_classifier/local_backbone.py"),
        Path("ocr_reranker/__init__.py"),
        Path("ocr_reranker/reranker.py"),
        Path("fusion_stage2/__init__.py"),
        Path("fusion_stage2/core.py"),
        Path("fusion_stage2/kaggle_io.py"),
        Path("fusion_stage2/train_fusion.py"),
        Path("fusion_stage2/README.md"),
        Path("kaggle_notebooks/Fusion-Stage2A-Frozen.ipynb"),
        Path("kaggle_notebooks/Fusion-Stage2B-Joint.ipynb"),
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
                "VIscaNEr Stage II fusion code",
                f"{args.kaggle_username}/viscaner-fusion-stage2-code",
            ), indent=2,
        ), encoding="utf-8",
    )
    common_datasets = [
        f"{args.kaggle_username}/viscaner-fusion-stage2-code",
        f"{args.kaggle_username}/viscaner-fusion-hardset-v1",
        f"{args.kaggle_username}/viscaner-dinov3-data",
        f"{args.kaggle_username}/viscaner-bottle-classifier-data",
    ]
    kernels = (
        (
            stage2a_kernel,
            project / "kaggle_notebooks" / "Fusion-Stage2A-Frozen.ipynb",
            {
                "id": f"{args.kaggle_username}/viscaner-fusion-stage-2a-frozen",
                "title": "VIscaNEr Fusion Stage 2A Frozen",
                "kernel_sources": [
                    "f1amex/viscaner-dinov3-b-bottles-hard-fine-tune",
                    "f1amex/viscaner-dinov3-b-labels-hard-fine-tune",
                ],
            },
        ),
        (
            stage2b_kernel,
            project / "kaggle_notebooks" / "Fusion-Stage2B-Joint.ipynb",
            {
                "id": f"{args.kaggle_username}/viscaner-fusion-stage-2b-joint",
                "title": "VIscaNEr Fusion Stage 2B Joint",
                "kernel_sources": [
                    "f1amex/viscaner-dinov3-b-bottles-hard-fine-tune",
                    "f1amex/viscaner-dinov3-b-labels-hard-fine-tune",
                    f"{args.kaggle_username}/viscaner-fusion-stage-2a-frozen",
                ],
            },
        ),
    )
    for kernel_dir, notebook, specific in kernels:
        link_or_copy(notebook, kernel_dir / notebook.name)
        kernel_metadata = {
            **specific,
            "code_file": notebook.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": common_datasets,
            "competition_sources": [],
        }
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(kernel_metadata, indent=2), encoding="utf-8"
        )
    if not args.code_only:
        print(f"Data bundle: {data_bundle}")
    print(f"Code bundle: {code_bundle}")
    print(f"Stage 2A kernel: {stage2a_kernel}")
    print(f"Stage 2B kernel: {stage2b_kernel}")


if __name__ == "__main__":
    main()
