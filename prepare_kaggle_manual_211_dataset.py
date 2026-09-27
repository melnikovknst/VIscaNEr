#!/usr/bin/env python3
"""Stage the reviewed 211-photo manual dataset for a private Kaggle upload."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--kaggle-username", required=True)
    args = parser.parse_args()

    project = args.project_root.expanduser().resolve()
    source = project / "datasets" / "manual_211"
    output = (args.output_root or project / "kaggle_upload").expanduser().resolve()
    destination = output / "viscaner-manual-211"
    summary_path = source / "build_summary.json"
    manifest_path = source / "manifest.csv"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run notebooks/Build-Manual-211.ipynb first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("complete") or int(summary.get("rows", 0)) != 211:
        raise RuntimeError(f"Manual-211 dataset is incomplete: {summary}")
    if int(summary.get("trainable_catalog_rows", 0)) != 99:
        raise RuntimeError(f"Unexpected catalog-labelled row count: {summary}")

    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            link_or_copy(path, destination / path.relative_to(source))
    metadata = {
        "title": "VIscaNEr Manual 211",
        "id": f"{args.kaggle_username}/viscaner-manual-211",
        "licenses": [{"name": "other"}],
    }
    (destination / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
