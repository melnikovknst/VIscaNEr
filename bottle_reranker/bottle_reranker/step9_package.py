"""Step 9 - assemble the handover manifest and the Kaggle staging folder.

The manifest is the single table the next stage (encoder training) reads. One
row per prepared bottle, with relative paths, identity, split, provenance group,
the coordinates and transforms that produced the crop, and the quality flags.
Relative paths keep the package relocatable: it works the same in the repo, in
/kaggle/input and on another machine.

The Kaggle staging folder is written but never uploaded automatically. Upload is
a deliberate, outward-facing act and stays a manual command, printed at the end.
Ownership is configurable - the read-only upstream account and the working copy
are two different names in the config, never hard-coded.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .common import (
    percentage,
    read_csv_rows,
    stage_record,
    summarise_counts,
    write_csv_rows,
    write_json,
    write_text,
)
from .config import Config

CONFIG_FINGERPRINT_KEYS = ("output.root", "output.manifest_name", "kaggle.owner", "kaggle.dataset_slug")

MANIFEST_COLUMNS = [
    "sample_id", "role", "source", "source_kind", "photographic",
    "wine_slug", "split", "provenance_group", "dino_unseen_identity",
    "bottle_path", "mask_path", "normalized_path", "top_path",
    "original_image_path", "source_width", "source_height", "exif_orientation",
    "label_box", "crop_window", "deskew_applied_deg", "axis_angle_deg",
    "axis_reliable", "orientation_confident", "shoulder_row", "shoulder_method",
    "top_mask_fill", "mask_hole_fraction", "sharpness_var_laplacian",
    "quality_flags", "usable_for_training",
]


def build_manifest(config: Config) -> tuple[Path, list[dict[str, Any]]]:
    audit = config.output_dir("audit", create=False)
    crops = read_csv_rows(audit / "query_crops.csv") if (audit / "query_crops.csv").is_file() else []
    refs = read_csv_rows(audit / "reference_crops.csv") if (audit / "reference_crops.csv").is_file() else []
    splits = (
        {f"{r['source']}:{r['image_relative_path']}": r for r in read_csv_rows(audit / "splits.csv")}
        if (audit / "splits.csv").is_file() else {}
    )

    rows: list[dict[str, Any]] = []
    for crop in crops:
        if crop.get("status") != "ok":
            continue
        key = f"{crop['source']}:{crop['image_relative_path']}"
        split_row = splits.get(key, {})
        flags = crop.get("quality_flags", "")
        rows.append({
            "sample_id": crop["stem"],
            "role": "query",
            "source": crop["source"],
            "source_kind": split_row.get("source", ""),
            "photographic": split_row.get("photographic", ""),
            "wine_slug": crop["wine_slug"],
            "split": split_row.get("split", "unassigned"),
            "provenance_group": split_row.get("provenance_group", ""),
            "dino_unseen_identity": split_row.get("dino_unseen_identity", ""),
            "bottle_path": crop.get("bottle_path", ""),
            "mask_path": crop.get("mask_path", ""),
            "normalized_path": crop.get("normalized_path", ""),
            "top_path": crop.get("top_path", ""),
            "original_image_path": crop.get("image_path", ""),
            "source_width": crop.get("source_width", ""),
            "source_height": crop.get("source_height", ""),
            "exif_orientation": crop.get("exif_orientation", ""),
            "label_box": crop.get("label_box", ""),
            "crop_window": crop.get("crop_window", ""),
            "deskew_applied_deg": crop.get("deskew_applied_deg", ""),
            "axis_angle_deg": crop.get("axis_angle_deg", ""),
            "axis_reliable": crop.get("axis_reliable", ""),
            "orientation_confident": crop.get("orientation_confident", ""),
            "shoulder_row": crop.get("shoulder_row", ""),
            "shoulder_method": crop.get("shoulder_method", ""),
            "top_mask_fill": crop.get("top_mask_fill", ""),
            "mask_hole_fraction": crop.get("mask_hole_fraction", ""),
            "sharpness_var_laplacian": crop.get("sharpness_var_laplacian", ""),
            "quality_flags": flags,
            "usable_for_training": not ({"crop_too_small", "top_part_empty"} & set(flags.split(";"))),
        })

    for ref in refs:
        if ref.get("status") != "ok":
            continue
        rows.append({
            "sample_id": f"ref__{ref['wine_slug']}",
            "role": "reference",
            "source": "catalog",
            "source_kind": "catalog_reference",
            "photographic": True,
            "wine_slug": ref["wine_slug"],
            "split": "gallery",
            "provenance_group": f"catalog:{ref['wine_slug']}",
            "dino_unseen_identity": "",
            "bottle_path": ref.get("bottle_path", ""),
            "mask_path": ref.get("mask_path", ""),
            "normalized_path": ref.get("normalized_path", ""),
            "top_path": ref.get("top_path", ""),
            "original_image_path": ref.get("reference_file", ""),
            "source_width": ref.get("reference_width", ""),
            "source_height": ref.get("reference_height", ""),
            "shoulder_method": ref.get("shoulder_method", ""),
            "top_mask_fill": ref.get("top_mask_fill", ""),
            "quality_flags": ref.get("quality_flags", ""),
            "usable_for_training": str(ref.get("usable_for_bottle_comparison", "")).lower() == "true",
        })

    destination = config.output_root / str(config.get("output.manifest_name"))
    write_csv_rows(destination, rows, fieldnames=MANIFEST_COLUMNS)
    return destination, rows


def _dataset_metadata(config: Config) -> dict[str, Any]:
    owner = str(config.get("kaggle.owner"))
    slug = str(config.get("kaggle.dataset_slug"))
    return {
        "title": str(config.get("kaggle.title")),
        "id": f"{owner}/{slug}",
        "licenses": [{"name": str(config.get("kaggle.licence", "other"))}],
        "isPrivate": bool(config.get("kaggle.private", True)),
    }


def run(config: Config, *, copy_images: bool = True) -> dict[str, Any]:
    manifest_path, manifest_rows = build_manifest(config)

    staging = config.resolve(config.get("kaggle.staging_root"))
    staging.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    payload = [
        ("queries", config.output_dir("queries", create=False)),
        ("references", config.output_dir("references", create=False)),
        ("pairs", config.output_dir("pairs", create=False)),
        ("plan", config.output_dir("plan", create=False)),
        ("audit", config.output_dir("audit", create=False)),
    ]
    for name, directory in payload:
        if not directory.is_dir():
            continue
        target = staging / name
        if copy_images or name not in {"queries", "references"}:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(directory, target, ignore=shutil.ignore_patterns("*_ledger.jsonl", "*.tmp"))
            copied.append(name)

    if manifest_path.is_file():
        shutil.copy2(manifest_path, staging / manifest_path.name)

    reports = config.reports_root
    if reports.is_dir():
        target = staging / "reports"
        target.mkdir(parents=True, exist_ok=True)
        for item in reports.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)

    write_json(staging / "dataset-metadata.json", _dataset_metadata(config))

    owner = str(config.get("kaggle.owner"))
    slug = str(config.get("kaggle.dataset_slug"))
    source_owner = str(config.get("kaggle.source_owner"))
    readme = f"""# {config.get('kaggle.title')}

Prepared by `bottle_reranker` from the VIscaNEr project. This package is DATA
for the next stage; no encoder has been trained on it yet.

## Layout

```
{config.get('output.manifest_name')}   one row per prepared bottle or reference
queries/bottle|mask|normalized|top/    query crops
references/bottle|mask|normalized|top/ catalog reference crops, same four views
pairs/organic_pairs.csv                candidate pairs as seen at inference
pairs/injected_triplets.csv            training triplets with the truth injected
plan/confusion_pairs_to_collect.csv    what to photograph next
audit/                                 correspondences, review queue, splits,
                                       segmentation and DINO candidate tables
reports/                               per-stage JSON reports and the review sheet
```

All paths in the manifest are relative to the package root, so it behaves the
same in the repository and under `/kaggle/input/{slug}`.

## Provenance

Read-only inputs come from the `{source_owner}` Kaggle account. This package is
published to `{owner}/{slug}`. Both names live in
`bottle_reranker/configs/bottle_reranker.yaml` and are not hard-coded anywhere.

## Reading the numbers

Similarity columns are raw cosine similarities between L2-normalised
embeddings. They are not probabilities. `dino_exposure` says whether a
prediction was made on an image the retrieval checkpoint trained on; thresholds
must not be tuned on those rows.

See `reports/FINDINGS.md` for what this data can and cannot support.
"""
    write_text(staging / "README.md", readme)

    by_role = summarise_counts(r["role"] for r in manifest_rows)
    by_split = summarise_counts(r["split"] for r in manifest_rows)
    usable = sum(1 for r in manifest_rows if r["usable_for_training"])

    report = stage_record(
        "step9_package",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        manifest={
            "path": config.relative(manifest_path),
            "rows": len(manifest_rows),
            "by_role": by_role,
            "by_split": by_split,
            "usable_for_training": usable,
            "usable_percent": percentage(usable, max(1, len(manifest_rows))),
            "flagged": summarise_counts(
                flag for r in manifest_rows for flag in (r.get("quality_flags") or "").split(";") if flag
            ),
        },
        kaggle={
            "staging_root": config.relative(staging),
            "dataset_id": f"{owner}/{slug}",
            "private": bool(config.get("kaggle.private", True)),
            "source_owner": source_owner,
            "directories_copied": copied,
            "upload_is_manual": True,
        },
        upload_commands=[
            f"kaggle datasets create -p {config.relative(staging)} -r zip   # first publication",
            f"kaggle datasets version -p {config.relative(staging)} -m \"<message>\" -r zip   # later versions",
        ],
    )
    write_json(config.report_path("step9_package.json"), report)
    print(f"step9: manifest with {len(manifest_rows)} rows -> {config.relative(manifest_path)}")
    print(f"       staging ready at {config.relative(staging)} for {owner}/{slug}")
    print("       upload is manual; run one of:")
    for command in report["upload_commands"]:
        print(f"         {command}")
    return report
