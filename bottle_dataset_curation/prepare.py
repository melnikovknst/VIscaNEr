#!/usr/bin/env python3
"""Prepare a portable review queue and ingest the labelled fresh-photo archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from .paths import (
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    dhash,
    dhash_file,
    file_sha256,
    hash_similarity,
    reference_lookup,
    resolve_recorded_image,
)


DECISION_COLUMNS = ["review_id", "decision", "note", "decided_at"]
def _stable_id(*values: object) -> str:
    payload = "\0".join(str(value) for value in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _portable_metadata(
    metadata_path: Path,
    crops_root: Path,
    source_manifest_path: Path,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path).copy()
    required = {
        "source_relative_path",
        "wine_slug",
        "crop_path",
        "status",
        "source_path",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_path} is missing {sorted(missing)}")
    if metadata["source_relative_path"].duplicated().any():
        raise ValueError("Bottle metadata has duplicate source_relative_path values")

    manifest = pd.read_csv(source_manifest_path)
    manifest_required = {"source_relative_path", "merged_path", "merged_filename"}
    missing = manifest_required.difference(manifest.columns)
    if missing:
        raise ValueError(f"{source_manifest_path} is missing {sorted(missing)}")
    metadata = metadata.merge(
        manifest[list(manifest_required)],
        on="source_relative_path",
        how="left",
        validate="one_to_one",
    )

    merged_root = source_manifest_path.parent

    def local_crop(row: pd.Series) -> str:
        path = resolve_recorded_image(row["crop_path"], crops_root, row["status"])
        return str(path) if path else ""

    def local_original(row: pd.Series) -> str:
        recorded = Path(str(row["source_path"]))
        if recorded.is_file():
            return str(recorded.resolve())
        merged = Path(str(row.get("merged_path", "")))
        if merged.is_file():
            return str(merged.resolve())
        candidate = merged_root / str(row.get("merged_filename", ""))
        return str(candidate.resolve()) if candidate.is_file() else ""

    metadata["local_crop_path"] = metadata.apply(local_crop, axis=1)
    metadata["local_original_path"] = metadata.apply(local_original, axis=1)
    metadata["crop_basename"] = metadata["crop_path"].fillna("").map(
        lambda value: Path(str(value)).name
    )
    if metadata.loc[metadata["crop_basename"].ne(""), "crop_basename"].duplicated().any():
        raise ValueError("Bottle crop basenames are not unique")
    return metadata


def build_review_queue(
    audit_path: Path,
    metadata_path: Path,
    crops_root: Path,
    refs_root: Path,
    source_manifest_path: Path,
    output_dir: Path,
    fresh_manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    audit = pd.read_csv(audit_path)
    required = {
        "split",
        "query_path",
        "true_slug",
        "rank",
        "true_similarity",
        "top1_top2_margin",
        "top1_slug",
        "top1_similarity",
        "top2_slug",
        "top2_similarity",
    }
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"{audit_path} is missing {sorted(missing)}")
    errors = audit.loc[audit["rank"].gt(1)].copy()
    errors["crop_basename"] = errors["query_path"].map(lambda value: Path(str(value)).name)

    metadata = _portable_metadata(metadata_path, crops_root, source_manifest_path)
    metadata_columns = [
        "crop_basename",
        "source_relative_path",
        "source_filename",
        "wine_slug",
        "local_crop_path",
        "local_original_path",
        "status",
        "confidence",
        "label_coverage",
        "score_margin",
        "num_detections",
        "num_eligible_candidates",
        "touches_frame",
        "vertically_truncated",
        "reject_reason",
    ]
    metadata_columns = [column for column in metadata_columns if column in metadata.columns]
    crop_metadata = metadata.loc[metadata["crop_basename"].ne(""), metadata_columns]
    errors = errors.merge(
        crop_metadata,
        on="crop_basename",
        how="left",
        validate="many_to_one",
    )
    fresh_mask = errors["split"].eq("fresh_intake")
    if fresh_mask.any():
        if fresh_manifest_path is None or not fresh_manifest_path.is_file():
            raise FileNotFoundError(
                "The audit contains fresh_intake rows but fresh_manifest.csv is unavailable"
            )
        fresh = pd.read_csv(fresh_manifest_path)
        fresh_lookup = fresh.set_index("image_path", drop=False)
        for index in errors.index[fresh_mask]:
            query_path = str(errors.at[index, "query_path"])
            if query_path not in fresh_lookup.index:
                raise KeyError(f"Fresh audit image is absent from fresh_manifest.csv: {query_path}")
            fresh_row = fresh_lookup.loc[query_path]
            errors.at[index, "local_crop_path"] = query_path
            errors.at[index, "local_original_path"] = query_path
            errors.at[index, "source_relative_path"] = (
                f"fresh/{fresh_row['archive_filename']}"
            )
            errors.at[index, "wine_slug"] = str(fresh_row["wine_slug"])
            errors.at[index, "status"] = "fresh_intake"
    identity_mismatch = errors["wine_slug"].notna() & errors["wine_slug"].ne(errors["true_slug"])
    if identity_mismatch.any():
        raise ValueError("Audit true_slug disagrees with bottle metadata wine_slug")

    refs = reference_lookup(refs_root)
    errors["true_reference_path"] = errors["true_slug"].map(
        lambda slug: str(refs.get(str(slug), ""))
    )
    errors["predicted_reference_path"] = errors["top1_slug"].map(
        lambda slug: str(refs.get(str(slug), ""))
    )
    missing_paths = {
        "crop": int(errors["local_crop_path"].fillna("").eq("").sum()),
        "original": int(errors["local_original_path"].fillna("").eq("").sum()),
        "true_reference": int(errors["true_reference_path"].eq("").sum()),
        "predicted_reference": int(errors["predicted_reference_path"].eq("").sum()),
    }
    if missing_paths["crop"] or missing_paths["true_reference"] or missing_paths["predicted_reference"]:
        raise FileNotFoundError(f"Review queue has unresolved required images: {missing_paths}")

    ref_hashes = {slug: dhash_file(path) for slug, path in refs.items()}
    errors["reference_dhash_similarity"] = errors.apply(
        lambda row: hash_similarity(
            ref_hashes[str(row["true_slug"])], ref_hashes[str(row["top1_slug"])]
        ),
        axis=1,
    )
    errors["confusion_pair"] = errors.apply(
        lambda row: " <> ".join(sorted((str(row["true_slug"]), str(row["top1_slug"])))),
        axis=1,
    )
    errors["pair_frequency"] = errors.groupby("confusion_pair")["confusion_pair"].transform("size")
    errors["wrong_lead"] = errors["top1_similarity"] - errors["true_similarity"]
    errors["review_id"] = errors.apply(
        lambda row: _stable_id(row["crop_basename"], row["true_slug"], row["top1_slug"]),
        axis=1,
    )

    likely_duplicate = (
        errors["rank"].eq(2)
        & errors["top1_top2_margin"].le(0.03)
        & errors["reference_dhash_similarity"].ge(0.875)
    )
    likely_data_problem = errors["rank"].ge(5) | errors["true_similarity"].lt(0.45)
    errors["suggested_bucket"] = np.select(
        [fresh_mask, likely_duplicate, likely_data_problem],
        ["inspect_fresh_label", "inspect_visual_duplicate", "inspect_crop_or_label"],
        default="inspect_hard_model_error",
    )
    errors = errors.sort_values(
        ["suggested_bucket", "pair_frequency", "rank", "wrong_lead"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    errors["review_order"] = np.arange(1, len(errors) + 1)

    preferred = [
        "review_order",
        "review_id",
        "suggested_bucket",
        "confusion_pair",
        "pair_frequency",
        "split",
        "source_relative_path",
        "local_original_path",
        "local_crop_path",
        "true_slug",
        "true_reference_path",
        "top1_slug",
        "predicted_reference_path",
        "rank",
        "true_similarity",
        "top1_similarity",
        "top2_similarity",
        "top1_top2_margin",
        "wrong_lead",
        "reference_dhash_similarity",
        "status",
        "confidence",
        "label_coverage",
        "score_margin",
        "num_detections",
        "num_eligible_candidates",
        "touches_frame",
        "vertically_truncated",
        "reject_reason",
        "query_path",
    ]
    queue = errors[[column for column in preferred if column in errors.columns]].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "review_queue.csv"
    queue.to_csv(queue_path, index=False)
    decisions_path = output_dir / "curation_decisions.csv"
    if not decisions_path.exists():
        pd.DataFrame(columns=DECISION_COLUMNS).to_csv(decisions_path, index=False)

    summary = {
        "audit_rows": int(len(audit)),
        "review_errors": int(len(queue)),
        "rank2_errors": int(queue["rank"].eq(2).sum()),
        "unique_confusion_pairs": int(queue["confusion_pair"].nunique()),
        "suggested_bucket_counts": queue["suggested_bucket"].value_counts().to_dict(),
        "missing_paths": missing_paths,
        "review_queue": str(queue_path.resolve()),
        "decisions": str(decisions_path.resolve()),
    }
    (output_dir / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return queue, summary


def ingest_fresh_archive(
    archive_path: Path,
    refs_root: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    refs = reference_lookup(refs_root)
    fresh_root = output_dir / "fresh_images"
    fresh_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    with ZipFile(archive_path) as archive:
        if "wines.csv" not in archive.namelist():
            raise FileNotFoundError("Вино.zip must contain wines.csv")
        catalog = pd.read_csv(archive.open("wines.csv"))
        required = {"Slug", "Название фото"}
        if missing := required.difference(catalog.columns):
            raise ValueError(f"wines.csv is missing {sorted(missing)}")
        members = {
            Path(info.filename).name: info
            for info in archive.infolist()
            if not info.is_dir()
        }
        if catalog["Slug"].duplicated().any():
            raise ValueError("wines.csv has duplicate Slug rows")

        for record in catalog.to_dict("records"):
            slug = str(record["Slug"])
            filename = str(record["Название фото"])
            info = members.get(filename)
            ref_path = refs.get(slug)
            if info is None or ref_path is None:
                rows.append(
                    {
                        "wine_slug": slug,
                        "archive_filename": filename,
                        "image_path": "",
                        "status": "missing_image" if info is None else "missing_reference",
                    }
                )
                continue
            suffix = Path(filename).suffix.lower() or ".jpg"
            destination = fresh_root / slug / f"fresh{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(info)
            destination.write_bytes(data)
            try:
                with Image.open(destination) as image:
                    image.load()
                    width, height = image.size
                    fresh_hash = dhash(image)
                ref_hash = dhash_file(ref_path)
                visual_similarity = hash_similarity(fresh_hash, ref_hash)
                status = "near_reference_duplicate" if visual_similarity >= 0.95 else "candidate"
                error = ""
            except Exception as exc:
                width = height = 0
                visual_similarity = np.nan
                status = "corrupt"
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "wine_slug": slug,
                    "archive_filename": filename,
                    "image_path": str(destination.resolve()),
                    "reference_path": str(ref_path),
                    "status": status,
                    "width": width,
                    "height": height,
                    "sha256": file_sha256(destination),
                    "reference_dhash_similarity": visual_similarity,
                    "error": error,
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["status", "wine_slug"]).reset_index(drop=True)
    manifest_path = output_dir / "fresh_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "archive": str(archive_path.resolve()),
        "catalog_rows": int(len(manifest)),
        "status_counts": manifest["status"].value_counts().to_dict(),
        "candidate_rows": int(manifest["status"].eq("candidate").sum()),
        "near_reference_duplicates": int(
            manifest["status"].eq("near_reference_duplicate").sum()
        ),
        "manifest": str(manifest_path.resolve()),
    }
    (output_dir / "fresh_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=PROJECT_ROOT
        / "datasets/bottle_dino_curation/source_kaggle/retrieval_audit.csv",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=PROJECT_ROOT / "datasets/bottle_classifier_crops/crops_metadata.csv",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=PROJECT_ROOT / "datasets/bottle_images_45k/bottle_images_manifest.csv",
    )
    parser.add_argument(
        "--crops-root",
        type=Path,
        default=PROJECT_ROOT / "datasets/bottle_classifier_crops",
    )
    parser.add_argument(
        "--refs-root",
        type=Path,
        default=PROJECT_ROOT / "datasets/bottle_classifier_crops/refs",
    )
    parser.add_argument(
        "--fresh-zip",
        type=Path,
        default=Path.home() / "Downloads/Вино.zip",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-fresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue, review_summary = build_review_queue(
        args.audit_csv,
        args.metadata_csv,
        args.crops_root,
        args.refs_root,
        args.source_manifest,
        args.output_dir,
        args.output_dir / "fresh_manifest.csv",
    )
    print(json.dumps(review_summary, ensure_ascii=False, indent=2), flush=True)
    if not args.skip_fresh:
        _, fresh_summary = ingest_fresh_archive(
            args.fresh_zip, args.refs_root, args.output_dir
        )
        print(json.dumps(fresh_summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Prepared {len(queue)} model errors for review", flush=True)


if __name__ == "__main__":
    main()
