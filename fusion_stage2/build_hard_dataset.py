#!/usr/bin/env python3
"""Build a portable 12k hard-pair dataset for Stage-II fusion.

Difficulty is measured with the existing label DINO-B predictions, a fresh
evaluation of the current whole-bottle DINO-B, branch disagreement, retrieval
margins and detector/crop quality. ``hard_train`` comes only from the original
retrieval train split, while ``hard_val`` comes from the old non-training
``val_seen`` split. The identity-disjoint ``val_unseen`` rows remain test-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cascade_resolver.config import CascadeConfig
from cascade_resolver.data import build_inventory, reference_lookup
from cascade_resolver.evaluation import rank_primary
from cascade_resolver.modeling import (
    choose_device,
    embed_paths,
    load_retrieval_model,
    release_accelerator_memory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "fusion_hardset_v1"
DEFAULT_ARCHIVE = PROJECT_ROOT / "datasets" / "fusion_hardset_v1.zip"


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def load_label_audit(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "source_relative_path",
        "b_true_rank",
        "b_top1_label_id",
        "b_top1_slug",
        "b_top1_similarity",
        "b_top2_similarity",
        "b_top1_top2_gap",
        "b_true_similarity",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Label audit is missing columns: {sorted(missing)}")
    if frame["source_relative_path"].duplicated().any():
        raise ValueError("Label audit contains duplicate source_relative_path rows")
    keep = [
        "source_relative_path",
        "b_true_rank",
        "b_top1_label_id",
        "b_top1_slug",
        "b_top1_similarity",
        "b_top2_similarity",
        "b_top1_top2_gap",
        "b_true_similarity",
    ]
    for position in range(1, 11):
        for suffix in ("label_id", "slug", "similarity"):
            column = f"b_top{position}_{suffix}"
            if column in frame.columns and column not in keep:
                keep.append(column)
    return frame[keep].rename(
        columns={column: column.replace("b_", "label_", 1) for column in keep if column.startswith("b_")}
    )


def add_original_photo_fallbacks(
    inventory: pd.DataFrame,
    bottle_crops_root: Path,
) -> tuple[pd.DataFrame, int]:
    """Use the original photo when YOLO produced no usable bottle box.

    This is the established production policy for sub-threshold bottle
    detections. A missing detection is the lowest-confidence case, so silently
    dropping it would make both the hard set and test set artificially easy.
    """
    result = inventory.copy()
    fallback_dir = bottle_crops_root / "failed_original_fallback"
    mask = result["label_crop_available"] & ~result["bottle_crop_available"]
    applied = 0
    for index, row in result.loc[mask].iterrows():
        source = Path(str(row["original_path"]))
        if not source.is_file():
            continue
        filename = portable_name(str(row["source_relative_path"]), str(source))
        destination = fallback_dir / filename
        link_or_copy(source, destination)
        result.at[index, "bottle_status"] = "failed_original_fallback"
        result.at[index, "bottle_recorded_path"] = filename
        result.at[index, "bottle_detector_confidence"] = 0.0
        result.at[index, "bottle_crop_path"] = str(destination.resolve())
        result.at[index, "bottle_crop_available"] = True
        applied += 1
    return result, applied


def evaluate_bottle_branch(
    inventory: pd.DataFrame,
    checkpoint: Path,
    refs_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cache_dir: Path,
) -> pd.DataFrame:
    refs = reference_lookup(refs_root)
    slugs = sorted(refs)
    label_by_slug = {slug: index for index, slug in enumerate(slugs)}
    query_identities = set(inventory["wine_slug"].astype(str))
    if missing := query_identities.difference(slugs):
        raise ValueError(f"Bottle gallery is missing query identities: {sorted(missing)[:5]}")
    rows = inventory.copy().reset_index(drop=True)
    rows["label_id"] = rows["wine_slug"].map(label_by_slug).astype(int)
    model, info = load_retrieval_model(checkpoint, "vitb16", device)
    image_size = int(info["image_size"])
    gallery_paths = [str(refs[slug]) for slug in slugs]
    gallery, gallery_valid, errors, _ = embed_paths(
        model,
        gallery_paths,
        device,
        image_size,
        batch_size,
        num_workers,
        "fusion_bottle_gallery",
        cache_dir,
        checkpoint,
    )
    if not bool(gallery_valid.all()):
        raise RuntimeError(f"Bottle gallery contains corrupt files: {errors[:5]}")
    query, valid, errors, _ = embed_paths(
        model,
        rows["bottle_crop_path"].astype(str).tolist(),
        device,
        image_size,
        batch_size,
        num_workers,
        "fusion_bottle_queries",
        cache_dir,
        checkpoint,
    )
    if not bool(valid.all()):
        bad = [error for error in errors if error][:5]
        raise RuntimeError(f"Bottle queries contain corrupt files: {bad}")
    ranked = rank_primary(query, gallery, rows["label_id"], slugs, 10, 512)
    ranked.insert(0, "source_relative_path", rows["source_relative_path"].to_numpy())
    ranked = ranked.rename(
        columns={column: column.replace("b_", "bottle_", 1) for column in ranked.columns if column.startswith("b_")}
    )
    release_accelerator_memory(device)
    return ranked


def normalized_rank_difficulty(rank: pd.Series) -> pd.Series:
    values = pd.to_numeric(rank, errors="coerce").fillna(2103).clip(lower=1)
    return (np.log2(values) / math.log2(10)).clip(0.0, 1.0)


def uncertainty(gap: pd.Series, scale: float = 0.03) -> pd.Series:
    values = pd.to_numeric(gap, errors="coerce").fillna(1.0).clip(lower=0.0)
    return np.exp(-values / scale)


def add_hardness(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    status_weight = result["bottle_status"].map(
        {
            "successful": 0.0,
            "low_confidence": 0.65,
            "partial": 0.85,
            "ambiguous": 1.0,
            "failed_original_fallback": 1.0,
        }
    ).fillna(1.0)
    label_conf = pd.to_numeric(result["label_detector_confidence"], errors="coerce").fillna(0.0)
    bottle_conf = pd.to_numeric(result["bottle_detector_confidence"], errors="coerce").fillna(0.0)
    detector_difficulty = (
        0.30 * (1.0 - label_conf.clip(0.0, 1.0))
        + 0.35 * (1.0 - bottle_conf.clip(0.0, 1.0))
        + 0.35 * status_weight
    )
    disagreement = result["label_top1_slug"].astype(str).ne(
        result["bottle_top1_slug"].astype(str)
    ).astype(float)
    result["hard_score"] = (
        0.25 * normalized_rank_difficulty(result["label_true_rank"])
        + 0.25 * normalized_rank_difficulty(result["bottle_true_rank"])
        + 0.15 * uncertainty(result["label_top1_top2_gap"])
        + 0.15 * uncertainty(result["bottle_top1_top2_gap"])
        + 0.10 * disagreement
        + 0.10 * detector_difficulty
    )
    result["branch_top1_disagreement"] = disagreement.astype(int)
    result["detector_difficulty"] = detector_difficulty
    return result


def choose_hard_rows(
    train: pd.DataFrame,
    target: int,
    minimum_per_identity: int,
    maximum_per_identity: int,
) -> pd.Index:
    if len(train) < target:
        raise ValueError(f"Only {len(train)} eligible train rows; requested {target}")
    ordered = train.sort_values(
        ["hard_score", "label_true_rank", "bottle_true_rank"],
        ascending=[False, False, False],
    )
    selected: list[int] = []
    selected_set: set[int] = set()
    counts: dict[str, int] = {}
    for _, group in ordered.groupby("wine_slug", sort=True):
        for index in group.head(minimum_per_identity).index:
            selected.append(int(index))
            selected_set.add(int(index))
            slug = str(train.at[index, "wine_slug"])
            counts[slug] = counts.get(slug, 0) + 1
    for index, row in ordered.iterrows():
        if len(selected) >= target:
            break
        index = int(index)
        if index in selected_set:
            continue
        slug = str(row["wine_slug"])
        if counts.get(slug, 0) >= maximum_per_identity:
            continue
        selected.append(index)
        selected_set.add(index)
        counts[slug] = counts.get(slug, 0) + 1
    if len(selected) != target:
        raise RuntimeError(
            f"Identity cap produced {len(selected)} rows instead of {target}; "
            "increase --max-per-identity"
        )
    return pd.Index(selected)


def portable_name(source_relative_path: str, path: str) -> str:
    digest = hashlib.sha1(source_relative_path.encode("utf-8")).hexdigest()[:12]
    suffix = Path(path).suffix.lower() or ".jpg"
    return f"{digest}{suffix}"


def materialize_hard_rows(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    result = frame.copy()
    label_rel: list[str] = []
    bottle_rel: list[str] = []
    for row in result.itertuples(index=False):
        filename = portable_name(str(row.source_relative_path), str(row.label_crop_path))
        label_destination = output / "label_crops" / filename
        bottle_destination = output / "bottle_crops" / filename
        link_or_copy(Path(str(row.label_crop_path)), label_destination)
        link_or_copy(Path(str(row.bottle_crop_path)), bottle_destination)
        label_rel.append(label_destination.relative_to(output).as_posix())
        bottle_rel.append(bottle_destination.relative_to(output).as_posix())
    result["portable_label_path"] = label_rel
    result["portable_bottle_path"] = bottle_rel
    return result


def write_summary(frame: pd.DataFrame, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary = {
        "complete": True,
        "rows": int(len(frame)),
        "identities": int(frame["wine_slug"].nunique()),
        "split_rows": frame["fusion_split"].value_counts().astype(int).to_dict(),
        "split_identities": frame.groupby("fusion_split")["wine_slug"].nunique().astype(int).to_dict(),
        "hard_rows": int(frame["fusion_split"].isin({"hard_train", "hard_val"}).sum()),
        "hard_train_rows": int(frame["fusion_split"].eq("hard_train").sum()),
        "hard_val_rows": int(frame["fusion_split"].eq("hard_val").sum()),
        "test_rows": int(frame["fusion_split"].str.startswith("test_").sum()),
        "bottle_status_counts": frame["bottle_status"].value_counts().astype(int).to_dict(),
        "parameters": {
            "hard_size": args.hard_size,
            "hard_val_size": args.hard_val_size,
            "minimum_per_identity": args.min_per_identity,
            "maximum_per_identity": args.max_per_identity,
            "seed": args.seed,
        },
        "leakage_checks": {
            "source_overlap_train_test": int(
                len(
                    set(frame.loc[frame.fusion_split.eq("hard_train"), "source_relative_path"])
                    & set(frame.loc[frame.fusion_split.str.startswith("test_"), "source_relative_path"])
                )
            ),
            "identity_overlap_train_test_unseen": int(
                len(
                    set(frame.loc[frame.fusion_split.eq("hard_train"), "wine_slug"])
                    & set(frame.loc[frame.fusion_split.eq("test_unseen"), "wine_slug"])
                )
            ),
            "hard_train_rows_not_from_original_train": int(
                (~frame.loc[frame.fusion_split.eq("hard_train"), "primary_split"].eq("train")).sum()
            ),
            "hard_val_rows_not_from_original_val_seen": int(
                (~frame.loc[frame.fusion_split.eq("hard_val"), "primary_split"].eq("val_seen")).sum()
            ),
        },
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cascade-config", type=Path, default=PROJECT_ROOT / "configs/dino_cascade.yaml")
    parser.add_argument("--label-audit", type=Path, default=PROJECT_ROOT / "runs/dino_cascade/predictions.csv")
    parser.add_argument(
        "--bottle-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models/trained_checkpoints/dinov3_vitb16_bottles_best_full.pt",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--hard-size", type=int, default=12000)
    parser.add_argument("--hard-val-size", type=int, default=1200)
    parser.add_argument("--min-per-identity", type=int, default=2)
    parser.add_argument("--max-per-identity", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-archive", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists; pass --force to replace it: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    cfg = CascadeConfig.load(args.cascade_config).with_overrides(
        resolver_checkpoint=str(args.bottle_checkpoint.resolve())
    )
    inventory, inventory_summary = build_inventory(cfg)
    inventory, fallback_count = add_original_photo_fallbacks(
        inventory, Path(cfg.bottle_crops_root)
    )
    paired = inventory[
        inventory["label_crop_available"] & inventory["bottle_crop_available"]
    ].copy().reset_index(drop=True)
    print(
        f"INVENTORY | paired={len(paired)} | original_fallbacks={fallback_count} "
        f"| {inventory_summary['primary_split_counts']}",
        flush=True,
    )

    label_audit = load_label_audit(args.label_audit.resolve())
    paired = paired.merge(label_audit, on="source_relative_path", how="inner", validate="one_to_one")
    if len(paired) < args.hard_size:
        raise RuntimeError(f"Only {len(paired)} paired rows have label-DINO audit data")

    device = choose_device(args.device)
    print(f"BOTTLE DINO-B | device={device} | queries={len(paired)}", flush=True)
    bottle_audit = evaluate_bottle_branch(
        paired,
        args.bottle_checkpoint.resolve(),
        Path(cfg.bottle_refs_root),
        device,
        args.batch_size,
        args.num_workers,
        output / "embedding_cache",
    )
    paired = paired.merge(bottle_audit, on="source_relative_path", how="inner", validate="one_to_one")
    paired = add_hardness(paired)

    train = paired[paired["primary_split"].eq("train")].copy()
    hard_train_size = args.hard_size - args.hard_val_size
    if hard_train_size <= 0:
        raise ValueError("--hard-size must be greater than --hard-val-size")
    hard_train_indices = choose_hard_rows(
        train,
        hard_train_size,
        args.min_per_identity,
        args.max_per_identity,
    )
    val_seen = paired[paired["primary_split"].eq("val_seen")].copy()
    hard_val_indices = choose_hard_rows(
        val_seen,
        args.hard_val_size,
        minimum_per_identity=0,
        maximum_per_identity=args.max_per_identity,
    )
    paired["fusion_split"] = np.select(
        [
            paired.index.isin(hard_train_indices),
            paired.index.isin(hard_val_indices),
            paired["primary_split"].eq("val_seen"),
            paired["primary_split"].eq("val_unseen"),
        ],
        ["hard_train", "hard_val", "test_seen_model_selection", "test_unseen"],
        default="test_seen_prior_exposure",
    )

    hard_materialized = materialize_hard_rows(
        paired[paired["fusion_split"].isin({"hard_train", "hard_val"})].copy(),
        output,
    )
    portable = paired.copy()
    portable["portable_label_path"] = ""
    portable["portable_bottle_path"] = ""
    portable.loc[hard_materialized.index, "portable_label_path"] = hard_materialized["portable_label_path"]
    portable.loc[hard_materialized.index, "portable_bottle_path"] = hard_materialized["portable_bottle_path"]
    portable["ocr_text"] = ""
    portable["ocr_available"] = 0
    portable = portable.sort_values(["fusion_split", "hard_score"], ascending=[True, False])
    portable.to_csv(output / "manifest.csv", index=False)
    summary = write_summary(portable, output, args)
    shutil.rmtree(output / "embedding_cache", ignore_errors=True)

    if not args.skip_archive:
        archive = args.archive.expanduser().resolve()
        archive.unlink(missing_ok=True)
        shutil.make_archive(str(archive.with_suffix("")), "zip", output.parent, output.name)
        summary["archive"] = str(archive)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
