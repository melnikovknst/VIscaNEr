#!/usr/bin/env python3
"""Evaluate the latest whole-bottle checkpoint on all existing and fresh images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from cascade_resolver.modeling import (
    choose_device,
    embed_paths,
    load_retrieval_model,
    release_accelerator_memory,
)

from .paths import DEFAULT_OUTPUT_DIR, PROJECT_ROOT, reference_lookup, resolve_recorded_image


def _localize_index(index: pd.DataFrame, crops_root: Path, refs_root: Path) -> pd.DataFrame:
    refs = reference_lookup(refs_root)
    localized = index.copy()
    if "filename" not in localized.columns:
        raise ValueError("The retrieval index must contain a filename column")

    def resolve(row: pd.Series) -> str:
        if str(row["split"]) == "gallery":
            path = refs.get(str(row["wine_slug"]))
            return str(path) if path else ""
        recorded = Path(str(row["filename"]))
        status = str(row.get("status", ""))
        status = status if status in {
            "successful",
            "low_confidence",
            "partial",
            "ambiguous",
        } else None
        path = resolve_recorded_image(recorded, crops_root, status)
        return str(path) if path else ""

    localized["image_path"] = localized.apply(resolve, axis=1)
    missing = localized["image_path"].eq("")
    if missing.any():
        examples = localized.loc[missing, ["split", "wine_slug"]].head().to_dict("records")
        raise FileNotFoundError(f"Could not localize {int(missing.sum())} index images: {examples}")
    return localized


def _rank_all(
    embeddings: torch.Tensor,
    valid: torch.Tensor,
    rows: pd.DataFrame,
    gallery_embeddings: torch.Tensor,
    gallery_slugs: list[str],
    gallery_paths: list[str],
    top_k: int,
    chunk_size: int,
) -> pd.DataFrame:
    gallery_embeddings = F.normalize(gallery_embeddings.float(), dim=1)
    slug_to_gallery = {slug: index for index, slug in enumerate(gallery_slugs)}
    output: list[dict[str, object]] = []
    effective_k = min(top_k, len(gallery_slugs))
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        query = F.normalize(embeddings[start:stop].float(), dim=1)
        similarities = query @ gallery_embeddings.T
        values, indices = similarities.topk(effective_k, dim=1)
        for offset, (_, row) in enumerate(rows.iloc[start:stop].iterrows()):
            true_slug = str(row["wine_slug"])
            target = slug_to_gallery[true_slug]
            ranking = torch.argsort(similarities[offset], descending=True)
            rank = int((ranking == target).nonzero(as_tuple=False)[0, 0]) + 1
            record: dict[str, object] = {
                "split": str(row["split"]),
                "query_path": str(row["image_path"]),
                "true_slug": true_slug,
                "true_reference_path": gallery_paths[target],
                "rank": rank,
                "true_similarity": float(similarities[offset, target]),
                "top1_top2_margin": float(values[offset, 0] - values[offset, 1]),
                "embedding_valid": bool(valid[start + offset]),
            }
            for position in range(effective_k):
                gallery_index = int(indices[offset, position])
                prefix = f"top{position + 1}"
                record[f"{prefix}_slug"] = gallery_slugs[gallery_index]
                record[f"{prefix}_path"] = gallery_paths[gallery_index]
                record[f"{prefix}_similarity"] = float(values[offset, position])
            output.append(record)
    return pd.DataFrame(output)


def _metric_rows(audit: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for split, frame in [("all", audit), *audit.groupby("split", sort=True)]:
        ranks = frame["rank"]
        result[str(split)] = {
            "num_queries": int(len(frame)),
            "accuracy": float(ranks.eq(1).mean()),
            "recall_at_1": float(ranks.le(1).mean()),
            "recall_at_2": float(ranks.le(2).mean()),
            "recall_at_5": float(ranks.le(5).mean()),
            "recall_at_10": float(ranks.le(10).mean()),
            "mean_rank": float(ranks.mean()),
            "errors": int(ranks.gt(1).sum()),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "models/trained_checkpoints/dinov3_vitb16_bottles_best_full.pt",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "source_kaggle/index.csv",
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
    parser.add_argument("--fresh-manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "fresh_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-fresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"DEVICE | {device}", flush=True)
    index = _localize_index(pd.read_csv(args.index_csv), args.crops_root, args.refs_root)
    gallery = index[index["split"].eq("gallery")].reset_index(drop=True)
    queries = index[~index["split"].eq("gallery")].reset_index(drop=True)

    if args.include_fresh:
        fresh = pd.read_csv(args.fresh_manifest)
        fresh = fresh[fresh["status"].isin({"candidate", "near_reference_duplicate"})].copy()
        fresh = fresh.rename(columns={"image_path": "image_path", "wine_slug": "wine_slug"})
        fresh["split"] = "fresh_intake"
        fresh["label_id"] = fresh["wine_slug"].map(
            gallery.set_index("wine_slug")["label_id"].to_dict()
        )
        fresh = fresh.dropna(subset=["label_id"])
        queries = pd.concat(
            [queries, fresh[["image_path", "wine_slug", "split", "label_id"]]],
            ignore_index=True,
        )

    model, model_info = load_retrieval_model(args.checkpoint, "vitb16", device)
    image_size = int(model_info["image_size"])
    cache_dir = args.output_dir / "embedding_cache"
    gallery_embeddings, gallery_valid, gallery_errors, _ = embed_paths(
        model,
        gallery["image_path"].tolist(),
        device,
        image_size,
        args.batch_size,
        args.num_workers,
        "curation_gallery",
        cache_dir,
        args.checkpoint,
    )
    if not bool(gallery_valid.all()):
        raise RuntimeError(f"Gallery contains corrupt images: {gallery_errors[:5]}")
    query_embeddings, query_valid, query_errors, _ = embed_paths(
        model,
        queries["image_path"].tolist(),
        device,
        image_size,
        args.batch_size,
        args.num_workers,
        "curation_queries",
        cache_dir,
        args.checkpoint,
    )
    audit = _rank_all(
        query_embeddings,
        query_valid,
        queries,
        gallery_embeddings,
        gallery["wine_slug"].astype(str).tolist(),
        gallery["image_path"].astype(str).tolist(),
        args.top_k,
        chunk_size=512,
    )
    audit["embedding_error"] = query_errors
    audit_path = args.output_dir / "all_retrieval_audit.csv"
    audit.to_csv(audit_path, index=False)
    metrics = _metric_rows(audit)
    payload = {"checkpoint": model_info, "metrics": metrics, "audit_csv": str(audit_path.resolve())}
    (args.output_dir / "all_evaluation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    release_accelerator_memory(device)


if __name__ == "__main__":
    main()
