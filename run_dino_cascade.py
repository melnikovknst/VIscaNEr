#!/usr/bin/env python3
"""Evaluate DINO-B label retrieval and conditionally resolve Top-2 with DINO-S."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cascade_resolver.config import CascadeConfig
from cascade_resolver.data import build_inventory, gallery_paths, limit_queries, reference_lookup
from cascade_resolver.evaluation import (
    cascade_diagnostics,
    initialize_final_columns,
    make_metrics_table,
    rank_primary,
    resolve_top_two,
    save_audit_visualizations,
)
from cascade_resolver.modeling import (
    checkpoint_fingerprint,
    choose_device,
    embed_paths,
    environment_summary,
    load_retrieval_model,
    release_accelerator_memory,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dino_cascade.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--ambiguity-margin", type=float)
    parser.add_argument("--limit", type=int, help="Deterministic query limit for a smoke run")
    parser.add_argument("--force", action="store_true", help="Ignore embedding caches")
    parser.add_argument("--audit-only", action="store_true", help="Validate data and checkpoints without inference")
    return parser.parse_args()


def _print_heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def _print_metrics(metrics: pd.DataFrame) -> None:
    columns = [
        "scope",
        "stage",
        "num_queries",
        "accuracy",
        "recall_at_1",
        "recall_at_2",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "median_rank",
        "mean_rank",
    ]
    printable = metrics[columns].copy()
    for column in ("accuracy", "recall_at_1", "recall_at_2", "recall_at_5", "recall_at_10", "mrr"):
        printable[column] = printable[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "n/a")
    print(printable.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    cfg = CascadeConfig.load(args.config).with_overrides(
        device=args.device,
        ambiguity_margin=args.ambiguity_margin,
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    started = time.perf_counter()

    _print_heading("DINO CASCADE | DATA AND CHECKPOINT AUDIT")
    inventory, inventory_summary = build_inventory(cfg)
    checkpoint_summary = {
        "dino_b_labels": checkpoint_fingerprint(cfg.primary_checkpoint),
        "dino_s_bottles": checkpoint_fingerprint(cfg.resolver_checkpoint),
    }
    inventory.to_csv(output_dir / "source_inventory.csv", index=False)
    save_json(output_dir / "audit_summary.json", {"inventory": inventory_summary, "checkpoints": checkpoint_summary})
    print(json.dumps(inventory_summary, ensure_ascii=False, indent=2), flush=True)
    print("CHECKPOINT | DINO-B", checkpoint_summary["dino_b_labels"]["sha256"], flush=True)
    print("CHECKPOINT | DINO-S", checkpoint_summary["dino_s_bottles"]["sha256"], flush=True)
    if args.audit_only:
        print(f"AUDIT ONLY  | outputs={output_dir}", flush=True)
        return

    device = choose_device(cfg.device)
    print("ENVIRONMENT | " + json.dumps(environment_summary(device), ensure_ascii=False), flush=True)
    queries = limit_queries(inventory, args.limit, cfg.seed)
    queries = queries.loc[queries["label_crop_available"]].reset_index(drop=True)
    if queries.empty:
        raise RuntimeError("No valid label crops are available for evaluation")
    print(
        f"EVALUATION  | queries={len(queries)} / sources={len(inventory)} "
        f"| margin<={cfg.ambiguity_margin:.4f}",
        flush=True,
    )

    label_slugs, label_gallery_paths = gallery_paths(cfg, "label")
    if label_slugs != sorted(inventory["wine_slug"].unique().tolist()):
        raise RuntimeError("DINO-B gallery order does not match label IDs")

    _print_heading("STAGE 1 | DINOv3-B/16 ON LABEL CROPS")
    primary, primary_info = load_retrieval_model(cfg.primary_checkpoint, "vitb16", device)
    if primary_info["num_classes"] != len(label_slugs):
        raise RuntimeError(
            f"DINO-B checkpoint has {primary_info['num_classes']} classes, dataset has {len(label_slugs)}"
        )
    print("MODEL       | " + json.dumps(primary_info, ensure_ascii=False), flush=True)
    label_gallery, gallery_valid, gallery_errors, gallery_timing = embed_paths(
        primary,
        label_gallery_paths,
        device,
        primary_info["image_size"],
        cfg.primary_batch_size,
        cfg.num_workers,
        "b_gallery",
        cache_dir,
        cfg.primary_checkpoint,
        args.force,
    )
    if not gallery_valid.all():
        broken = [path for path, valid, error in zip(label_gallery_paths, gallery_valid, gallery_errors, strict=True) if not valid]
        raise RuntimeError(f"DINO-B gallery contains unreadable files: {broken[:5]}")
    label_queries, query_valid, query_errors, query_timing = embed_paths(
        primary,
        queries["label_crop_path"].tolist(),
        device,
        primary_info["image_size"],
        cfg.primary_batch_size,
        cfg.num_workers,
        "b_queries",
        cache_dir,
        cfg.primary_checkpoint,
        args.force,
    )
    if not query_valid.all():
        failed = queries.loc[~query_valid.numpy(), ["source_relative_path", "label_crop_path"]].copy()
        failed["error"] = [error for error, valid in zip(query_errors, query_valid.tolist(), strict=True) if not valid]
        failed.to_csv(output_dir / "unreadable_label_crops.csv", index=False)
        print(f"WARNING     | dropping {len(failed)} unreadable label crops", flush=True)
        keep = query_valid.numpy()
        queries = queries.loc[keep].reset_index(drop=True)
        label_queries = label_queries[query_valid]

    primary_ranks = rank_primary(
        label_queries,
        label_gallery,
        queries["label_id"].tolist(),
        label_slugs,
        cfg.top_k,
        cfg.ranking_batch_size,
    )
    predictions = pd.concat([queries.reset_index(drop=True), primary_ranks], axis=1)
    predictions = initialize_final_columns(predictions)
    predictions["resolver_ambiguous"] = predictions["b_top1_top2_gap"].le(cfg.ambiguity_margin)
    predictions["resolver_missing_bottle_crop"] = (
        predictions["resolver_ambiguous"] & ~predictions["bottle_crop_available"]
    )
    predictions["resolver_invoked"] = (
        predictions["resolver_ambiguous"] & predictions["bottle_crop_available"]
    )
    invoked_indices = predictions.index[predictions["resolver_invoked"]].tolist()
    print(
        f"DINO-B      | Top-1={predictions['b_true_rank'].eq(1).mean():.4f} "
        f"| Top-2={predictions['b_true_rank'].le(2).mean():.4f} "
        f"| ambiguous={int(predictions['resolver_ambiguous'].sum())} "
        f"| resolver-ready={len(invoked_indices)}",
        flush=True,
    )
    del primary, label_queries
    gc.collect()
    release_accelerator_memory(device)

    resolver_info: dict[str, Any] | None = None
    resolver_timings: dict[str, Any] = {}
    if invoked_indices:
        _print_heading("STAGE 2 | DINOv3-S/16 RESOLVES ONLY DINO-B TOP-1/TOP-2")
        bottle_slugs, bottle_gallery_paths = gallery_paths(cfg, "bottle")
        if bottle_slugs != label_slugs:
            raise RuntimeError("DINO-S and DINO-B gallery identity order differs")
        resolver, resolver_info = load_retrieval_model(cfg.resolver_checkpoint, "vits16", device)
        if resolver_info["num_classes"] != len(bottle_slugs):
            raise RuntimeError(
                f"DINO-S checkpoint has {resolver_info['num_classes']} classes, dataset has {len(bottle_slugs)}"
            )
        print("MODEL       | " + json.dumps(resolver_info, ensure_ascii=False), flush=True)
        bottle_gallery, s_gallery_valid, s_gallery_errors, s_gallery_timing = embed_paths(
            resolver,
            bottle_gallery_paths,
            device,
            resolver_info["image_size"],
            cfg.resolver_batch_size,
            cfg.num_workers,
            "s_gallery",
            cache_dir,
            cfg.resolver_checkpoint,
            args.force,
        )
        if not s_gallery_valid.all():
            broken = [path for path, valid in zip(bottle_gallery_paths, s_gallery_valid.tolist(), strict=True) if not valid]
            raise RuntimeError(f"DINO-S gallery contains unreadable files: {broken[:5]}")
        invoked = predictions.loc[invoked_indices].copy()
        resolver_queries, resolver_valid, resolver_errors, s_query_timing = embed_paths(
            resolver,
            invoked["bottle_crop_path"].tolist(),
            device,
            resolver_info["image_size"],
            cfg.resolver_batch_size,
            cfg.num_workers,
            "s_queries",
            cache_dir,
            cfg.resolver_checkpoint,
            args.force,
        )
        resolved = resolve_top_two(invoked, resolver_queries, bottle_gallery, resolver_valid)
        update_columns = [
            "resolver_embedding_valid",
            "resolver_swapped",
            "s_candidate1_similarity",
            "s_candidate2_similarity",
            "s_candidate_gap",
            "final_top1_label_id",
            "final_top2_label_id",
            "final_top1_slug",
            "final_top2_slug",
            "final_true_rank",
        ]
        predictions.loc[invoked_indices, update_columns] = resolved[update_columns].to_numpy()
        if not resolver_valid.all():
            failed = invoked.loc[~resolver_valid.numpy(), ["source_relative_path", "bottle_crop_path"]].copy()
            failed["error"] = [error for error, valid in zip(resolver_errors, resolver_valid.tolist(), strict=True) if not valid]
            failed.to_csv(output_dir / "unreadable_bottle_crops.csv", index=False)
        resolver_timings = {"gallery": s_gallery_timing, "queries": s_query_timing}
        del resolver, resolver_queries, bottle_gallery
        gc.collect()
        release_accelerator_memory(device)
    else:
        print("STAGE 2     | skipped: no query satisfied the ambiguity rule", flush=True)

    _print_heading("RESULTS | DINO-B VS CONDITIONAL DINO-S CASCADE")
    for column in ("b_true_rank", "final_true_rank", "final_top1_label_id", "final_top2_label_id"):
        predictions[column] = predictions[column].astype(int)
    metrics = make_metrics_table(predictions)
    diagnostics = cascade_diagnostics(predictions)
    _print_metrics(metrics)
    print("\nCASCADE     | " + json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "stage_metrics.csv", index=False)
    label_ref_lookup = reference_lookup(cfg.label_refs_root)
    bottle_ref_lookup = reference_lookup(cfg.bottle_refs_root)
    visualization_paths = save_audit_visualizations(
        predictions,
        output_dir / "visualizations",
        label_ref_lookup,
        bottle_ref_lookup,
        cfg.visualization_examples,
    )
    summary: dict[str, Any] = {
        "config": cfg.__dict__,
        "limited_run": args.limit is not None,
        "inventory": inventory_summary,
        "checkpoints": checkpoint_summary,
        "models": {"primary": primary_info, "resolver": resolver_info},
        "timings": {
            "primary_gallery": gallery_timing,
            "primary_queries": query_timing,
            "resolver": resolver_timings,
            "total_seconds": time.perf_counter() - started,
        },
        "diagnostics": diagnostics,
        "metrics": metrics.to_dict("records"),
        "visualizations": visualization_paths,
    }
    save_json(output_dir / "summary.json", summary)
    print(f"\nOUTPUTS     | {output_dir}", flush=True)
    print(f"PREDICTIONS | {output_dir / 'predictions.csv'}", flush=True)
    print(f"METRICS     | {output_dir / 'stage_metrics.csv'}", flush=True)
    print(f"FIGURES     | {output_dir / 'visualizations'}", flush=True)
    print(
        "NOTE        | all_eligible/train are descriptive because they contain training images; "
        "use val_unseen for the least optimistic available estimate.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        raise
