#!/usr/bin/env python3
"""Evaluate Stage-2C MLP and Candidate Transformer on all Manual-211 photos.

Predictions are produced for every one of the 211 reviewed photos. Retrieval
metrics are reported only for rows with at least one accepted catalog slug;
out-of-catalog, unsure and unresolved rows have no valid closed-set target and
remain available in the per-query audit without fabricated accuracy labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cascade_resolver.modeling import choose_device, load_retrieval_model
from fusion_stage2.core import FEATURE_NAMES, FusionMLP, ranking_metrics, resolve_refs
from fusion_stage2.train_manual_211_adaptation import (
    ManualDataset,
    candidate_features,
    load_manual_frame,
    refresh_galleries,
    split_manual,
)
from fusion_stage2.train_transformer_ranker import CandidateTransformerRanker


MODEL_NAMES = ("transformer", "fusion_mlp", "label_dino_b", "bottle_dino_b")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def positive_rank(scores: torch.Tensor, positive_ids: list[int]) -> int | None:
    if not positive_ids:
        return None
    return min(1 + int((scores > scores[index]).sum()) for index in positive_ids)


def candidate_rank(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_ids: list[int],
    missing_rank: int,
) -> int | None:
    if not positive_ids:
        return None
    valid_positions = candidate_mask.nonzero(as_tuple=False).flatten()
    order = valid_positions[logits[valid_positions].argsort(descending=True)]
    ordered_ids = candidate_ids[order].tolist()
    ranks = [ordered_ids.index(index) + 1 for index in positive_ids if index in ordered_ids]
    return min(ranks) if ranks else missing_rank


def top_items(scores: torch.Tensor, slugs: list[str], limit: int) -> list[dict[str, Any]]:
    count = min(limit, len(scores))
    values, indices = scores.topk(count)
    return [
        {"slug": slugs[int(index)], "score": float(value)}
        for value, index in zip(values, indices, strict=True)
    ]


def candidate_top_items(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    slugs: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    valid_positions = candidate_mask.nonzero(as_tuple=False).flatten()
    order = valid_positions[logits[valid_positions].argsort(descending=True)][:limit]
    return [
        {"slug": slugs[int(candidate_ids[position])], "score": float(logits[position])}
        for position in order
    ]


def metric_block(frame: pd.DataFrame, indices: Iterable[int], gallery_size: int) -> dict[str, Any]:
    selected = list(indices)
    if not selected:
        return {"num_queries": 0}
    report: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        ranks = torch.tensor([int(frame.at[index, f"{model_name}_rank"]) for index in selected], dtype=torch.long)
        metrics = ranking_metrics(ranks)
        if model_name in {"transformer", "fusion_mlp"}:
            metrics["candidate_recall"] = float(ranks.le(gallery_size).float().mean())
        report[model_name] = metrics
    mlp_ranks = torch.tensor([int(frame.at[index, "fusion_mlp_rank"]) for index in selected])
    transformer_ranks = torch.tensor([int(frame.at[index, "transformer_rank"]) for index in selected])
    fixed = int(((mlp_ranks > 1) & transformer_ranks.eq(1)).sum())
    harmed = int((mlp_ranks.eq(1) & (transformer_ranks > 1)).sum())
    report["comparison"] = {
        "fixed_top1": fixed,
        "harmed_top1": harmed,
        "net_top1": fixed - harmed,
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    print("=" * 116, flush=True)
    print("MANUAL-211 | TRANSFORMER VS STAGE-2C FUSION MLP", flush=True)
    coverage = report["coverage"]
    print(
        f"COVERAGE | all={coverage['rows_total']} | scored_catalog={coverage['scored_catalog_rows']} "
        f"| notcat={coverage['identity_status_counts'].get('notcat', 0)} "
        f"| unsure={coverage['identity_status_counts'].get('unsure', 0)} "
        f"| unresolved={coverage['identity_status_counts'].get('unresolved', 0)}",
        flush=True,
    )
    for split_name, split_report in report["results"].items():
        for model_name in MODEL_NAMES:
            metrics = split_report[model_name]
            print(
                f"{split_name:<30} | {model_name:<15} | Acc/R@1={100*metrics['recall_at_1']:6.2f}% "
                f"| R@2={100*metrics['recall_at_2']:6.2f}% | R@5={100*metrics['recall_at_5']:6.2f}% "
                f"| R@10={100*metrics['recall_at_10']:6.2f}% | MRR={metrics['mrr']:.4f} "
                f"| mean_rank={metrics['mean_rank']:.3f} | N={metrics['num_queries']}",
                flush=True,
            )
        comparison = split_report["comparison"]
        print(
            f"{'':30} | Transformer vs MLP | fixed={comparison['fixed_top1']} "
            f"harmed={comparison['harmed_top1']} net={comparison['net_top1']:+d}",
            flush=True,
        )
    print("=" * 116, flush=True)


def save_metric_plot(report: dict[str, Any], output: Path) -> None:
    split_names = [name for name in ("manual_val", "manual_all_catalog") if name in report["results"]]
    figure, axes = plt.subplots(1, len(split_names), figsize=(7 * len(split_names), 5), squeeze=False)
    colors = {"transformer": "#5B5BD6", "fusion_mlp": "#18A999", "label_dino_b": "#E08E45", "bottle_dino_b": "#C8553D"}
    for axis, split_name in zip(axes[0], split_names, strict=True):
        names = list(MODEL_NAMES)
        values = [100 * report["results"][split_name][name]["recall_at_1"] for name in names]
        bars = axis.bar(names, values, color=[colors[name] for name in names])
        axis.bar_label(bars, fmt="%.2f%%", padding=3)
        axis.set_ylim(0, 100)
        axis.set_ylabel("Recall@1 / accuracy, %")
        axis.set_title(f"{split_name} (N={report['results'][split_name]['transformer']['num_queries']})")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Manual-211 closed-set catalog evaluation")
    figure.tight_layout()
    figure.savefig(output / "manual_211_model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    slugs, label_ref_paths = resolve_refs(Path(args.label_refs_root))
    bottle_slugs, bottle_ref_paths = resolve_refs(Path(args.bottle_refs_root))
    if slugs != bottle_slugs:
        raise ValueError("Label and bottle reference galleries are not aligned")
    slug_to_id = {slug: index for index, slug in enumerate(slugs)}

    manual = load_manual_frame(Path(args.manual_root), slugs)
    manual_train, manual_val = split_manual(manual, args.seed)
    manual_train_ids = set(manual_train["sample_id"].astype(str))
    manual_val_ids = set(manual_val["sample_id"].astype(str))

    label_model, label_info = load_retrieval_model(args.label_base_checkpoint, "vitb16", device)
    bottle_model, bottle_info = load_retrieval_model(args.bottle_base_checkpoint, "vitb16", device)
    adapted = torch.load(args.adapted_checkpoint, map_location="cpu", weights_only=False)
    label_model.load_state_dict(adapted["label_model_state_dict"], strict=True)
    bottle_model.load_state_dict(adapted["bottle_model_state_dict"], strict=True)
    for model in (label_model, bottle_model):
        model.set_backbone_trainable(0)
        model.eval()

    fusion_state = adapted["fusion_state_dict"]
    hidden_dim = int(fusion_state["network.1.weight"].shape[0])
    fusion = FusionMLP(hidden_dim=hidden_dim, dropout=args.dropout).to(device)
    fusion.load_state_dict(fusion_state, strict=True)
    fusion.eval()

    transformer_payload = torch.load(args.transformer_checkpoint, map_location="cpu", weights_only=False)
    config = transformer_payload["config"]
    transformer = CandidateTransformerRanker(
        torch.zeros(len(FEATURE_NAMES)),
        torch.ones(len(FEATURE_NAMES)),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        feedforward_dim=int(config["feedforward_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    transformer.load_state_dict(transformer_payload["model_state_dict"], strict=True)
    transformer.eval()

    image_size = int(label_info["image_size"])
    if int(bottle_info["image_size"]) != image_size:
        raise ValueError("Label and bottle image sizes differ")
    label_gallery, bottle_gallery = refresh_galleries(
        label_model,
        bottle_model,
        label_ref_paths,
        bottle_ref_paths,
        image_size,
        device,
        args.batch_size,
        args.num_workers,
    )

    dataset = ManualDataset(manual, image_size, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows: list[dict[str, Any]] = []
    missing_rank = len(slugs) + 1
    print(f"EVAL | embedding and ranking all {len(manual)} Manual-211 rows", flush=True)
    for batch_number, batch in enumerate(loader, start=1):
        indices = batch["index"].tolist()
        label_embeddings, _ = label_model(batch["label"].to(device))
        bottle_embeddings, _ = bottle_model(batch["bottle"].to(device))
        features, mask, candidate_ids, label_scores, bottle_scores = candidate_features(
            label_embeddings,
            bottle_embeddings,
            label_gallery,
            bottle_gallery,
            manual,
            indices,
            args.top_k,
            None,
        )
        fusion_logits = fusion(features, mask)
        transformer_logits = transformer(features, mask)
        for local, frame_index in enumerate(indices):
            source = manual.iloc[frame_index]
            positive_slugs = list(source["positive_slugs"])
            positive_ids = [slug_to_id[slug] for slug in positive_slugs]
            model_tops = {
                "transformer": candidate_top_items(transformer_logits[local], candidate_ids[local], mask[local], slugs, args.report_top_k),
                "fusion_mlp": candidate_top_items(fusion_logits[local], candidate_ids[local], mask[local], slugs, args.report_top_k),
                "label_dino_b": top_items(label_scores[local], slugs, args.report_top_k),
                "bottle_dino_b": top_items(bottle_scores[local], slugs, args.report_top_k),
            }
            model_ranks = {
                "transformer": candidate_rank(transformer_logits[local], candidate_ids[local], mask[local], positive_ids, missing_rank),
                "fusion_mlp": candidate_rank(fusion_logits[local], candidate_ids[local], mask[local], positive_ids, missing_rank),
                "label_dino_b": positive_rank(label_scores[local], positive_ids),
                "bottle_dino_b": positive_rank(bottle_scores[local], positive_ids),
            }
            row = {
                "sample_id": str(source["sample_id"]),
                "source_group": str(source["source_group"]),
                "source_filename": str(source["source_filename"]),
                "identity_status": str(source["identity_status"]),
                "accepted_slugs": ";".join(positive_slugs),
                "scored_catalog": bool(source["trainable_catalog"] and positive_slugs),
                "manual_split": "manual_val" if str(source["sample_id"]) in manual_val_ids else "manual_train" if str(source["sample_id"]) in manual_train_ids else "unscored",
                "label_path": str(source["label_path"]),
                "bottle_path": str(source["bottle_path"]),
            }
            for model_name in MODEL_NAMES:
                row[f"{model_name}_prediction"] = model_tops[model_name][0]["slug"]
                row[f"{model_name}_rank"] = model_ranks[model_name]
                row[f"{model_name}_top10"] = json.dumps(model_tops[model_name], ensure_ascii=False)
            rows.append(row)
        print(f"EVAL | batch {batch_number}/{len(loader)} | rows={min(batch_number * args.batch_size, len(manual))}/{len(manual)}", flush=True)

    audit = pd.DataFrame(rows)
    scored_indices = audit.index[audit["scored_catalog"]].tolist()
    train_indices = audit.index[audit["manual_split"].eq("manual_train")].tolist()
    val_indices = audit.index[audit["manual_split"].eq("manual_val")].tolist()
    results: dict[str, Any] = {
        "manual_all_catalog": metric_block(audit, scored_indices, len(slugs)),
        "manual_train_prior_exposure": metric_block(audit, train_indices, len(slugs)),
        "manual_val": metric_block(audit, val_indices, len(slugs)),
    }
    for source_group, group in audit[audit["scored_catalog"]].groupby("source_group", sort=True):
        results[f"source_group::{source_group}"] = metric_block(audit, group.index.tolist(), len(slugs))

    coverage = {
        "rows_total": len(audit),
        "predictions_generated": {name: int(audit[f"{name}_prediction"].notna().sum()) for name in MODEL_NAMES},
        "scored_catalog_rows": len(scored_indices),
        "manual_train_rows": len(train_indices),
        "manual_val_rows": len(val_indices),
        "identity_status_counts": {str(key): int(value) for key, value in audit["identity_status"].value_counts().items()},
        "unscored_reason": "Closed-set retrieval has no valid target for notcat, unsure or unresolved rows.",
    }
    report = {
        "stage2c_checkpoint": str(args.adapted_checkpoint),
        "transformer_checkpoint": str(args.transformer_checkpoint),
        "transformer_best_epoch": int(transformer_payload.get("epoch", -1)),
        "coverage": coverage,
        "results": results,
    }
    audit.to_csv(output / "manual_211_predictions.csv", index=False)
    save_json(output / "manual_211_metrics.json", report)
    save_metric_plot(report, output)
    print_report(report)
    print("FINAL MANUAL-211 TRANSFORMER EVAL |", json.dumps(report, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-root", required=True)
    parser.add_argument("--label-refs-root", required=True)
    parser.add_argument("--bottle-refs-root", required=True)
    parser.add_argument("--label-base-checkpoint", required=True)
    parser.add_argument("--bottle-base-checkpoint", required=True)
    parser.add_argument("--adapted-checkpoint", required=True)
    parser.add_argument("--transformer-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report-top-k", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
