#!/usr/bin/env python3
"""Train a candidate Transformer and compare it with the frozen Fusion MLP.

Both DINOv3-B branches are restored from the Stage-2C checkpoint and remain
frozen.  Their Top-K union and the exact 19 candidate features used by the MLP
are recomputed once.  A small permutation-equivariant Transformer then learns
to rank candidates jointly.  ``hard_val`` selects the checkpoint; test splits
are evaluated only after the best checkpoint is restored.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from cascade_resolver.modeling import choose_device, load_retrieval_model
from fusion_stage2.core import FEATURE_NAMES, FeatureCache, FusionMLP, branch_metrics, load_manifest, ranking_metrics
from fusion_stage2.train_fusion import TEST_SPLITS, cache_subset, embed_cache, validate_gallery_alignment


class CandidateTransformerRanker(nn.Module):
    """Contextual candidate scorer without positional-index leakage."""

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        d_model: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_std", feature_std.float().clamp_min(1e-5))
        self.input_projection = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.context_score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.direct_score = nn.Linear(len(FEATURE_NAMES), 1, bias=False)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = (features.float() - self.feature_mean) / self.feature_std
        encoded = self.encoder(self.input_projection(normalized), src_key_padding_mask=~mask)
        logits = self.context_score(encoded).squeeze(-1) + self.direct_score(normalized).squeeze(-1)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def feature_statistics(cache: FeatureCache) -> tuple[torch.Tensor, torch.Tensor]:
    values = cache.features[cache.candidate_mask]
    return values.mean(dim=0), values.std(dim=0).clamp_min(1e-5)


@torch.inference_mode()
def rank_cache(
    model: nn.Module,
    cache: FeatureCache,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    model.eval()
    all_ranks, all_predictions = [], []
    for start in range(0, len(cache.features), batch_size):
        stop = min(start + batch_size, len(cache.features))
        mask = cache.candidate_mask[start:stop].to(device)
        logits = model(cache.features[start:stop].to(device), mask)
        order = logits.argsort(dim=1, descending=True).cpu()
        candidate_ids = cache.candidate_ids[start:stop]
        all_predictions.append(candidate_ids.gather(1, order[:, :1]).squeeze(1))
        truth = cache.true_positions[start:stop]
        present = truth.ge(0)
        ranks = torch.full((stop - start,), len(cache.slugs) + 1, dtype=torch.long)
        if present.any():
            ranks[present] = 1 + order[present].eq(truth[present, None]).nonzero(as_tuple=False)[:, 1]
        all_ranks.append(ranks)
    ranks = torch.cat(all_ranks)
    metrics = ranking_metrics(ranks)
    metrics["candidate_recall"] = float(cache.true_positions.ge(0).float().mean())
    return metrics, ranks, torch.cat(all_predictions)


def report_for_splits(
    transformer: nn.Module,
    mlp: FusionMLP,
    cache: FeatureCache,
    frame: pd.DataFrame,
    device: torch.device,
    splits: tuple[str, ...],
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    report, details = {}, {}
    for split in splits:
        subset = cache_subset(cache, frame, split)
        transformer_metrics, transformer_ranks, transformer_predictions = rank_cache(transformer, subset, device, batch_size)
        mlp_metrics, mlp_ranks, mlp_predictions = rank_cache(mlp, subset, device, batch_size)
        fixed = int(((mlp_ranks > 1) & (transformer_ranks == 1)).sum())
        harmed = int(((mlp_ranks == 1) & (transformer_ranks > 1)).sum())
        report[split] = {
            "transformer": transformer_metrics,
            "fusion_mlp": mlp_metrics,
            **branch_metrics(subset),
            "comparison": {"fixed_top1": fixed, "harmed_top1": harmed, "net_top1": fixed - harmed},
        }
        details[split] = {
            "transformer_ranks": transformer_ranks,
            "transformer_predictions": transformer_predictions,
            "mlp_ranks": mlp_ranks,
            "mlp_predictions": mlp_predictions,
            "row_indices": subset.row_indices,
        }
    return report, details


def print_report(title: str, report: dict[str, Any]) -> None:
    print("=" * 116, flush=True); print(title, flush=True)
    for split, values in report.items():
        for name in ("transformer", "fusion_mlp", "label_dino_b", "bottle_dino_b"):
            metrics = values[name]
            print(
                f"{split:<30} | {name:<15} | Acc/R@1={100*float(metrics['recall_at_1']):6.2f}% "
                f"| R@2={100*float(metrics['recall_at_2']):6.2f}% | R@5={100*float(metrics['recall_at_5']):6.2f}% "
                f"| R@10={100*float(metrics['recall_at_10']):6.2f}% | MRR={float(metrics['mrr']):.4f} "
                f"| N={int(metrics['num_queries'])}", flush=True,
            )
        comparison = values["comparison"]
        print(f"{'':30} | fixed={comparison['fixed_top1']} harmed={comparison['harmed_top1']} net={comparison['net_top1']:+d}", flush=True)
    print("=" * 116, flush=True)


def save_audit_and_visualization(
    frame: pd.DataFrame,
    cache: FeatureCache,
    details: dict[str, dict[str, torch.Tensor]],
    output: Path,
) -> None:
    rows = []
    for split, values in details.items():
        for local, frame_index in enumerate(values["row_indices"].tolist()):
            row = frame.iloc[frame_index]
            rows.append({
                "source_relative_path": row["source_relative_path"],
                "fusion_split": split,
                "true_slug": row["wine_slug"],
                "transformer_prediction": cache.slugs[int(values["transformer_predictions"][local])],
                "fusion_mlp_prediction": cache.slugs[int(values["mlp_predictions"][local])],
                "transformer_rank": int(values["transformer_ranks"][local]),
                "fusion_mlp_rank": int(values["mlp_ranks"][local]),
                "label_path": row["label_path"],
                "bottle_path": row["bottle_path"],
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "transformer_audit.csv", index=False)
    unseen = audit[audit["fusion_split"].eq("test_unseen")].copy()
    unseen["harmed"] = unseen["fusion_mlp_rank"].eq(1) & unseen["transformer_rank"].gt(1)
    unseen["fixed"] = unseen["fusion_mlp_rank"].gt(1) & unseen["transformer_rank"].eq(1)
    selected = pd.concat([
        unseen[unseen["harmed"]].sort_values("transformer_rank", ascending=False).head(6),
        unseen[unseen["fixed"]].sort_values("fusion_mlp_rank", ascending=False).head(6),
    ]).drop_duplicates("source_relative_path").head(12)
    if selected.empty:
        selected = unseen.sort_values("transformer_rank", ascending=False).head(12)
    figure, axes = plt.subplots(max(len(selected), 1), 2, figsize=(12, max(4, 4 * len(selected))))
    axes = np.asarray(axes).reshape(-1, 2)
    for axis_pair, (_, row) in zip(axes, selected.iterrows(), strict=False):
        for axis, path, label in zip(axis_pair, (row["label_path"], row["bottle_path"]), ("label", "bottle"), strict=True):
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(label); axis.axis("off")
        status = "FIXED" if row["fixed"] else "HARMED" if row["harmed"] else "ERROR"
        axis_pair[0].set_ylabel(
            f"{status}\ntrue={row['true_slug']}\ntransformer={row['transformer_prediction']} (r{row['transformer_rank']})\n"
            f"mlp={row['fusion_mlp_prediction']} (r{row['fusion_mlp_rank']})",
            fontsize=8,
        )
    figure.suptitle("Transformer vs Fusion MLP — test_unseen", fontsize=14)
    figure.tight_layout(); figure.savefig(output / "transformer_comparison.png", dpi=150, bbox_inches="tight"); plt.close(figure)


def main() -> None:
    args = parse_args(); seed_everything(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    frame = load_manifest(args.manifest, args.hard_root, args.label_data_root, args.bottle_data_root)
    slugs, label_refs, bottle_refs = validate_gallery_alignment(Path(args.label_refs_root), Path(args.bottle_refs_root))
    label_model, label_info = load_retrieval_model(args.label_base_checkpoint, "vitb16", device)
    bottle_model, bottle_info = load_retrieval_model(args.bottle_base_checkpoint, "vitb16", device)
    adapted = torch.load(args.adapted_checkpoint, map_location="cpu", weights_only=False)
    label_model.load_state_dict(adapted["label_model_state_dict"], strict=True)
    bottle_model.load_state_dict(adapted["bottle_model_state_dict"], strict=True)
    for model in (label_model, bottle_model):
        model.set_backbone_trainable(0); model.eval()
    fusion_state = adapted["fusion_state_dict"]
    hidden_dim = int(fusion_state["network.1.weight"].shape[0])
    mlp = FusionMLP(hidden_dim=hidden_dim, dropout=args.dropout).to(device)
    mlp.load_state_dict(fusion_state, strict=True); mlp.eval()

    cache_args = argparse.Namespace(
        output_dir=str(output), eval_batch_size=args.embedding_batch_size,
        num_workers=args.num_workers, top_k=args.top_k,
        label_checkpoint=args.adapted_checkpoint, bottle_checkpoint=args.adapted_checkpoint,
    )
    print("CACHE | recomputing frozen Stage-2C DINO features", flush=True)
    cache, _, _ = embed_cache(
        frame, label_model, bottle_model, label_info, bottle_info, slugs,
        label_refs, bottle_refs, device, cache_args, "transformer", allow_disk_cache=False,
    )
    if cache is None:
        raise RuntimeError("Feature cache was not created")
    cache.save(output / "transformer_features.pt")
    train_cache = cache_subset(cache, frame, "hard_train")
    val_cache = cache_subset(cache, frame, "hard_val")
    mean, std = feature_statistics(train_cache)
    model = CandidateTransformerRanker(
        mean, std, d_model=args.d_model, num_heads=args.num_heads,
        num_layers=args.num_layers, feedforward_dim=args.feedforward_dim, dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"MODEL | CandidateTransformerRanker | parameters={parameter_count:,}", flush=True)

    present = train_cache.true_positions.ge(0)
    dataset = TensorDataset(
        train_cache.features[present], train_cache.candidate_mask[present], train_cache.true_positions[present]
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    best_score, best_r1, bad, history = -math.inf, -math.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; total_rows = 0; total_correct = 0
        for features, mask, truth in loader:
            features, mask, truth = features.to(device), mask.to(device), truth.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, mask)
            ce = F.cross_entropy(logits, truth)
            positive = logits.gather(1, truth[:, None]).squeeze(1)
            negatives = logits.masked_fill(F.one_hot(truth, logits.shape[1]).bool(), torch.finfo(logits.dtype).min).max(dim=1).values
            margin = F.softplus(negatives - positive + args.margin).mean()
            loss = ce + args.pairwise_weight * margin
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            count = len(truth); total_rows += count; total_loss += float(loss.detach()) * count
            total_correct += int(logits.argmax(dim=1).eq(truth).sum())
        scheduler.step()
        metrics, _, _ = rank_cache(model, val_cache, device, args.eval_batch_size)
        r1, mrr = float(metrics["recall_at_1"]), float(metrics["mrr"])
        score = r1 + 1e-3 * mrr
        improved = score > best_score + args.min_delta
        if improved:
            best_score, best_r1, bad = score, r1, 0
            atomic_save({
                "format": "candidate-transformer-v1", "epoch": epoch,
                "model_state_dict": model.state_dict(), "feature_names": FEATURE_NAMES,
                "config": {"d_model": args.d_model, "num_heads": args.num_heads, "num_layers": args.num_layers,
                           "feedforward_dim": args.feedforward_dim, "dropout": args.dropout},
                "monitor_split": "hard_val", "monitor_metric": "recall_at_1", "monitor_value": r1,
                "parameter_count": parameter_count,
            }, output / "transformer_best.pt")
        else:
            bad += 1
        row = {"epoch": epoch, "loss": total_loss / max(total_rows, 1), "train_accuracy": total_correct / max(total_rows, 1),
               "learning_rate": optimizer.param_groups[0]["lr"], "hard_val": metrics, "patience": bad}
        history.append(row); (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            f"EPOCH {epoch:02d}/{args.epochs} | loss={row['loss']:.4f} train_acc={100*row['train_accuracy']:.2f}% "
            f"| LR={row['learning_rate']:.2e} | hard_val R@1={100*r1:.2f}% R@2={100*float(metrics['recall_at_2']):.2f}% "
            f"R@5={100*float(metrics['recall_at_5']):.2f}% R@10={100*float(metrics['recall_at_10']):.2f}% "
            f"| best={100*best_r1:.2f}% patience={bad}/{args.patience}", flush=True,
        )
        if epoch >= args.min_epochs and bad >= args.patience:
            print("EARLY STOPPING | candidate Transformer", flush=True); break

    checkpoint = torch.load(output / "transformer_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    splits = ("hard_val",) + TEST_SPLITS
    report, details = report_for_splits(model, mlp, cache, frame, device, splits, args.eval_batch_size)
    print_report("FINAL TRANSFORMER VS FUSION MLP", report)
    payload = {"best_epoch": checkpoint["epoch"], "parameter_count": parameter_count, "results": report}
    (output / "transformer_final_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    save_audit_and_visualization(frame, cache, details, output)
    print("FINAL TRANSFORMER |", json.dumps(payload), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True); parser.add_argument("--hard-root", required=True)
    parser.add_argument("--label-data-root", required=True); parser.add_argument("--bottle-data-root", required=True)
    parser.add_argument("--label-refs-root", required=True); parser.add_argument("--bottle-refs-root", required=True)
    parser.add_argument("--label-base-checkpoint", required=True); parser.add_argument("--bottle-base-checkpoint", required=True)
    parser.add_argument("--adapted-checkpoint", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40); parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=7); parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512); parser.add_argument("--embedding-batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4); parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=256); parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--pairwise-weight", type=float, default=0.15); parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--min-delta", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
