#!/usr/bin/env python3
"""Train Stage-II visual fusion, then optionally jointly tune both DINOv3-B branches.

Stage ``frozen`` embeds every image once, trains a small candidate-scoring MLP,
and optionally fits a CatBoostRanker control model. Stage ``joint`` starts from
that MLP, unfreezes the last blocks of both DINOv3-B encoders and optimizes the
fusion ranking loss together with one classification/retrieval loss per branch.

The test partitions are evaluated before training and once after the best
checkpoint is restored. They are deliberately never used for early stopping.
OCR is represented by two missing-aware feature columns. If ``ocr_text`` is
absent in the manifest, those columns stay zero and no OCR result is invented.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cascade_resolver.modeling import choose_device, embed_paths, load_retrieval_model
from fusion_stage2.core import (
    FEATURE_NAMES,
    FeatureCache,
    FusionMLP,
    PairedImageDataset,
    branch_metrics,
    build_feature_cache,
    frame_fingerprint,
    load_manifest,
    predictions_from_cache,
    ranking_metrics,
    resolve_refs,
    save_json,
    score_cache,
)


TEST_SPLITS = (
    "test_seen_prior_exposure",
    "test_seen_model_selection",
    "test_unseen",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def cache_subset(cache: FeatureCache, frame: pd.DataFrame, split: str) -> FeatureCache:
    indices = np.flatnonzero(frame["fusion_split"].eq(split).to_numpy()).tolist()
    if not indices:
        raise ValueError(f"Fusion split is empty: {split}")
    return cache.subset(indices)


def metrics_for_splits(
    model: FusionMLP,
    cache: FeatureCache,
    frame: pd.DataFrame,
    device: torch.device,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in splits:
        subset = cache_subset(cache, frame, split)
        fusion_metrics, _ = score_cache(model, subset, device)
        report[split] = {"fusion_mlp": fusion_metrics, **branch_metrics(subset)}
    return report


def print_report(title: str, report: dict[str, Any]) -> None:
    print("=" * 110, flush=True)
    print(title, flush=True)
    for split, models in report.items():
        for name, values in models.items():
            print(
                f"{split:<30} | {name:<15} | "
                f"Acc/R@1={100 * float(values['recall_at_1']):6.2f}% "
                f"| R@2={100 * float(values['recall_at_2']):6.2f}% "
                f"| R@5={100 * float(values['recall_at_5']):6.2f}% "
                f"| R@10={100 * float(values['recall_at_10']):6.2f}% "
                f"| MRR={float(values['mrr']):.4f} "
                f"| N={int(values['num_queries'])}",
                flush=True,
            )
    print("=" * 110, flush=True)


def validate_gallery_alignment(
    label_refs_root: Path, bottle_refs_root: Path
) -> tuple[list[str], list[str], list[str]]:
    label_slugs, label_paths = resolve_refs(label_refs_root)
    bottle_slugs, bottle_paths = resolve_refs(bottle_refs_root)
    if label_slugs != bottle_slugs:
        missing_label = sorted(set(bottle_slugs) - set(label_slugs))[:5]
        missing_bottle = sorted(set(label_slugs) - set(bottle_slugs))[:5]
        raise ValueError(
            "Label and bottle galleries are not identity-aligned: "
            f"missing_label={missing_label}, missing_bottle={missing_bottle}"
        )
    return label_slugs, label_paths, bottle_paths


def embed_cache(
    frame: pd.DataFrame,
    label_model: torch.nn.Module,
    bottle_model: torch.nn.Module,
    label_info: dict[str, Any],
    bottle_info: dict[str, Any],
    slugs: list[str],
    label_ref_paths: list[str],
    bottle_ref_paths: list[str],
    device: torch.device,
    args: argparse.Namespace,
    description: str,
    allow_disk_cache: bool,
) -> tuple[FeatureCache | None, torch.Tensor, torch.Tensor]:
    # Gallery/query caches must be deterministic: projection dropout is
    # disabled here even when this helper is called between joint epochs.
    label_model.eval()
    bottle_model.eval()
    cache_dir = Path(args.output_dir) / "embedding_cache" if allow_disk_cache else None
    kwargs = dict(
        device=device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        cache_dir=cache_dir,
        force=not allow_disk_cache,
    )
    label_gallery, label_valid, label_errors, _ = embed_paths(
        label_model, label_ref_paths, image_size=int(label_info["image_size"]),
        description=f"{description}_label_gallery",
        checkpoint_path=args.label_checkpoint, **kwargs,
    )
    bottle_gallery, bottle_valid, bottle_errors, _ = embed_paths(
        bottle_model, bottle_ref_paths, image_size=int(bottle_info["image_size"]),
        description=f"{description}_bottle_gallery",
        checkpoint_path=args.bottle_checkpoint, **kwargs,
    )
    if not bool(label_valid.all()) or not bool(bottle_valid.all()):
        raise RuntimeError(
            "A gallery image could not be decoded: "
            f"label={label_errors}, bottle={bottle_errors}"
        )
    if frame.empty:
        return None, label_gallery, bottle_gallery
    label_queries, label_query_valid, label_query_errors, _ = embed_paths(
        label_model, frame["label_path"].tolist(),
        image_size=int(label_info["image_size"]),
        description=f"{description}_label_queries",
        checkpoint_path=args.label_checkpoint, **kwargs,
    )
    bottle_queries, bottle_query_valid, bottle_query_errors, _ = embed_paths(
        bottle_model, frame["bottle_path"].tolist(),
        image_size=int(bottle_info["image_size"]),
        description=f"{description}_bottle_queries",
        checkpoint_path=args.bottle_checkpoint, **kwargs,
    )
    valid = label_query_valid & bottle_query_valid
    if not bool(valid.all()):
        examples = [
            {
                "source_relative_path": frame.iloc[index]["source_relative_path"],
                "label_error": label_query_errors[index],
                "bottle_error": bottle_query_errors[index],
            }
            for index in torch.where(~valid)[0].tolist()[:10]
        ]
        raise RuntimeError(f"Paired crop decode failures: {examples}")
    cache = build_feature_cache(
        frame, label_queries, bottle_queries, label_gallery, bottle_gallery,
        slugs, top_k=args.top_k,
    )
    return cache, label_gallery, bottle_gallery


def save_audit(
    model: FusionMLP,
    cache: FeatureCache,
    frame: pd.DataFrame,
    device: torch.device,
    path: Path,
) -> None:
    prediction_ids, ranks, scores = predictions_from_cache(model, cache, device)
    audit = frame[["source_relative_path", "wine_slug", "fusion_split"]].copy()
    audit["prediction"] = [cache.slugs[index] for index in prediction_ids.tolist()]
    audit["rank"] = ranks.numpy()
    audit["fusion_score"] = scores.numpy()
    audit["correct_top1"] = audit["rank"].eq(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path, index=False)


def flatten_catboost(cache: FeatureCache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, labels, groups = [], [], []
    for query in range(len(cache.features)):
        count = int(cache.candidate_mask[query].sum())
        rows.append(cache.features[query, :count].numpy())
        labels.extend((torch.arange(count) == cache.true_positions[query]).int().tolist())
        groups.extend([query] * count)
    return np.concatenate(rows), np.asarray(labels), np.asarray(groups)


def run_catboost(
    train_cache: FeatureCache,
    val_cache: FeatureCache,
    evaluation_caches: dict[str, FeatureCache],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError:
        print("CATBOOST | skipped (package is not installed)", flush=True)
        return {"status": "skipped", "reason": "catboost is not installed"}
    eligible_train = train_cache.subset(torch.where(train_cache.true_positions.ge(0))[0].tolist())
    eligible_val = val_cache.subset(torch.where(val_cache.true_positions.ge(0))[0].tolist())
    x_train, y_train, groups_train = flatten_catboost(eligible_train)
    x_val_fit, y_val, groups_val = flatten_catboost(eligible_val)
    model = CatBoostRanker(
        iterations=600, depth=7, learning_rate=0.035, loss_function="YetiRank",
        eval_metric="NDCG:top=10", random_seed=seed, verbose=100,
        allow_writing_files=False,
    )
    train_pool = Pool(x_train, y_train, group_id=groups_train, feature_names=list(FEATURE_NAMES))
    val_pool = Pool(x_val_fit, y_val, group_id=groups_val, feature_names=list(FEATURE_NAMES))
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=60)
    model.save_model(str(output_dir / "catboost_ranker.cbm"))
    report: dict[str, Any] = {}
    for split, current in evaluation_caches.items():
        x_eval, _, _ = flatten_catboost(current)
        scores = model.predict(x_eval)
        ranks = []
        offset = 0
        for query in range(len(current.features)):
            count = int(current.candidate_mask[query].sum())
            truth = int(current.true_positions[query])
            if truth < 0:
                ranks.append(len(current.slugs) + 1)
            else:
                order = np.argsort(-scores[offset:offset + count])
                ranks.append(int(np.flatnonzero(order == truth)[0]) + 1)
            offset += count
        report[split] = ranking_metrics(torch.tensor(ranks))
    save_json(output_dir / "catboost_final_metrics.json", report)
    return {"status": "complete", "metrics": report}


def train_frozen(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    cache: FeatureCache,
    device: torch.device,
) -> None:
    output_dir = Path(args.output_dir)
    train_cache = cache_subset(cache, frame, "hard_train")
    val_cache = cache_subset(cache, frame, "hard_val")
    trainable = train_cache.true_positions.ge(0)
    if not bool(trainable.any()):
        raise ValueError("No hard-train ground truth occurs in the candidate union")
    model = FusionMLP(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.fusion_lr, weight_decay=1e-3)
    loader = DataLoader(
        TensorDataset(torch.where(trainable)[0]), batch_size=args.fusion_batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
    )
    best, bad, history = -math.inf, 0, []
    for epoch in range(1, args.frozen_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        total_correct = 0
        for (indices,) in loader:
            features = train_cache.features[indices].to(device)
            mask = train_cache.candidate_mask[indices].to(device)
            truth = train_cache.true_positions[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, mask)
            loss = F.cross_entropy(logits, truth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total_correct += int(logits.argmax(dim=1).eq(truth).sum())
            total_rows += len(indices)
        val_metrics, _ = score_cache(model, val_cache, device)
        monitor = float(val_metrics["recall_at_1"])
        improved = monitor > best + args.min_delta
        if improved:
            best, bad = monitor, 0
            atomic_torch_save(
                {
                    "format": "fusion-stage2a-v1", "model_state_dict": model.state_dict(),
                    "feature_names": FEATURE_NAMES, "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout, "epoch": epoch,
                    "monitor_split": "hard_val", "monitor_metric": "recall_at_1",
                    "monitor_value": best,
                },
                output_dir / "fusion_mlp_best.pt",
            )
        else:
            bad += 1
        row = {
            "epoch": epoch, "train_loss": total_loss / max(total_rows, 1),
            "train_accuracy": total_correct / max(total_rows, 1),
            "hard_val": val_metrics, "best_hard_val_recall_at_1": best,
            "no_improvement": bad,
        }
        history.append(row)
        save_json(output_dir / "frozen_history.json", history)
        print(
            f"EPOCH {epoch:02d}/{args.frozen_epochs} | stage=2A frozen "
            f"| lr={args.fusion_lr:.2e} | loss={row['train_loss']:.4f} "
            f"train_acc={100*row['train_accuracy']:.2f}% "
            f"| hard_val Acc/R@1={100*monitor:.2f}% "
            f"| R@2={100*float(val_metrics['recall_at_2']):.2f}% "
            f"| R@5={100*float(val_metrics['recall_at_5']):.2f}% "
            f"| R@10={100*float(val_metrics['recall_at_10']):.2f}% "
            f"| best={100*best:.2f}% | patience={bad}/{args.patience}",
            flush=True,
        )
        if epoch >= args.min_epochs and bad >= args.patience:
            print("EARLY STOPPING | stage=2A | monitor=hard_val.recall_at_1", flush=True)
            break
    checkpoint = torch.load(output_dir / "fusion_mlp_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final = metrics_for_splits(model, cache, frame, device, ("hard_val",) + TEST_SPLITS)
    print_report("FINAL STAGE 2A | TEST USED ONCE AFTER MODEL SELECTION", final)
    save_json(output_dir / "frozen_final_metrics.json", final)
    save_audit(model, cache, frame, device, output_dir / "frozen_predictions.csv")
    if args.catboost:
        evaluation_caches = {
            split: cache_subset(cache, frame, split)
            for split in ("hard_val",) + TEST_SPLITS
        }
        result = run_catboost(
            train_cache, val_cache, evaluation_caches, output_dir, args.seed
        )
        save_json(output_dir / "catboost_summary.json", result)


def dynamic_features(
    base_features: torch.Tensor,
    candidate_ids: torch.Tensor,
    mask: torch.Tensor,
    label_embeddings: torch.Tensor,
    bottle_embeddings: torch.Tensor,
    label_gallery: torch.Tensor,
    bottle_gallery: torch.Tensor,
) -> torch.Tensor:
    safe_ids = candidate_ids.clamp_min(0)
    label_candidates = label_gallery[safe_ids]
    bottle_candidates = bottle_gallery[safe_ids]
    label_scores = torch.einsum("bd,bkd->bk", label_embeddings, label_candidates)
    bottle_scores = torch.einsum("bd,bkd->bk", bottle_embeddings, bottle_candidates)
    label_scores = label_scores.masked_fill(~mask, -1.0)
    bottle_scores = bottle_scores.masked_fill(~mask, -1.0)
    features = base_features.clone()
    features[:, :, 0] = label_scores
    features[:, :, 1] = bottle_scores
    features[:, :, 2] = label_scores - label_scores.max(dim=1, keepdim=True).values
    features[:, :, 3] = bottle_scores - bottle_scores.max(dim=1, keepdim=True).values
    return features


@torch.inference_mode()
def evaluate_joint(
    fusion: FusionMLP,
    label_model: torch.nn.Module,
    bottle_model: torch.nn.Module,
    dataset: PairedImageDataset,
    cache: FeatureCache,
    label_gallery: torch.Tensor,
    bottle_gallery: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    fusion.eval(); label_model.eval(); bottle_model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    ranks = []
    for batch in loader:
        indices = batch["index"].long()
        label_embeddings, _ = label_model(batch["label"].to(device, non_blocking=True))
        bottle_embeddings, _ = bottle_model(batch["bottle"].to(device, non_blocking=True))
        mask = cache.candidate_mask[indices].to(device)
        ids = cache.candidate_ids[indices].to(device)
        features = dynamic_features(
            cache.features[indices].to(device), ids, mask,
            label_embeddings, bottle_embeddings, label_gallery, bottle_gallery,
        )
        order = fusion(features, mask).argsort(dim=1, descending=True).cpu()
        truth = cache.true_positions[indices]
        batch_ranks = torch.full((len(indices),), len(cache.slugs) + 1, dtype=torch.long)
        present = truth.ge(0)
        if present.any():
            batch_ranks[present] = order[present].eq(truth[present, None]).float().argmax(1).long() + 1
        ranks.append(batch_ranks)
    metrics = ranking_metrics(torch.cat(ranks))
    metrics["candidate_recall"] = float(cache.true_positions.ge(0).float().mean())
    return metrics


def train_joint(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    initial_cache: FeatureCache,
    label_model: torch.nn.Module,
    bottle_model: torch.nn.Module,
    label_gallery: torch.Tensor,
    bottle_gallery: torch.Tensor,
    slugs: list[str],
    label_ref_paths: list[str],
    bottle_ref_paths: list[str],
    label_info: dict[str, Any],
    bottle_info: dict[str, Any],
    device: torch.device,
) -> None:
    output_dir = Path(args.output_dir)
    frozen_checkpoint = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False)
    if tuple(frozen_checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Fusion checkpoint feature schema does not match this code")
    fusion = FusionMLP(
        hidden_dim=int(frozen_checkpoint["hidden_dim"]),
        dropout=float(frozen_checkpoint["dropout"]),
    ).to(device)
    fusion.load_state_dict(frozen_checkpoint["model_state_dict"])
    for model in (label_model, bottle_model):
        model.set_backbone_trainable(args.unfreeze_last_blocks)
        model.train()
    train_frame = frame[frame["fusion_split"].eq("hard_train")].reset_index(drop=True)
    val_frame = frame[frame["fusion_split"].eq("hard_val")].reset_index(drop=True)
    train_cache = cache_subset(initial_cache, frame, "hard_train")
    val_cache = cache_subset(initial_cache, frame, "hard_val")
    train_dataset = PairedImageDataset(train_frame, args.image_size, train=True)
    val_dataset = PairedImageDataset(val_frame, args.image_size, train=False)
    loader = DataLoader(
        train_dataset, batch_size=args.joint_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    parameter_groups = [
        {"params": fusion.parameters(), "lr": args.fusion_lr},
        {"params": [p for p in label_model.projection.parameters() if p.requires_grad] +
                   [p for p in label_model.classifier.parameters() if p.requires_grad], "lr": args.head_lr},
        {"params": [p for p in bottle_model.projection.parameters() if p.requires_grad] +
                   [p for p in bottle_model.classifier.parameters() if p.requires_grad], "lr": args.head_lr},
        {"params": [p for p in label_model.backbone.parameters() if p.requires_grad], "lr": args.backbone_lr},
        {"params": [p for p in bottle_model.backbone.parameters() if p.requires_grad], "lr": args.backbone_lr},
    ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    label_gallery = label_gallery.to(device)
    bottle_gallery = bottle_gallery.to(device)
    best, bad, history = -math.inf, 0, []
    for epoch in range(1, args.joint_epochs + 1):
        fusion.train(); label_model.train(); bottle_model.train()
        totals = {
            "loss": 0.0, "fusion": 0.0, "label": 0.0, "bottle": 0.0,
            "rows": 0, "fusion_correct": 0, "fusion_rows": 0,
            "label_correct": 0, "bottle_correct": 0,
        }
        for batch in loader:
            indices = batch["index"].long()
            mask = train_cache.candidate_mask[indices].to(device)
            candidate_ids = train_cache.candidate_ids[indices].to(device)
            truth_position = train_cache.true_positions[indices].to(device)
            truth_id = train_cache.true_label_ids[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device.type == "cuda" and args.amp):
                label_embeddings, label_logits = label_model(batch["label"].to(device, non_blocking=True))
                bottle_embeddings, bottle_logits = bottle_model(batch["bottle"].to(device, non_blocking=True))
                features = dynamic_features(
                    train_cache.features[indices].to(device), candidate_ids, mask,
                    label_embeddings, bottle_embeddings, label_gallery, bottle_gallery,
                )
                fusion_logits = fusion(features, mask)
                present = truth_position.ge(0)
                fusion_loss = (
                    F.cross_entropy(fusion_logits[present], truth_position[present])
                    if present.any() else fusion_logits.sum() * 0.0
                )
                label_loss = F.cross_entropy(label_logits, truth_id)
                bottle_loss = F.cross_entropy(bottle_logits, truth_id)
                loss = fusion_loss + args.label_loss_weight * label_loss + args.bottle_loss_weight * bottle_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(fusion.parameters()) + list(label_model.parameters()) + list(bottle_model.parameters()), 2.0
            )
            scaler.step(optimizer); scaler.update()
            count = len(indices)
            totals["loss"] += float(loss.detach()) * count
            totals["fusion"] += float(fusion_loss.detach()) * count
            totals["label"] += float(label_loss.detach()) * count
            totals["bottle"] += float(bottle_loss.detach()) * count
            totals["rows"] += count
            totals["fusion_correct"] += int(
                fusion_logits[present].argmax(dim=1).eq(truth_position[present]).sum()
            ) if present.any() else 0
            totals["fusion_rows"] += int(present.sum())
            totals["label_correct"] += int(label_logits.argmax(dim=1).eq(truth_id).sum())
            totals["bottle_correct"] += int(bottle_logits.argmax(dim=1).eq(truth_id).sum())
        val_metrics = evaluate_joint(
            fusion, label_model, bottle_model, val_dataset, val_cache,
            label_gallery, bottle_gallery, device, args.eval_batch_size, args.num_workers,
        )
        monitor = float(val_metrics["recall_at_1"])
        improved = monitor > best + args.min_delta
        if improved:
            best, bad = monitor, 0
            atomic_torch_save(
                {
                    "format": "fusion-stage2b-v1", "fusion_state_dict": fusion.state_dict(),
                    "label_model_state_dict": label_model.state_dict(),
                    "bottle_model_state_dict": bottle_model.state_dict(),
                    "feature_names": FEATURE_NAMES, "epoch": epoch,
                    "unfreeze_last_blocks": args.unfreeze_last_blocks,
                    "monitor_split": "hard_val", "monitor_metric": "recall_at_1",
                    "monitor_value": best,
                },
                output_dir / "joint_best.pt",
            )
        else:
            bad += 1
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(totals["rows"], 1),
            "fusion": totals["fusion"] / max(totals["rows"], 1),
            "label": totals["label"] / max(totals["rows"], 1),
            "bottle": totals["bottle"] / max(totals["rows"], 1),
            "train_fusion_accuracy": totals["fusion_correct"] / max(totals["fusion_rows"], 1),
            "train_label_accuracy": totals["label_correct"] / max(totals["rows"], 1),
            "train_bottle_accuracy": totals["bottle_correct"] / max(totals["rows"], 1),
            "hard_val": val_metrics, "best_hard_val_recall_at_1": best,
            "no_improvement": bad,
        }
        history.append(row); save_json(output_dir / "joint_history.json", history)
        print(
            f"EPOCH {epoch:02d}/{args.joint_epochs} | stage=2B joint "
            f"| loss={row['loss']:.4f} fusion={row['fusion']:.4f} "
            f"label={row['label']:.4f} bottle={row['bottle']:.4f} "
            f"| train_acc fusion={100*row['train_fusion_accuracy']:.2f}% "
            f"label={100*row['train_label_accuracy']:.2f}% "
            f"bottle={100*row['train_bottle_accuracy']:.2f}% "
            f"| LR fusion={args.fusion_lr:.2e} head={args.head_lr:.2e} backbone={args.backbone_lr:.2e} "
            f"| hard_val Acc/R@1={100*monitor:.2f}% R@2={100*float(val_metrics['recall_at_2']):.2f}% "
            f"R@5={100*float(val_metrics['recall_at_5']):.2f}% R@10={100*float(val_metrics['recall_at_10']):.2f}% "
            f"| best={100*best:.2f}% patience={bad}/{args.patience}",
            flush=True,
        )
        atomic_torch_save(
            {
                "format": "fusion-stage2b-resume-v1", "epoch": epoch,
                "fusion_state_dict": fusion.state_dict(),
                "label_model_state_dict": label_model.state_dict(),
                "bottle_model_state_dict": bottle_model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "history": history, "best_monitor": best, "no_improvement": bad,
            }, output_dir / "joint_last.pt",
        )
        if epoch < args.joint_epochs and epoch % args.remine_every == 0:
            print(
                f"RE-MINING | epoch={epoch} | refreshing both galleries and "
                "Top-K candidate unions on hard_train + hard_val",
                flush=True,
            )
            hard_frame = frame[
                frame["fusion_split"].isin(["hard_train", "hard_val"])
            ].reset_index(drop=True)
            refreshed, refreshed_label_gallery, refreshed_bottle_gallery = embed_cache(
                hard_frame, label_model, bottle_model, label_info, bottle_info,
                slugs, label_ref_paths, bottle_ref_paths, device, args,
                f"joint_epoch_{epoch}", False,
            )
            if refreshed is None:
                raise RuntimeError("Candidate refresh unexpectedly returned no cache")
            train_cache = cache_subset(refreshed, hard_frame, "hard_train")
            val_cache = cache_subset(refreshed, hard_frame, "hard_val")
            label_gallery = refreshed_label_gallery.to(device)
            bottle_gallery = refreshed_bottle_gallery.to(device)
        if epoch >= args.min_epochs and bad >= args.patience:
            print("EARLY STOPPING | stage=2B | monitor=hard_val.recall_at_1", flush=True)
            break
    best_payload = torch.load(output_dir / "joint_best.pt", map_location="cpu", weights_only=False)
    fusion.load_state_dict(best_payload["fusion_state_dict"])
    label_model.load_state_dict(best_payload["label_model_state_dict"])
    bottle_model.load_state_dict(best_payload["bottle_model_state_dict"])
    print(
        "FINAL REFRESH | rebuilding both galleries and all candidate unions once "
        "before the untouched test partitions",
        flush=True,
    )
    final_cache, refreshed_label_gallery, refreshed_bottle_gallery = embed_cache(
        frame, label_model, bottle_model, label_info, bottle_info,
        slugs, label_ref_paths, bottle_ref_paths, device, args,
        "joint_final", False,
    )
    if final_cache is None:
        raise RuntimeError("Final candidate refresh unexpectedly returned no cache")
    label_gallery = refreshed_label_gallery.to(device)
    bottle_gallery = refreshed_bottle_gallery.to(device)
    final_cache.save(output_dir / "joint_final_features.pt")
    final: dict[str, Any] = {}
    for split in ("hard_val",) + TEST_SPLITS:
        split_frame = frame[frame["fusion_split"].eq(split)].reset_index(drop=True)
        split_cache = cache_subset(final_cache, frame, split)
        split_dataset = PairedImageDataset(split_frame, args.image_size, train=False)
        final[split] = {
            "fusion_joint": evaluate_joint(
                fusion, label_model, bottle_model, split_dataset, split_cache,
                label_gallery, bottle_gallery, device, args.eval_batch_size, args.num_workers,
            ),
            **branch_metrics(split_cache),
        }
    print_report("FINAL STAGE 2B | TEST USED ONCE AFTER MODEL SELECTION", final)
    save_json(output_dir / "joint_final_metrics.json", final)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("frozen", "joint"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hard-root", required=True)
    parser.add_argument("--label-data-root", required=True)
    parser.add_argument("--bottle-data-root", required=True)
    parser.add_argument("--label-refs-root", required=True)
    parser.add_argument("--bottle-refs-root", required=True)
    parser.add_argument("--label-checkpoint", required=True)
    parser.add_argument("--bottle-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fusion-checkpoint")
    parser.add_argument("--feature-cache")
    parser.add_argument(
        "--ocr-csv",
        help="Optional CSV with unique source_relative_path and verified ocr_text columns",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--fusion-batch-size", type=int, default=512)
    parser.add_argument("--joint-batch-size", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--fusion-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=5e-7)
    parser.add_argument("--label-loss-weight", type=float, default=0.25)
    parser.add_argument("--bottle-loss-weight", type=float, default=0.25)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=2)
    parser.add_argument("--remine-every", type=int, default=3)
    parser.add_argument("--frozen-epochs", type=int, default=40)
    parser.add_argument("--joint-epochs", type=int, default=18)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--catboost", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "joint" and not args.fusion_checkpoint:
        raise SystemExit("--fusion-checkpoint is required for joint Stage 2B")
    if args.remine_every < 1:
        raise SystemExit("--remine-every must be positive")
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type != "cuda":
        print(f"WARNING | device={device}; Stage II is designed for Kaggle CUDA", flush=True)
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_manifest(args.manifest, args.hard_root, args.label_data_root, args.bottle_data_root)
    if args.ocr_csv:
        ocr = pd.read_csv(args.ocr_csv, dtype=str).fillna("")
        required_ocr = {"source_relative_path", "ocr_text"}
        if missing := required_ocr.difference(ocr.columns):
            raise ValueError(f"OCR CSV is missing columns: {sorted(missing)}")
        if ocr["source_relative_path"].duplicated().any():
            raise ValueError("OCR CSV contains duplicate source_relative_path values")
        frame = frame.drop(columns=["ocr_text", "ocr_available"], errors="ignore").merge(
            ocr[["source_relative_path", "ocr_text"]],
            on="source_relative_path", how="left", validate="one_to_one",
        )
        frame["ocr_text"] = frame["ocr_text"].fillna("")
        frame["ocr_available"] = frame["ocr_text"].str.strip().ne("").astype(int)
    slugs, label_refs, bottle_refs = validate_gallery_alignment(
        Path(args.label_refs_root), Path(args.bottle_refs_root)
    )
    unknown = sorted(set(frame["wine_slug"]) - set(slugs))
    if unknown:
        raise ValueError(f"Manifest identities missing from gallery: {unknown[:5]}")
    label_model, label_info = load_retrieval_model(args.label_checkpoint, "vitb16", device)
    bottle_model, bottle_info = load_retrieval_model(args.bottle_checkpoint, "vitb16", device)
    if int(label_info["image_size"]) != args.image_size or int(bottle_info["image_size"]) != args.image_size:
        raise ValueError(
            f"--image-size={args.image_size} differs from Stage-I checkpoints: "
            f"label={label_info['image_size']}, bottle={bottle_info['image_size']}"
        )
    if int(label_info["num_classes"]) != len(slugs) or int(bottle_info["num_classes"]) != len(slugs):
        raise ValueError("A Stage-I checkpoint classifier does not match the shared gallery")
    print(
        f"SETUP | stage={args.stage} device={device} rows={len(frame)} identities={len(slugs)} "
        f"OCR rows={int(frame.get('ocr_text', pd.Series(dtype=str)).astype(bool).sum()) if 'ocr_text' in frame else 0}",
        flush=True,
    )
    cache_path = Path(args.feature_cache).resolve() if args.feature_cache else output_dir / "fusion_features.pt"
    if cache_path.is_file():
        cache = FeatureCache.load(cache_path)
        _, label_gallery, bottle_gallery = embed_cache(
            frame.iloc[:0], label_model, bottle_model, label_info, bottle_info,
            slugs, label_refs, bottle_refs, device, args, "gallery_only", True,
        )
    else:
        cache, label_gallery, bottle_gallery = embed_cache(
            frame, label_model, bottle_model, label_info, bottle_info,
            slugs, label_refs, bottle_refs, device, args, "stage2", True,
        )
        if cache is None:
            raise RuntimeError("Full manifest embedding unexpectedly returned no cache")
        cache.save(cache_path)
    if cache is None:
        raise RuntimeError("Feature cache is unavailable")
    if (
        len(cache.features) != len(frame)
        or cache.slugs != slugs
        or cache.data_fingerprint != frame_fingerprint(frame)
    ):
        raise ValueError(
            "Feature cache does not match the attached manifest/gallery; "
            "attach the Stage-2A output produced from this exact hard-set version"
        )
    baseline = {split: branch_metrics(cache_subset(cache, frame, split)) for split in ("hard_val",) + TEST_SPLITS}
    save_json(output_dir / "visual_branch_baseline.json", baseline)
    print_report("BASELINE VISUAL BRANCHES | TEST SNAPSHOT BEFORE STAGE II", baseline)
    if args.stage == "frozen":
        train_frozen(args, frame, cache, device)
    else:
        train_joint(
            args, frame, cache, label_model, bottle_model,
            label_gallery, bottle_gallery, slugs, label_refs, bottle_refs,
            label_info, bottle_info, device,
        )


if __name__ == "__main__":
    main()
