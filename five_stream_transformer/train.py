#!/usr/bin/env python3
"""Train a from-scratch residual Transformer over frozen Stage-2C DINO-B."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from five_stream_transformer.data import combine_frames, load_hard_rows, load_manual_rows, resolve_refs
from five_stream_transformer.dino import choose_device
from five_stream_transformer.features import FiveStreamCache, build_cache
from five_stream_transformer.model import (
    STREAM_NAMES,
    FiveStreamResidualTransformer,
    residual_ranking_loss,
)
from five_stream_transformer.ocr import fill_ocr_text
from five_stream_transformer.stage2c import load_stage2c_models


class IndexDataset(Dataset):
    def __init__(self, indices: torch.Tensor) -> None:
        self.indices = indices.long()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.indices[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def batch_from_cache(cache: FiveStreamCache, indices: torch.Tensor, device: torch.device):
    candidate_ids = cache.candidate_ids[indices]
    query = {
        name: cache.query_streams[name][indices].float().to(device, non_blocking=True)
        for name in STREAM_NAMES
    }
    references = {
        name: cache.gallery_streams[name][candidate_ids].float().to(device, non_blocking=True)
        for name in STREAM_NAMES
    }
    return (
        query,
        references,
        cache.candidate_mask[indices].to(device, non_blocking=True),
        cache.positive_mask[indices].to(device, non_blocking=True),
        cache.query_stream_available[indices].to(device, non_blocking=True),
    )


def loader_for(
    cache: FiveStreamCache,
    indices: torch.Tensor,
    batch_size: int,
    num_workers: int,
    training: bool,
    manual_repeat: float,
) -> DataLoader:
    dataset = IndexDataset(indices)
    sampler = None
    if training:
        weights = torch.tensor(
            [manual_repeat if cache.source_groups[index] == "manual_211" else 1.0 for index in indices],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def ranks_from_logits(logits: torch.Tensor, positive_mask: torch.Tensor, missing_rank: int) -> torch.Tensor:
    order = logits.argsort(dim=1, descending=True)
    ranked_positive = positive_mask.gather(1, order)
    any_positive = ranked_positive.any(dim=1)
    first = ranked_positive.float().argmax(dim=1) + 1
    return torch.where(any_positive, first, torch.full_like(first, missing_rank))


def summarize_ranks(
    ranks: torch.Tensor, candidate_recall: float, missing_rank: int
) -> dict[str, Any]:
    ranks = ranks.float()
    metrics: dict[str, Any] = {
        "num_queries": int(len(ranks)),
        "accuracy": float((ranks == 1).float().mean()),
        "mrr": float(torch.where(ranks < missing_rank, 1.0 / ranks, 0.0).mean()),
        "mean_rank_in_pool": float(ranks.mean()),
        "candidate_recall": float(candidate_recall),
    }
    for k in (1, 2, 5, 10):
        metrics[f"recall_at_{k}"] = float((ranks <= k).float().mean())
    return metrics


@torch.inference_mode()
def evaluate(
    model: FiveStreamResidualTransformer,
    cache: FiveStreamCache,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    model.eval()
    final_ranks: list[torch.Tensor] = []
    base_ranks: list[torch.Tensor] = []
    natural: list[torch.Tensor] = []
    fixed = 0
    harmed = 0
    missing_rank = cache.candidate_ids.shape[1] + 1
    for batch_indices in loader_for(cache, indices, batch_size, num_workers, False, 1.0):
        query, refs, candidate_mask, positive_mask, available = batch_from_cache(cache, batch_indices, device)
        final_logits, base_logits, _ = model(query, refs, candidate_mask, available)
        final = ranks_from_logits(final_logits, positive_mask, missing_rank)
        base = ranks_from_logits(base_logits, positive_mask, missing_rank)
        final_ranks.append(final.cpu())
        base_ranks.append(base.cpu())
        natural.append(cache.natural_positive_in_pool[batch_indices])
        fixed += int(((base != 1) & (final == 1)).sum())
        harmed += int(((base == 1) & (final != 1)).sum())
    final = torch.cat(final_ranks)
    base = torch.cat(base_ranks)
    candidate_recall = float(torch.cat(natural).float().mean())
    return {
        "residual_transformer": summarize_ranks(final, candidate_recall, missing_rank),
        "bottle_dino_base": summarize_ranks(base, candidate_recall, missing_rank),
        "comparison": {"fixed_top1": fixed, "harmed_top1": harmed, "net_top1": fixed - harmed},
    }


def train_epoch(
    model: FiveStreamResidualTransformer,
    cache: FiveStreamCache,
    indices: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0, "ranking": 0.0, "pairwise": 0.0, "protection": 0.0,
        "residual_guard": 0.0, "correct": 0.0, "base_correct": 0.0, "rows": 0.0,
    }
    loader = loader_for(cache, indices, args.batch_size, args.num_workers, True, args.manual_repeat)
    for batch_indices in loader:
        query, refs, candidate_mask, positive_mask, available = batch_from_cache(cache, batch_indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda" and args.amp):
            final_logits, base_logits, residual = model(query, refs, candidate_mask, available)
            loss, parts = residual_ranking_loss(
                final_logits, base_logits, residual, positive_mask, candidate_mask,
                pairwise_weight=args.pairwise_weight,
                protection_weight=args.protection_weight,
                residual_weight=args.residual_weight,
                pairwise_margin=args.pairwise_margin,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        count = len(batch_indices)
        totals["rows"] += count
        totals["loss"] += float(loss.detach()) * count
        for name in ("ranking", "pairwise", "protection", "residual_guard"):
            totals[name] += float(parts[name]) * count
        prediction = final_logits.argmax(dim=1)
        totals["correct"] += int(positive_mask.gather(1, prediction[:, None]).sum())
        totals["base_correct"] += float(parts["base_correct_rate"]) * count
    rows = max(totals.pop("rows"), 1.0)
    return {name: value / rows for name, value in totals.items()}


def print_metrics(prefix: str, values: dict[str, Any]) -> None:
    learned = values["residual_transformer"]
    base = values["bottle_dino_base"]
    comparison = values["comparison"]
    print(
        f"{prefix} | residual R@1={100*learned['recall_at_1']:.2f}% "
        f"R@2={100*learned['recall_at_2']:.2f}% R@5={100*learned['recall_at_5']:.2f}% "
        f"R@10={100*learned['recall_at_10']:.2f}% MRR={learned['mrr']:.4f} "
        f"| bottle_base R@1={100*base['recall_at_1']:.2f}% "
        f"| fixed={comparison['fixed_top1']} harmed={comparison['harmed_top1']} "
        f"net={comparison['net_top1']} cand={100*learned['candidate_recall']:.2f}% "
        f"N={learned['num_queries']}", flush=True,
    )


def checkpoint_payload(
    model: FiveStreamResidualTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    stage2c_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "stage2c-five-stream-residual-v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
        "stage2c_source": stage2c_info,
        "architecture": {
            "streams": list(STREAM_NAMES), "model_dim": 256, "num_heads": 8,
            "num_layers": 3, "feedforward_dim": 768, "dropout": args.dropout,
            "base_temperature": args.base_temperature, "max_residual": args.max_residual,
            "residual_head_zero_initialized": True,
        },
        "encoders_frozen": True,
        "transformer_initialized_from_scratch": True,
        "args": vars(args),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--hard-root", type=Path, required=True)
    parser.add_argument("--label-data-root", type=Path, required=True)
    parser.add_argument("--bottle-data-root", type=Path, required=True)
    parser.add_argument("--manual-root", type=Path, required=True)
    parser.add_argument("--easyocr-model-dir", type=Path, required=True)
    parser.add_argument("--ocr-cache-every", type=int, default=25)
    parser.add_argument("--stage2c-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--feature-batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--manual-repeat", type=float, default=20.0)
    parser.add_argument("--base-temperature", type=float, default=0.07)
    parser.add_argument("--max-residual", type=float, default=2.5)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--protection-weight", type=float, default=0.75)
    parser.add_argument("--residual-weight", type=float, default=0.02)
    parser.add_argument("--pairwise-margin", type=float, default=0.5)
    parser.add_argument("--maximum-hard-val-drop", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Training requires a CUDA GPU")

    label_slugs, label_refs = resolve_refs(args.label_data_root / "refs")
    bottle_slugs, bottle_refs = resolve_refs(args.bottle_data_root / "refs")
    if label_slugs != bottle_slugs:
        raise ValueError("Label and bottle reference galleries are not aligned")

    print("=" * 112, flush=True)
    print("SETUP | STAGE-2C FIVE-STREAM RESIDUAL TRANSFORMER", flush=True)
    print(
        "SETUP | DINO-B branches=complete Stage-2C state, frozen | "
        "Transformer=random initialization | data=stored 45k crops + stored Manual-211 crops",
        flush=True,
    )
    hard = load_hard_rows(
        args.hard_manifest, args.hard_root, args.label_data_root, args.bottle_data_root
    )
    manual = load_manual_rows(args.manual_root, label_slugs, args.seed)
    frame = combine_frames(hard, manual)
    frame = fill_ocr_text(
        frame, output / "ocr_text_cache.csv", set(frame["source_group"]),
        args.easyocr_model_dir, gpu=True, cache_every=args.ocr_cache_every,
    )
    frame.assign(
        positive_slugs=frame["positive_slugs"].map(lambda values: ";".join(values))
    ).to_csv(output / "combined_manifest.csv", index=False)
    print(
        "DATA | " + json.dumps({
            "rows": len(frame),
            "split_counts": frame["five_stream_split"].value_counts().to_dict(),
            "source_counts": frame["source_group"].value_counts().to_dict(),
            "ocr_rows": int(frame["ocr_text"].str.strip().ne("").sum()),
        }, ensure_ascii=False), flush=True,
    )

    print("DINO | loading both complete frozen Stage-2C branches after OCR release", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    label_model, bottle_model, stage2c_info = load_stage2c_models(
        args.stage2c_checkpoint, device, args.image_size
    )
    print("STAGE2C | " + json.dumps(stage2c_info, ensure_ascii=False), flush=True)

    cache_path = output / f"features_stage2c_{stage2c_info['sha256'][:12]}.pt"
    if cache_path.is_file():
        print(f"FEATURE | loading cache {cache_path}", flush=True)
        cache = FiveStreamCache.load(cache_path)
    else:
        cache = build_cache(
            frame, label_model, bottle_model, label_refs, bottle_refs, label_slugs,
            args.image_size, device, args.feature_batch_size, args.num_workers, args.top_k,
        )
        cache.save(cache_path)
    del label_model, bottle_model
    gc.collect()
    torch.cuda.empty_cache()

    train_indices = cache.indices("train")
    validation_indices = cache.indices(("val_hard", "val_manual"))
    train_indices = train_indices[cache.natural_positive_in_pool[train_indices]]
    validation_indices = validation_indices[cache.natural_positive_in_pool[validation_indices]]
    print(
        f"CANDIDATES | width={cache.candidate_ids.shape[1]} train={len(train_indices)} "
        f"validation={len(validation_indices)} overall_recall="
        f"{100*float(cache.natural_positive_in_pool.float().mean()):.2f}%", flush=True,
    )

    model = FiveStreamResidualTransformer(
        dropout=args.dropout, max_candidates=cache.candidate_ids.shape[1],
        base_temperature=args.base_temperature, max_residual=args.max_residual,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    initial_validation = evaluate(
        model, cache, validation_indices, device, args.batch_size * 2, args.num_workers
    )
    initial_hard = evaluate(
        model, cache, cache.indices("val_hard"), device, args.batch_size * 2, args.num_workers
    )
    print_metrics("BASELINE combined_validation", initial_validation)
    print_metrics("BASELINE val_hard", initial_hard)
    hard_floor = float(initial_hard["bottle_dino_base"]["recall_at_1"]) - args.maximum_hard_val_drop
    best_score = float(initial_validation["residual_transformer"]["recall_at_1"])
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    atomic_save(
        output / "residual_best.pt",
        checkpoint_payload(model, optimizer, 0, args, initial_validation, stage2c_info),
    )

    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model, cache, train_indices, optimizer, scaler, device, args)
        validation = evaluate(
            model, cache, validation_indices, device, args.batch_size * 2, args.num_workers
        )
        hard_metrics = evaluate(
            model, cache, cache.indices("val_hard"), device, args.batch_size * 2, args.num_workers
        )
        score = float(validation["residual_transformer"]["recall_at_1"])
        hard_r1 = float(hard_metrics["residual_transformer"]["recall_at_1"])
        guardrail_passed = hard_r1 >= hard_floor
        improved = guardrail_passed and score > best_score + args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            atomic_save(
                output / "residual_best.pt",
                checkpoint_payload(model, optimizer, epoch, args, validation, stage2c_info),
            )
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch, "losses": losses, "validation": validation,
            "val_hard": hard_metrics, "guardrail_passed": guardrail_passed,
            "best_score": best_score, "best_epoch": best_epoch, "patience": bad_epochs,
        }
        history.append(row)
        save_json(output / "history.json", history)
        learned = validation["residual_transformer"]
        base = validation["bottle_dino_base"]
        comparison = validation["comparison"]
        print(
            f"EPOCH {epoch:02d}/{args.epochs} | loss={losses['loss']:.4f} "
            f"rank={losses['ranking']:.4f} pair={losses['pairwise']:.4f} "
            f"protect={losses['protection']:.4f} "
            f"| train R@1={100*losses['correct']:.2f}% base={100*losses['base_correct']:.2f}% "
            f"| val residual R@1={100*learned['recall_at_1']:.2f}% "
            f"R@2={100*learned['recall_at_2']:.2f}% bottle_base={100*base['recall_at_1']:.2f}% "
            f"fixed={comparison['fixed_top1']} harmed={comparison['harmed_top1']} "
            f"net={comparison['net_top1']} | hard={100*hard_r1:.2f}% "
            f"floor={100*hard_floor:.2f}% | LR={optimizer.param_groups[0]['lr']:.2e} "
            f"best={100*best_score:.2f}%@{best_epoch} patience={bad_epochs}/{args.patience}", flush=True,
        )
        scheduler.step()
        if bad_epochs >= args.patience:
            print(f"EARLY STOP | no guardrail-safe improvement for {args.patience} epochs", flush=True)
            break

    checkpoint = torch.load(output / "residual_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    report: dict[str, Any] = {}
    for split in ("val_hard", "val_manual", "test_unseen"):
        indices = cache.indices(split)
        if len(indices):
            report[split] = evaluate(
                model, cache, indices, device, args.batch_size * 2, args.num_workers
            )
            print_metrics(f"FINAL {split}", report[split])
    save_json(output / "evaluation.json", report)
    print(
        "FINAL STAGE2C RESIDUAL FIVE STREAM | " + json.dumps({
            "best_epoch": best_epoch, "best_validation_r1": best_score,
            "hard_guardrail_floor": hard_floor,
            "stage2c_sha256": stage2c_info["sha256"], "results": report,
        }, ensure_ascii=False), flush=True,
    )


if __name__ == "__main__":
    main()
