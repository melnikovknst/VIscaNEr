#!/usr/bin/env python3
"""Domain-adapt a completed Stage-II model on reviewed real shelf photos.

Only rows with reviewed catalog identities are optimized.  Out-of-catalog and
unsure rows remain in the output audit, but are never assigned invented class
labels.  A deterministic within-identity validation subset measures adaptation;
the original Stage-II hard validation split is a forgetting guardrail.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cascade_resolver.modeling import choose_device, embed_paths, load_retrieval_model
from dinov3_retrieval import build_transforms
from fusion_stage2.core import (
    FEATURE_NAMES,
    FusionMLP,
    build_feature_cache,
    load_manifest,
    ranking_metrics,
    resolve_refs,
    score_cache,
)


class ManualDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, train: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        train_transform, eval_transform = build_transforms(image_size)
        self.transform = train_transform if train else eval_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(str(row["label_path"])) as image:
            label = self.transform(image.convert("RGB"))
        with Image.open(str(row["bottle_path"])) as image:
            bottle = self.transform(image.convert("RGB"))
        return {"label": label, "bottle": bottle, "index": index}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_manual_frame(root: Path, gallery_slugs: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(root / "manifest.csv").fillna("")
    if len(frame) != 211:
        raise ValueError(f"Expected 211 manual rows, found {len(frame)}")
    for column in ("label_path", "bottle_path", "accepted_slugs", "trainable_catalog"):
        if column not in frame:
            raise ValueError(f"Manual-211 manifest is missing {column}")
    frame["label_path"] = frame["label_path"].map(lambda value: str((root / str(value)).resolve()))
    frame["bottle_path"] = frame["bottle_path"].map(lambda value: str((root / str(value)).resolve()))
    gallery = set(gallery_slugs)
    frame["positive_slugs"] = frame["accepted_slugs"].map(
        lambda value: tuple(part.strip() for part in str(value).split(";") if part.strip())
    )
    missing = sorted({slug for values in frame["positive_slugs"] for slug in values if slug not in gallery})
    if missing:
        raise ValueError(f"Reviewed slugs missing from gallery: {missing}")
    frame["trainable_catalog"] = frame["trainable_catalog"].astype(str).str.lower().isin({"true", "1"})
    return frame


def split_manual(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = frame[frame["trainable_catalog"] & frame["positive_slugs"].map(bool)].copy()
    usable["split_key"] = usable["positive_slugs"].map(lambda values: values[0])
    rng = random.Random(seed)
    validation_indices: list[int] = []
    for _, group in usable.groupby("split_key", sort=True):
        indices = group.index.tolist()
        rng.shuffle(indices)
        if len(indices) >= 2:
            validation_indices.append(indices[0])
    validation = usable.loc[sorted(validation_indices)].copy()
    training = usable.drop(index=validation_indices).copy()
    if training.empty or validation.empty:
        raise ValueError("Manual train/validation split is empty")
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def positive_id_lists(frame: pd.DataFrame, slug_to_id: dict[str, int]) -> list[list[int]]:
    return [[slug_to_id[slug] for slug in values] for values in frame["positive_slugs"]]


def multi_positive_ce(logits: torch.Tensor, positives: list[list[int]]) -> torch.Tensor:
    losses = []
    for row, ids in zip(logits, positives, strict=True):
        positive = torch.as_tensor(ids, device=row.device, dtype=torch.long)
        losses.append(torch.logsumexp(row, dim=0) - torch.logsumexp(row[positive], dim=0))
    return torch.stack(losses).mean()


def candidate_features(
    label_embeddings: torch.Tensor,
    bottle_embeddings: torch.Tensor,
    label_gallery: torch.Tensor,
    bottle_gallery: torch.Tensor,
    frame: pd.DataFrame,
    row_indices: list[int],
    top_k: int,
    positive_ids: list[list[int]] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    label_scores = F.normalize(label_embeddings.float(), dim=1) @ F.normalize(label_gallery.float(), dim=1).T
    bottle_scores = F.normalize(bottle_embeddings.float(), dim=1) @ F.normalize(bottle_gallery.float(), dim=1).T
    label_values, label_top = label_scores.topk(top_k, dim=1)
    bottle_values, bottle_top = bottle_scores.topk(top_k, dim=1)
    unions: list[list[int]] = []
    for local in range(len(row_indices)):
        union: list[int] = []
        for candidate in torch.cat([label_top[local], bottle_top[local]]).tolist():
            if int(candidate) not in union:
                union.append(int(candidate))
        if positive_ids is not None:
            for candidate in positive_ids[local]:
                if candidate not in union:
                    union.append(candidate)
        unions.append(union)
    width = max(len(values) for values in unions)
    candidate_ids = torch.full((len(unions), width), -1, dtype=torch.long, device=label_scores.device)
    mask = torch.zeros((len(unions), width), dtype=torch.bool, device=label_scores.device)
    features = torch.zeros((len(unions), width, len(FEATURE_NAMES)), device=label_scores.device)
    for local, ids in enumerate(unions):
        count = len(ids)
        tensor_ids = torch.as_tensor(ids, dtype=torch.long, device=label_scores.device)
        candidate_ids[local, :count] = tensor_ids
        mask[local, :count] = True
        label_candidate = label_scores[local, tensor_ids]
        bottle_candidate = bottle_scores[local, tensor_ids]
        label_rank = {int(value): rank + 1 for rank, value in enumerate(label_top[local].tolist())}
        bottle_rank = {int(value): rank + 1 for rank, value in enumerate(bottle_top[local].tolist())}
        row = frame.iloc[row_indices[local]]
        agreement = float(int(label_top[local, 0]) == int(bottle_top[local, 0]))
        label_conf = float(row.get("label_confidence", 0.0) or 0.0)
        bottle_conf = float(row.get("bottle_confidence", 0.0) or 0.0)
        original = float(str(row.get("bottle_image_mode", "")).startswith("original_"))
        for position, candidate in enumerate(ids):
            lr, br = label_rank.get(candidate, 0), bottle_rank.get(candidate, 0)
            features[local, position] = torch.stack([
                label_candidate[position], bottle_candidate[position],
                label_candidate[position] - label_values[local, 0],
                bottle_candidate[position] - bottle_values[local, 0],
                label_candidate.new_tensor(1.0 / lr if lr else 0.0),
                label_candidate.new_tensor(1.0 / br if br else 0.0),
                label_candidate.new_tensor(float(lr == 1)), label_candidate.new_tensor(float(br == 1)),
                label_candidate.new_tensor(label_conf), label_candidate.new_tensor(bottle_conf),
                label_candidate.new_tensor(0.0), label_candidate.new_tensor(0.0),
                label_candidate.new_tensor(0.0), label_candidate.new_tensor(original),
                label_candidate.new_tensor(agreement),
                label_candidate.new_tensor(0.0), label_candidate.new_tensor(0.0),
                label_candidate.new_tensor(0.0), label_candidate.new_tensor(0.0),
            ])
    return features, mask, candidate_ids, label_scores, bottle_scores


def positive_rank(scores: torch.Tensor, positives: list[list[int]]) -> torch.Tensor:
    ranks = []
    for row, ids in zip(scores, positives, strict=True):
        best = min(1 + int((row > row[index]).sum()) for index in ids)
        ranks.append(best)
    return torch.tensor(ranks, dtype=torch.long)


@torch.inference_mode()
def evaluate_manual(
    frame: pd.DataFrame,
    label_model: torch.nn.Module,
    bottle_model: torch.nn.Module,
    fusion: FusionMLP,
    label_gallery: torch.Tensor,
    bottle_gallery: torch.Tensor,
    slug_to_id: dict[str, int],
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    top_k: int,
) -> dict[str, Any]:
    label_model.eval(); bottle_model.eval(); fusion.eval()
    dataset = ManualDataset(frame, image_size, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    label_ranks, bottle_ranks, fusion_ranks = [], [], []
    all_positives = positive_id_lists(frame, slug_to_id)
    for batch in loader:
        indices = batch["index"].tolist()
        label_embeddings, _ = label_model(batch["label"].to(device))
        bottle_embeddings, _ = bottle_model(batch["bottle"].to(device))
        positives = [all_positives[index] for index in indices]
        features, mask, candidate_ids, label_scores, bottle_scores = candidate_features(
            label_embeddings, bottle_embeddings, label_gallery, bottle_gallery,
            frame, indices, top_k, None,
        )
        fusion_logits = fusion(features, mask)
        order = fusion_logits.argsort(dim=1, descending=True)
        for local, ids in enumerate(positives):
            positions = [
                int(match[0]) for positive in ids
                for match in (candidate_ids[local].eq(positive).nonzero(as_tuple=False),)
                if len(match)
            ]
            if positions:
                ranked_positions = order[local].tolist()
                fusion_ranks.append(min(ranked_positions.index(position) + 1 for position in positions))
            else:
                fusion_ranks.append(len(slug_to_id) + 1)
        label_ranks.append(positive_rank(label_scores.cpu(), positives))
        bottle_ranks.append(positive_rank(bottle_scores.cpu(), positives))
    return {
        "fusion_joint": ranking_metrics(torch.tensor(fusion_ranks)),
        "label_dino_b": ranking_metrics(torch.cat(label_ranks)),
        "bottle_dino_b": ranking_metrics(torch.cat(bottle_ranks)),
    }


def refresh_galleries(
    label_model: torch.nn.Module,
    bottle_model: torch.nn.Module,
    label_paths: list[str],
    bottle_paths: list[str],
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    label_model.eval(); bottle_model.eval()
    common = dict(device=device, image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    label_gallery, label_valid, label_errors, _ = embed_paths(label_model, label_paths, description="manual_211_label_gallery", **common)
    bottle_gallery, bottle_valid, bottle_errors, _ = embed_paths(bottle_model, bottle_paths, description="manual_211_bottle_gallery", **common)
    if not bool(label_valid.all()) or not bool(bottle_valid.all()):
        raise RuntimeError(f"Gallery decode failure: label={label_errors}, bottle={bottle_errors}")
    return label_gallery.to(device), bottle_gallery.to(device)


def evaluate_hard_val(
    hard_frame: pd.DataFrame,
    label_model: torch.nn.Module,
    bottle_model: torch.nn.Module,
    fusion: FusionMLP,
    label_gallery: torch.Tensor,
    bottle_gallery: torch.Tensor,
    slugs: list[str],
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    top_k: int,
) -> dict[str, Any]:
    label_model.eval(); bottle_model.eval(); fusion.eval()
    common = dict(device=device, image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    label_embeddings, valid_l, errors_l, _ = embed_paths(label_model, hard_frame["label_path"].tolist(), description="manual_211_hard_label", **common)
    bottle_embeddings, valid_b, errors_b, _ = embed_paths(bottle_model, hard_frame["bottle_path"].tolist(), description="manual_211_hard_bottle", **common)
    if not bool((valid_l & valid_b).all()):
        raise RuntimeError(f"Hard-val decode failure: label={errors_l}, bottle={errors_b}")
    cache = build_feature_cache(
        hard_frame, label_embeddings, bottle_embeddings,
        label_gallery.cpu(), bottle_gallery.cpu(), slugs, top_k=top_k,
    )
    metrics, _ = score_cache(fusion, cache, device)
    return metrics


def main() -> None:
    args = parse_args(); seed_everything(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    slugs, label_ref_paths = resolve_refs(Path(args.label_refs_root))
    bottle_slugs, bottle_ref_paths = resolve_refs(Path(args.bottle_refs_root))
    if slugs != bottle_slugs:
        raise ValueError("Label and bottle galleries are not aligned")
    slug_to_id = {slug: index for index, slug in enumerate(slugs)}
    manual = load_manual_frame(Path(args.manual_root), slugs)
    train_frame, val_frame = split_manual(manual, args.seed)
    hard = load_manifest(args.hard_manifest, args.hard_root, args.label_data_root, args.bottle_data_root)
    hard_val = hard[hard["fusion_split"].eq("hard_val")].reset_index(drop=True)

    label_model, label_info = load_retrieval_model(args.label_base_checkpoint, "vitb16", device)
    bottle_model, bottle_info = load_retrieval_model(args.bottle_base_checkpoint, "vitb16", device)
    joint = torch.load(args.joint_checkpoint, map_location="cpu", weights_only=False)
    label_model.load_state_dict(joint["label_model_state_dict"], strict=True)
    bottle_model.load_state_dict(joint["bottle_model_state_dict"], strict=True)
    fusion_state = joint["fusion_state_dict"]
    hidden_dim = int(fusion_state["network.1.weight"].shape[0])
    fusion = FusionMLP(hidden_dim=hidden_dim, dropout=args.dropout).to(device)
    fusion.load_state_dict(fusion_state, strict=True)
    for model in (label_model, bottle_model):
        model.set_backbone_trainable(args.unfreeze_last_blocks)

    image_size = int(label_info["image_size"])
    if int(bottle_info["image_size"]) != image_size:
        raise ValueError("Branch image sizes differ")
    baseline_label_gallery, baseline_bottle_gallery = refresh_galleries(
        label_model, bottle_model, label_ref_paths, bottle_ref_paths,
        image_size, device, args.eval_batch_size, args.num_workers,
    )
    baseline_manual = evaluate_manual(
        val_frame, label_model, bottle_model, fusion,
        baseline_label_gallery, baseline_bottle_gallery, slug_to_id,
        image_size, device, args.eval_batch_size, args.num_workers, args.top_k,
    )
    baseline_hard = evaluate_hard_val(
        hard_val, label_model, bottle_model, fusion,
        baseline_label_gallery, baseline_bottle_gallery, slugs,
        image_size, device, args.eval_batch_size, args.num_workers, args.top_k,
    )
    print("BASELINE |", json.dumps({"manual_val": baseline_manual, "hard_val": baseline_hard}), flush=True)

    train_dataset = ManualDataset(train_frame, image_size, train=True)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    parameter_groups = [
        {"params": fusion.parameters(), "lr": args.fusion_lr},
        {"params": list(label_model.projection.parameters()) + list(label_model.classifier.parameters()), "lr": args.head_lr},
        {"params": list(bottle_model.projection.parameters()) + list(bottle_model.classifier.parameters()), "lr": args.head_lr},
        {"params": [p for p in label_model.backbone.parameters() if p.requires_grad], "lr": args.backbone_lr},
        {"params": [p for p in bottle_model.backbone.parameters() if p.requires_grad], "lr": args.backbone_lr},
    ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    train_positives = positive_id_lists(train_frame, slug_to_id)
    best_score, bad, history = -math.inf, 0, []
    hard_floor = float(baseline_hard["recall_at_1"]) - args.maximum_hard_val_drop
    for epoch in range(1, args.epochs + 1):
        label_gallery, bottle_gallery = refresh_galleries(
            label_model, bottle_model, label_ref_paths, bottle_ref_paths,
            image_size, device, args.eval_batch_size, args.num_workers,
        )
        fusion.train(); label_model.train(); bottle_model.train()
        totals = {"loss": 0.0, "fusion": 0.0, "label": 0.0, "bottle": 0.0, "rows": 0}
        for batch in loader:
            indices = batch["index"].tolist()
            positives = [train_positives[index] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda" and args.amp):
                label_embeddings, label_logits = label_model(batch["label"].to(device))
                bottle_embeddings, bottle_logits = bottle_model(batch["bottle"].to(device))
                features, mask, candidate_ids, _, _ = candidate_features(
                    label_embeddings, bottle_embeddings, label_gallery, bottle_gallery,
                    train_frame, indices, args.top_k, positives,
                )
                fusion_logits = fusion(features, mask)
                candidate_positives = []
                for local, ids in enumerate(positives):
                    positions = [int(candidate_ids[local].eq(value).nonzero(as_tuple=False)[0, 0]) for value in ids]
                    candidate_positives.append(positions)
                fusion_loss = multi_positive_ce(fusion_logits, candidate_positives)
                label_loss = multi_positive_ce(label_logits, positives)
                bottle_loss = multi_positive_ce(bottle_logits, positives)
                loss = fusion_loss + args.branch_loss_weight * (label_loss + bottle_loss)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(fusion.parameters()) + list(label_model.parameters()) + list(bottle_model.parameters()), 2.0
            )
            scaler.step(optimizer); scaler.update()
            count = len(indices); totals["rows"] += count
            for key, value in (("loss", loss), ("fusion", fusion_loss), ("label", label_loss), ("bottle", bottle_loss)):
                totals[key] += float(value.detach()) * count

        label_gallery, bottle_gallery = refresh_galleries(
            label_model, bottle_model, label_ref_paths, bottle_ref_paths,
            image_size, device, args.eval_batch_size, args.num_workers,
        )
        manual_metrics = evaluate_manual(
            val_frame, label_model, bottle_model, fusion,
            label_gallery, bottle_gallery, slug_to_id,
            image_size, device, args.eval_batch_size, args.num_workers, args.top_k,
        )
        hard_metrics = evaluate_hard_val(
            hard_val, label_model, bottle_model, fusion,
            label_gallery, bottle_gallery, slugs,
            image_size, device, args.eval_batch_size, args.num_workers, args.top_k,
        )
        manual_r1 = float(manual_metrics["fusion_joint"]["recall_at_1"])
        hard_r1 = float(hard_metrics["recall_at_1"])
        eligible = hard_r1 >= hard_floor
        score = manual_r1 + args.hard_val_weight * hard_r1 if eligible else -math.inf
        improved = score > best_score + args.min_delta
        if improved:
            best_score, bad = score, 0
            atomic_save({
                "format": "manual-stage2c-v1", "epoch": epoch,
                "fusion_state_dict": fusion.state_dict(),
                "label_model_state_dict": label_model.state_dict(),
                "bottle_model_state_dict": bottle_model.state_dict(),
                "manual_val": manual_metrics, "hard_val": hard_metrics,
                "baseline_hard_val": baseline_hard, "baseline_manual_val": baseline_manual,
                "train_rows": len(train_frame), "manual_val_rows": len(val_frame),
                "feature_names": FEATURE_NAMES,
            }, output / "manual_stage2c_best.pt")
        else:
            bad += 1
        row = {
            "epoch": epoch,
            **{key: value / max(totals["rows"], 1) for key, value in totals.items() if key != "rows"},
            "manual_val": manual_metrics, "hard_val": hard_metrics,
            "guardrail_passed": eligible, "best_score": best_score, "patience": bad,
        }
        history.append(row); save_json(output / "history.json", history)
        print(
            f"EPOCH {epoch:02d}/{args.epochs} | loss={row['loss']:.4f} "
            f"fusion={row['fusion']:.4f} label={row['label']:.4f} bottle={row['bottle']:.4f} "
            f"| manual_val R@1={100*manual_r1:.2f}% R@2={100*manual_metrics['fusion_joint']['recall_at_2']:.2f}% "
            f"| hard_val R@1={100*hard_r1:.2f}% floor={100*hard_floor:.2f}% "
            f"| best={best_score:.4f} patience={bad}/{args.patience}", flush=True,
        )
        if epoch >= args.min_epochs and bad >= args.patience:
            print("EARLY STOPPING | manual Stage 2C adaptation", flush=True); break

    if not (output / "manual_stage2c_best.pt").is_file():
        raise RuntimeError("No checkpoint passed the hard-val forgetting guardrail")
    best = torch.load(output / "manual_stage2c_best.pt", map_location="cpu", weights_only=False)
    save_json(output / "final_metrics.json", {
        "manual_rows_total": len(manual), "trainable_catalog_rows": int(manual["trainable_catalog"].sum()),
        "train_rows": len(train_frame), "manual_val_rows": len(val_frame),
        "best_epoch": int(best["epoch"]), "manual_val": best["manual_val"],
        "hard_val": best["hard_val"], "baseline_manual_val": baseline_manual,
        "baseline_hard_val": baseline_hard,
    })
    print("FINAL MANUAL STAGE 2C |", (output / "final_metrics.json").read_text(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-root", required=True)
    parser.add_argument("--hard-manifest", required=True)
    parser.add_argument("--hard-root", required=True)
    parser.add_argument("--label-data-root", required=True)
    parser.add_argument("--bottle-data-root", required=True)
    parser.add_argument("--label-refs-root", required=True)
    parser.add_argument("--bottle-refs-root", required=True)
    parser.add_argument("--label-base-checkpoint", required=True)
    parser.add_argument("--bottle-base-checkpoint", required=True)
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=2)
    parser.add_argument("--fusion-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-6)
    parser.add_argument("--backbone-lr", type=float, default=1e-7)
    parser.add_argument("--branch-loss-weight", type=float, default=0.25)
    parser.add_argument("--hard-val-weight", type=float, default=0.5)
    parser.add_argument("--maximum-hard-val-drop", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
