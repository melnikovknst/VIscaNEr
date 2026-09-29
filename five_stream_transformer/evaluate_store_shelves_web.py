#!/usr/bin/env python3
"""Evaluate frozen Stage-2C bottle DINO-B and the latest residual ranker.

The web shelf dataset is a strict holdout. Its annotated target coordinates are
never used for detector selection or cropping; they are retained only in the
saved prediction table for later error analysis.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

from five_stream_transformer.data import resolve_refs
from five_stream_transformer.dino import choose_device
from five_stream_transformer.features import extract_visual_streams, mine_candidates
from five_stream_transformer.model import STREAM_NAMES, FiveStreamResidualTransformer
from five_stream_transformer.ocr import fill_ocr_text
from five_stream_transformer.stage2c import _build_branch
from five_stream_transformer.text import frozen_char_ngram_embedding
from infer_wine import box_iou, crop_with_policy
from joint_yolo.infer import (
    pair_bottle,
    padded_crop,
    predict_detections,
    select_label_candidates,
)
from yolo_target_selection import confidence_axis_score


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def accepted_slugs(value: object) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(";") if part.strip())


def select_bottles(
    detections: list[dict[str, Any]],
    image_width: int,
    candidate_confidence: float,
    ambiguity_confidence: float,
    ambiguity_margin: float,
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    for detection in detections:
        if detection["class_id"] != 0 or detection["confidence"] < candidate_confidence:
            continue
        score, distance, proximity = confidence_axis_score(
            detection["confidence"], detection["box"], image_width, axis_x=0.50
        )
        candidates.append({**detection, "selection_score": score, "axis_distance": distance,
                           "axis_proximity": proximity})
    candidates.sort(key=lambda row: (row["selection_score"], row["confidence"]), reverse=True)
    if not candidates:
        return [], 0
    selected = [candidates[0]]
    if len(candidates) >= 2:
        first, second = candidates[:2]
        ambiguous = (
            min(first["confidence"], second["confidence"]) >= ambiguity_confidence
            and abs(first["selection_score"] - second["selection_score"]) <= ambiguity_margin
            and abs(first["axis_distance"] - second["axis_distance"]) <= 0.08
            and box_iou(first["box"], second["box"]) < 0.80
        )
        if ambiguous:
            selected.append(second)
    return selected, len(candidates)


def save_image(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=95)
    return str(path.resolve())


def prepare_queries(
    dataset_root: Path,
    detector_path: Path,
    output: Path,
    device: torch.device,
    candidate_confidence: float,
    bottle_threshold: float,
    ambiguity_margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    labels = pd.read_csv(dataset_root / "labels.csv").fillna("")
    required = {"query_id", "image_path", "scored", "accepted_slugs", "training_use"}
    if missing := required.difference(labels.columns):
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")
    if len(labels) != 130 or not labels["query_id"].is_unique:
        raise ValueError(f"Expected 130 unique holdout rows, got {len(labels)}")
    if set(labels["training_use"].astype(str)) != {"holdout"}:
        raise ValueError("store_shelves_web must remain a pure holdout")
    labels["positive_slugs"] = labels["accepted_slugs"].map(accepted_slugs)
    if not labels["positive_slugs"].map(bool).all():
        raise ValueError("Every scored row must have at least one accepted slug")

    detector = YOLO(str(detector_path.resolve()))
    names = detector.names
    class_zero = names.get(0) if isinstance(names, dict) else names[0]
    class_one = names.get(1) if isinstance(names, dict) else names[1]
    if str(class_zero).lower() != "bottle" or "label" not in str(class_one).lower():
        raise ValueError(f"Unexpected joint YOLO classes: {names}")
    yolo_device: str | int = device.index or 0 if device.type == "cuda" else device.type

    bottle_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    diagnostics = {
        "images": 0, "no_bottle_detection": 0, "no_label_detection": 0,
        "bottle_original_fallback": 0, "ambiguous_bottle_images": 0,
        "ambiguous_label_images": 0,
    }
    crop_root = output / "crops"
    print(f"YOLO | joint detector | 0/{len(labels)} images", flush=True)
    for number, (_, row) in enumerate(labels.iterrows(), start=1):
        source = dataset_root / "queries" / str(row["image_path"])
        if not source.is_file() or source.suffix.lower() not in SUPPORTED:
            raise FileNotFoundError(source)
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        detections = predict_detections(detector, image, yolo_device, candidate_confidence)
        bottles = [item for item in detections if item["class_id"] == 0]
        label_detections = [item for item in detections if item["class_id"] == 1]
        selected_bottles, bottle_candidate_count = select_bottles(
            detections, image.width, candidate_confidence, bottle_threshold, ambiguity_margin
        )
        if not selected_bottles:
            diagnostics["no_bottle_detection"] += 1
        if len(selected_bottles) == 2:
            diagnostics["ambiguous_bottle_images"] += 1
        bottle_views = selected_bottles or [None]
        for view_number, selected in enumerate(bottle_views, start=1):
            crop, mode, crop_box, confidence = crop_with_policy(
                image, selected, bottle_threshold, 0.06
            )
            if mode != "yolo_bottle_crop":
                diagnostics["bottle_original_fallback"] += 1
            path = save_image(crop, crop_root / "bottle_only" / f"{row['query_id']}__v{view_number}.jpg")
            bottle_rows.append({
                "query_id": str(row["query_id"]), "positive_slugs": row["positive_slugs"],
                "path": path, "view": view_number, "mode": mode,
                "confidence": confidence, "crop_box": crop_box,
                "candidate_count": bottle_candidate_count,
            })

        selected_labels, label_ambiguous = select_label_candidates(
            label_detections,
            (image.width * 0.5, image.height * 0.5),
            math.hypot(image.width, image.height), ambiguity_margin,
            image_width=image.width,
        )
        if not selected_labels:
            diagnostics["no_label_detection"] += 1
            # Preserve coverage without consulting GT: use the full frame as a
            # weak label view and the production bottle-only view as bottle input.
            selected_labels = [None]
        if label_ambiguous:
            diagnostics["ambiguous_label_images"] += 1
        for view_number, label in enumerate(selected_labels, start=1):
            if label is None:
                label_crop = image.copy()
                label_box = None
                label_confidence = None
                paired = selected_bottles[0] if selected_bottles else None
            else:
                label_crop, label_box = padded_crop(image, label["box"], 0.10)
                label_confidence = label["confidence"]
                paired = pair_bottle(label, bottles)
            bottle_crop, bottle_mode, bottle_box, bottle_confidence = crop_with_policy(
                image, paired, bottle_threshold, 0.06
            )
            label_path = save_image(
                label_crop, crop_root / "pairs" / f"{row['query_id']}__v{view_number}__label.jpg"
            )
            bottle_path = save_image(
                bottle_crop, crop_root / "pairs" / f"{row['query_id']}__v{view_number}__bottle.jpg"
            )
            pair_rows.append({
                "sample_id": f"{row['query_id']}::v{view_number}",
                "query_id": str(row["query_id"]), "positive_slugs": row["positive_slugs"],
                "source_group": "store_shelves_web", "label_path": label_path,
                "bottle_path": bottle_path, "ocr_text": "",
                "label_confidence": label_confidence, "label_box": label_box,
                "bottle_confidence": bottle_confidence, "bottle_box": bottle_box,
                "bottle_mode": bottle_mode,
            })
        diagnostics["images"] += 1
        if number % 13 == 0 or number == len(labels):
            print(f"YOLO | {number}/{len(labels)} ({100*number/len(labels):5.1f}%)", flush=True)
    del detector
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    labels.to_csv(output / "holdout_labels.csv", index=False)
    return pd.DataFrame(bottle_rows), pd.DataFrame(pair_rows), diagnostics


def load_stage2c_fast(
    checkpoint_path: Path, device: torch.device, image_size: int
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"epoch", "label_model_state_dict", "bottle_model_state_dict"}
    if missing := required.difference(payload):
        raise ValueError(f"Incomplete Stage-2C checkpoint: {sorted(missing)}")
    label = _build_branch(payload["label_model_state_dict"], image_size, device)
    bottle = _build_branch(payload["bottle_model_state_dict"], image_size, device)
    return label, bottle, {
        "path": str(checkpoint_path), "epoch": int(payload["epoch"]),
        "num_classes": int(payload["bottle_model_state_dict"]["classifier.weight"].shape[0]),
        "embedding_dim": int(payload["bottle_model_state_dict"]["classifier.weight"].shape[1]),
        "hash_check": "skipped_for_fast_evaluation",
    }


def first_rank(order: Iterable[str], positives: set[str], missing_rank: int) -> int:
    for rank, slug in enumerate(order, start=1):
        if slug in positives:
            return rank
    return missing_rank


def summarize(ranks: list[int], candidate_recall: float | None = None) -> dict[str, Any]:
    tensor = torch.tensor(ranks, dtype=torch.float32)
    missing_rank = 2104
    result: dict[str, Any] = {
        "num_queries": len(ranks),
        "accuracy": float((tensor == 1).float().mean()),
        "mrr": float(torch.where(tensor < missing_rank, 1.0 / tensor, 0.0).mean()),
        "mean_rank": float(tensor.mean()),
    }
    for k in (1, 2, 5, 10):
        result[f"recall_at_{k}"] = float((tensor <= k).float().mean())
    if candidate_recall is not None:
        result["candidate_recall"] = float(candidate_recall)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--label-data-root", type=Path, required=True)
    parser.add_argument("--bottle-data-root", type=Path, required=True)
    parser.add_argument("--stage2c-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-yolo", type=Path, required=True)
    parser.add_argument("--easyocr-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--feature-batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--candidate-confidence", type=float, default=0.05)
    parser.add_argument("--bottle-threshold", type=float, default=0.75)
    parser.add_argument("--ambiguity-margin", type=float, default=0.06)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Fast evaluation requires a CUDA GPU")
    print("=" * 112, flush=True)
    print("STORE SHELVES WEB | STRICT HOLDOUT EVALUATION", flush=True)
    print("LEAKAGE GUARD | labels x/y are not used for YOLO selection or crops", flush=True)

    bottle_rows, pair_rows, diagnostics = prepare_queries(
        args.dataset_root.resolve(), args.joint_yolo.resolve(), output, device,
        args.candidate_confidence, args.bottle_threshold, args.ambiguity_margin,
    )
    pair_rows = fill_ocr_text(
        pair_rows, output / "ocr_cache.csv", {"store_shelves_web"},
        args.easyocr_model_dir.resolve(), gpu=True, cache_every=20,
    )
    pair_rows.assign(
        positive_slugs=pair_rows["positive_slugs"].map(lambda values: ";".join(values))
    ).to_csv(output / "query_pairs.csv", index=False)

    label_slugs, label_refs = resolve_refs(args.label_data_root.resolve())
    bottle_slugs, bottle_refs = resolve_refs(args.bottle_data_root.resolve())
    if label_slugs != bottle_slugs:
        raise ValueError("Label and bottle galleries are not aligned")
    gallery_slugs = label_slugs
    gallery_set = set(gallery_slugs)
    holdout = pd.read_csv(args.dataset_root / "labels.csv").fillna("")
    holdout["positive_slugs"] = holdout["accepted_slugs"].map(accepted_slugs)
    missing_targets = sorted({slug for values in holdout["positive_slugs"] for slug in values} - gallery_set)
    if missing_targets:
        raise ValueError(f"Holdout targets absent from gallery: {missing_targets[:10]}")

    print("DINO | loading complete Stage-2C label+bottle branches (no SHA pass)", flush=True)
    label_model, bottle_model, stage2c_info = load_stage2c_fast(
        args.stage2c_checkpoint.resolve(), device, args.image_size
    )
    bottle_only_raw, bottle_only_embedding = extract_visual_streams(
        bottle_model, bottle_rows["path"].tolist(), args.image_size, device,
        args.feature_batch_size, args.num_workers, "holdout_bottle_only_queries",
    )
    del bottle_only_raw
    label_raw, label_embedding = extract_visual_streams(
        label_model, pair_rows["label_path"].tolist(), args.image_size, device,
        args.feature_batch_size, args.num_workers, "holdout_label_queries",
    )
    bottle_raw, bottle_embedding = extract_visual_streams(
        bottle_model, pair_rows["bottle_path"].tolist(), args.image_size, device,
        args.feature_batch_size, args.num_workers, "holdout_pair_bottle_queries",
    )
    label_ref_raw, label_ref_embedding = extract_visual_streams(
        label_model, label_refs, args.image_size, device,
        args.feature_batch_size, args.num_workers, "label_gallery",
    )
    bottle_ref_raw, bottle_ref_embedding = extract_visual_streams(
        bottle_model, bottle_refs, args.image_size, device,
        args.feature_batch_size, args.num_workers, "bottle_gallery",
    )
    del label_model, bottle_model
    gc.collect()
    torch.cuda.empty_cache()

    # Honest Stage-2C bottle retrieval across the complete gallery. Ambiguous
    # views are fused by maximum cosine similarity exactly as in production.
    bottle_scores = bottle_only_embedding.float() @ bottle_ref_embedding.float().T
    bottle_ranks: list[int] = []
    bottle_top10: dict[str, list[str]] = {}
    for query_id, group in bottle_rows.groupby("query_id", sort=False):
        fused = bottle_scores[group.index].max(dim=0).values
        order_ids = fused.argsort(descending=True).tolist()
        order = [gallery_slugs[index] for index in order_ids]
        positives = set(group.iloc[0]["positive_slugs"])
        bottle_ranks.append(first_rank(order, positives, len(gallery_slugs) + 1))
        bottle_top10[str(query_id)] = order[:10]

    ocr_texts = pair_rows["ocr_text"].fillna("").astype(str).tolist()
    query_ocr = frozen_char_ngram_embedding(ocr_texts).half()
    gallery_ocr = frozen_char_ngram_embedding(gallery_slugs).half()
    ocr_available = torch.tensor([bool(text.strip()) for text in ocr_texts], dtype=torch.bool)
    checkpoint = torch.load(args.residual_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "stage2c-five-stream-residual-v1":
        raise ValueError("Residual checkpoint format mismatch")
    architecture = checkpoint["architecture"]
    max_candidates = int(checkpoint["model_state_dict"]["rank_embedding.weight"].shape[0])
    top_k = int(checkpoint.get("args", {}).get("top_k", max_candidates // 3))
    model = FiveStreamResidualTransformer(
        model_dim=int(architecture["model_dim"]), num_heads=int(architecture["num_heads"]),
        num_layers=int(architecture["num_layers"]),
        feedforward_dim=int(architecture["feedforward_dim"]),
        dropout=float(architecture["dropout"]), max_candidates=max_candidates,
        base_temperature=float(architecture["base_temperature"]),
        max_residual=float(architecture["max_residual"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    candidate_ids, candidate_mask, _, natural = mine_candidates(
        label_embedding, bottle_embedding, query_ocr,
        label_ref_embedding, bottle_ref_embedding, gallery_ocr,
        pair_rows["positive_slugs"].tolist(), gallery_slugs, ocr_available, top_k,
    )
    query_streams = {
        "bottle_crop": bottle_raw.float().to(device),
        "label_crop": label_raw.float().to(device),
        "bottle_dino": bottle_embedding.float().to(device),
        "label_dino": label_embedding.float().to(device),
        "ocr_text": query_ocr.float().to(device),
    }
    gallery_streams = {
        "bottle_crop": bottle_ref_raw,
        "label_crop": label_ref_raw,
        "bottle_dino": bottle_ref_embedding,
        "label_dino": label_ref_embedding,
        "ocr_text": gallery_ocr,
    }
    references = {
        name: gallery_streams[name][candidate_ids].float().to(device)
        for name in STREAM_NAMES
    }
    available = torch.ones(len(pair_rows), len(STREAM_NAMES), dtype=torch.bool)
    available[:, STREAM_NAMES.index("ocr_text")] = ocr_available
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        final_logits, _, _ = model(
            query_streams, references, candidate_mask.to(device), available.to(device)
        )
    final_logits = final_logits.float().cpu()

    # Fuse one/two label-pair views by maximum final logit for every candidate.
    transformer_ranks: list[int] = []
    transformer_top10: dict[str, list[str]] = {}
    candidate_hit_by_query: dict[str, bool] = {}
    for query_id, group in pair_rows.groupby("query_id", sort=False):
        score_by_slug: dict[str, float] = {}
        positives = set(group.iloc[0]["positive_slugs"])
        hit = False
        for index in group.index:
            valid_ids = candidate_ids[index][candidate_mask[index]].tolist()
            valid_scores = final_logits[index][candidate_mask[index]].tolist()
            for candidate, score in zip(valid_ids, valid_scores, strict=True):
                slug = gallery_slugs[candidate]
                score_by_slug[slug] = max(score_by_slug.get(slug, float("-inf")), float(score))
            hit = hit or bool(natural[index])
        order = sorted(score_by_slug, key=score_by_slug.get, reverse=True)
        transformer_ranks.append(first_rank(order, positives, len(gallery_slugs) + 1))
        transformer_top10[str(query_id)] = order[:10]
        candidate_hit_by_query[str(query_id)] = hit

    bottle_metrics = summarize(bottle_ranks)
    transformer_metrics = summarize(
        transformer_ranks,
        sum(candidate_hit_by_query.values()) / max(len(candidate_hit_by_query), 1),
    )
    fixed = sum(base != 1 and learned == 1 for base, learned in zip(bottle_ranks, transformer_ranks, strict=True))
    harmed = sum(base == 1 and learned != 1 for base, learned in zip(bottle_ranks, transformer_ranks, strict=True))
    predictions: list[dict[str, Any]] = []
    for row, bottle_rank, transformer_rank in zip(
        holdout.to_dict("records"), bottle_ranks, transformer_ranks, strict=True
    ):
        query_id = str(row["query_id"])
        predictions.append({
            "query_id": query_id, "image_path": row["image_path"],
            "accepted_slugs": row["accepted_slugs"], "note": row.get("note", ""),
            "annotated_x": row.get("x", ""), "annotated_y": row.get("y", ""),
            "stage2c_bottle_rank": bottle_rank,
            "stage2c_bottle_top10": ";".join(bottle_top10[query_id]),
            "residual_transformer_rank": transformer_rank,
            "residual_transformer_top10": ";".join(transformer_top10[query_id]),
            "transformer_candidate_hit": candidate_hit_by_query[query_id],
        })
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    report = {
        "dataset": {"name": "store_shelves_web", "rows": len(holdout), "training_use": "holdout"},
        "leakage_guard": "annotated x/y excluded from detection, cropping and ranking",
        "models": {
            "stage2c": stage2c_info,
            "residual_transformer": {
                "path": str(args.residual_checkpoint), "format": checkpoint["format"],
                "epoch": int(checkpoint["epoch"]), "top_k_per_stream": top_k,
                "max_candidates": max_candidates,
            },
            "joint_yolo": str(args.joint_yolo),
        },
        "detection": diagnostics,
        "results": {
            "stage2c_bottle_dino_b": bottle_metrics,
            "latest_residual_five_stream_transformer": transformer_metrics,
            "comparison": {"fixed_top1": fixed, "harmed_top1": harmed, "net_top1": fixed - harmed},
        },
    }
    save_json(output / "evaluation.json", report)
    print("=" * 112, flush=True)
    print("FINAL STORE SHELVES WEB | " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
