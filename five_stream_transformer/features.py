"""Frozen DINO/OCR feature extraction and candidate mining."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from five_stream_transformer.dino import ImagePathDataset, synchronize
from five_stream_transformer.model import STREAM_NAMES
from five_stream_transformer.text import frozen_char_ngram_embedding


@dataclass
class FiveStreamCache:
    query_streams: dict[str, torch.Tensor]
    gallery_streams: dict[str, torch.Tensor]
    query_stream_available: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_mask: torch.Tensor
    positive_mask: torch.Tensor
    natural_positive_in_pool: torch.Tensor
    splits: list[str]
    sample_ids: list[str]
    source_groups: list[str]
    slugs: list[str]
    ocr_texts: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.__dict__, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "FiveStreamCache":
        return cls(**torch.load(path, map_location="cpu", weights_only=False))

    def indices(self, split: str | Sequence[str]) -> torch.Tensor:
        wanted = {split} if isinstance(split, str) else set(split)
        return torch.tensor([index for index, value in enumerate(self.splits) if value in wanted])


@torch.inference_mode()
def extract_visual_streams(
    model: torch.nn.Module,
    paths: Sequence[str],
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pre-head DINO features and final retrieval embeddings."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(
        ImagePathDataset(paths, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    raw_chunks: list[torch.Tensor] = []
    embedding_chunks: list[torch.Tensor] = []
    total = len(loader)
    interval = max(1, total // 10)
    started = time.perf_counter()
    print(f"FEATURE | {description} | 0/{total} batches", flush=True)
    for number, batch in enumerate(loader, start=1):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            raw = model.backbone_features(pixels)
            embedding = F.normalize(model.projection(raw), dim=1)
        valid = batch["valid"].bool()
        raw = raw.float().cpu()
        embedding = embedding.float().cpu()
        raw[~valid] = 0
        embedding[~valid] = 0
        raw_chunks.append(raw.half())
        embedding_chunks.append(embedding.half())
        if number % interval == 0 or number == total:
            print(
                f"FEATURE | {description} | {number}/{total} ({100.0 * number / total:5.1f}%) "
                f"| elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    synchronize(device)
    return torch.cat(raw_chunks), torch.cat(embedding_chunks)


def mine_candidates(
    query_label: torch.Tensor,
    query_bottle: torch.Tensor,
    query_ocr: torch.Tensor,
    gallery_label: torch.Tensor,
    gallery_bottle: torch.Tensor,
    gallery_ocr: torch.Tensor,
    positive_slugs: Sequence[Sequence[str]],
    slugs: list[str],
    query_ocr_available: torch.Tensor,
    top_k: int,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    slug_to_id = {slug: index for index, slug in enumerate(slugs)}
    max_candidates = top_k * 3
    all_ids: list[list[int]] = []
    all_positive_masks: list[list[bool]] = []
    all_natural: list[bool] = []
    for start in range(0, len(query_label), chunk_size):
        stop = min(len(query_label), start + chunk_size)
        label_scores = query_label[start:stop].float() @ gallery_label.float().T
        bottle_scores = query_bottle[start:stop].float() @ gallery_bottle.float().T
        ocr_scores = query_ocr[start:stop].float() @ gallery_ocr.float().T
        label_top = label_scores.topk(top_k, dim=1).indices
        bottle_top = bottle_scores.topk(top_k, dim=1).indices
        ocr_top = ocr_scores.topk(top_k, dim=1).indices
        for local in range(stop - start):
            global_index = start + local
            pool = set(label_top[local].tolist()) | set(bottle_top[local].tolist())
            if bool(query_ocr_available[global_index]):
                pool |= set(ocr_top[local].tolist())
            ranked = sorted(
                pool,
                key=lambda candidate: max(
                    float(label_scores[local, candidate]),
                    float(bottle_scores[local, candidate]),
                    float(ocr_scores[local, candidate]) if bool(query_ocr_available[global_index]) else -1.0,
                ),
                reverse=True,
            )[:max_candidates]
            positives = {slug_to_id[slug] for slug in positive_slugs[global_index] if slug in slug_to_id}
            natural = any(candidate in positives for candidate in ranked)
            all_ids.append(ranked)
            all_positive_masks.append([candidate in positives for candidate in ranked])
            all_natural.append(natural)
    width = max(len(row) for row in all_ids)
    candidate_ids = torch.zeros(len(all_ids), width, dtype=torch.long)
    candidate_mask = torch.zeros(len(all_ids), width, dtype=torch.bool)
    positive_mask = torch.zeros(len(all_ids), width, dtype=torch.bool)
    for row, (ids, positives) in enumerate(zip(all_ids, all_positive_masks, strict=True)):
        candidate_ids[row, : len(ids)] = torch.tensor(ids)
        candidate_mask[row, : len(ids)] = True
        positive_mask[row, : len(ids)] = torch.tensor(positives)
    return candidate_ids, candidate_mask, positive_mask, torch.tensor(all_natural, dtype=torch.bool)


def build_cache(
    frame: pd.DataFrame,
    label_model: torch.nn.Module,
    bottle_model: torch.nn.Module,
    label_ref_paths: list[str],
    bottle_ref_paths: list[str],
    slugs: list[str],
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    top_k: int,
) -> FiveStreamCache:
    label_raw, label_embedding = extract_visual_streams(
        label_model, frame["label_path"].tolist(), image_size, device,
        batch_size, num_workers, "label_queries",
    )
    bottle_raw, bottle_embedding = extract_visual_streams(
        bottle_model, frame["bottle_path"].tolist(), image_size, device,
        batch_size, num_workers, "bottle_queries",
    )
    label_ref_raw, label_ref_embedding = extract_visual_streams(
        label_model, label_ref_paths, image_size, device,
        batch_size, num_workers, "label_gallery",
    )
    bottle_ref_raw, bottle_ref_embedding = extract_visual_streams(
        bottle_model, bottle_ref_paths, image_size, device,
        batch_size, num_workers, "bottle_gallery",
    )
    ocr_texts = frame["ocr_text"].fillna("").astype(str).tolist()
    query_ocr = frozen_char_ngram_embedding(ocr_texts).half()
    gallery_ocr = frozen_char_ngram_embedding(slugs).half()
    ocr_available = torch.tensor([bool(text.strip()) for text in ocr_texts], dtype=torch.bool)
    candidate_ids, candidate_mask, positive_mask, natural = mine_candidates(
        label_embedding, bottle_embedding, query_ocr,
        label_ref_embedding, bottle_ref_embedding, gallery_ocr,
        frame["positive_slugs"].tolist(), slugs, ocr_available, top_k,
    )
    available = torch.ones(len(frame), len(STREAM_NAMES), dtype=torch.bool)
    available[:, STREAM_NAMES.index("ocr_text")] = ocr_available
    return FiveStreamCache(
        query_streams={
            "bottle_crop": bottle_raw,
            "label_crop": label_raw,
            "bottle_dino": bottle_embedding,
            "label_dino": label_embedding,
            "ocr_text": query_ocr,
        },
        gallery_streams={
            "bottle_crop": bottle_ref_raw,
            "label_crop": label_ref_raw,
            "bottle_dino": bottle_ref_embedding,
            "label_dino": label_ref_embedding,
            "ocr_text": gallery_ocr,
        },
        query_stream_available=available,
        candidate_ids=candidate_ids,
        candidate_mask=candidate_mask,
        positive_mask=positive_mask,
        natural_positive_in_pool=natural,
        splits=frame["five_stream_split"].astype(str).tolist(),
        sample_ids=frame["sample_id"].astype(str).tolist(),
        source_groups=frame["source_group"].astype(str).tolist(),
        slugs=slugs,
        ocr_texts=ocr_texts,
    )
