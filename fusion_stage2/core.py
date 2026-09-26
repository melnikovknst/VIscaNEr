"""Shared data, candidate-feature and ranking utilities for Stage-II fusion."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from dinov3_retrieval import build_transforms
from ocr_reranker.reranker import build_idf, lexical_score


FEATURE_NAMES = (
    "label_similarity",
    "bottle_similarity",
    "label_delta_from_top1",
    "bottle_delta_from_top1",
    "label_reciprocal_topk_rank",
    "bottle_reciprocal_topk_rank",
    "label_is_top1",
    "bottle_is_top1",
    "label_detector_confidence",
    "bottle_detector_confidence",
    "bottle_status_partial",
    "bottle_status_ambiguous",
    "bottle_status_low_confidence",
    "bottle_status_original_fallback",
    "branch_top1_agreement",
    "ocr_lexical_score",
    "ocr_reciprocal_topk_rank",
    "ocr_is_top1",
    "ocr_available",
)
def first_file(candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"None of the candidate files exists: {list(candidates)}")


def resolve_crop(root: Path, status: str, recorded: str, portable: str, hard_root: Path) -> Path:
    if portable:
        candidate = hard_root / portable
        if candidate.is_file():
            return candidate.resolve()
    name = Path(recorded).name
    return first_file((root / "crops" / status / name, root / status / name))


def resolve_refs(root: Path) -> tuple[list[str], list[str]]:
    refs_root = root / "refs" if (root / "refs").is_dir() else root
    files = sorted(
        path for path in refs_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not files:
        raise FileNotFoundError(f"No reference images under {refs_root}")
    slugs = [path.stem for path in files]
    if len(slugs) != len(set(slugs)):
        raise ValueError(f"Duplicate reference slugs under {refs_root}")
    return slugs, [str(path.resolve()) for path in files]


def load_manifest(
    manifest_path: str | Path,
    hard_root: str | Path,
    label_data_root: str | Path,
    bottle_data_root: str | Path,
) -> pd.DataFrame:
    manifest_path = Path(manifest_path).resolve()
    hard_root = Path(hard_root).resolve()
    label_root = Path(label_data_root).resolve()
    bottle_root = Path(bottle_data_root).resolve()
    frame = pd.read_csv(manifest_path, low_memory=False).fillna("")
    required = {
        "source_relative_path", "wine_slug", "fusion_split",
        "label_recorded_path", "label_status", "bottle_recorded_path", "bottle_status",
        "label_detector_confidence", "bottle_detector_confidence",
        "portable_label_path", "portable_bottle_path",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Fusion manifest is missing columns: {sorted(missing)}")
    if frame["source_relative_path"].duplicated().any():
        raise ValueError("Fusion manifest contains duplicate source_relative_path rows")

    frame["label_path"] = frame.apply(
        lambda row: str(
            resolve_crop(
                label_root,
                str(row["label_status"]),
                str(row["label_recorded_path"]),
                str(row["portable_label_path"]),
                hard_root,
            )
        ),
        axis=1,
    )
    frame["bottle_path"] = frame.apply(
        lambda row: str(
            resolve_crop(
                bottle_root,
                str(row["bottle_status"]),
                str(row["bottle_recorded_path"]),
                str(row["portable_bottle_path"]),
                hard_root,
            )
        ),
        axis=1,
    )
    return frame.reset_index(drop=True)


class FusionMLP(nn.Module):
    def __init__(self, feature_dim: int = len(FEATURE_NAMES), hidden_dim: int = 96, dropout: float = 0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.network(features).squeeze(-1)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


@dataclass
class FeatureCache:
    candidate_ids: torch.Tensor
    candidate_mask: torch.Tensor
    features: torch.Tensor
    true_positions: torch.Tensor
    true_label_ids: torch.Tensor
    label_true_ranks: torch.Tensor
    bottle_true_ranks: torch.Tensor
    row_indices: torch.Tensor
    slugs: list[str]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    data_fingerprint: str = ""

    def subset(self, indices: Sequence[int]) -> "FeatureCache":
        take = torch.as_tensor(indices, dtype=torch.long)
        return FeatureCache(
            candidate_ids=self.candidate_ids[take],
            candidate_mask=self.candidate_mask[take],
            features=self.features[take],
            true_positions=self.true_positions[take],
            true_label_ids=self.true_label_ids[take],
            label_true_ranks=self.label_true_ranks[take],
            bottle_true_ranks=self.bottle_true_ranks[take],
            row_indices=self.row_indices[take],
            slugs=self.slugs,
            feature_names=self.feature_names,
            data_fingerprint=self.data_fingerprint,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureCache":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["feature_names"] = tuple(payload.get("feature_names", FEATURE_NAMES))
        payload["data_fingerprint"] = str(payload.get("data_fingerprint", ""))
        return cls(**payload)


def frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["source_relative_path", "wine_slug", "ocr_text"]].fillna("").itertuples(index=False):
        digest.update("\0".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _rank_of_truth(scores: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    truth_scores = scores.gather(1, truth[:, None]).squeeze(1)
    return 1 + (scores > truth_scores[:, None]).sum(dim=1)


def build_feature_cache(
    frame: pd.DataFrame,
    label_embeddings: torch.Tensor,
    bottle_embeddings: torch.Tensor,
    label_gallery: torch.Tensor,
    bottle_gallery: torch.Tensor,
    slugs: list[str],
    top_k: int = 10,
    chunk_size: int = 512,
) -> FeatureCache:
    if len(frame) != len(label_embeddings) or len(frame) != len(bottle_embeddings):
        raise ValueError("Manifest and query embedding counts differ")
    if len(slugs) != len(label_gallery) or len(slugs) != len(bottle_gallery):
        raise ValueError("Gallery counts differ")
    slug_to_id = {slug: index for index, slug in enumerate(slugs)}
    truth = torch.tensor([slug_to_id[str(value)] for value in frame["wine_slug"]], dtype=torch.long)
    max_candidates = top_k * 3
    all_ids: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_features: list[torch.Tensor] = []
    all_true_positions: list[torch.Tensor] = []
    all_label_ranks: list[torch.Tensor] = []
    all_bottle_ranks: list[torch.Tensor] = []
    idf = build_idf(slugs)

    label_gallery = F.normalize(label_gallery.float(), dim=1)
    bottle_gallery = F.normalize(bottle_gallery.float(), dim=1)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        label_scores = F.normalize(label_embeddings[start:stop].float(), dim=1) @ label_gallery.T
        bottle_scores = F.normalize(bottle_embeddings[start:stop].float(), dim=1) @ bottle_gallery.T
        label_values, label_top = label_scores.topk(top_k, dim=1)
        bottle_values, bottle_top = bottle_scores.topk(top_k, dim=1)
        batch_size = stop - start
        ids = torch.full((batch_size, max_candidates), -1, dtype=torch.long)
        mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
        features = torch.zeros((batch_size, max_candidates, len(FEATURE_NAMES)), dtype=torch.float32)
        true_positions = torch.full((batch_size,), -1, dtype=torch.long)
        batch_truth = truth[start:stop]
        label_ranks = _rank_of_truth(label_scores, batch_truth)
        bottle_ranks = _rank_of_truth(bottle_scores, batch_truth)
        for local in range(batch_size):
            union: list[int] = []
            for candidate in torch.cat([label_top[local], bottle_top[local]]).tolist():
                if candidate not in union:
                    union.append(int(candidate))
            candidate_ids = union[:max_candidates]
            count = len(candidate_ids)
            ids[local, :count] = torch.tensor(candidate_ids)
            mask[local, :count] = True
            candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)
            label_candidate_scores = label_scores[local, candidate_tensor]
            bottle_candidate_scores = bottle_scores[local, candidate_tensor]
            label_rank_lookup = {int(value): rank + 1 for rank, value in enumerate(label_top[local].tolist())}
            bottle_rank_lookup = {int(value): rank + 1 for rank, value in enumerate(bottle_top[local].tolist())}
            row = frame.iloc[start + local]
            label_conf = float(row.get("label_detector_confidence", 0.0) or 0.0)
            bottle_conf = float(row.get("bottle_detector_confidence", 0.0) or 0.0)
            status = str(row.get("bottle_status", ""))
            ocr_text = str(row.get("ocr_text", "") or "").strip()
            ocr_available = float(bool(ocr_text))
            ocr_scores_by_id: dict[int, float] = {}
            ocr_rank_lookup: dict[int, int] = {}
            if ocr_available:
                all_ocr_scores = np.asarray(
                    [lexical_score(ocr_text, slug, idf) for slug in slugs],
                    dtype=np.float32,
                )
                ocr_top = np.argsort(-all_ocr_scores, kind="stable")[:top_k].tolist()
                for rank, candidate in enumerate(ocr_top, start=1):
                    ocr_rank_lookup[int(candidate)] = rank
                    ocr_scores_by_id[int(candidate)] = float(all_ocr_scores[candidate])
                    if int(candidate) not in union:
                        union.append(int(candidate))
                candidate_ids = union[:max_candidates]
                count = len(candidate_ids)
                ids[local].fill_(-1)
                mask[local].fill_(False)
                ids[local, :count] = torch.tensor(candidate_ids)
                mask[local, :count] = True
                candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)
                label_candidate_scores = label_scores[local, candidate_tensor]
                bottle_candidate_scores = bottle_scores[local, candidate_tensor]
            agreement = float(int(label_top[local, 0]) == int(bottle_top[local, 0]))
            for position, candidate in enumerate(candidate_ids):
                lr = label_rank_lookup.get(candidate, 0)
                br = bottle_rank_lookup.get(candidate, 0)
                ocr_rank = ocr_rank_lookup.get(candidate, 0)
                ocr_score = (
                    ocr_scores_by_id.get(candidate)
                    if candidate in ocr_scores_by_id
                    else lexical_score(ocr_text, slugs[candidate], idf) if ocr_available else 0.0
                )
                values = (
                    float(label_candidate_scores[position]),
                    float(bottle_candidate_scores[position]),
                    float(label_candidate_scores[position] - label_values[local, 0]),
                    float(bottle_candidate_scores[position] - bottle_values[local, 0]),
                    1.0 / lr if lr else 0.0,
                    1.0 / br if br else 0.0,
                    float(lr == 1),
                    float(br == 1),
                    label_conf,
                    bottle_conf,
                    float(status == "partial"),
                    float(status == "ambiguous"),
                    float(status == "low_confidence"),
                    float(status == "failed_original_fallback"),
                    agreement,
                    float(ocr_score),
                    1.0 / ocr_rank if ocr_rank else 0.0,
                    float(ocr_rank == 1),
                    ocr_available,
                )
                features[local, position] = torch.tensor(values)
            matches = (ids[local] == batch_truth[local]).nonzero(as_tuple=False)
            if len(matches):
                true_positions[local] = int(matches[0, 0])
        all_ids.append(ids)
        all_masks.append(mask)
        all_features.append(features)
        all_true_positions.append(true_positions)
        all_label_ranks.append(label_ranks)
        all_bottle_ranks.append(bottle_ranks)
    return FeatureCache(
        candidate_ids=torch.cat(all_ids),
        candidate_mask=torch.cat(all_masks),
        features=torch.cat(all_features),
        true_positions=torch.cat(all_true_positions),
        true_label_ids=truth,
        label_true_ranks=torch.cat(all_label_ranks),
        bottle_true_ranks=torch.cat(all_bottle_ranks),
        row_indices=torch.arange(len(frame), dtype=torch.long),
        slugs=slugs,
        data_fingerprint=frame_fingerprint(frame),
    )


def ranking_metrics(ranks: torch.Tensor) -> dict[str, float | int]:
    ranks = ranks.detach().cpu().float()
    result: dict[str, float | int] = {
        "num_queries": int(len(ranks)),
        "accuracy": float((ranks == 1).float().mean()) if len(ranks) else math.nan,
        "mrr": float((1.0 / ranks).mean()) if len(ranks) else math.nan,
        "mean_rank": float(ranks.mean()) if len(ranks) else math.nan,
    }
    for k in (1, 2, 5, 10):
        result[f"recall_at_{k}"] = float((ranks <= k).float().mean()) if len(ranks) else math.nan
    return result


def branch_metrics(cache: FeatureCache) -> dict[str, dict[str, float | int]]:
    """Return directly comparable retrieval metrics for both visual branches."""
    return {
        "label_dino_b": ranking_metrics(cache.label_true_ranks),
        "bottle_dino_b": ranking_metrics(cache.bottle_true_ranks),
    }


@torch.inference_mode()
def predictions_from_cache(
    model: FusionMLP,
    cache: FeatureCache,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predicted gallery ids, ranks, and top scores for an audit table."""
    model.eval()
    predictions: list[torch.Tensor] = []
    ranks: list[torch.Tensor] = []
    top_scores: list[torch.Tensor] = []
    for start in range(0, len(cache.features), batch_size):
        stop = min(start + batch_size, len(cache.features))
        mask = cache.candidate_mask[start:stop].to(device)
        logits = model(cache.features[start:stop].to(device), mask)
        order = logits.argsort(dim=1, descending=True).cpu()
        ids = cache.candidate_ids[start:stop]
        predictions.append(ids.gather(1, order[:, :1]).squeeze(1))
        top_scores.append(logits.gather(1, order[:, :1].to(device)).squeeze(1).cpu())
        truth = cache.true_positions[start:stop]
        present = truth.ge(0)
        batch_ranks = torch.full((stop - start,), len(cache.slugs) + 1, dtype=torch.long)
        if present.any():
            match = order[present].eq(truth[present, None])
            batch_ranks[present] = match.float().argmax(dim=1).long() + 1
        ranks.append(batch_ranks)
    return torch.cat(predictions), torch.cat(ranks), torch.cat(top_scores)


@torch.inference_mode()
def score_cache(model: FusionMLP, cache: FeatureCache, device: torch.device, batch_size: int = 512) -> tuple[dict[str, Any], torch.Tensor]:
    model.eval()
    ranks: list[torch.Tensor] = []
    for start in range(0, len(cache.features), batch_size):
        stop = min(start + batch_size, len(cache.features))
        features = cache.features[start:stop].to(device)
        mask = cache.candidate_mask[start:stop].to(device)
        logits = model(features, mask).cpu()
        order = logits.argsort(dim=1, descending=True)
        truth = cache.true_positions[start:stop]
        present = truth.ge(0)
        batch_ranks = torch.full((stop - start,), len(cache.slugs) + 1, dtype=torch.long)
        if present.any():
            batch_ranks[present] = 1 + (order[present] == truth[present, None]).nonzero(as_tuple=False)[:, 1]
        ranks.append(batch_ranks)
    all_ranks = torch.cat(ranks)
    metrics = ranking_metrics(all_ranks)
    metrics["candidate_recall"] = float(cache.true_positions.ge(0).float().mean())
    return metrics, all_ranks


class PairedImageDataset(Dataset):
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


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
