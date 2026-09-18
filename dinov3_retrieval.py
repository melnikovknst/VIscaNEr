"""DINOv3 ViT-B/16 metric-learning pipeline for wine-label retrieval.

The module is intentionally independent of Hugging Face Hub at runtime.  It
constructs the official DINOv3 ViT-B/16 architecture and loads the local
``model.safetensors`` checkpoint supplied with the project.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from transformers import DINOv3ViTConfig, DINOv3ViTModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class PipelineConfig:
    project_root: str
    weights_path: str = "models/dinov3/model.safetensors"
    crops_metadata_path: str = "datasets/yolo_label_detector/crops/crops_metadata.csv"
    bottle_manifest_path: str = "datasets/bottle_images_45k/bottle_images_manifest.csv"
    crops_root: str = "datasets/yolo_label_detector/crops"
    refs_root: str = "datasets/wine-scanner/data/refs/rgb"
    index_path: str = "datasets/dinov3_retrieval/index.csv"
    split_summary_path: str = "datasets/dinov3_retrieval/split_summary.json"
    models_dir: str = "models/dinov3_retrieval"
    runs_dir: str = "runs/dinov3_retrieval"
    run_name: str = "dinov3_vitb16_wine_retrieval"
    seed: int = 42
    image_size: int = 224
    embedding_dim: int = 256
    projection_hidden_dim: int = 512
    unseen_identity_fraction: float = 0.15
    val_seen_per_identity: int = 2
    include_low_confidence_in_train: bool = False
    stage1_epochs: int = 5
    stage2_epochs: int = 12
    stage1_identities_per_batch: int = 16
    stage1_images_per_identity: int = 4
    stage2_identities_per_batch: int = 8
    stage2_images_per_identity: int = 2
    head_lr: float = 3e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 0.05
    unfreeze_last_blocks: int = 2
    supcon_temperature: float = 0.07
    ce_temperature: float = 0.07
    ce_weight: float = 0.5
    num_workers: int = 4
    eval_batch_size: int = 64
    patience: int = 5
    device: str = "auto"
    amp: bool = True

    def resolved(self) -> "PipelineConfig":
        root = Path(self.project_root).expanduser().resolve()
        data = asdict(self)
        data["project_root"] = str(root)
        for key in (
            "weights_path",
            "crops_metadata_path",
            "bottle_manifest_path",
            "crops_root",
            "refs_root",
            "index_path",
            "split_summary_path",
            "models_dir",
            "runs_dir",
        ):
            path = Path(data[key]).expanduser()
            if not path.is_absolute():
                path = root / path
            data[key] = str(path.resolve())
        return PipelineConfig(**data)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment_summary(device: torch.device | None = None) -> dict[str, Any]:
    device = device or choose_device()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }


def _stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _reference_lookup(refs_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in sorted(refs_root.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if path.stem in lookup:
                raise ValueError(f"Duplicate reference stem: {path.stem}")
            lookup[path.stem] = path
    return lookup


def prepare_retrieval_index(config: PipelineConfig, validate_files: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join YOLO crops to wine identities and create leakage-safe splits.

    Seen identities contribute training images plus two held-out query images.
    A deterministic identity-level holdout never contributes training images.
    Low-confidence crops are kept as a separate hard-query benchmark.
    """
    cfg = config.resolved()
    metadata_path = Path(cfg.crops_metadata_path)
    manifest_path = Path(cfg.bottle_manifest_path)
    crops_root = Path(cfg.crops_root)
    refs_root = Path(cfg.refs_root)

    for required in (metadata_path, manifest_path, crops_root, refs_root, Path(cfg.weights_path)):
        if not required.exists():
            raise FileNotFoundError(f"Required input does not exist: {required}")

    metadata = pd.read_csv(metadata_path)
    manifest = pd.read_csv(manifest_path)
    required_meta = {"source_path", "crop_path", "status", "confidence"}
    required_manifest = {"source_path", "wine_slug"}
    if missing := required_meta - set(metadata.columns):
        raise ValueError(f"Crop metadata is missing columns: {sorted(missing)}")
    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f"Bottle manifest is missing columns: {sorted(missing)}")
    if metadata["source_path"].duplicated().any():
        raise ValueError("Crop metadata contains duplicate source_path rows")
    if manifest["source_path"].duplicated().any():
        raise ValueError("Bottle manifest contains duplicate source_path rows")

    crops = metadata.merge(
        manifest[["source_path", "wine_slug"]],
        on="source_path",
        how="left",
        validate="one_to_one",
    )
    if crops["wine_slug"].isna().any():
        examples = crops.loc[crops["wine_slug"].isna(), "source_path"].head(5).tolist()
        raise ValueError(f"Some crops cannot be mapped to an identity: {examples}")

    refs = _reference_lookup(refs_root)
    identities = sorted(crops["wine_slug"].astype(str).unique())
    missing_refs = sorted(set(identities) - set(refs))
    if missing_refs:
        raise ValueError(f"Missing reference images for {len(missing_refs)} identities: {missing_refs[:5]}")
    label_by_slug = {slug: idx for idx, slug in enumerate(identities)}
    unseen = {
        slug
        for slug in identities
        if _stable_unit_interval(slug, cfg.seed) < cfg.unseen_identity_fraction
    }
    # Avoid a pathological empty split when testing with a tiny fixture.
    if len(identities) > 1 and not unseen:
        unseen.add(min(identities, key=lambda x: _stable_unit_interval(x, cfg.seed)))

    rows: list[dict[str, Any]] = []
    broken: list[str] = []
    successful = crops[crops["status"].eq("successful")].copy()
    successful["sort_key"] = successful.apply(
        lambda row: _stable_unit_interval(f"{row['wine_slug']}:{row['source_path']}", cfg.seed), axis=1
    )
    for slug, group in successful.groupby("wine_slug", sort=True):
        group = group.sort_values("sort_key").reset_index(drop=True)
        if slug in unseen:
            split_values = ["val_unseen"] * len(group)
        else:
            n_val = min(cfg.val_seen_per_identity, max(1, len(group) - 2))
            split_values = ["val_seen"] * n_val + ["train"] * (len(group) - n_val)
        for row, split in zip(group.to_dict("records"), split_values, strict=True):
            path = crops_root / "successful" / Path(str(row["crop_path"])).name
            if validate_files and not path.is_file():
                broken.append(str(path))
                continue
            rows.append(
                {
                    "split": split,
                    "path_kind": "crop",
                    "filename": path.name,
                    "status": "successful",
                    "wine_slug": slug,
                    "label_id": label_by_slug[slug],
                    "confidence": float(row["confidence"]),
                    "source_path": str(row["source_path"]),
                }
            )

    low = crops[crops["status"].eq("low_confidence")]
    for row in low.to_dict("records"):
        slug = str(row["wine_slug"])
        path = crops_root / "low_confidence" / Path(str(row["crop_path"])).name
        if validate_files and not path.is_file():
            broken.append(str(path))
            continue
        split = "train" if cfg.include_low_confidence_in_train and slug not in unseen else "val_hard"
        rows.append(
            {
                "split": split,
                "path_kind": "crop",
                "filename": path.name,
                "status": "low_confidence",
                "wine_slug": slug,
                "label_id": label_by_slug[slug],
                "confidence": float(row["confidence"]),
                "source_path": str(row["source_path"]),
            }
        )

    for slug in identities:
        ref = refs[slug]
        rows.append(
            {
                "split": "gallery",
                "path_kind": "reference",
                "filename": ref.name,
                "status": "reference",
                "wine_slug": slug,
                "label_id": label_by_slug[slug],
                "confidence": np.nan,
                "source_path": "",
            }
        )

    if broken:
        raise FileNotFoundError(f"Missing {len(broken)} crop files. Examples: {broken[:5]}")
    index = pd.DataFrame(rows).sort_values(["split", "label_id", "filename"]).reset_index(drop=True)
    index_path = Path(cfg.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_path, index=False)

    split_counts = index.groupby("split").size().astype(int).to_dict()
    identity_counts = index.groupby("split")["wine_slug"].nunique().astype(int).to_dict()
    summary = {
        "index_path": str(index_path),
        "num_identities": len(identities),
        "seen_identities": len(identities) - len(unseen),
        "unseen_identities": len(unseen),
        "split_rows": split_counts,
        "split_identities": identity_counts,
        "source_status_counts": metadata["status"].value_counts(dropna=False).astype(int).to_dict(),
        "leakage_check_train_vs_unseen": len(
            set(index.loc[index.split.eq("train"), "wine_slug"])
            & set(index.loc[index.split.eq("val_unseen"), "wine_slug"])
        ),
    }
    summary_path = Path(cfg.split_summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return index, summary


class PadToSquare:
    def __init__(self, fill: tuple[int, int, int] = (124, 116, 104)) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        size = max(width, height)
        left = (size - width) // 2
        top = (size - height) // 2
        return ImageOps.expand(
            image,
            border=(left, top, size - width - left, size - height - top),
            fill=self.fill,
        )


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    if image_size % 16 != 0:
        raise ValueError("DINOv3 ViT-B/16 image_size must be divisible by 16")
    common_end = [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    train_transform = transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomApply(
                [transforms.RandomPerspective(distortion_scale=0.08, p=1.0, fill=tuple(int(v * 255) for v in IMAGENET_MEAN))],
                p=0.20,
            ),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.04, 0.04),
                scale=(0.92, 1.08),
                shear=(-2, 2, -2, 2),
                interpolation=InterpolationMode.BICUBIC,
                fill=tuple(int(v * 255) for v in IMAGENET_MEAN),
            ),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.12, hue=0.02),
            transforms.RandomGrayscale(p=0.03),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.10),
            *common_end,
        ]
    )
    eval_transform = transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            *common_end,
        ]
    )
    return train_transform, eval_transform


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class WineRetrievalDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        crops_root: str | Path,
        refs_root: str | Path,
        transform: Any,
        two_views: bool = False,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.crops_root = Path(crops_root)
        self.refs_root = Path(refs_root)
        self.transform = transform
        self.two_views = two_views

    def __len__(self) -> int:
        return len(self.records)

    def path_for_row(self, row: pd.Series) -> Path:
        if row["path_kind"] == "reference":
            return self.refs_root / row["filename"]
        return self.crops_root / row["status"] / row["filename"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        path = self.path_for_row(row)
        image = open_rgb(path)
        item: dict[str, Any] = {
            "label": int(row["label_id"]),
            "path": str(path),
            "wine_slug": str(row["wine_slug"]),
        }
        if self.two_views:
            item["view1"] = self.transform(image)
            item["view2"] = self.transform(image.copy())
        else:
            item["image"] = self.transform(image)
        return item


class IdentityBatchSampler(Sampler[list[int]]):
    """P x K sampler: every batch contains positives for metric learning."""

    def __init__(
        self,
        labels: Sequence[int],
        identities_per_batch: int,
        images_per_identity: int,
        seed: int = 42,
        batches_per_epoch: int | None = None,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.identities_per_batch = identities_per_batch
        self.images_per_identity = images_per_identity
        self.seed = seed
        self.epoch = 0
        self.by_label: dict[int, np.ndarray] = {
            int(label): np.flatnonzero(self.labels == label) for label in np.unique(self.labels)
        }
        if len(self.by_label) < identities_per_batch:
            raise ValueError(
                f"Need at least {identities_per_batch} identities, found {len(self.by_label)}"
            )
        batch_size = identities_per_batch * images_per_identity
        self.batches_per_epoch = batches_per_epoch or math.ceil(len(labels) / batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        identities = np.asarray(sorted(self.by_label), dtype=np.int64)
        for _ in range(self.batches_per_epoch):
            selected_labels = rng.choice(identities, self.identities_per_batch, replace=False)
            batch: list[int] = []
            for label in selected_labels:
                candidates = self.by_label[int(label)]
                replace = len(candidates) < self.images_per_identity
                batch.extend(rng.choice(candidates, self.images_per_identity, replace=replace).tolist())
            rng.shuffle(batch)
            yield batch


def make_dinov3_vitb16_config(image_size: int = 224) -> DINOv3ViTConfig:
    return DINOv3ViTConfig(
        image_size=image_size,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_register_tokens=4,
        rope_theta=100.0,
        layerscale_value=1e-5,
        pos_embed_rescale=2.0,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        layer_norm_eps=1e-5,
    )


def load_local_dinov3_backbone(weights_path: str | Path, image_size: int = 224) -> DINOv3ViTModel:
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    backbone = DINOv3ViTModel(make_dinov3_vitb16_config(image_size))
    state = load_file(str(weights_path), device="cpu")
    # DINOv3 encoder keys changed between Transformers releases: some builds
    # expose ``layer.*`` while others expose ``model.layer.*``.  Normalize the
    # checkpoint to the state-dict layout expected by the installed build.
    expected_keys = set(backbone.state_dict())
    expects_model_prefix = any(key.startswith("model.layer.") for key in expected_keys)
    checkpoint_has_model_prefix = any(key.startswith("model.layer.") for key in state)
    if expects_model_prefix and not checkpoint_has_model_prefix:
        state = {(f"model.{key}" if key.startswith("layer.") else key): value for key, value in state.items()}
    elif checkpoint_has_model_prefix and not expects_model_prefix:
        state = {(key.removeprefix("model.") if key.startswith("model.layer.") else key): value for key, value in state.items()}
    result = backbone.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")
    return backbone


class DINOv3RetrievalModel(nn.Module):
    def __init__(
        self,
        backbone: DINOv3ViTModel,
        num_classes: int,
        embedding_dim: int = 256,
        projection_hidden_dim: int = 512,
        ce_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_register_tokens = int(backbone.config.num_register_tokens)
        feature_dim = int(backbone.config.hidden_size) * 2
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_hidden_dim),
            nn.LayerNorm(projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(projection_hidden_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.ce_temperature = ce_temperature
        self.backbone_frozen = False

    def set_backbone_trainable(self, last_n_blocks: int) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if last_n_blocks > 0:
            # Transformers releases expose DINOv3 encoder blocks either as
            # ``model.layer`` or directly as ``layer``.
            encoder = getattr(self.backbone, "model", self.backbone)
            blocks = encoder.layer
            for block in blocks[-last_n_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True
        self.backbone_frozen = last_n_blocks == 0

    def backbone_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.backbone_frozen else nullcontext()
        with context:
            hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
            cls = hidden[:, 0]
            patch_start = 1 + self.num_register_tokens
            mean_patch = hidden[:, patch_start:].mean(dim=1)
        return torch.cat([cls, mean_patch], dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone_features(pixel_values)
        embeddings = F.normalize(self.projection(features), dim=-1)
        normalized_weights = F.normalize(self.classifier.weight, dim=-1)
        logits = embeddings @ normalized_weights.T / self.ce_temperature
        return embeddings, logits


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid = positive_mask.sum(dim=1) > 0
    if not valid.any():
        raise ValueError("Supervised contrastive loss needs at least two samples per identity")
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_positive_log_prob[valid].mean()


def _autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def create_train_loader(
    records: pd.DataFrame,
    cfg: PipelineConfig,
    stage: int,
    epoch: int = 0,
) -> tuple[DataLoader, IdentityBatchSampler]:
    train_transform, _ = build_transforms(cfg.image_size)
    dataset = WineRetrievalDataset(records, cfg.crops_root, cfg.refs_root, train_transform, two_views=True)
    if stage == 1:
        identities_per_batch = cfg.stage1_identities_per_batch
        images_per_identity = cfg.stage1_images_per_identity
    else:
        identities_per_batch = cfg.stage2_identities_per_batch
        images_per_identity = cfg.stage2_images_per_identity
    sampler = IdentityBatchSampler(
        records["label_id"].astype(int).tolist(),
        identities_per_batch=identities_per_batch,
        images_per_identity=images_per_identity,
        seed=cfg.seed,
    )
    sampler.set_epoch(epoch)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    return loader, sampler


def create_eval_loader(records: pd.DataFrame, cfg: PipelineConfig) -> DataLoader:
    _, eval_transform = build_transforms(cfg.image_size)
    dataset = WineRetrievalDataset(records, cfg.crops_root, cfg.refs_root, eval_transform, two_views=False)
    return DataLoader(
        dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )


def build_optimizer(model: DINOv3RetrievalModel, cfg: PipelineConfig) -> torch.optim.Optimizer:
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    head_parameters = [
        p for module in (model.projection, model.classifier) for p in module.parameters() if p.requires_grad
    ]
    groups: list[dict[str, Any]] = [{"params": head_parameters, "lr": cfg.head_lr}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": cfg.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def train_one_epoch(
    model: DINOv3RetrievalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: PipelineConfig,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    if model.backbone_frozen:
        model.backbone.eval()
    totals = defaultdict(float)
    seen = 0
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        view1 = batch["view1"].to(device, non_blocking=True)
        view2 = batch["view2"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, cfg.amp):
            emb1, logits1 = model(view1)
            emb2, logits2 = model(view2)
            embeddings = torch.cat([emb1, emb2], dim=0)
            repeated_labels = labels.repeat(2)
            supcon = supervised_contrastive_loss(embeddings, repeated_labels, cfg.supcon_temperature)
            ce = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) * 0.5
            loss = supcon + cfg.ce_weight * ce
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        batch_size = len(labels)
        seen += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["supcon_loss"] += float(supcon.detach()) * batch_size
        totals["ce_loss"] += float(ce.detach()) * batch_size
        accuracy = (logits1.argmax(dim=1) == labels).float().mean()
        totals["train_accuracy"] += float(accuracy.detach()) * batch_size
        progress.set_postfix(loss=f"{totals['loss'] / seen:.4f}")
    return {key: value / max(seen, 1) for key, value in totals.items()}


@torch.inference_mode()
def embed_loader(
    model: DINOv3RetrievalModel,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    desc: str = "embed",
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    paths: list[str] = []
    slugs: list[str] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with _autocast_context(device, amp):
            batch_embeddings, _ = model(images)
        embeddings.append(batch_embeddings.float().cpu())
        labels.append(batch["label"].long().cpu())
        paths.extend(batch["path"])
        slugs.extend(batch["wine_slug"])
    return torch.cat(embeddings), torch.cat(labels), paths, slugs


def retrieval_metrics(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
    chunk_size: int = 512,
) -> tuple[dict[str, float], torch.Tensor]:
    if len(torch.unique(gallery_labels)) != len(gallery_labels):
        raise ValueError("Gallery must contain exactly one reference per identity")
    gallery_by_label = {int(label): idx for idx, label in enumerate(gallery_labels.tolist())}
    missing = sorted(set(query_labels.tolist()) - set(gallery_by_label))
    if missing:
        raise ValueError(f"Gallery is missing labels: {missing[:5]}")
    ranks: list[torch.Tensor] = []
    gallery_embeddings = F.normalize(gallery_embeddings.float(), dim=-1)
    for start in range(0, len(query_embeddings), chunk_size):
        query = F.normalize(query_embeddings[start : start + chunk_size].float(), dim=-1)
        labels = query_labels[start : start + chunk_size]
        similarities = query @ gallery_embeddings.T
        target_indices = torch.tensor([gallery_by_label[int(x)] for x in labels], dtype=torch.long)
        target_scores = similarities[torch.arange(len(similarities)), target_indices]
        ranks.append((similarities > target_scores[:, None]).sum(dim=1) + 1)
    rank_tensor = torch.cat(ranks).float()
    metrics = {f"recall_at_{k}": float((rank_tensor <= k).float().mean()) for k in ks}
    metrics.update(
        {
            "mrr": float((1.0 / rank_tensor).mean()),
            "median_rank": float(rank_tensor.median()),
            "mean_rank": float(rank_tensor.mean()),
            "num_queries": int(len(rank_tensor)),
        }
    )
    return metrics, rank_tensor.long()


def evaluate_splits(
    model: DINOv3RetrievalModel,
    index: pd.DataFrame,
    cfg: PipelineConfig,
    device: torch.device,
    splits: Sequence[str] = ("val_seen", "val_unseen", "val_hard"),
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    gallery_records = index[index["split"].eq("gallery")]
    gallery = embed_loader(model, create_eval_loader(gallery_records, cfg), device, cfg.amp, "gallery")
    gallery_embeddings, gallery_labels, gallery_paths, gallery_slugs = gallery
    metrics: dict[str, dict[str, float]] = {}
    details: dict[str, Any] = {
        "gallery_embeddings": gallery_embeddings,
        "gallery_labels": gallery_labels,
        "gallery_paths": gallery_paths,
        "gallery_slugs": gallery_slugs,
    }
    for split in splits:
        records = index[index["split"].eq(split)]
        if records.empty:
            continue
        query = embed_loader(model, create_eval_loader(records, cfg), device, cfg.amp, split)
        query_embeddings, query_labels, query_paths, query_slugs = query
        split_metrics, ranks = retrieval_metrics(
            query_embeddings, query_labels, gallery_embeddings, gallery_labels
        )
        metrics[split] = split_metrics
        details[split] = {
            "embeddings": query_embeddings,
            "labels": query_labels,
            "paths": query_paths,
            "slugs": query_slugs,
            "ranks": ranks,
        }
    return metrics, details


def _save_checkpoint(
    path: Path,
    model: DINOv3RetrievalModel,
    cfg: PipelineConfig,
    epoch: int,
    stage: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "stage": stage,
            "metrics": metrics,
        },
        path,
    )


def load_trained_model(
    checkpoint_path: str | Path,
    weights_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[DINOv3RetrievalModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = PipelineConfig(**checkpoint["config"]).resolved()
    backbone = load_local_dinov3_backbone(weights_path, cfg.image_size)
    model = DINOv3RetrievalModel(
        backbone,
        num_classes=int(checkpoint["model_state_dict"]["classifier.weight"].shape[0]),
        embedding_dim=cfg.embedding_dim,
        projection_hidden_dim=cfg.projection_hidden_dim,
        ce_temperature=cfg.ce_temperature,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    return model, checkpoint


def train_pipeline(
    config: PipelineConfig,
    quick_smoke: bool = False,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config.resolved()
    seed_everything(cfg.seed)
    device = choose_device(cfg.device)
    Path(cfg.models_dir).mkdir(parents=True, exist_ok=True)
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(cfg.index_path)
    if not index_path.exists():
        index, split_summary = prepare_retrieval_index(cfg)
    else:
        index = pd.read_csv(index_path)
        split_summary = json.loads(Path(cfg.split_summary_path).read_text(encoding="utf-8"))
    train_records = index[index["split"].eq("train")].reset_index(drop=True)
    if quick_smoke:
        selected_ids = sorted(train_records["label_id"].unique())[: max(16, cfg.stage1_identities_per_batch)]
        train_records = train_records[train_records["label_id"].isin(selected_ids)].groupby("label_id").head(4)
        cfg.stage1_epochs = 1
        cfg.stage2_epochs = 0
        cfg.num_workers = 0
        cfg.stage1_identities_per_batch = min(8, len(selected_ids))
        cfg.stage1_images_per_identity = 2
        cfg.eval_batch_size = 8

    num_classes = int(index["label_id"].max()) + 1
    backbone = load_local_dinov3_backbone(cfg.weights_path, cfg.image_size)
    model = DINOv3RetrievalModel(
        backbone,
        num_classes=num_classes,
        embedding_dim=cfg.embedding_dim,
        projection_hidden_dim=cfg.projection_hidden_dim,
        ce_temperature=cfg.ce_temperature,
    ).to(device)
    resume_payload: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        print(
            f"Resuming from {resume_path} "
            f"(epoch={resume_payload.get('epoch')}, stage={resume_payload.get('stage')})"
        )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg.amp))
    history_path = run_dir / "history.csv"
    if resume_payload is not None and history_path.is_file():
        history = pd.read_csv(history_path).to_dict("records")
    else:
        history: list[dict[str, Any]] = []
    global_epoch = int(resume_payload.get("epoch", 0)) if resume_payload else 0
    best_recall = -1.0
    best_path = Path(cfg.models_dir) / "best.pt"
    if resume_payload is not None and best_path.is_file():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        best_recall = float(
            best_payload.get("metrics", {}).get("val_seen", {}).get("recall_at_1", -1.0)
        )
    elif resume_payload is not None:
        best_recall = float(
            resume_payload.get("metrics", {}).get("val_seen", {}).get("recall_at_1", -1.0)
        )

    for stage, epochs, unfrozen in (
        (1, cfg.stage1_epochs, 0),
        (2, cfg.stage2_epochs, cfg.unfreeze_last_blocks),
    ):
        if epochs <= 0:
            continue
        completed_stage_epochs = 0
        if resume_payload is not None:
            resume_stage = int(resume_payload.get("stage", 1))
            if stage < resume_stage:
                completed_stage_epochs = epochs
            elif stage == resume_stage:
                if stage == 1:
                    completed_stage_epochs = min(global_epoch, cfg.stage1_epochs)
                else:
                    completed_stage_epochs = min(
                        max(global_epoch - cfg.stage1_epochs, 0), cfg.stage2_epochs
                    )
        if completed_stage_epochs >= epochs:
            print(f"Skipping completed stage {stage} ({completed_stage_epochs}/{epochs} epochs)")
            continue
        stage_epochs_without_improvement = 0
        model.set_backbone_trainable(unfrozen)
        optimizer = build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        for stage_epoch in range(completed_stage_epochs, epochs):
            global_epoch += 1
            loader, sampler = create_train_loader(train_records, cfg, stage, stage_epoch)
            sampler.set_epoch(global_epoch)
            started = time.perf_counter()
            train_metrics = train_one_epoch(model, loader, optimizer, device, cfg, scaler)
            scheduler.step()
            if quick_smoke:
                val_metrics: dict[str, dict[str, float]] = {}
            else:
                val_metrics, _ = evaluate_splits(
                    model, index, cfg, device, splits=("val_seen", "val_unseen")
                )
            monitor = val_metrics.get("val_seen", {}).get("recall_at_1", -train_metrics["loss"])
            row = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch + 1,
                "elapsed_seconds": time.perf_counter() - started,
                **train_metrics,
            }
            for split, values in val_metrics.items():
                row.update({f"{split}_{key}": value for key, value in values.items()})
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            _save_checkpoint(Path(cfg.models_dir) / "last.pt", model, cfg, global_epoch, stage, val_metrics)
            if monitor > best_recall:
                best_recall = monitor
                stage_epochs_without_improvement = 0
                _save_checkpoint(Path(cfg.models_dir) / "best.pt", model, cfg, global_epoch, stage, val_metrics)
            else:
                stage_epochs_without_improvement += 1
            print(json.dumps(row, indent=2))
            if stage_epochs_without_improvement >= cfg.patience:
                print(f"Early stopping after {cfg.patience} epochs without improvement")
                break

    if quick_smoke:
        final_metrics: dict[str, dict[str, float]] = {}
        details: dict[str, Any] = {}
    else:
        best_model, _ = load_trained_model(Path(cfg.models_dir) / "best.pt", cfg.weights_path, device)
        final_metrics, details = evaluate_splits(best_model, index, cfg, device)
        (run_dir / "final_metrics.json").write_text(
            json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        export_gallery_embeddings(details, run_dir / "gallery_embeddings.pt")
        plot_training_history(run_dir / "history.csv", run_dir / "training_curves.png")
        plot_retrieval_failures(details, run_dir / "retrieval_failures.png")

    result = {
        "device": str(device),
        "best_checkpoint": str(Path(cfg.models_dir) / "best.pt"),
        "last_checkpoint": str(Path(cfg.models_dir) / "last.pt"),
        "run_dir": str(run_dir),
        "split_summary": split_summary,
        "final_metrics": final_metrics,
        "quick_smoke": quick_smoke,
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def export_gallery_embeddings(details: dict[str, Any], output_path: str | Path) -> None:
    required = {"gallery_embeddings", "gallery_labels", "gallery_paths", "gallery_slugs"}
    if not required.issubset(details):
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": details["gallery_embeddings"],
            "labels": details["gallery_labels"],
            "paths": details["gallery_paths"],
            "wine_slugs": details["gallery_slugs"],
        },
        output_path,
    )


def plot_training_history(history_path: str | Path, output_path: str | Path) -> None:
    history = pd.read_csv(history_path)
    if history.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for column in ("loss", "supcon_loss", "ce_loss"):
        if column in history:
            axes[0].plot(history["epoch"], history[column], marker="o", label=column)
    axes[0].set(title="Training losses", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    for column in ("val_seen_recall_at_1", "val_unseen_recall_at_1", "val_seen_recall_at_5"):
        if column in history:
            axes[1].plot(history["epoch"], history[column], marker="o", label=column)
    axes[1].set(title="Retrieval validation", xlabel="Epoch", ylabel="Recall")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_failures(details: dict[str, Any], output_path: str | Path, n: int = 12) -> None:
    split = "val_seen" if "val_seen" in details else next(
        (key for key in ("val_unseen", "val_hard") if key in details), None
    )
    if split is None:
        return
    payload = details[split]
    ranks = payload["ranks"]
    worst = torch.argsort(ranks, descending=True)[:n]
    gallery_embeddings = F.normalize(details["gallery_embeddings"].float(), dim=-1)
    gallery_paths = details["gallery_paths"]
    gallery_slugs = details["gallery_slugs"]
    cols = 4
    rows = math.ceil(len(worst) / cols)
    fig, axes = plt.subplots(rows * 2, cols, figsize=(16, rows * 6))
    axes = np.atleast_2d(axes)
    for column, query_index in enumerate(worst.tolist()):
        grid_row = (column // cols) * 2
        grid_col = column % cols
        query_embedding = F.normalize(payload["embeddings"][query_index].float(), dim=-1)
        predicted_index = int((query_embedding @ gallery_embeddings.T).argmax())
        query_image = open_rgb(Path(payload["paths"][query_index]))
        predicted_image = open_rgb(Path(gallery_paths[predicted_index]))
        axes[grid_row, grid_col].imshow(query_image)
        axes[grid_row, grid_col].set_title(
            f"Query rank={int(ranks[query_index])}\ntrue={payload['slugs'][query_index][:28]}"
        )
        axes[grid_row + 1, grid_col].imshow(predicted_image)
        axes[grid_row + 1, grid_col].set_title(f"Top-1={gallery_slugs[predicted_index][:28]}")
        axes[grid_row, grid_col].axis("off")
        axes[grid_row + 1, grid_col].axis("off")
    for axis in axes.ravel():
        if not axis.has_data():
            axis.axis("off")
    fig.suptitle(f"Worst retrieval examples: {split}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_transforms(config: PipelineConfig, n: int = 6) -> None:
    cfg = config.resolved()
    index = pd.read_csv(cfg.index_path)
    records = index[index["split"].eq("train")].sample(n=min(n, sum(index.split.eq("train"))), random_state=cfg.seed)
    train_transform, _ = build_transforms(cfg.image_size)
    dataset = WineRetrievalDataset(records, cfg.crops_root, cfg.refs_root, train_transform, two_views=True)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    fig, axes = plt.subplots(len(dataset), 2, figsize=(7, 3 * len(dataset)))
    axes = np.atleast_2d(axes)
    for idx in range(len(dataset)):
        item = dataset[idx]
        for col, key in enumerate(("view1", "view2")):
            image = (item[key] * std + mean).clamp(0, 1).permute(1, 2, 0)
            axes[idx, col].imshow(image)
            axes[idx, col].set_title(f"{key}: {item['wine_slug'][:42]}")
            axes[idx, col].axis("off")
    fig.tight_layout()
    plt.show()


def benchmark_model(
    model: DINOv3RetrievalModel,
    loader: DataLoader,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, float]:
    model.eval()
    iterator = iter(loader)
    latencies: list[float] = []

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    with torch.inference_mode():
        for idx in range(warmup + iterations):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            images = batch["image"].to(device)
            sync()
            start = time.perf_counter()
            model(images)
            sync()
            elapsed_ms = (time.perf_counter() - start) * 1000 / len(images)
            if idx >= warmup:
                latencies.append(elapsed_ms)
    values = np.asarray(latencies)
    return {
        "average_ms_per_image": float(values.mean()),
        "p50_ms_per_image": float(np.percentile(values, 50)),
        "p95_ms_per_image": float(np.percentile(values, 95)),
        "images_per_second": float(1000.0 / values.mean()),
    }
