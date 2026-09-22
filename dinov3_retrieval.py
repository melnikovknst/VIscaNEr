"""DINOv3 metric-learning pipeline for wine retrieval.

The module is intentionally independent of external model hubs at runtime. It
constructs a supported DINOv3 architecture and loads a project-local
``safetensors`` checkpoint. ViT-B/16 remains the default; other local
backbones can be installed by a task-specific entrypoint before training.
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
from transformers import DINOv3ViTConfig, DINOv3ViTModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class PipelineConfig:
    project_root: str
    weights_path: str = "models/dinov3/model.safetensors"
    crops_metadata_path: str = "datasets/dinov3_target_crops/crops_metadata.csv"
    bottle_manifest_path: str = "datasets/bottle_images_45k/bottle_images_manifest.csv"
    crops_root: str = "datasets/dinov3_target_crops"
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
    # Optional full-backbone phase. Zero keeps the established two-stage run.
    stage3_epochs: int = 0
    stage3_head_lr: float = 1e-4
    stage3_backbone_lr: float = 5e-6
    stage3_patience: int = 12
    stage3_min_epochs: int = 15
    stage3_warmup_epochs: int = 3
    max_session_hours: float = 10.5
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
    early_stopping_split: str = "val_unseen"
    early_stopping_metric: str = "recall_at_1"
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

    print(
        f"Preparing retrieval index from {metadata_path} "
        f"(validate_files={validate_files})",
        flush=True,
    )

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

    # New target-aligned crop metadata carries the identity explicitly. Keep
    # the original manifest as the source of truth and verify that the builder
    # did not associate a crop with a different wine. Older crop metadata did
    # not contain ``wine_slug`` and remains supported by the same code path.
    metadata_identity = metadata.get("wine_slug")
    metadata_for_merge = metadata.drop(columns=["wine_slug"], errors="ignore")
    crops = metadata_for_merge.merge(
        manifest[["source_path", "wine_slug"]],
        on="source_path",
        how="left",
        validate="one_to_one",
    )
    if crops["wine_slug"].isna().any():
        examples = crops.loc[crops["wine_slug"].isna(), "source_path"].head(5).tolist()
        raise ValueError(f"Some crops cannot be mapped to an identity: {examples}")
    if metadata_identity is not None:
        expected = metadata_identity.astype(str).reset_index(drop=True)
        actual = crops["wine_slug"].astype(str).reset_index(drop=True)
        mismatch = expected.ne(actual)
        if mismatch.any():
            examples = crops.loc[mismatch, ["source_path", "wine_slug"]].head(5).to_dict("records")
            raise ValueError(
                "Crop metadata identities disagree with the bottle manifest. "
                f"Examples: {examples}"
            )

    refs = _reference_lookup(refs_root)
    identities = sorted(crops["wine_slug"].astype(str).unique())
    missing_refs = sorted(set(identities) - set(refs))
    if missing_refs:
        raise ValueError(f"Missing reference images for {len(missing_refs)} identities: {missing_refs[:5]}")
    label_by_slug = {slug: idx for idx, slug in enumerate(identities)}
    available_crop_names: dict[str, set[str]] = {}
    if validate_files:
        statuses = {
            str(status)
            for status in crops["status"].dropna().unique()
            if str(status) in {"successful", "low_confidence"}
        }
        for status in sorted(statuses):
            status_dir = crops_root / status
            if not status_dir.is_dir():
                raise FileNotFoundError(f"Crop directory does not exist: {status_dir}")
            with os.scandir(status_dir) as entries:
                available_crop_names[status] = {
                    entry.name for entry in entries if entry.is_file()
                }
        print(
            "Crop inventory loaded: "
            + ", ".join(
                f"{status}={len(names)}"
                for status, names in sorted(available_crop_names.items())
            ),
            flush=True,
        )
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
            if validate_files and path.name not in available_crop_names["successful"]:
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
        if validate_files and path.name not in available_crop_names["low_confidence"]:
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
    print(
        f"Retrieval index ready: {len(index)} rows, {len(identities)} identities",
        flush=True,
    )
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
        backbone: nn.Module,
        num_classes: int,
        embedding_dim: int = 256,
        projection_hidden_dim: int = 512,
        ce_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_register_tokens = int(
            getattr(backbone.config, "num_register_tokens", 0)
        )
        if hasattr(backbone.config, "hidden_size"):
            feature_width = int(backbone.config.hidden_size)
        elif hasattr(backbone.config, "hidden_sizes"):
            feature_width = int(backbone.config.hidden_sizes[-1])
        else:
            raise TypeError(
                "Unsupported DINOv3 backbone config: expected hidden_size or hidden_sizes"
            )
        feature_dim = feature_width * 2
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
        if last_n_blocks == -1:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
        elif last_n_blocks < -1:
            raise ValueError(
                "Use -1 for the full backbone, 0 to freeze it, or a positive block count"
            )
        elif last_n_blocks > 0:
            encoder = getattr(self.backbone, "model", self.backbone)
            if hasattr(encoder, "layer"):
                # DINOv3 ViT: Transformers releases expose blocks either as
                # ``model.layer`` or directly as ``layer``.
                blocks = list(encoder.layer)
                for block in blocks[-last_n_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
                for parameter in self.backbone.norm.parameters():
                    parameter.requires_grad = True
            elif hasattr(encoder, "stages"):
                # DINOv3 ConvNeXt: count residual blocks across all stages.
                # If a selected block belongs to a stage, train that stage's
                # downsampling transition as well so its representation can
                # adapt coherently.
                staged_blocks = [
                    (stage, block)
                    for stage in encoder.stages
                    for block in stage.layers
                ]
                selected = staged_blocks[-last_n_blocks:]
                selected_stage_ids = {id(stage) for stage, _ in selected}
                for stage, block in selected:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
                for stage in encoder.stages:
                    if id(stage) in selected_stage_ids:
                        for parameter in stage.downsample_layers.parameters():
                            parameter.requires_grad = True
                for parameter in self.backbone.layer_norm.parameters():
                    parameter.requires_grad = True
            else:
                raise TypeError(
                    "Unsupported DINOv3 backbone: expected ViT layers or ConvNeXt stages"
                )
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
    total_batches = len(loader)
    progress_interval = max(1, math.ceil(total_batches / 10))
    print(f"  TRAIN       | 0/{total_batches} batches", flush=True)
    for batch_index, batch in enumerate(loader, start=1):
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
        if batch_index % progress_interval == 0 or batch_index == total_batches:
            percent = 100.0 * batch_index / max(total_batches, 1)
            print(
                f"  TRAIN {percent:5.1f}% | {batch_index}/{total_batches} batches "
                f"| loss={totals['loss'] / seen:.4f}",
                flush=True,
            )
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
    total_batches = len(loader)
    progress_interval = max(1, math.ceil(total_batches / 4))
    label = desc.upper()
    print(f"  {label:<11} | 0/{total_batches} batches", flush=True)
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        with _autocast_context(device, amp):
            batch_embeddings, _ = model(images)
        embeddings.append(batch_embeddings.float().cpu())
        labels.append(batch["label"].long().cpu())
        paths.extend(batch["path"])
        slugs.extend(batch["wine_slug"])
        if batch_index % progress_interval == 0 or batch_index == total_batches:
            percent = 100.0 * batch_index / max(total_batches, 1)
            print(
                f"  {label:<11} | {percent:5.1f}% "
                f"| {batch_index}/{total_batches} batches",
                flush=True,
            )
    return torch.cat(embeddings), torch.cat(labels), paths, slugs


def retrieval_metrics(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    ks: Sequence[int] = (1, 2, 5, 10),
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
    # Retrieval Top-1 accuracy is exactly Recall@1 when every query has one
    # correct gallery identity. Keep the explicit alias in every validation
    # and final metrics payload so downstream reports do not have to infer it.
    metrics["accuracy"] = metrics["recall_at_1"]
    metrics.update(
        {
            "mrr": float((1.0 / rank_tensor).mean()),
            "median_rank": float(rank_tensor.median()),
            "mean_rank": float(rank_tensor.mean()),
            "num_queries": int(len(rank_tensor)),
        }
    )
    return metrics, rank_tensor.long()


def print_epoch_report(
    row: dict[str, Any],
    val_metrics: dict[str, dict[str, float]],
    stage_epochs: int,
    best_monitor: float,
    improved: bool,
    epochs_without_improvement: int,
    total_stages: int = 3,
) -> None:
    """Print one stable, human-readable epoch block for notebook logs."""

    separator = "=" * 96
    stage = int(row["stage"])
    stage_epoch = int(row["stage_epoch"])
    print(separator, flush=True)
    print(
        f"EPOCH {int(row['epoch'])} | STAGE {stage}/{total_stages} "
        f"| STAGE EPOCH {stage_epoch}/{stage_epochs} "
        f"| {float(row['elapsed_seconds']) / 60.0:.1f} min",
        flush=True,
    )
    backbone_lr = float(row.get("backbone_lr", 0.0))
    backbone_lr_text = f"{backbone_lr:.3e}" if backbone_lr > 0 else "frozen"
    print(
        f"LR            | head={float(row.get('head_lr', 0.0)):.3e} "
        f"| backbone={backbone_lr_text}",
        flush=True,
    )
    print(
        "TRAIN         | "
        f"loss={float(row['loss']):.4f} "
        f"| supcon={float(row['supcon_loss']):.4f} "
        f"| ce={float(row['ce_loss']):.4f} "
        f"| accuracy={100.0 * float(row['train_accuracy']):.2f}%",
        flush=True,
    )
    for split in ("val_seen", "val_unseen", "val_hard"):
        values = val_metrics.get(split)
        if values is None:
            continue
        print(
            f"{split.upper():<13} | "
            f"accuracy={100.0 * float(values['accuracy']):.2f}% "
            f"| R@1={100.0 * float(values['recall_at_1']):.2f}% "
            f"| R@2={100.0 * float(values['recall_at_2']):.2f}% "
            f"| R@5={100.0 * float(values['recall_at_5']):.2f}% "
            f"| R@10={100.0 * float(values['recall_at_10']):.2f}% "
            f"| MRR={float(values['mrr']):.4f} "
            f"| median_rank={float(values['median_rank']):.1f} "
            f"| mean_rank={float(values['mean_rank']):.1f} "
            f"| queries={int(values['num_queries'])}",
            flush=True,
        )
    monitor_name = f"{row['early_stopping_split']}.{row['early_stopping_metric']}"
    print(
        f"MONITOR       | {monitor_name}="
        f"{100.0 * float(row['early_stopping_value']):.2f}% "
        f"| best={100.0 * best_monitor:.2f}% "
        f"| new_best={'yes' if improved else 'no'} "
        f"| no_improvement={epochs_without_improvement}",
        flush=True,
    )
    print(separator, flush=True)


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


def normalize_retrieval_checkpoint_state_dict(
    model: DINOv3RetrievalModel,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Adapt trained checkpoints across Transformers DINOv3 key layouts.

    Transformers releases expose encoder blocks either as ``backbone.layer``
    or ``backbone.model.layer``.  The tensors are equivalent; only the module
    path changed.  Keep strict loading after the targeted rename so genuine
    architecture mismatches still fail loudly.
    """

    expected_keys = set(model.state_dict())
    checkpoint_keys = set(state_dict)
    expected_nested = any(key.startswith("backbone.model.layer.") for key in expected_keys)
    checkpoint_nested = any(
        key.startswith("backbone.model.layer.") for key in checkpoint_keys
    )

    normalized = state_dict
    if expected_nested and not checkpoint_nested:
        normalized = {
            (
                key.replace("backbone.layer.", "backbone.model.layer.", 1)
                if key.startswith("backbone.layer.")
                else key
            ): value
            for key, value in state_dict.items()
        }
    elif checkpoint_nested and not expected_nested:
        normalized = {
            (
                key.replace("backbone.model.layer.", "backbone.layer.", 1)
                if key.startswith("backbone.model.layer.")
                else key
            ): value
            for key, value in state_dict.items()
        }

    normalized_keys = set(normalized)
    missing = sorted(expected_keys - normalized_keys)
    unexpected = sorted(normalized_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint architecture mismatch after DINOv3 key normalization. "
            f"Missing ({len(missing)}): {missing[:10]}; "
            f"unexpected ({len(unexpected)}): {unexpected[:10]}"
        )
    return normalized


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
    model_state = normalize_retrieval_checkpoint_state_dict(
        model, checkpoint["model_state_dict"]
    )
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    return model, checkpoint


def train_pipeline(
    config: PipelineConfig,
    quick_smoke: bool = False,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if config.stage3_epochs > 0:
        # Keep the default two-stage trainer compact while making the long,
        # exactly-resumable full-backbone schedule explicitly opt-in.
        from full_finetune import train_full_pipeline

        return train_full_pipeline(config, quick_smoke, resume_checkpoint)
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
    print(f"Loading DINOv3 backbone from {cfg.weights_path}", flush=True)
    backbone = load_local_dinov3_backbone(cfg.weights_path, cfg.image_size)
    print("DINOv3 backbone loaded", flush=True)
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
        resume_state = normalize_retrieval_checkpoint_state_dict(
            model, resume_payload["model_state_dict"]
        )
        model.load_state_dict(resume_state, strict=True)
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
    best_monitor = -1.0
    best_path = Path(cfg.models_dir) / "best.pt"
    if resume_payload is not None and best_path.is_file():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        best_monitor = float(
            best_payload.get("metrics", {})
            .get(cfg.early_stopping_split, {})
            .get(cfg.early_stopping_metric, -1.0)
        )
    elif resume_payload is not None:
        best_monitor = float(
            resume_payload.get("metrics", {})
            .get(cfg.early_stopping_split, {})
            .get(cfg.early_stopping_metric, -1.0)
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
            used_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            started = time.perf_counter()
            train_metrics = train_one_epoch(model, loader, optimizer, device, cfg, scaler)
            scheduler.step()
            if quick_smoke:
                val_metrics: dict[str, dict[str, float]] = {}
            else:
                val_metrics, _ = evaluate_splits(
                    model, index, cfg, device, splits=("val_seen", "val_unseen")
                )
            if quick_smoke:
                monitor = -train_metrics["loss"]
            else:
                monitored_split = val_metrics.get(cfg.early_stopping_split)
                if monitored_split is None:
                    raise ValueError(
                        f"Early-stopping split is unavailable: {cfg.early_stopping_split}"
                    )
                if cfg.early_stopping_metric not in monitored_split:
                    raise ValueError(
                        "Early-stopping metric is unavailable: "
                        f"{cfg.early_stopping_split}.{cfg.early_stopping_metric}"
                    )
                monitor = float(monitored_split[cfg.early_stopping_metric])
            row = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch + 1,
                "elapsed_seconds": time.perf_counter() - started,
                "head_lr": used_lrs[0],
                "backbone_lr": used_lrs[-1] if stage > 1 else 0.0,
                "early_stopping_split": cfg.early_stopping_split,
                "early_stopping_metric": cfg.early_stopping_metric,
                "early_stopping_value": monitor,
                **train_metrics,
            }
            for split, values in val_metrics.items():
                row.update({f"{split}_{key}": value for key, value in values.items()})
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            _save_checkpoint(Path(cfg.models_dir) / "last.pt", model, cfg, global_epoch, stage, val_metrics)
            improved = monitor > best_monitor
            if improved:
                best_monitor = monitor
                stage_epochs_without_improvement = 0
                _save_checkpoint(Path(cfg.models_dir) / "best.pt", model, cfg, global_epoch, stage, val_metrics)
            else:
                stage_epochs_without_improvement += 1
            print_epoch_report(
                row,
                val_metrics,
                stage_epochs=epochs,
                best_monitor=best_monitor,
                improved=improved,
                epochs_without_improvement=stage_epochs_without_improvement,
                total_stages=2,
            )
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
    for column in (
        "val_seen_recall_at_1",
        "val_seen_recall_at_2",
        "val_unseen_recall_at_1",
        "val_unseen_recall_at_2",
        "val_seen_recall_at_5",
    ):
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


def build_retrieval_audit(
    details: dict[str, Any],
    splits: Sequence[str] = ("val_seen", "val_unseen", "val_hard"),
    top_k: int = 5,
    chunk_size: int = 256,
) -> pd.DataFrame:
    """Create one inspectable row per query with its ranked gallery matches."""
    gallery_embeddings = F.normalize(details["gallery_embeddings"].float(), dim=-1)
    gallery_labels = details["gallery_labels"].long()
    gallery_paths = details["gallery_paths"]
    gallery_slugs = details["gallery_slugs"]
    gallery_by_label = {int(label): idx for idx, label in enumerate(gallery_labels.tolist())}
    effective_top_k = min(top_k, len(gallery_embeddings))
    rows: list[dict[str, Any]] = []

    for split in splits:
        if split not in details:
            continue
        payload = details[split]
        query_embeddings = F.normalize(payload["embeddings"].float(), dim=-1)
        query_labels = payload["labels"].long()
        for start in range(0, len(query_embeddings), chunk_size):
            stop = min(start + chunk_size, len(query_embeddings))
            similarities = query_embeddings[start:stop] @ gallery_embeddings.T
            top_scores, top_indices = torch.topk(similarities, k=effective_top_k, dim=1)
            labels = query_labels[start:stop]
            true_indices = torch.tensor(
                [gallery_by_label[int(label)] for label in labels], dtype=torch.long
            )
            true_scores = similarities[torch.arange(stop - start), true_indices]

            for local_index in range(stop - start):
                query_index = start + local_index
                ranked_indices = top_indices[local_index].tolist()
                ranked_scores = top_scores[local_index].tolist()
                row: dict[str, Any] = {
                    "split": split,
                    "query_path": payload["paths"][query_index],
                    "true_slug": payload["slugs"][query_index],
                    "true_reference_path": gallery_paths[int(true_indices[local_index])],
                    "rank": int(payload["ranks"][query_index]),
                    "true_similarity": float(true_scores[local_index]),
                    "top1_top2_margin": (
                        float(ranked_scores[0] - ranked_scores[1])
                        if len(ranked_scores) > 1
                        else float("nan")
                    ),
                }
                for position in range(effective_top_k):
                    gallery_index = int(ranked_indices[position])
                    prefix = f"top{position + 1}"
                    row[f"{prefix}_slug"] = gallery_slugs[gallery_index]
                    row[f"{prefix}_path"] = gallery_paths[gallery_index]
                    row[f"{prefix}_similarity"] = float(ranked_scores[position])
                rows.append(row)
    return pd.DataFrame(rows)


def _short_slug(value: str, limit: int = 42) -> str:
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def plot_retrieval_audit_group(
    records: pd.DataFrame,
    output_path: str | Path,
    title: str,
    n: int = 8,
) -> Path | None:
    """Plot query, true reference, Top-1 and Top-2 for selected audit rows."""
    if records.empty:
        return None
    selected = records.head(n).reset_index(drop=True)
    fig, axes = plt.subplots(len(selected), 4, figsize=(16, max(4, 3.6 * len(selected))))
    axes = np.atleast_2d(axes)
    for row_index, row in selected.iterrows():
        panels = (
            (
                row["query_path"],
                f"Query · rank={int(row['rank'])}\n{_short_slug(str(row['true_slug']))}",
            ),
            (
                row["true_reference_path"],
                f"Ground truth · sim={row['true_similarity']:.3f}\n"
                f"{_short_slug(str(row['true_slug']))}",
            ),
            (
                row["top1_path"],
                f"Top-1 · sim={row['top1_similarity']:.3f}\n"
                f"{_short_slug(str(row['top1_slug']))}",
            ),
            (
                row["top2_path"],
                f"Top-2 · sim={row['top2_similarity']:.3f}\n"
                f"{_short_slug(str(row['top2_slug']))}",
            ),
        )
        for column_index, (path, panel_title) in enumerate(panels):
            axes[row_index, column_index].imshow(open_rgb(Path(path)))
            axes[row_index, column_index].set_title(panel_title, fontsize=9)
            axes[row_index, column_index].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def save_retrieval_audit(
    details: dict[str, Any],
    output_dir: str | Path,
    splits: Sequence[str] = ("val_seen", "val_unseen", "val_hard"),
    n: int = 8,
) -> dict[str, Any]:
    """Save full ranking CSV plus worst-error and rank-2 visual sheets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_retrieval_audit(details, splits=splits, top_k=5)
    csv_path = output_dir / "retrieval_audit.csv"
    audit.to_csv(csv_path, index=False)
    images: dict[str, str] = {}
    category_counts: dict[str, dict[str, int]] = {}

    for split in splits:
        split_rows = audit[audit["split"].eq(split)].copy()
        if split_rows.empty:
            continue
        errors = split_rows[split_rows["rank"].gt(1)].copy()
        worst = errors.sort_values(
            ["rank", "true_similarity"], ascending=[False, True]
        )
        rank2 = errors[errors["rank"].eq(2)].copy()
        rank2["wrong_lead"] = rank2["top1_similarity"] - rank2["true_similarity"]
        rank2 = rank2.sort_values("wrong_lead", ascending=False)
        category_counts[split] = {
            "queries": int(len(split_rows)),
            "errors": int(len(errors)),
            "rank2_errors": int(len(rank2)),
        }

        for category, records, title in (
            ("worst", worst, f"{split}: worst retrieval errors"),
            ("rank2", rank2, f"{split}: ground truth is Top-2"),
        ):
            image_path = plot_retrieval_audit_group(
                records,
                output_dir / f"{split}_{category}.png",
                title,
                n=n,
            )
            if image_path is not None:
                images[f"{split}_{category}"] = str(image_path.resolve())

    summary = {
        "audit_csv": str(csv_path.resolve()),
        "images": images,
        "category_counts": category_counts,
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


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
