"""Self-contained DINOv3-B retrieval architecture used by Stage-2C."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import DINOv3ViTConfig, DINOv3ViTModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def make_vitb16_config(image_size: int = 224) -> DINOv3ViTConfig:
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


class RetrievalModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        embedding_dim: int,
        projection_hidden_dim: int,
        ce_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_register_tokens = int(getattr(backbone.config, "num_register_tokens", 0))
        feature_dim = int(backbone.config.hidden_size) * 2
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_hidden_dim),
            nn.LayerNorm(projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(projection_hidden_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.ce_temperature = float(ce_temperature)
        self.backbone_frozen = True

    def backbone_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.backbone_frozen else nullcontext()
        with context:
            hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
            cls = hidden[:, 0]
            patch_start = 1 + self.num_register_tokens
            mean_patch = hidden[:, patch_start:].mean(dim=1)
        return torch.cat((cls, mean_patch), dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone_features(pixel_values)
        embeddings = F.normalize(self.projection(features), dim=-1)
        normalized_weights = F.normalize(self.classifier.weight, dim=-1)
        logits = embeddings @ normalized_weights.T / self.ce_temperature
        return embeddings, logits


def normalize_checkpoint_state(
    model: RetrievalModel, state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    expected = set(model.state_dict())
    checkpoint = set(state)
    expected_nested = any(key.startswith("backbone.model.layer.") for key in expected)
    checkpoint_nested = any(key.startswith("backbone.model.layer.") for key in checkpoint)
    normalized = state
    if expected_nested and not checkpoint_nested:
        normalized = {
            (key.replace("backbone.layer.", "backbone.model.layer.", 1)
             if key.startswith("backbone.layer.") else key): value
            for key, value in state.items()
        }
    elif checkpoint_nested and not expected_nested:
        normalized = {
            (key.replace("backbone.model.layer.", "backbone.layer.", 1)
             if key.startswith("backbone.model.layer.") else key): value
            for key, value in state.items()
        }
    if set(normalized) != expected:
        missing = sorted(expected - set(normalized))
        unexpected = sorted(set(normalized) - expected)
        raise RuntimeError(
            f"Checkpoint architecture mismatch; missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return normalized


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


def eval_transform(image_size: int = 224) -> transforms.Compose:
    if image_size % 16 != 0:
        raise ValueError("DINOv3-B/16 image size must be divisible by 16")
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], image_size: int) -> None:
        self.paths = list(paths)
        self.transform = eval_transform(image_size)
        self.blank = torch.zeros(3, image_size, image_size, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = Path(self.paths[index])
        try:
            with Image.open(path) as image:
                tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
            return {"pixel_values": tensor, "index": index, "valid": True, "error": ""}
        except Exception as exc:
            return {
                "pixel_values": self.blank.clone(), "index": index, "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
