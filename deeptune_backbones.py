"""Project-local DINOv3-B backbone loaders for deep retrieval fine-tuning.

No model hub or network lookup is used. Every checkpoint is identified by an
explicit SHA256 digest and loaded strictly into its matching architecture.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import torch.nn as nn
from safetensors.torch import load_file
from transformers import (
    DINOv3ConvNextConfig,
    DINOv3ConvNextModel,
    DINOv3ViTConfig,
    DINOv3ViTModel,
)

import dinov3_retrieval as retrieval


MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "vitb16": {
        "display_name": "DINOv3 ViT-B/16",
        "filename": "model.safetensors",
        "sha256": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
    },
    "convnextb": {
        "display_name": "DINOv3 ConvNeXt-B",
        "filename": "dinov3-convnext-b.safetensors",
        "sha256": "ec90bd798b5fc5b8e30443796a6c24a7a73e28ad85c6c0ceda78b1d249a694cc",
    },
}


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_weights(weights_path: str | Path, variant: str) -> Path:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown deep-tuning backbone: {variant}")
    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights do not exist: {path}")
    expected = str(MODEL_VARIANTS[variant]["sha256"])
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"Wrong or corrupted weights for {variant}: expected {expected}, got {actual}"
        )
    return path


def _vit_config(image_size: int) -> DINOv3ViTConfig:
    return retrieval.make_dinov3_vitb16_config(image_size)


def _convnext_config(image_size: int) -> DINOv3ConvNextConfig:
    return DINOv3ConvNextConfig(
        image_size=image_size,
        num_channels=3,
        hidden_sizes=[128, 256, 512, 1024],
        depths=[3, 3, 27, 3],
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.0,
    )


def _normalize_vit_state(
    state: dict[str, Any], model: DINOv3ViTModel
) -> dict[str, Any]:
    expected = set(model.state_dict())
    expects_model_prefix = any(key.startswith("model.layer.") for key in expected)
    has_model_prefix = any(key.startswith("model.layer.") for key in state)
    if expects_model_prefix and not has_model_prefix:
        return {
            (f"model.{key}" if key.startswith("layer.") else key): value
            for key, value in state.items()
        }
    if has_model_prefix and not expects_model_prefix:
        return {
            (key.removeprefix("model.") if key.startswith("model.layer.") else key): value
            for key, value in state.items()
        }
    return state


def _normalize_convnext_state(state: dict[str, Any]) -> dict[str, Any]:
    # The original checkpoint stores encoder tensors as ``stages.*`` while
    # Transformers nests the encoder below ``model``.
    return {
        (f"model.{key}" if key.startswith("stages.") else key): value
        for key, value in state.items()
    }


def make_loader(variant: str) -> Callable[[str | Path, int], nn.Module]:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown deep-tuning backbone: {variant}")

    def load_backbone(weights_path: str | Path, image_size: int = 224) -> nn.Module:
        path = validate_weights(weights_path, variant)
        raw_state = load_file(str(path), device="cpu")
        if variant == "vitb16":
            model: nn.Module = DINOv3ViTModel(_vit_config(image_size))
            state = _normalize_vit_state(raw_state, model)
        else:
            model = DINOv3ConvNextModel(_convnext_config(image_size))
            state = _normalize_convnext_state(raw_state)
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: {result}")
        return model

    return load_backbone


def install_local_backbone_loader(variant: str) -> dict[str, Any]:
    """Install one loader for the lifetime of the current training process."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown deep-tuning backbone: {variant}")
    retrieval.load_local_dinov3_backbone = make_loader(variant)
    return dict(MODEL_VARIANTS[variant])
