"""Load the two project-local DINOv3 ViT-S backbones from safetensors files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

import dinov3_retrieval as retrieval


MODEL_VARIANTS = {
    "vits16": {
        "display_name": "DINOv3 ViT-S/16",
        "filename": "model-s.safetensors",
        "sha256": "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d",
        "hidden_act": "gelu",
        "use_gated_mlp": False,
        "parameter_count": 21_000_000,
    },
    "vits16plus": {
        "display_name": "DINOv3 ViT-S+/16",
        "filename": "model-s_plus.safetensors",
        "sha256": "208146e499dace99e4c9376ddb8a26f77d64c31c46c4dc4b86ff8bc63b0235e2",
        "hidden_act": "silu",
        "use_gated_mlp": True,
        "parameter_count": 29_000_000,
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
        raise ValueError(f"Unknown model variant: {variant}")
    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights do not exist: {path}")
    expected = str(MODEL_VARIANTS[variant]["sha256"])
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"Wrong or corrupted weights for {variant}: expected SHA256 {expected}, got {actual}"
        )
    return path


def make_config(variant: str, image_size: int) -> DINOv3ViTConfig:
    if image_size % 16:
        raise ValueError("DINOv3 ViT input size must be divisible by patch size 16")
    spec = MODEL_VARIANTS[variant]
    return DINOv3ViTConfig(
        image_size=image_size,
        patch_size=16,
        num_channels=3,
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=12,
        num_attention_heads=6,
        num_register_tokens=4,
        rope_theta=100.0,
        layerscale_value=1.0,
        pos_embed_rescale=2.0,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        layer_norm_eps=1e-5,
        hidden_act=str(spec["hidden_act"]),
        use_gated_mlp=bool(spec["use_gated_mlp"]),
    )


def normalize_state_dict(
    state: dict[str, Any], model: DINOv3ViTModel
) -> dict[str, Any]:
    expected_keys = set(model.state_dict())
    expects_model_prefix = any(key.startswith("model.layer.") for key in expected_keys)
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


def make_loader(variant: str) -> Callable[[str | Path, int], DINOv3ViTModel]:
    def load_backbone(
        weights_path: str | Path, image_size: int = 224
    ) -> DINOv3ViTModel:
        path = validate_weights(weights_path, variant)
        model = DINOv3ViTModel(make_config(variant, image_size))
        state = normalize_state_dict(load_file(str(path), device="cpu"), model)
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: {result}")
        return model

    return load_backbone


def install_local_backbone_loader(variant: str) -> dict[str, Any]:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    retrieval.load_local_dinov3_backbone = make_loader(variant)
    return dict(MODEL_VARIANTS[variant])
