"""Load gated Hugging Face DINOv3 backbones into the shared retrieval engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from transformers import AutoModel

import dinov3_retrieval as retrieval


MODEL_VARIANTS = {
    "vits16": {
        "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "parameter_count": 21_000_000,
    },
    "vits16plus": {
        "model_id": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "parameter_count": 29_000_000,
    },
}


def validate_snapshot(snapshot: str | Path, expected_model_id: str) -> Path:
    root = Path(snapshot).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DINOv3 snapshot directory does not exist: {root}")
    for filename in ("config.json", "model.safetensors"):
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Incomplete DINOv3 snapshot, missing {filename}: {root}")
    marker_path = root / "viscaner_model_source.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        actual_model_id = marker.get("model_id")
        if actual_model_id != expected_model_id:
            raise ValueError(
                f"Wrong DINOv3 snapshot: expected {expected_model_id}, found {actual_model_id}"
            )
    return root


def make_loader(expected_model_id: str) -> Callable[[str | Path, int], Any]:
    def load_backbone(weights_path: str | Path, image_size: int = 224) -> Any:
        if image_size % 16:
            raise ValueError("DINOv3 ViT input size must be divisible by patch size 16")
        snapshot = validate_snapshot(weights_path, expected_model_id)
        backbone = AutoModel.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        config = backbone.config
        required = ("hidden_size", "num_hidden_layers", "num_register_tokens")
        if missing := [name for name in required if not hasattr(config, name)]:
            raise TypeError(f"Snapshot is not a supported DINOv3 ViT: missing {missing}")
        if int(config.hidden_size) != 384:
            raise ValueError(
                f"Expected a DINOv3 ViT-S/S+ hidden size of 384, got {config.hidden_size}"
            )
        if int(config.num_register_tokens) != 4:
            raise ValueError(
                f"Expected four DINOv3 register tokens, got {config.num_register_tokens}"
            )
        return backbone

    return load_backbone


def install_huggingface_backbone_loader(variant: str) -> str:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    model_id = str(MODEL_VARIANTS[variant]["model_id"])
    # The established trainer calls this module-level function both for fresh
    # initialization and checkpoint evaluation. Replacing the loader keeps all
    # split, loss, metric, visualization and resume behavior identical while
    # leaving the label-DINO implementation untouched on disk.
    retrieval.load_local_dinov3_backbone = make_loader(model_id)
    return model_id
