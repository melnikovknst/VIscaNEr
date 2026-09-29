"""Strict loader for the joint Stage-2C DINOv3-B checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from transformers import DINOv3ViTModel

from five_stream_transformer.dino import (
    RetrievalModel,
    make_vitb16_config,
    normalize_checkpoint_state,
)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _build_branch(
    state: dict[str, torch.Tensor], image_size: int, device: torch.device
) -> RetrievalModel:
    required = {"classifier.weight", "projection.0.weight", "projection.4.weight"}
    if missing := required.difference(state):
        raise ValueError(f"Stage-2C branch is missing tensors: {sorted(missing)}")
    classifier = state["classifier.weight"]
    projection_in = state["projection.0.weight"]
    projection_out = state["projection.4.weight"]
    num_classes = int(classifier.shape[0])
    embedding_dim = int(classifier.shape[1])
    projection_hidden_dim = int(projection_in.shape[0])
    if int(projection_out.shape[0]) != embedding_dim:
        raise ValueError("Stage-2C projection and classifier embedding dimensions differ")
    backbone = DINOv3ViTModel(make_vitb16_config(image_size))
    model = RetrievalModel(
        backbone=backbone,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        projection_hidden_dim=projection_hidden_dim,
        ce_temperature=0.07,
    )
    normalized = normalize_checkpoint_state(model, state)
    result = model.load_state_dict(normalized, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Stage-2C checkpoint mismatch: {result}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.backbone_frozen = True
    return model


def load_stage2c_models(
    checkpoint_path: str | Path,
    device: torch.device,
    image_size: int = 224,
    verify_sha256: bool = True,
) -> tuple[RetrievalModel, RetrievalModel, dict[str, Any]]:
    """Load both complete DINO-B branches from one Stage-2C artifact.

    No Stage-I checkpoint is consulted. The Stage-2C artifact contains the
    complete backbone, projection and classifier state for both branches.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"epoch", "label_model_state_dict", "bottle_model_state_dict"}
    if missing := required.difference(payload):
        raise ValueError(f"Not a complete Stage-2C checkpoint; missing: {sorted(missing)}")
    label_model = _build_branch(payload["label_model_state_dict"], image_size, device)
    bottle_model = _build_branch(payload["bottle_model_state_dict"], image_size, device)
    info = {
        "path": str(path),
        "sha256": file_sha256(path) if verify_sha256 else None,
        "sha256_verified": bool(verify_sha256),
        "format": "complete-stage2c-dual-dino-v1",
        "epoch": int(payload["epoch"]),
        "image_size": int(image_size),
        "num_classes": int(payload["bottle_model_state_dict"]["classifier.weight"].shape[0]),
        "embedding_dim": int(payload["bottle_model_state_dict"]["classifier.weight"].shape[1]),
        "manual_val": payload.get("manual_val"),
        "hard_val": payload.get("hard_val"),
    }
    return label_model, bottle_model, info
