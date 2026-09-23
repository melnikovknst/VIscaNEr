"""Configuration loading for the local DINOv3 cascade evaluator."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CascadeConfig:
    project_root: str = "."
    primary_checkpoint: str = "models/trained_checkpoints/dinov3_vitb16_labels_best_full.pt"
    resolver_checkpoint: str = "models/trained_checkpoints/dinov3_vits16_bottles_best_full.pt"
    label_metadata: str = "datasets/dinov3_target_crops/crops_metadata.csv"
    bottle_metadata: str = "datasets/bottle_classifier_crops/crops_metadata.csv"
    bottle_training_metadata: str = "datasets/bottle_classifier_crops/training_metadata.csv"
    source_manifest: str = "datasets/bottle_images_45k/bottle_images_manifest.csv"
    label_crops_root: str = "datasets/dinov3_target_crops"
    bottle_crops_root: str = "datasets/bottle_classifier_crops"
    label_refs_root: str = "datasets/wine-scanner/data/refs/rgb"
    bottle_refs_root: str = "datasets/bottle_classifier_crops/refs"
    output_dir: str = "runs/dino_cascade"
    device: str = "auto"
    ambiguity_margin: float = 0.03
    primary_batch_size: int = 24
    resolver_batch_size: int = 48
    ranking_batch_size: int = 512
    num_workers: int = 2
    top_k: int = 10
    seed: int = 42
    unseen_identity_fraction: float = 0.15
    val_seen_per_identity: int = 2
    visualization_examples: int = 12

    @classmethod
    def load(cls, path: str | Path) -> "CascadeConfig":
        config_path = Path(path).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        valid = {field.name for field in fields(cls)}
        if unknown := set(payload).difference(valid):
            raise ValueError(f"Unknown cascade config keys: {sorted(unknown)}")
        root = Path(payload.get("project_root", ".")).expanduser()
        if not root.is_absolute():
            root = (config_path.parent / root).resolve()
        payload["project_root"] = str(root)
        return cls(**payload).resolved()

    def with_overrides(self, **overrides: Any) -> "CascadeConfig":
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        payload.update({key: value for key, value in overrides.items() if value is not None})
        return CascadeConfig(**payload).resolved()

    def resolved(self) -> "CascadeConfig":
        root = Path(self.project_root).expanduser().resolve()
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        payload["project_root"] = str(root)
        path_fields = {
            "primary_checkpoint",
            "resolver_checkpoint",
            "label_metadata",
            "bottle_metadata",
            "bottle_training_metadata",
            "source_manifest",
            "label_crops_root",
            "bottle_crops_root",
            "label_refs_root",
            "bottle_refs_root",
            "output_dir",
        }
        for key in path_fields:
            value = Path(str(payload[key])).expanduser()
            if not value.is_absolute():
                value = root / value
            payload[key] = str(value.resolve())
        if not 0.0 <= float(payload["ambiguity_margin"]) <= 2.0:
            raise ValueError("ambiguity_margin must be in [0, 2]")
        if int(payload["top_k"]) < 10:
            raise ValueError("top_k must be at least 10 for Recall@10")
        return CascadeConfig(**payload)
