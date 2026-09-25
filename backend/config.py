from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISCANER_", env_file=ROOT / ".env", extra="ignore")

    model_provider: Literal["demo", "local", "cascade", "remote"] = "demo"
    catalog_path: Path = ROOT / "datasets/wine-scanner/data/catalog.csv"
    catalog_archive: Path = ROOT / "datasets/wine-scanner_code-catalog.zip"
    refs_root: Path = ROOT / "datasets/wine-scanner/data/refs"
    data_dir: Path = ROOT / "backend/data"
    checkpoint_path: Path = ROOT / "models/dinov3_retrieval/best.pt"
    gallery_path: Path = ROOT / "runs/dinov3_retrieval/gallery_embeddings.pt"
    detector_path: Path | None = None
    # Two-stage cascade (provider "cascade"). Defaults point at the checkpoints
    # the colleagues pushed and the galleries those checkpoints were trained on.
    primary_checkpoint: Path = ROOT / "models/trained_checkpoints/dinov3_vitb16_labels_best_full.pt"
    # Whole-bottle resolver. vitb16 is built the way it was trained: the pinned
    # DINOv3-B backbone via deeptune_backbones, then the fine-tuned checkpoint.
    resolver_checkpoint: Path | None = ROOT / "models/trained_checkpoints/dinov3_vitb16_bottles_best_full.pt"
    resolver_variant: Literal["vitb16", "vits16"] = "vitb16"
    resolver_backbone_path: Path | None = ROOT / "models/dinov3/model.safetensors"
    label_refs_root: Path = ROOT / "datasets/wine-scanner/data/refs/rgb"
    bottle_refs_root: Path | None = ROOT / "datasets/bottle_classifier_crops/refs"
    label_detector_path: Path | None = ROOT / "models/yolo_label_detector/best.pt"
    bottle_detector_path: Path | None = ROOT / "models/bottle_reranker/best_bottle_detector.pt"
    # Stage-1 gap at or below which the bottle model is consulted. 0.01525 is
    # the value tune_cascade_threshold.py selected on val_seen.
    ambiguity_margin: float = Field(default=0.01525, ge=0, le=1)
    # Separation the bottle model must show before its answer is accepted.
    min_resolver_margin: float = Field(default=0.02, ge=0, le=2)
    device: str = "auto"
    remote_url: str = ""
    remote_api_key: SecretStr = SecretStr("")
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    min_similarity: float = Field(default=0.65, ge=-1, le=1)
    min_margin: float = Field(default=0.04, ge=0, le=2)
    # Below min_similarity but at or above this, the top candidates are offered as a
    # choice instead of "not found". None = no such band.
    min_suggest_similarity: float | None = Field(default=None, ge=-1, le=1)
    max_upload_mb: int = Field(default=12, ge=1, le=30)
    max_pixels: int = Field(default=24_000_000, ge=1)
    history_limit: int = Field(default=100, ge=1, le=1000)
    # Local LLM sommelier (backend/sommelier.py). Off unless the weights are present.
    sommelier_enabled: bool = False
    sommelier_llm_path: Path = ROOT / "models/llm/yandexgpt5-lite-8b-instruct"
    sommelier_embedder_path: Path = ROOT / "models/llm/bge-m3"
    sommelier_profiles_path: Path = ROOT / "runs/sommelier/profiles.jsonl"
