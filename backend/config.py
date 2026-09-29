"""Service settings. Every field can be set from the environment as VISCANER_<NAME> or in .env."""
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISCANER_", env_file=ROOT / ".env", extra="ignore")

    # five_stream: the recognition pipeline. demo: catalog only, uploads return 503.
    model_provider: Literal["five_stream", "demo"] = "five_stream"
    device: str = "auto"  # auto | cuda | cpu
    catalog_archive: Path = ROOT / "data/catalog.zip"
    data_dir: Path = ROOT / "backend/data"

    # When to answer. Confidence is the softmax over the Transformer's top-10
    # ranking logits. Chosen on held-out shelf photos with wines in and out of the
    # catalog, trading right answers against honest nulls.
    min_confidence: float = Field(default=0.45, ge=0, le=1)
    # Between this and min_confidence the site asks "which of these is yours?";
    # below it, "not recognised". None disables the choice.
    min_suggest_confidence: float | None = Field(default=0.20, ge=0, le=1)
    # Required gap between the top two confidences. Softmax already accounts for
    # the runner-up, so 0 by default.
    min_margin: float = Field(default=0.0, ge=0, le=1)

    max_upload_mb: int = Field(default=12, ge=1, le=30)
    max_pixels: int = Field(default=24_000_000, ge=1)
    history_limit: int = Field(default=100, ge=1, le=1000)
    timeout_seconds: float = Field(default=30, gt=0, le=300)

    # Sommelier (backend/sommelier.py): bge-m3 retrieval over the catalog, then an
    # LLM explains the choice. openrouter: any chat model via the OpenRouter API.
    # local: YandexGPT-5 Lite, 4-bit on the GPU (weights are not in the repo).
    sommelier_enabled: bool = False
    sommelier_backend: Literal["openrouter", "local"] = "openrouter"
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_model: str = "anthropic/claude-sonnet-5.5"
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    sommelier_llm_path: Path = ROOT / "models/llm/yandexgpt5-lite-8b-instruct"
    # A local directory or a Hugging Face model id (downloaded once, ~2.3 GB).
    sommelier_embedder: str = "BAAI/bge-m3"
    sommelier_profiles_path: Path = ROOT / "data/sommelier/profiles.jsonl"
