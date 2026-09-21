from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISCANER_", env_file=ROOT / ".env", extra="ignore")

    model_provider: Literal["demo", "local", "remote"] = "demo"
    catalog_path: Path = ROOT / "datasets/wine-scanner/data/catalog.csv"
    catalog_archive: Path = ROOT / "datasets/wine-scanner_code-catalog.zip"
    refs_root: Path = ROOT / "datasets/wine-scanner/data/refs"
    data_dir: Path = ROOT / "backend/data"
    checkpoint_path: Path = ROOT / "models/dinov3_retrieval/best.pt"
    gallery_path: Path = ROOT / "runs/dinov3_retrieval/gallery_embeddings.pt"
    detector_path: Path | None = None
    device: str = "auto"
    remote_url: str = ""
    remote_api_key: SecretStr = SecretStr("")
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    min_similarity: float = Field(default=0.65, ge=-1, le=1)
    min_margin: float = Field(default=0.04, ge=0, le=2)
    max_upload_mb: int = Field(default=12, ge=1, le=30)
    max_pixels: int = Field(default=24_000_000, ge=1)
    history_limit: int = Field(default=100, ge=1, le=1000)
