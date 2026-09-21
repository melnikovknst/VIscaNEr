from typing import Literal
from pydantic import BaseModel, Field


class Wine(BaseModel):
    slug: str
    name: str
    category: str
    color: str
    region: str
    grapes: str
    description: str
    winery: str
    image_url: str
    source_url: str
    near_dup_group: str = ""
    rosquality_rating: float | None = None


class Candidate(BaseModel):
    slug: str = Field(min_length=1, max_length=300)
    similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)


class Prediction(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list, max_length=20)
    model_version: str = Field(default="unknown", max_length=100)
    # Explicit abstention from an upstream detector takes precedence over scores.
    abstain: bool = False


class Match(BaseModel):
    wine: Wine
    similarity: float


class ScanResult(BaseModel):
    id: str
    status: Literal["matched", "uncertain", "not_found", "demo"]
    wine: Wine | None
    candidates: list[Match]
    similarity: float | None
    margin: float | None
    elapsed_ms: int
    model_version: str
    provider: str
    created_at: str
    message: str
    # F1 is a dataset metric, never an individual prediction probability.
    metrics: dict = Field(default_factory=lambda: {"f1_top1": None, "f1_top5": None, "source": "not_evaluated"})


class PairingRequest(BaseModel):
    dish: Literal["meat", "fish", "cheese", "vegetables", "dessert"]
    preference: Literal["any", "red", "white", "rose", "sparkling"] = "any"
