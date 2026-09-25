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
    # A cascade returns candidates already in their final order: re-sorting by
    # stage-1 similarity would undo a stage-2 decision.
    ranked: bool = False
    # Which stage made the call, and the margin that stage decided by. Absent
    # for single-stage providers, where top1 - top2 similarity is the margin.
    decision_basis: Literal["label", "resolver"] = "label"
    decision_margin: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    # Boxes, detector confidences and stage-2 details, for the result screen.
    pipeline: dict | None = None


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
    decision_basis: Literal["label", "resolver"] = "label"
    pipeline: dict | None = None
    # F1 is a dataset metric, never an individual prediction probability.
    metrics: dict = Field(default_factory=lambda: {"f1_top1": None, "f1_top5": None, "source": "not_evaluated"})


class PairingRequest(BaseModel):
    dish: Literal["meat", "fish", "cheese", "vegetables", "dessert"]
    preference: Literal["any", "red", "white", "rose", "sparkling"] = "any"
