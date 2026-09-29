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
    # Softmax over the Transformer's top-10 ranking logits: 0..1, not a calibrated probability.
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class Prediction(BaseModel):
    # Best first, as the recognition pipeline ranked them.
    candidates: list[Candidate] = Field(default_factory=list, max_length=20)
    model_version: str = Field(default="unknown", max_length=100)
    # The pipeline found nothing to rank.
    abstain: bool = False
    # Detections, OCR text and raw logits, for debugging the result screen.
    pipeline: dict | None = None


class Match(BaseModel):
    wine: Wine
    confidence: float


class ScanResult(BaseModel):
    id: str
    status: Literal["matched", "uncertain", "not_found", "demo"]
    wine: Wine | None
    candidates: list[Match]
    confidence: float | None
    margin: float | None
    elapsed_ms: int
    model_version: str
    provider: str
    created_at: str
    message: str
    pipeline: dict | None = None


class SommelierRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    wine_slug: str | None = Field(default=None, max_length=300)


class SommelierAnswer(BaseModel):
    answer: str
    wines: list[Wine]
    cited: list[int]
    elapsed_ms: int
    model: str


class PairingRequest(BaseModel):
    dish: Literal["meat", "fish", "cheese", "vegetables", "dessert"]
    preference: Literal["any", "red", "white", "rose", "sparkling"] = "any"
