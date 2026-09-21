import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="allow")
    f1_top1: float = Field(ge=0, le=1, allow_inf_nan=False)
    f1_top5: float = Field(ge=0, le=1, allow_inf_nan=False)
    samples: int = Field(gt=0)
    model_versions: list[str]
    source: str = "evaluation"


def read_metrics(path: Path) -> dict:
    empty = {"f1_top1": None, "f1_top5": None, "source": "not_evaluated"}
    if not path.is_file():
        return empty
    try:
        return EvaluationReport.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
    except (OSError, ValidationError, json.JSONDecodeError):
        return {**empty, "source": "invalid_report"}
