"""Lightweight learned gate deciding whether the whole-bottle resolver is useful."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


BASE_NUMERIC_FEATURES = [
    *(f"b_top{index}_similarity" for index in range(1, 11)),
    "b_top1_top2_gap",
    "label_detector_confidence",
    "bottle_detector_confidence",
]
CATEGORICAL_FEATURES = ["bottle_status"]


def make_gate_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot build learned-gate features; missing columns: {sorted(missing)}")
    features = frame[BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for index in range(2, 11):
        features[f"b_top1_top{index}_gap"] = (
            frame["b_top1_similarity"] - frame[f"b_top{index}_similarity"]
        )
    features["b_top2_top3_gap"] = frame["b_top2_similarity"] - frame["b_top3_similarity"]
    return features


def expected_utility_from_classifier(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.named_steps["model"].classes_)
    positive = probabilities[:, np.flatnonzero(classes == 1)[0]]
    negative = probabilities[:, np.flatnonzero(classes == -1)[0]]
    return positive - negative


def save_gate_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_gate_artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Learned gate not found: {path}. Run `python train_cascade_gate.py` first."
        )
    payload = joblib.load(path)
    required = {"model", "model_type", "decision_threshold", "max_margin", "feature_columns"}
    if missing := required.difference(payload):
        raise ValueError(f"Invalid learned-gate artifact; missing keys: {sorted(missing)}")
    return payload


def predict_gate_scores(payload: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    features = make_gate_features(frame)
    model = payload["model"]
    model_type = str(payload["model_type"])
    if model_type == "logistic_expected_utility":
        return expected_utility_from_classifier(model, features)
    if model_type == "ridge_utility":
        return np.asarray(model.predict(features), dtype=np.float64)
    raise ValueError(f"Unsupported learned-gate model type: {model_type}")
