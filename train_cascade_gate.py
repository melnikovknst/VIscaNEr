#!/usr/bin/env python3
"""Train and validate a lightweight gate for conditional DINO-S execution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cascade_resolver.gate import (
    CATEGORICAL_FEATURES,
    expected_utility_from_classifier,
    make_gate_features,
    save_gate_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="runs/dino_cascade/predictions.csv")
    parser.add_argument("--model-output", default="models/cascade_gate/dino_s_gate.joblib")
    parser.add_argument("--output-dir", default="runs/dino_cascade/gate_experiments")
    parser.add_argument("--max-margin", type=float, default=0.03)
    return parser.parse_args()


def add_utility_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    primary_correct = result["label_id"].eq(result["b_top1_label_id"])
    resolver_top1 = result["b_top2_label_id"].where(
        result["s_candidate2_similarity"] > result["s_candidate1_similarity"],
        result["b_top1_label_id"],
    )
    resolver_correct = result["label_id"].eq(resolver_top1)
    result["gate_utility"] = resolver_correct.astype(int) - primary_correct.astype(int)
    result["resolver_candidate_top1"] = resolver_top1.astype(int)
    return result


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = [column for column in CATEGORICAL_FEATURES if column in features]
    numeric = [column for column in features.columns if column not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def tune_decision_threshold(
    frame: pd.DataFrame,
    scores: np.ndarray,
    split: str = "val_seen",
) -> dict[str, Any]:
    mask = frame["primary_split"].eq(split).to_numpy()
    if not mask.any():
        raise ValueError(f"Empty gate calibration split: {split}")
    truth = frame["label_id"].to_numpy(dtype=np.int64)
    primary = frame["b_top1_label_id"].to_numpy(dtype=np.int64)
    resolver = frame["resolver_candidate_top1"].to_numpy(dtype=np.int64)
    thresholds = np.r_[scores[mask].max() + 1.0, np.unique(scores[mask])]
    best: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    for threshold in thresholds:
        invoked = scores >= threshold
        final = np.where(invoked, resolver, primary)
        correct = final == truth
        scoped_correct = correct[mask]
        scoped_invoked = invoked[mask]
        row = {
            "decision_threshold": float(threshold),
            "recall_at_1": float(scoped_correct.mean()),
            "invocation_rate_within_candidate_pool": float(scoped_invoked.mean()),
            "invocations": int(scoped_invoked.sum()),
            "fixed": int(((primary != truth) & (final == truth) & mask).sum()),
            "harmed": int(((primary == truth) & (final != truth) & mask).sum()),
        }
        row["net_corrections"] = row["fixed"] - row["harmed"]
        key = (row["recall_at_1"], -row["invocation_rate_within_candidate_pool"], -float(threshold))
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    return best[1]


def evaluate_on_full_split(
    all_rows: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_scores: np.ndarray,
    threshold: float,
    split: str,
) -> dict[str, Any]:
    scoped_all = all_rows["primary_split"].eq(split).to_numpy()
    truth = all_rows["label_id"].to_numpy(dtype=np.int64)
    primary = all_rows["b_top1_label_id"].to_numpy(dtype=np.int64)
    final = primary.copy()
    invoked = np.zeros(len(all_rows), dtype=bool)
    candidate_invoke = candidate_scores >= threshold
    candidate_positions = candidates["_all_row_position"].to_numpy(dtype=np.int64)
    invoked[candidate_positions[candidate_invoke]] = True
    final[candidate_positions[candidate_invoke]] = candidates.loc[
        candidate_invoke, "resolver_candidate_top1"
    ].to_numpy(dtype=np.int64)
    base_correct = primary == truth
    final_correct = final == truth
    n = int(scoped_all.sum())
    return {
        "split": split,
        "num_queries": n,
        "baseline_recall_at_1": float(base_correct[scoped_all].mean()),
        "recall_at_1": float(final_correct[scoped_all].mean()),
        "delta_recall_at_1": float(final_correct[scoped_all].mean() - base_correct[scoped_all].mean()),
        "invocations": int((invoked & scoped_all).sum()),
        "invocation_rate": float(invoked[scoped_all].mean()),
        "fixed": int((~base_correct & final_correct & scoped_all).sum()),
        "harmed": int((base_correct & ~final_correct & scoped_all).sum()),
        "net_corrections": int((final_correct & scoped_all).sum() - (base_correct & scoped_all).sum()),
    }


def train_candidates(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
) -> list[dict[str, Any]]:
    train_mask = candidates["primary_split"].eq("train").to_numpy()
    y = candidates["gate_utility"].to_numpy(dtype=np.int64)
    specs: list[tuple[str, str, Any, Callable[[Any, pd.DataFrame], np.ndarray]]] = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        specs.append(
            (
                f"ridge_alpha_{alpha:g}",
                "ridge_utility",
                Ridge(alpha=alpha),
                lambda model, values: np.asarray(model.predict(values), dtype=np.float64),
            )
        )
    for c_value in (0.01, 0.1, 1.0, 10.0):
        specs.append(
            (
                f"logistic_c_{c_value:g}",
                "logistic_expected_utility",
                LogisticRegression(C=c_value, max_iter=2000),
                expected_utility_from_classifier,
            )
        )

    trained: list[dict[str, Any]] = []
    for name, model_type, estimator, scorer in specs:
        pipeline = Pipeline([("preprocessor", make_preprocessor(features)), ("model", estimator)])
        pipeline.fit(features.loc[train_mask], y[train_mask])
        scores = scorer(pipeline, features)
        operating_point = tune_decision_threshold(candidates, scores, "val_seen")
        trained.append(
            {
                "name": name,
                "model_type": model_type,
                "model": pipeline,
                "scores": scores,
                **operating_point,
            }
        )
    return trained


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if hasattr(value, "item"):
        return _clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    all_rows = pd.read_csv(predictions_path).reset_index(drop=True)
    all_rows["_all_row_position"] = np.arange(len(all_rows))
    expected_candidates = (
        all_rows["b_top1_top2_gap"].le(args.max_margin + 1e-12)
        & all_rows["bottle_crop_available"].astype(bool)
    )
    complete_scores = (
        all_rows["resolver_embedding_valid"].astype(bool)
        & all_rows["s_candidate1_similarity"].notna()
        & all_rows["s_candidate2_similarity"].notna()
    )
    missing_supervision = expected_candidates & ~complete_scores
    if missing_supervision.any():
        raise RuntimeError(
            f"Missing DINO-S supervision for {int(missing_supervision.sum())} candidate rows. "
            "Generate an unbiased broad-margin table first: "
            f"python run_dino_cascade.py --gating-strategy margin "
            f"--ambiguity-margin {args.max_margin}"
        )
    candidate_mask = (
        expected_candidates & complete_scores
    )
    candidates = add_utility_target(all_rows.loc[candidate_mask].copy()).reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError(
            "No DINO-S training targets are available. First run "
            f"`python run_dino_cascade.py --ambiguity-margin {args.max_margin}`."
        )
    features = make_gate_features(candidates)
    trained = train_candidates(candidates, features)
    trained.sort(
        key=lambda item: (
            float(item["recall_at_1"]),
            -float(item["invocation_rate_within_candidate_pool"]),
        ),
        reverse=True,
    )
    winner = trained[0]

    report_rows: list[dict[str, Any]] = []
    for item in trained:
        row = {
            key: value
            for key, value in item.items()
            if key not in {"model", "scores"}
        }
        for split in ("val_seen", "val_unseen"):
            metrics = evaluate_on_full_split(
                all_rows,
                candidates,
                item["scores"],
                float(item["decision_threshold"]),
                split,
            )
            row.update({f"{split}_{key}": value for key, value in metrics.items() if key != "split"})
        report_rows.append(row)
    report = pd.DataFrame(report_rows).sort_values(
        ["val_seen_recall_at_1", "val_seen_invocation_rate"], ascending=[False, True]
    )

    artifact = {
        "model": winner["model"],
        "model_name": winner["name"],
        "model_type": winner["model_type"],
        "decision_threshold": float(winner["decision_threshold"]),
        "max_margin": float(args.max_margin),
        "feature_columns": features.columns.tolist(),
        "training_split": "train",
        "calibration_split": "val_seen",
        "evaluation_split": "val_unseen",
    }
    save_gate_artifact(args.model_output, artifact)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "model_comparison.csv", index=False)

    winner_report = report.loc[report["name"].eq(winner["name"])].iloc[0].to_dict()
    summary = {"winner": winner_report, "artifact": {key: value for key, value in artifact.items() if key != "model"}}
    (output_dir / "gate_summary.json").write_text(
        json.dumps(_clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    Path(args.model_output).with_suffix(".json").write_text(
        json.dumps(_clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(
        100 * report["val_seen_invocation_rate"],
        100 * report["val_seen_recall_at_1"],
        c=report["model_type"].eq("logistic_expected_utility").map({True: "tab:orange", False: "tab:blue"}),
    )
    for _, row in report.iterrows():
        axis.annotate(row["name"], (100 * row["val_seen_invocation_rate"], 100 * row["val_seen_recall_at_1"]), fontsize=7)
    axis.set_xlabel("DINO-S invocation rate on val_seen, %")
    axis.set_ylabel("Cascade Recall@1 on val_seen, %")
    axis.grid(alpha=0.25)
    axis.set_title("Learned gate model selection (selection split only)")
    figure.tight_layout()
    figure.savefig(output_dir / "gate_model_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    print("=" * 92)
    print("LEARNED DINO-S GATE | train=train, select=val_seen, evaluate=val_unseen")
    print("=" * 92)
    print(
        f"WINNER      | {winner['name']} ({winner['model_type']}) "
        f"| score_threshold={winner['decision_threshold']:.6f}"
    )
    print(
        f"VAL_SEEN    | R@1={winner_report['val_seen_recall_at_1']:.4f} "
        f"| calls={winner_report['val_seen_invocation_rate']:.2%} "
        f"| net={int(winner_report['val_seen_net_corrections']):+d}"
    )
    print(
        f"VAL_UNSEEN  | R@1={winner_report['val_unseen_recall_at_1']:.4f} "
        f"| baseline={winner_report['val_unseen_baseline_recall_at_1']:.4f} "
        f"| calls={winner_report['val_unseen_invocation_rate']:.2%} "
        f"| net={int(winner_report['val_unseen_net_corrections']):+d}"
    )
    print(f"MODEL       | {Path(args.model_output)}")
    print(f"REPORT      | {output_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
