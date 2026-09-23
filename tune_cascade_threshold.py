#!/usr/bin/env python3
"""Select a resource/quality operating point for the conditional DINO resolver."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "primary_split",
    "label_id",
    "b_top1_label_id",
    "b_top2_label_id",
    "b_true_rank",
    "b_top1_top2_gap",
    "bottle_crop_available",
    "resolver_embedding_valid",
    "s_candidate1_similarity",
    "s_candidate2_similarity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="runs/dino_cascade/predictions.csv")
    parser.add_argument("--output-dir", default="runs/dino_cascade/threshold_experiments")
    parser.add_argument("--calibration-split", default="val_seen")
    parser.add_argument("--evaluation-split", default="val_unseen")
    parser.add_argument("--max-threshold", type=float, default=0.03)
    parser.add_argument("--step", type=float, default=0.00025)
    return parser.parse_args()


def _bool_series(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy()
    return values.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()


def load_predictions(path: str | Path, max_threshold: float) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Predictions not found: {path}. Run `python run_dino_cascade.py` first."
        )
    frame = pd.read_csv(path, usecols=lambda column: column in REQUIRED_COLUMNS)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    frame["bottle_crop_available"] = _bool_series(frame["bottle_crop_available"])
    frame["resolver_embedding_valid"] = _bool_series(frame["resolver_embedding_valid"])
    should_have_scores = (
        frame["bottle_crop_available"]
        & frame["b_top1_top2_gap"].le(max_threshold + 1e-12)
    )
    has_scores = (
        frame["resolver_embedding_valid"]
        & frame["s_candidate1_similarity"].notna()
        & frame["s_candidate2_similarity"].notna()
    )
    missing_scores = should_have_scores & ~has_scores
    if missing_scores.any():
        largest_supported = frame.loc[has_scores, "b_top1_top2_gap"].max()
        raise RuntimeError(
            f"{int(missing_scores.sum())} rows need DINO-S scores for max_threshold={max_threshold:.5f}. "
            f"Current predictions support only about {largest_supported:.5f}. First run: "
            f"python run_dino_cascade.py --ambiguity-margin {max_threshold:.5f}"
        )
    return frame


def simulate_threshold(frame: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gap = frame["b_top1_top2_gap"].to_numpy()
    available = frame["bottle_crop_available"].to_numpy(dtype=bool)
    valid = frame["resolver_embedding_valid"].to_numpy(dtype=bool)
    score1 = frame["s_candidate1_similarity"].to_numpy()
    score2 = frame["s_candidate2_similarity"].to_numpy()
    invoked = (gap <= threshold) & available & valid & np.isfinite(score1) & np.isfinite(score2)
    swapped = invoked & (score2 > score1)

    candidate1 = frame["b_top1_label_id"].to_numpy(dtype=np.int64)
    candidate2 = frame["b_top2_label_id"].to_numpy(dtype=np.int64)
    truth = frame["label_id"].to_numpy(dtype=np.int64)
    final_top1 = np.where(swapped, candidate2, candidate1)
    final_top2 = np.where(swapped, candidate1, candidate2)
    ranks = frame["b_true_rank"].to_numpy(dtype=np.int64).copy()
    ranks[truth == final_top1] = 1
    ranks[truth == final_top2] = 2
    return ranks, invoked, final_top1


def metrics_for_scope(
    frame: pd.DataFrame,
    mask: np.ndarray,
    ranks: np.ndarray,
    invoked: np.ndarray,
    final_top1: np.ndarray,
    threshold: float,
    scope: str,
) -> dict[str, Any]:
    scoped_ranks = ranks[mask]
    scoped_invoked = invoked[mask]
    truth = frame["label_id"].to_numpy(dtype=np.int64)[mask]
    primary_top1 = frame["b_top1_label_id"].to_numpy(dtype=np.int64)[mask]
    scoped_final = final_top1[mask]
    primary_correct = primary_top1 == truth
    final_correct = scoped_final == truth
    n = int(mask.sum())
    recall1 = float(final_correct.mean()) if n else math.nan
    return {
        "threshold": threshold,
        "scope": scope,
        "num_queries": n,
        "accuracy": recall1,
        "recall_at_1": recall1,
        "recall_at_2": float(np.mean(scoped_ranks <= 2)) if n else math.nan,
        "recall_at_5": float(np.mean(scoped_ranks <= 5)) if n else math.nan,
        "recall_at_10": float(np.mean(scoped_ranks <= 10)) if n else math.nan,
        "mrr": float(np.mean(1.0 / scoped_ranks)) if n else math.nan,
        "invocations": int(scoped_invoked.sum()),
        "invocation_rate": float(scoped_invoked.mean()) if n else math.nan,
        "fixed": int((~primary_correct & final_correct).sum()),
        "harmed": int((primary_correct & ~final_correct).sum()),
        "net_corrections": int(final_correct.sum() - primary_correct.sum()),
        "recall_at_1_standard_error": (
            float(math.sqrt(recall1 * (1.0 - recall1) / n)) if n else math.nan
        ),
    }


def sweep_thresholds(
    frame: pd.DataFrame,
    max_threshold: float,
    step: float,
    calibration_split: str,
    evaluation_split: str,
) -> pd.DataFrame:
    if step <= 0 or max_threshold <= 0:
        raise ValueError("step and max_threshold must be positive")
    thresholds = np.arange(0.0, max_threshold + step / 2.0, step)
    thresholds = np.unique(np.append(thresholds[thresholds <= max_threshold], max_threshold))
    scopes = {
        "all_eligible": np.ones(len(frame), dtype=bool),
        calibration_split: frame["primary_split"].eq(calibration_split).to_numpy(),
        evaluation_split: frame["primary_split"].eq(evaluation_split).to_numpy(),
    }
    if not scopes[calibration_split].any() or not scopes[evaluation_split].any():
        raise ValueError("Calibration or evaluation split is empty")

    rows: list[dict[str, Any]] = []
    # Explicit no-resolver baseline. Threshold zero is still a valid operating
    # point because it resolves exact B-score ties.
    for threshold in np.insert(thresholds, 0, -1.0):
        ranks, invoked, final_top1 = simulate_threshold(frame, float(threshold))
        for scope, mask in scopes.items():
            rows.append(
                metrics_for_scope(
                    frame,
                    mask,
                    ranks,
                    invoked,
                    final_top1,
                    float(threshold),
                    scope,
                )
            )
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame, calibration_split: str) -> pd.Series:
    candidates = sweep[(sweep["scope"] == calibration_split) & sweep["threshold"].ge(0)].copy()
    # Highest calibration Recall@1 wins. Within numerical ties, prefer fewer
    # resolver calls and then the smaller threshold.
    candidates = candidates.sort_values(
        ["recall_at_1", "invocation_rate", "threshold"],
        ascending=[False, True, True],
    )
    return candidates.iloc[0]


def save_plot(
    sweep: pd.DataFrame,
    chosen_threshold: float,
    calibration_split: str,
    evaluation_split: str,
    path: Path,
) -> None:
    data = sweep[sweep["threshold"].ge(0)]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for scope, style in ((calibration_split, "-"), (evaluation_split, "--")):
        subset = data[data["scope"] == scope]
        axes[0].plot(subset["threshold"], 100 * subset["recall_at_1"], style, label=scope)
        axes[1].plot(subset["threshold"], 100 * subset["invocation_rate"], style, label=scope)
    for axis in axes:
        axis.axvline(chosen_threshold, color="black", alpha=0.55, linestyle=":", label="selected")
        axis.grid(alpha=0.25)
        axis.set_xlabel("DINO-B Top-1/Top-2 cosine gap threshold")
        axis.legend()
    axes[0].set_ylabel("Recall@1, %")
    axes[0].set_title("Quality")
    axes[1].set_ylabel("DINO-S invocation rate, %")
    axes[1].set_title("Compute cost")
    figure.suptitle("Conditional DINO resolver threshold sweep")
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


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
    frame = load_predictions(args.predictions, args.max_threshold)
    sweep = sweep_thresholds(
        frame,
        args.max_threshold,
        args.step,
        args.calibration_split,
        args.evaluation_split,
    )
    chosen = choose_threshold(sweep, args.calibration_split)
    threshold = float(chosen["threshold"])
    baseline = sweep[sweep["threshold"].eq(-1.0)].set_index("scope")
    selected = sweep[np.isclose(sweep["threshold"], threshold)].set_index("scope")
    evaluation = selected.loc[args.evaluation_split]
    calibration = selected.loc[args.calibration_split]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    save_plot(
        sweep,
        threshold,
        args.calibration_split,
        args.evaluation_split,
        output_dir / "threshold_tradeoff.png",
    )
    summary = {
        "selection_rule": "maximize calibration Recall@1; tie-break by lower invocation rate and threshold",
        "calibration_split": args.calibration_split,
        "evaluation_split": args.evaluation_split,
        "max_supported_threshold": args.max_threshold,
        "step": args.step,
        "recommended_threshold": threshold,
        "calibration": calibration.to_dict(),
        "evaluation": evaluation.to_dict(),
        "baseline_calibration": baseline.loc[args.calibration_split].to_dict(),
        "baseline_evaluation": baseline.loc[args.evaluation_split].to_dict(),
    }
    (output_dir / "recommended_threshold.json").write_text(
        json.dumps(_clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("=" * 88)
    print("THRESHOLD SWEEP | selection uses val_seen; val_unseen stays evaluation-only")
    print("=" * 88)
    print(
        f"SELECTED     | threshold={threshold:.5f} "
        f"| {args.calibration_split} R@1={calibration['recall_at_1']:.4f} "
        f"| calls={calibration['invocation_rate']:.2%}"
    )
    print(
        f"EVALUATION   | {args.evaluation_split} R@1={evaluation['recall_at_1']:.4f} "
        f"| baseline={baseline.loc[args.evaluation_split, 'recall_at_1']:.4f} "
        f"| delta={evaluation['recall_at_1'] - baseline.loc[args.evaluation_split, 'recall_at_1']:+.4f} "
        f"| calls={evaluation['invocation_rate']:.2%}"
    )
    print(
        f"CORRECTIONS  | fixed={int(evaluation['fixed'])} "
        f"| harmed={int(evaluation['harmed'])} "
        f"| net={int(evaluation['net_corrections']):+d}"
    )
    print(f"CSV          | {output_dir / 'threshold_sweep.csv'}")
    print(f"PLOT         | {output_dir / 'threshold_tradeoff.png'}")
    print(f"CONFIG       | set ambiguity_margin: {threshold:.5f}")


if __name__ == "__main__":
    main()
