from __future__ import annotations

import pandas as pd
import torch

from cascade_resolver.evaluation import initialize_final_columns, ranking_metrics, resolve_top_two


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label_id": [1, 0, 2],
            "b_top1_label_id": [0, 0, 0],
            "b_top2_label_id": [1, 1, 1],
            "b_top1_slug": ["a", "a", "a"],
            "b_top2_slug": ["b", "b", "b"],
            "b_true_rank": [2, 1, 3],
        }
    )


def test_resolver_only_reorders_primary_top_two() -> None:
    predictions = initialize_final_columns(_predictions())
    query = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8]])
    gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    result = resolve_top_two(predictions, query, gallery, torch.tensor([True, True, True]))
    assert result["final_top1_label_id"].tolist() == [1, 0, 1]
    assert result["final_true_rank"].tolist() == [1, 1, 3]
    assert set(result["final_top1_label_id"]).issubset({0, 1})


def test_invalid_resolver_embedding_falls_back_to_primary() -> None:
    predictions = initialize_final_columns(_predictions().iloc[:1])
    result = resolve_top_two(
        predictions,
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        torch.tensor([False]),
    )
    assert result.loc[0, "final_top1_label_id"] == 0
    assert result.loc[0, "final_true_rank"] == 2


def test_ranking_metrics_include_accuracy_and_recall_at_two() -> None:
    metrics = ranking_metrics([1, 2, 7, 11])
    assert metrics["accuracy"] == 0.25
    assert metrics["recall_at_1"] == 0.25
    assert metrics["recall_at_2"] == 0.5
    assert metrics["recall_at_10"] == 0.75
