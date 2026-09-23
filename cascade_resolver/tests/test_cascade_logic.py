from __future__ import annotations

import pandas as pd
import torch

from dinov3_retrieval import retrieval_metrics
from cascade_resolver.evaluation import (
    initialize_final_columns,
    rank_primary,
    ranking_metrics,
    resolve_top_two,
)
from tune_cascade_threshold import simulate_threshold


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


def test_primary_ranking_breaks_exact_ties_by_gallery_index() -> None:
    queries = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    gallery = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = rank_primary(
        queries,
        gallery,
        true_label_ids=[0, 1],
        slugs=["a", "b", "c"],
        top_k=3,
        batch_size=2,
    )
    assert result["b_top1_label_id"].tolist() == [0, 0]
    assert result["b_true_rank"].tolist() == [1, 2]


def test_threshold_simulation_invokes_only_rows_inside_margin() -> None:
    frame = pd.DataFrame(
        {
            "label_id": [1, 1],
            "b_top1_label_id": [0, 0],
            "b_top2_label_id": [1, 1],
            "b_true_rank": [2, 2],
            "b_top1_top2_gap": [0.01, 0.02],
            "bottle_crop_available": [True, True],
            "resolver_embedding_valid": [True, True],
            "s_candidate1_similarity": [0.2, 0.2],
            "s_candidate2_similarity": [0.8, 0.8],
        }
    )
    ranks, invoked, final_top1 = simulate_threshold(frame, 0.01525)
    assert invoked.tolist() == [True, False]
    assert final_top1.tolist() == [1, 0]
    assert ranks.tolist() == [1, 2]


def test_training_metrics_use_same_deterministic_tie_policy() -> None:
    metrics, ranks = retrieval_metrics(
        query_embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        query_labels=torch.tensor([0, 1]),
        gallery_embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        gallery_labels=torch.tensor([0, 1]),
        ks=(1, 2),
    )
    assert ranks.tolist() == [1.0, 2.0]
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_2"] == 1.0
