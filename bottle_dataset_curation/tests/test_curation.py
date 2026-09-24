from pathlib import Path

import pandas as pd
import torch

from bottle_dataset_curation.evaluate_all import _rank_all
from bottle_dataset_curation.export import balanced_replay
from bottle_dataset_curation.paths import hash_similarity


def test_hash_similarity_extremes() -> None:
    assert hash_similarity(0, 0) == 1.0
    assert hash_similarity(0, (1 << 64) - 1) == 0.0


def test_balanced_replay_is_bounded_and_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "true_slug": ["a"] * 6 + ["b"] * 6,
            "query_path": [str(Path("x") / f"{index}.jpg") for index in range(12)],
        }
    )
    first = balanced_replay(frame, count=6, seed=42)
    second = balanced_replay(frame, count=6, seed=42)
    assert len(first) == 6
    assert first["query_path"].tolist() == second["query_path"].tolist()
    assert set(first["true_slug"]) == {"a", "b"}


def test_rank_all_breaks_exact_ties_consistently() -> None:
    gallery = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    queries = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    rows = pd.DataFrame(
        {
            "split": ["test", "test"],
            "image_path": ["q0.jpg", "q1.jpg"],
            "wine_slug": ["first", "second"],
        }
    )
    audit = _rank_all(
        queries,
        torch.tensor([True, True]),
        rows,
        gallery,
        ["first", "second"],
        ["first.jpg", "second.jpg"],
        top_k=2,
        chunk_size=2,
    )
    assert audit["top1_slug"].tolist() == ["first", "first"]
    assert audit["rank"].tolist() == [1, 2]
