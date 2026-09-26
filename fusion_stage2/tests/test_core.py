from __future__ import annotations

import pandas as pd
import torch

from fusion_stage2.core import FEATURE_NAMES, FusionMLP, build_feature_cache, score_cache


def test_candidate_union_and_fusion_shapes() -> None:
    slugs = ["a", "b", "c", "d"]
    frame = pd.DataFrame(
        {
            "source_relative_path": ["train/a/0.jpg", "train/c/0.jpg"],
            "wine_slug": ["a", "c"],
            "label_detector_confidence": [0.9, 0.8],
            "bottle_detector_confidence": [0.7, 0.6],
            "bottle_status": ["successful", "partial"],
            "ocr_text": ["", "d"],
        }
    )
    gallery = torch.eye(4)
    label_queries = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
    bottle_queries = torch.tensor([[0.9, 0.1, 0, 0], [0, 0.2, 0.8, 0]])
    cache = build_feature_cache(
        frame, label_queries, bottle_queries, gallery, gallery, slugs, top_k=2
    )
    assert cache.features.shape == (2, 6, len(FEATURE_NAMES))
    assert cache.true_positions.ge(0).all()
    assert 3 in cache.candidate_ids[1].tolist()  # OCR contributes its own candidate.
    model = FusionMLP(hidden_dim=16, dropout=0.0)
    metrics, ranks = score_cache(model, cache, torch.device("cpu"), batch_size=2)
    assert len(ranks) == 2
    assert metrics["num_queries"] == 2
