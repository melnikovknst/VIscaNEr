from __future__ import annotations

import torch

from five_stream_transformer.model import (
    STREAM_DIMS,
    STREAM_NAMES,
    FiveStreamResidualTransformer,
    residual_ranking_loss,
)
from five_stream_transformer.text import frozen_char_ngram_embedding


def random_streams(batch: int, candidates: int):
    query = {name: torch.randn(batch, width) for name, width in STREAM_DIMS.items()}
    refs = {name: torch.randn(batch, candidates, width) for name, width in STREAM_DIMS.items()}
    return query, refs


def test_model_shape_mask_and_backward() -> None:
    torch.manual_seed(7)
    model = FiveStreamResidualTransformer(
        model_dim=32, num_heads=4, num_layers=2, feedforward_dim=64,
        dropout=0.0, max_candidates=5,
    )
    query, refs = random_streams(2, 5)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    available = torch.ones(2, len(STREAM_NAMES), dtype=torch.bool)
    available[0, STREAM_NAMES.index("ocr_text")] = False
    logits, base_logits, residual = model(query, refs, mask, available)
    assert logits.shape == base_logits.shape == residual.shape == (2, 5)
    assert torch.isneginf(logits[0, 3:]).all()
    assert torch.allclose(logits, base_logits)
    assert residual.count_nonzero() == 0
    positives = torch.tensor(
        [[False, True, False, False, False], [True, False, True, False, False]]
    )
    loss, parts = residual_ranking_loss(logits, base_logits, residual, positives, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {
        "ranking", "pairwise", "protection", "residual_guard", "base_correct_rate"
    }
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_frozen_text_embedding_transliterates_and_handles_empty() -> None:
    embeddings = frozen_char_ngram_embedding(["Массандра мускатель белый", "massandra muskatel belyi", ""])
    assert embeddings.shape == (3, 512)
    assert float(embeddings[0] @ embeddings[1]) > 0.5
    assert embeddings[2].count_nonzero() == 0
