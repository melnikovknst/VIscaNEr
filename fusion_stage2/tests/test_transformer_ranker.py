import torch

from fusion_stage2.core import FEATURE_NAMES, FeatureCache
from fusion_stage2.train_transformer_ranker import CandidateTransformerRanker, feature_statistics, rank_cache


def synthetic_cache() -> FeatureCache:
    torch.manual_seed(3)
    features = torch.randn(4, 5, len(FEATURE_NAMES))
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.bool)
    return FeatureCache(
        candidate_ids=torch.arange(5).repeat(4, 1),
        candidate_mask=mask,
        features=features,
        true_positions=torch.tensor([0, 2, 1, 4]),
        true_label_ids=torch.tensor([0, 2, 1, 4]),
        label_true_ranks=torch.tensor([1, 2, 1, 3]),
        bottle_true_ranks=torch.tensor([2, 1, 2, 1]),
        row_indices=torch.arange(4),
        slugs=[f"wine-{index}" for index in range(5)],
    )


def test_transformer_masks_padding_and_scores_all_rows() -> None:
    cache = synthetic_cache()
    mean, std = feature_statistics(cache)
    model = CandidateTransformerRanker(mean, std, d_model=32, num_heads=4, num_layers=1, feedforward_dim=64, dropout=0.0)
    logits = model(cache.features, cache.candidate_mask)
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits[cache.candidate_mask]).all()
    assert (logits[~cache.candidate_mask] < -1e20).all()
    metrics, ranks, predictions = rank_cache(model, cache, torch.device("cpu"), batch_size=2)
    assert metrics["num_queries"] == 4
    assert ranks.shape == predictions.shape == (4,)


def test_transformer_is_permutation_equivariant_without_positional_embeddings() -> None:
    cache = synthetic_cache()
    mean, std = feature_statistics(cache)
    model = CandidateTransformerRanker(mean, std, d_model=32, num_heads=4, num_layers=1, feedforward_dim=64, dropout=0.0)
    model.eval()
    permutation = torch.tensor([2, 0, 1, 4, 3])
    inverse = torch.argsort(permutation)
    with torch.inference_mode():
        original = model(cache.features, cache.candidate_mask)
        permuted = model(cache.features[:, permutation], cache.candidate_mask[:, permutation])[:, inverse]
    valid = cache.candidate_mask
    assert torch.allclose(original[valid], permuted[valid], atol=1e-5)
