"""Residual candidate Transformer fed by five frozen encoder streams.

Each stream compares a query representation with the corresponding candidate
reference representation.  The trainable part consists only of the five pair
projections, stream/rank embeddings, a three-layer Transformer encoder and the
final residual head. DINO and OCR encoders live outside this module and are
always executed under inference mode. The strong whole-bottle DINO score is
the immutable base ranking; the Transformer can add only a bounded correction.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


STREAM_NAMES = (
    "bottle_crop",
    "label_crop",
    "bottle_dino",
    "label_dino",
    "ocr_text",
)
STREAM_DIMS = {
    "bottle_crop": 1536,
    "label_crop": 1536,
    "bottle_dino": 256,
    "label_dino": 256,
    "ocr_text": 512,
}


class PairProjection(nn.Module):
    """Project q/ref pair statistics into one candidate-stream token."""

    def __init__(self, input_dim: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        pair_dim = input_dim * 4
        self.network = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        query = query[:, None, :].expand_as(reference)
        pair = torch.cat((query, reference, (query - reference).abs(), query * reference), dim=-1)
        return self.network(pair)


class FiveStreamResidualTransformer(nn.Module):
    """Apply a bounded learned correction to the bottle-DINO ranking."""

    def __init__(
        self,
        stream_dims: Mapping[str, int] = STREAM_DIMS,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        feedforward_dim: int = 768,
        dropout: float = 0.15,
        max_candidates: int = 64,
        base_temperature: float = 0.07,
        max_residual: float = 2.5,
    ) -> None:
        super().__init__()
        self.stream_names = tuple(STREAM_NAMES)
        if set(self.stream_names) != set(stream_dims):
            raise ValueError(f"stream_dims must contain exactly {self.stream_names}")
        self.model_dim = int(model_dim)
        self.max_candidates = int(max_candidates)
        self.base_temperature = float(base_temperature)
        self.max_residual = float(max_residual)
        if self.base_temperature <= 0:
            raise ValueError("base_temperature must be positive")
        if self.max_residual <= 0:
            raise ValueError("max_residual must be positive")
        self.projections = nn.ModuleDict(
            {
                name: PairProjection(int(stream_dims[name]), model_dim, dropout)
                for name in self.stream_names
            }
        )
        self.stream_embedding = nn.Parameter(torch.empty(len(self.stream_names), model_dim))
        self.rank_embedding = nn.Embedding(max_candidates, model_dim)
        self.stream_gate = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim))
        self.residual_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )
        nn.init.normal_(self.stream_embedding, std=0.02)
        # Training starts exactly from the frozen bottle-DINO ranking.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        query_streams: Mapping[str, torch.Tensor],
        reference_streams: Mapping[str, torch.Tensor],
        candidate_mask: torch.Tensor,
        query_stream_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(final_logits, base_logits, bounded_residual)``.

        Query tensors are ``[B, D]`` and reference tensors ``[B, K, D]``.
        ``query_stream_available`` is ``[B, 5]`` and lets missing OCR results
        disappear from the learned stream pooling without inventing text.
        """

        if candidate_mask.ndim != 2:
            raise ValueError("candidate_mask must have shape [batch, candidates]")
        batch, candidates = candidate_mask.shape
        if candidates > self.max_candidates:
            raise ValueError(f"Got {candidates} candidates; max_candidates={self.max_candidates}")
        if query_stream_available is None:
            query_stream_available = torch.ones(
                batch, len(self.stream_names), dtype=torch.bool, device=candidate_mask.device
            )
        if query_stream_available.shape != (batch, len(self.stream_names)):
            raise ValueError("query_stream_available must have shape [batch, 5]")

        tokens = []
        for stream_index, name in enumerate(self.stream_names):
            projected = self.projections[name](query_streams[name], reference_streams[name])
            tokens.append(projected + self.stream_embedding[stream_index])
        stacked = torch.stack(tokens, dim=2)  # [B, K, S, D]
        available = query_stream_available[:, None, :].expand(batch, candidates, -1)
        gate_logits = self.stream_gate(stacked).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~available, torch.finfo(gate_logits.dtype).min)
        weights = torch.softmax(gate_logits, dim=2)
        weights = torch.where(available, weights, torch.zeros_like(weights))
        denominator = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        candidate_tokens = (stacked * (weights / denominator).unsqueeze(-1)).sum(dim=2)
        ranks = torch.arange(candidates, device=candidate_tokens.device)
        candidate_tokens = candidate_tokens + self.rank_embedding(ranks)[None, :, :]
        encoded = self.transformer(candidate_tokens, src_key_padding_mask=~candidate_mask)
        raw_residual = self.residual_head(encoded).squeeze(-1)
        residual = self.max_residual * torch.tanh(raw_residual)
        bottle_query = query_streams["bottle_dino"][:, None, :]
        bottle_reference = reference_streams["bottle_dino"]
        base_logits = (bottle_query * bottle_reference).sum(dim=-1) / self.base_temperature
        final_logits = base_logits + residual
        negative_infinity = float("-inf")
        return (
            final_logits.masked_fill(~candidate_mask, negative_infinity),
            base_logits.masked_fill(~candidate_mask, negative_infinity),
            residual.masked_fill(~candidate_mask, 0.0),
        )


def soft_target_cross_entropy(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Listwise cross entropy supporting more than one accepted identity."""

    if bool((positive_mask.sum(dim=1) == 0).any()):
        raise ValueError("Every training row must have at least one positive candidate")
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()


def residual_ranking_loss(
    final_logits: torch.Tensor,
    base_logits: torch.Tensor,
    residual: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    pairwise_weight: float = 0.25,
    protection_weight: float = 0.75,
    residual_weight: float = 0.02,
    pairwise_margin: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Listwise loss with an explicit do-no-harm guardrail.

    Rows where the frozen bottle branch is already correct receive KL
    distillation plus a small residual penalty. Wrong bottle predictions are
    optimized with listwise CE and a true-vs-hardest-negative margin.
    """

    ranking = soft_target_cross_entropy(final_logits, positive_mask)
    positive_logits = final_logits.masked_fill(~positive_mask, float("-inf"))
    negative_mask = candidate_mask & ~positive_mask
    negative_logits = final_logits.masked_fill(~negative_mask, float("-inf"))
    best_positive = positive_logits.max(dim=1).values
    best_negative = negative_logits.max(dim=1).values
    has_negative = negative_mask.any(dim=1)
    pairwise = torch.nn.functional.softplus(
        pairwise_margin - best_positive[has_negative] + best_negative[has_negative]
    ).mean() if bool(has_negative.any()) else ranking.new_zeros(())

    base_prediction = base_logits.argmax(dim=1)
    base_correct = positive_mask.gather(1, base_prediction[:, None]).squeeze(1)
    if bool(base_correct.any()):
        # KL with literal -inf padding can produce 0 * inf -> NaN. Preserve the
        # mask semantics with a large finite value for this term only.
        protected_base = base_logits[base_correct].masked_fill(
            ~candidate_mask[base_correct], -1e4
        )
        protected_final = final_logits[base_correct].masked_fill(
            ~candidate_mask[base_correct], -1e4
        )
        base_log_probs = torch.log_softmax(protected_base, dim=1)
        base_probs = base_log_probs.exp().detach()
        final_log_probs = torch.log_softmax(protected_final, dim=1)
        protection = torch.nn.functional.kl_div(
            final_log_probs, base_probs, reduction="batchmean"
        )
        residual_guard = residual[base_correct].pow(2).mean()
    else:
        protection = ranking.new_zeros(())
        residual_guard = ranking.new_zeros(())
    total = (
        ranking
        + float(pairwise_weight) * pairwise
        + float(protection_weight) * protection
        + float(residual_weight) * residual_guard
    )
    return total, {
        "ranking": ranking.detach(),
        "pairwise": pairwise.detach(),
        "protection": protection.detach(),
        "residual_guard": residual_guard.detach(),
        "base_correct_rate": base_correct.float().mean().detach(),
    }


# Backward-compatible import for the local unit tests and older analysis code.
FiveStreamCandidateTransformer = FiveStreamResidualTransformer
