"""PaddleOCR-VL text reranking for the label-DINO Top-5."""

from .reranker import DEFAULT_ALPHA, DEFAULT_EVIDENCE_GATE, rerank_top5

__all__ = ["DEFAULT_ALPHA", "DEFAULT_EVIDENCE_GATE", "rerank_top5"]
