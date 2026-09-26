"""Pure text-scoring code used after label DINO retrieval."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable

from rapidfuzz import fuzz
from unidecode import unidecode


DEFAULT_ALPHA = 0.90
DEFAULT_EVIDENCE_GATE = 0.10
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "vino", "vina", "vinograda", "sort", "sorta", "input", "sukhoe", "suhoe",
    "polusukhoe", "polusuhoe", "sladkoe", "polusladkoe", "krasnoe", "beloe",
    "rozovoe", "igristoe", "stolovoe", "zashchishchennoe", "naimenovanie",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return " ".join(TOKEN_RE.findall(unidecode(text)))


def tokens(text: str) -> list[str]:
    return [token for token in normalize(text).split() if len(token) >= 2]


def build_idf(slugs: Iterable[str]) -> dict[str, float]:
    slugs = list(slugs)
    document_frequency: Counter[str] = Counter()
    for slug in slugs:
        document_frequency.update(set(tokens(slug)))
    n = max(len(slugs), 1)
    return {
        token: math.log((n + 1) / (count + 1)) + 1
        for token, count in document_frequency.items()
    }


def lexical_score(query_text: str, slug: str, idf: dict[str, float]) -> float:
    query = normalize(query_text)
    candidate = normalize(slug)
    if not query or not candidate:
        return 0.0
    query_tokens = [token for token in query.split() if len(token) >= 3]
    candidate_tokens = [
        token for token in candidate.split()
        if len(token) >= 3 and token not in STOPWORDS
    ]
    weighted_total = sum(idf.get(token, 1.0) for token in candidate_tokens) or 1.0
    weighted_match = 0.0
    for candidate_token in candidate_tokens:
        best = max(
            (fuzz.ratio(candidate_token, query_token) for query_token in query_tokens),
            default=0,
        )
        if best >= 78:
            weighted_match += idf.get(candidate_token, 1.0) * ((best - 70) / 30)
    coverage = min(1.0, weighted_match / weighted_total)
    token_set = fuzz.token_set_ratio(query, candidate) / 100.0
    partial = fuzz.partial_ratio(query, candidate) / 100.0
    return float(0.50 * coverage + 0.30 * token_set + 0.20 * partial)


def rerank_top5(
    predictions: list[dict[str, Any]],
    ocr_text: str,
    gallery_slugs: Iterable[str],
    *,
    alpha: float = DEFAULT_ALPHA,
    evidence_gate: float = DEFAULT_EVIDENCE_GATE,
) -> list[dict[str, Any]]:
    """Rerank only the supplied DINO candidates; never introduce a new wine."""
    candidates = [dict(item) for item in predictions[:5]]
    if not candidates:
        return []
    normalized = normalize(ocr_text)
    if normalized in {"", "ocr", "text"}:
        normalized = ""
    idf = build_idf(gallery_slugs)
    similarities = [float(item["similarity"]) for item in candidates]
    low, high = min(similarities), max(similarities)
    spread = max(high - low, 1e-8)
    rescored: list[dict[str, Any]] = []
    for dino_rank, item in enumerate(candidates, start=1):
        current = dict(item)
        current["dino_rank"] = dino_rank
        current["dino_score_normalized"] = (float(current["similarity"]) - low) / spread
        current["ocr_score"] = lexical_score(normalized, current["wine_slug"], idf)
        current["combined_score"] = current["dino_score_normalized"] + alpha * current["ocr_score"]
        rescored.append(current)

    ordered = sorted(rescored, key=lambda item: item["combined_score"], reverse=True)
    dino_top, proposed = rescored[0], ordered[0]
    evidence = (
        (float(proposed["ocr_score"]) - float(dino_top["ocr_score"]))
        / max(float(proposed["ocr_score"]), 0.10)
    )
    if proposed["wine_slug"] != dino_top["wine_slug"] and evidence < evidence_gate:
        ordered = sorted(rescored, key=lambda item: item["dino_rank"])
    for rank, item in enumerate(ordered, start=1):
        item["ocr_rank"] = rank
    return ordered
