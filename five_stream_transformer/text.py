"""Frozen text normalization and deterministic OCR embeddings."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from unidecode import unidecode


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower()
    value = unidecode(value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _index_and_sign(token: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "little", signed=False)
    return number % dimensions, 1.0 if number & (1 << 63) else -1.0


def frozen_char_ngram_embedding(
    texts: Sequence[str], dimensions: int = 512, ngram_range: tuple[int, int] = (2, 5)
) -> torch.Tensor:
    """Hash normalized character n-grams without trainable parameters.

    Cyrillic OCR is transliterated to Latin so it can be compared with the
    catalog's transliterated slugs.  Empty OCR output maps to an all-zero row.
    """

    output = torch.zeros(len(texts), dimensions, dtype=torch.float32)
    for row, raw in enumerate(texts):
        text = f" {normalize_text(raw)} "
        if not text.strip():
            continue
        for ngram_size in range(ngram_range[0], ngram_range[1] + 1):
            for start in range(max(0, len(text) - ngram_size + 1)):
                token = text[start : start + ngram_size]
                index, sign = _index_and_sign(token, dimensions)
                output[row, index] += sign
    non_empty = output.norm(dim=1) > 0
    output[non_empty] = F.normalize(output[non_empty], dim=1)
    return output
