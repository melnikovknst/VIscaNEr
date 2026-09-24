"""Portable paths and image utilities used by the curation workflow."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "bottle_dino_curation"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}


def image_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def reference_lookup(root: str | Path) -> dict[str, Path]:
    refs: dict[str, Path] = {}
    for path in image_files(root):
        if path.stem in refs:
            raise ValueError(f"Duplicate reference slug: {path.stem}")
        refs[path.stem] = path.resolve()
    if not refs:
        raise FileNotFoundError(f"No reference images under {root}")
    return refs


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image, size: int = 8) -> int:
    gray = ImageOps.exif_transpose(image).convert("L").resize(
        (size + 1, size), Image.Resampling.LANCZOS
    )
    values = np.asarray(gray)
    bits = (values[:, 1:] > values[:, :-1]).reshape(-1)
    return sum(int(value) << index for index, value in enumerate(bits))


def dhash_file(path: str | Path) -> int:
    with Image.open(path) as image:
        return dhash(image)


def hash_similarity(left: int, right: int, bits: int = 64) -> float:
    return 1.0 - ((left ^ right).bit_count() / bits)


def resolve_recorded_image(
    recorded: object,
    root: str | Path,
    status: object | None = None,
) -> Path | None:
    """Resolve an old absolute/Kaggle path by its stable basename."""

    if recorded is None or str(recorded).strip() in {"", "nan", "None"}:
        return None
    candidate = Path(str(recorded))
    if candidate.is_file():
        return candidate.resolve()
    root = Path(root)
    if status is not None and str(status).strip() not in {"", "nan", "None"}:
        candidate = root / str(status) / candidate.name
        if candidate.is_file():
            return candidate.resolve()
    matches = list(root.rglob(Path(str(recorded)).name))
    if len(matches) == 1:
        return matches[0].resolve()
    return None


def first_existing(candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path.resolve()
    return None
