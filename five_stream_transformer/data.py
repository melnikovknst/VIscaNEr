"""Reference gallery layout: one image per wine, named by its catalog slug."""

from __future__ import annotations

from pathlib import Path


def resolve_refs(root: Path) -> tuple[list[str], list[str]]:
    """Resolve one reference image per wine without importing legacy fusion code."""

    refs_root = root / "refs" if (root / "refs").is_dir() else root
    files = sorted(
        path for path in refs_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not files:
        raise FileNotFoundError(f"No reference images under {refs_root}")
    slugs = [path.stem for path in files]
    if len(slugs) != len(set(slugs)):
        raise ValueError(f"Duplicate reference slugs under {refs_root}")
    return slugs, [str(path.resolve()) for path in files]
