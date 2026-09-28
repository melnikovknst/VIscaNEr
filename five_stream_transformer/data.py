"""Assemble already-cropped 45k and Manual-211 datasets.

This training pipeline consumes persisted label and whole-bottle crops only.
Image detection is deliberately outside its scope.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def _first_file(candidates: Iterable[Path]) -> Path:
    candidates = tuple(candidates)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"None of the candidate files exists: {list(candidates)}")


def _resolve_crop(root: Path, status: str, recorded: str, portable: str, hard_root: Path) -> Path:
    if portable:
        candidate = hard_root / portable
        if candidate.is_file():
            return candidate.resolve()
    name = Path(recorded).name
    return _first_file((root / "crops" / status / name, root / status / name))


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


def _load_manifest(
    manifest_path: Path,
    hard_root: Path,
    label_data_root: Path,
    bottle_data_root: Path,
) -> pd.DataFrame:
    hard_root = hard_root.resolve()
    label_root = label_data_root.resolve()
    bottle_root = bottle_data_root.resolve()
    frame = pd.read_csv(manifest_path.resolve(), low_memory=False).fillna("")
    required = {
        "source_relative_path", "wine_slug", "fusion_split",
        "label_recorded_path", "label_status", "bottle_recorded_path", "bottle_status",
        "label_detector_confidence", "bottle_detector_confidence",
        "portable_label_path", "portable_bottle_path",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Five-stream manifest is missing columns: {sorted(missing)}")
    if frame["source_relative_path"].duplicated().any():
        raise ValueError("Five-stream manifest contains duplicate source_relative_path rows")
    frame["label_path"] = frame.apply(
        lambda row: str(_resolve_crop(
            label_root, str(row["label_status"]), str(row["label_recorded_path"]),
            str(row["portable_label_path"]), hard_root,
        )),
        axis=1,
    )
    frame["bottle_path"] = frame.apply(
        lambda row: str(_resolve_crop(
            bottle_root, str(row["bottle_status"]), str(row["bottle_recorded_path"]),
            str(row["portable_bottle_path"]), hard_root,
        )),
        axis=1,
    )
    return frame.reset_index(drop=True)


def load_hard_rows(
    manifest: Path,
    hard_root: Path,
    label_data_root: Path,
    bottle_data_root: Path,
) -> pd.DataFrame:
    frame = _load_manifest(manifest, hard_root, label_data_root, bottle_data_root)
    frame["sample_id"] = "hard::" + frame["source_relative_path"].astype(str)
    frame["positive_slugs"] = frame["wine_slug"].map(lambda value: (str(value),))
    mapping = {
        "hard_train": "train",
        "test_seen_prior_exposure": "train",
        "test_seen_model_selection": "train",
        "hard_val": "val_hard",
        "test_unseen": "test_unseen",
    }
    frame["five_stream_split"] = frame["fusion_split"].map(mapping)
    frame["source_group"] = "main_45k"
    if "ocr_text" not in frame:
        frame["ocr_text"] = ""
    else:
        frame["ocr_text"] = frame["ocr_text"].fillna("").astype(str)
    return frame[
        [
            "sample_id", "label_path", "bottle_path", "positive_slugs",
            "five_stream_split", "source_group", "ocr_text",
        ]
    ].copy()


def load_manual_rows(root: Path, gallery_slugs: Sequence[str], seed: int) -> pd.DataFrame:
    frame = pd.read_csv(root / "manifest.csv").fillna("")
    frame["trainable_catalog"] = frame["trainable_catalog"].astype(str).str.lower().isin({"true", "1"})
    frame["positive_slugs"] = frame["accepted_slugs"].map(
        lambda value: tuple(part.strip() for part in str(value).split(";") if part.strip())
    )
    gallery = set(gallery_slugs)
    frame["positive_slugs"] = frame["positive_slugs"].map(
        lambda values: tuple(value for value in values if value in gallery)
    )
    frame = frame[frame["trainable_catalog"] & frame["positive_slugs"].map(bool)].copy()
    validation_indices: list[int] = []
    rng = random.Random(seed)
    frame["split_key"] = frame["positive_slugs"].map(lambda values: values[0])
    for _, group in frame.groupby("split_key", sort=True):
        indices = group.index.tolist()
        rng.shuffle(indices)
        if len(indices) >= 2:
            validation_indices.append(indices[0])
    frame["five_stream_split"] = "train"
    frame.loc[validation_indices, "five_stream_split"] = "val_manual"
    frame["sample_id"] = "manual::" + frame["sample_id"].astype(str)
    frame["label_path"] = frame["label_path"].map(lambda value: str((root / str(value)).resolve()))
    frame["bottle_path"] = frame["bottle_path"].map(lambda value: str((root / str(value)).resolve()))
    frame["source_group"] = "manual_211"
    frame["ocr_text"] = ""
    return frame[
        [
            "sample_id", "label_path", "bottle_path", "positive_slugs",
            "five_stream_split", "source_group", "ocr_text",
        ]
    ].reset_index(drop=True)


def combine_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Combined dataset contains duplicate sample_id values")
    for column in ("label_path", "bottle_path"):
        missing = [value for value in frame[column] if not Path(value).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {column}: {missing[:3]}")
    return frame.reset_index(drop=True)
