"""Dataset inventory and leakage-aware evaluation splits for the cascade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dinov3_retrieval import _stable_unit_interval

from .config import CascadeConfig


USABLE_BOTTLE_STATUSES = {"successful", "low_confidence", "partial", "ambiguous"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _read_csv(path: str | Path, required: set[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _assert_unique(frame: pd.DataFrame, key: str, name: str) -> None:
    if frame[key].isna().any():
        raise ValueError(f"{name} contains empty {key} values")
    duplicated = frame.loc[frame[key].duplicated(keep=False), key]
    if not duplicated.empty:
        raise ValueError(f"{name} contains duplicate {key}: {duplicated.head(5).tolist()}")


def reference_lookup(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    refs: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in refs:
                raise ValueError(f"Duplicate reference identity in {root}: {path.stem}")
            refs[path.stem] = path
    if not refs:
        raise ValueError(f"No reference images found in {root}")
    return refs


def _portable_crop_path(root: Path, status: Any, recorded_path: Any) -> str:
    if pd.isna(recorded_path) or pd.isna(status):
        return ""
    candidate = root / str(status) / Path(str(recorded_path)).name
    return str(candidate) if candidate.is_file() else ""


def _assign_primary_splits(
    inventory: pd.DataFrame,
    seed: int,
    unseen_identity_fraction: float,
    val_seen_per_identity: int,
) -> pd.Series:
    """Reproduce the split used by ``prepare_retrieval_index`` exactly."""

    splits = pd.Series("unavailable", index=inventory.index, dtype="object")
    eligible = inventory["label_crop_available"]
    identities = sorted(inventory["wine_slug"].astype(str).unique())
    unseen = {
        slug
        for slug in identities
        if _stable_unit_interval(slug, seed) < unseen_identity_fraction
    }
    if len(identities) > 1 and not unseen:
        unseen.add(min(identities, key=lambda value: _stable_unit_interval(value, seed)))

    for slug, group in inventory.loc[eligible].groupby("wine_slug", sort=True):
        ordered = group.assign(
            _sort_key=group["source_path"].map(
                lambda source: _stable_unit_interval(f"{slug}:{source}", seed)
            )
        ).sort_values("_sort_key")
        if slug in unseen:
            splits.loc[ordered.index] = "val_unseen"
            continue
        n_val = min(val_seen_per_identity, max(1, len(ordered) - 2))
        splits.loc[ordered.index[:n_val]] = "val_seen"
        splits.loc[ordered.index[n_val:]] = "train"
    return splits


def build_inventory(cfg: CascadeConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one portable row per original image and validate identity alignment."""

    manifest = _read_csv(
        cfg.source_manifest,
        {"source_path", "source_relative_path", "wine_slug", "merged_filename"},
    )
    labels = _read_csv(
        cfg.label_metadata,
        {"source_path", "source_relative_path", "wine_slug", "crop_path", "status"},
    )
    bottles = _read_csv(
        cfg.bottle_metadata,
        {"source_path", "source_relative_path", "wine_slug", "crop_path", "status"},
    )
    bottle_train = _read_csv(
        cfg.bottle_training_metadata,
        {"source_relative_path", "wine_slug", "status"},
    )
    for frame, name in ((manifest, "manifest"), (labels, "label metadata"), (bottles, "bottle metadata")):
        _assert_unique(frame, "source_relative_path", name)

    manifest = manifest.copy()
    manifest["source_relative_path"] = manifest["source_relative_path"].astype(str)
    labels = labels.copy()
    labels["source_relative_path"] = labels["source_relative_path"].astype(str)
    bottles = bottles.copy()
    bottles["source_relative_path"] = bottles["source_relative_path"].astype(str)

    inventory = manifest.merge(
        labels[["source_relative_path", "wine_slug", "crop_path", "status", "confidence"]].rename(
            columns={
                "wine_slug": "label_wine_slug",
                "crop_path": "label_recorded_path",
                "status": "label_status",
                "confidence": "label_detector_confidence",
            }
        ),
        on="source_relative_path",
        how="left",
        validate="one_to_one",
    ).merge(
        bottles[["source_relative_path", "wine_slug", "crop_path", "status", "confidence"]].rename(
            columns={
                "wine_slug": "bottle_wine_slug",
                "crop_path": "bottle_recorded_path",
                "status": "bottle_status",
                "confidence": "bottle_detector_confidence",
            }
        ),
        on="source_relative_path",
        how="left",
        validate="one_to_one",
    )

    for column in ("label_wine_slug", "bottle_wine_slug"):
        present = inventory[column].notna()
        mismatch = present & inventory[column].astype(str).ne(inventory["wine_slug"].astype(str))
        if mismatch.any():
            examples = inventory.loc[mismatch, ["source_relative_path", "wine_slug", column]].head(5)
            raise ValueError(f"Identity mismatch in {column}: {examples.to_dict('records')}")

    label_root = Path(cfg.label_crops_root)
    bottle_root = Path(cfg.bottle_crops_root)
    original_root = Path(cfg.source_manifest).parent
    inventory["original_path"] = inventory["merged_filename"].map(
        lambda name: str(original_root / str(name))
    )
    inventory["label_crop_path"] = inventory.apply(
        lambda row: _portable_crop_path(label_root, row["label_status"], row["label_recorded_path"]),
        axis=1,
    )
    inventory["bottle_crop_path"] = inventory.apply(
        lambda row: (
            _portable_crop_path(bottle_root, row["bottle_status"], row["bottle_recorded_path"])
            if str(row["bottle_status"]) in USABLE_BOTTLE_STATUSES
            else ""
        ),
        axis=1,
    )
    inventory["label_crop_available"] = (
        inventory["label_status"].eq("successful") & inventory["label_crop_path"].ne("")
    )
    inventory["bottle_crop_available"] = inventory["bottle_crop_path"].ne("")
    inventory["primary_split"] = _assign_primary_splits(
        inventory,
        seed=cfg.seed,
        unseen_identity_fraction=cfg.unseen_identity_fraction,
        val_seen_per_identity=cfg.val_seen_per_identity,
    )

    exposed_sources = set(
        bottle_train.loc[
            bottle_train["status"].isin({"successful", "low_confidence"}),
            "source_relative_path",
        ].dropna().astype(str)
    )
    inventory["resolver_query_seen_during_training"] = inventory["source_relative_path"].isin(
        exposed_sources
    )

    label_refs = reference_lookup(cfg.label_refs_root)
    bottle_refs = reference_lookup(cfg.bottle_refs_root)
    identities = sorted(inventory["wine_slug"].astype(str).unique())
    if set(identities) != set(label_refs):
        raise ValueError(
            "Label-reference identities do not match the source manifest: "
            f"missing={sorted(set(identities) - set(label_refs))[:5]}, "
            f"extra={sorted(set(label_refs) - set(identities))[:5]}"
        )
    if set(identities) != set(bottle_refs):
        raise ValueError(
            "Bottle-reference identities do not match the source manifest: "
            f"missing={sorted(set(identities) - set(bottle_refs))[:5]}, "
            f"extra={sorted(set(bottle_refs) - set(identities))[:5]}"
        )
    label_by_slug = {slug: index for index, slug in enumerate(identities)}
    inventory["label_id"] = inventory["wine_slug"].map(label_by_slug).astype(int)
    inventory = inventory.sort_values(["label_id", "source_relative_path"]).reset_index(drop=True)

    missing_originals = (~inventory["original_path"].map(lambda value: Path(value).is_file())).sum()
    if missing_originals:
        raise FileNotFoundError(f"Missing {missing_originals} original images under {original_root}")

    summary = {
        "total_source_images": int(len(inventory)),
        "num_identities": int(len(identities)),
        "label_crop_available": int(inventory["label_crop_available"].sum()),
        "label_crop_unavailable": int((~inventory["label_crop_available"]).sum()),
        "bottle_crop_available": int(inventory["bottle_crop_available"].sum()),
        "eligible_with_bottle_crop": int(
            (inventory["label_crop_available"] & inventory["bottle_crop_available"]).sum()
        ),
        "primary_split_counts": inventory["primary_split"].value_counts().astype(int).to_dict(),
        "label_status_counts": inventory["label_status"].fillna("missing").value_counts().astype(int).to_dict(),
        "bottle_status_counts": inventory["bottle_status"].fillna("missing").value_counts().astype(int).to_dict(),
        "resolver_query_seen_during_training": int(
            inventory["resolver_query_seen_during_training"].sum()
        ),
    }
    return inventory, summary


def gallery_paths(cfg: CascadeConfig, kind: str) -> tuple[list[str], list[str]]:
    if kind not in {"label", "bottle"}:
        raise ValueError("kind must be 'label' or 'bottle'")
    lookup = reference_lookup(cfg.label_refs_root if kind == "label" else cfg.bottle_refs_root)
    slugs = sorted(lookup)
    return slugs, [str(lookup[slug]) for slug in slugs]


def limit_queries(inventory: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """Keep full inventory for audit, or a deterministic stratified smoke subset."""

    if limit is None:
        return inventory
    eligible = inventory[inventory["label_crop_available"]]
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit >= len(eligible):
        return inventory
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible.index.to_numpy(), size=limit, replace=False)
    result = inventory.loc[sorted(chosen)].copy()
    return result.reset_index(drop=True)
