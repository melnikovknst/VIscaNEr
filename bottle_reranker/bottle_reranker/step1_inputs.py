"""Step 1 - find out what input data actually exists.

This stage never creates data. It looks for every input the later stages need,
reports each one as present or missing with the exact path that was checked, and
- importantly - tells apart a *photograph* from a *render*. Ready-made label
crops and a manifest of paths do not substitute for original frames: a neck and
a bottle silhouette cannot be recovered from a label crop.

The stage also unpacks nothing. Archives are reported as archives, with their
contents listed, so the operator decides where to extract them.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .common import csv_header, percentage, sha256_file, stage_record, write_json
from .config import Config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}

CONFIG_FINGERPRINT_KEYS = (
    "catalog.catalog_csv",
    "labels.crops_metadata_csv",
    "dino.checkpoint",
    "dino.index_csv",
    "segmentation.model",
    "output.root",
)


@dataclass
class InputCheck:
    """One required input, and whether it is usable."""

    key: str
    purpose: str
    path: str
    required_by: list[str]
    exists: bool
    kind: str                       # file | directory | archive | absent
    detail: dict[str, Any]
    blocking: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _describe(path: Path, *, count_images: bool, hash_small_files: bool) -> tuple[str, dict[str, Any]]:
    if not path.exists():
        return "absent", {}
    if path.is_dir():
        detail: dict[str, Any] = {}
        if count_images:
            images = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            detail["image_count"] = len(images)
            detail["example"] = images[0].name if images else None
        else:
            detail["entry_count"] = sum(1 for _ in path.iterdir())
        return "directory", detail
    size = path.stat().st_size
    detail = {"size_bytes": size}
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
            detail["archive_entries"] = len(names)
            detail["archive_image_entries"] = sum(
                1 for n in names if Path(n).suffix.lower() in IMAGE_EXTENSIONS
            )
            detail["archive_top_level"] = sorted({n.split("/")[0] for n in names})[:5]
        except zipfile.BadZipFile:
            detail["archive_error"] = "unreadable zip"
        return "archive", detail
    if path.suffix.lower() == ".csv":
        try:
            detail["columns"] = csv_header(path)
        except (OSError, UnicodeDecodeError) as error:
            detail["csv_error"] = str(error)
    if hash_small_files and size <= 64 * 1024 * 1024:
        detail["sha256"] = sha256_file(path)
    return "file", detail


def collect_checks(config: Config, *, hash_small_files: bool = False) -> list[InputCheck]:
    checks: list[InputCheck] = []

    def add(key: str, purpose: str, path: Path, required_by: list[str], *,
            blocking: bool, count_images: bool = False) -> None:
        kind, detail = _describe(path, count_images=count_images, hash_small_files=hash_small_files)
        checks.append(InputCheck(
            key=key,
            purpose=purpose,
            path=config.relative(path),
            required_by=required_by,
            exists=kind != "absent",
            kind=kind,
            detail=detail,
            blocking=blocking,
        ))

    # -- identities -------------------------------------------------------
    add("catalog.catalog_csv", "Wine identities, near-duplicate and same-image groups",
        config.path("catalog.catalog_csv"), ["step2", "step5", "step6", "step8"], blocking=True)
    add("catalog.near_duplicates_json", "Pre-computed near-duplicate label series",
        config.path("catalog.near_duplicates_json"), ["step2", "step8"], blocking=False)
    add("catalog.refs_rgb_root", "Catalog reference on white, used by the DINO gallery",
        config.path("catalog.refs_rgb_root"), ["step4", "step6"], blocking=True, count_images=True)
    add("catalog.refs_rgba_root", "Cut-out catalog reference with an exact alpha mask",
        config.path("catalog.refs_rgba_root"), ["step4"], blocking=True, count_images=True)

    # -- query sources ----------------------------------------------------
    for spec in config.sources(enabled_only=False):
        prefix = f"sources.{spec.name}"
        add(f"{prefix}.images_root",
            f"Original frames of source {spec.name} (kind={spec.kind}, photographic={spec.photographic})",
            spec.images_root, ["step3", "step4"], blocking=spec.enabled, count_images=True)
        add(f"{prefix}.manifest_csv", f"Frame -> wine_slug mapping for {spec.name}",
            spec.manifest_csv, ["step2", "step5"], blocking=spec.enabled)
        if spec.bottle_manifest_csv is not None:
            add(f"{prefix}.bottle_manifest_csv",
                f"Flat mirror of {spec.name} frames (prepare_bottle_images.py)",
                spec.bottle_manifest_csv, ["step2"], blocking=False)

    # -- chosen label boxes ----------------------------------------------
    add("labels.crops_metadata_csv",
        "Label box chosen by the production scanner; the anchor for picking the target bottle mask",
        config.path("labels.crops_metadata_csv"), ["step3"], blocking=True)
    add("labels.crops_root", "Existing label crops (traceability only, not a source of bottle shape)",
        config.path("labels.crops_root"), ["step3", "step7"], blocking=False)

    # -- models -----------------------------------------------------------
    segmentation_model = Path(config.get("segmentation.model"))
    if not segmentation_model.is_absolute() and segmentation_model.parent != Path("."):
        segmentation_model = config.resolve(segmentation_model)
    add("segmentation.model", "Instance segmentation checkpoint that produces bottle masks",
        segmentation_model, ["step3"], blocking=True)
    add("dino.checkpoint", "Trained DINOv3 retrieval checkpoint whose errors define the hard pairs",
        config.path("dino.checkpoint"), ["step6"], blocking=True)
    add("dino.index_csv", "Gallery index matching that checkpoint",
        config.path("dino.index_csv"), ["step6"], blocking=True)
    add("dino.retrieval_config", "Preprocessing that the checkpoint was trained with",
        config.path("dino.retrieval_config"), ["step6"], blocking=True)

    return checks


def _photographic_inventory(config: Config, checks: list[InputCheck]) -> dict[str, Any]:
    """How many genuine photographs exist, as opposed to renders.

    A whole-bottle reranker is meant to compare a photographed bottle against a
    catalog photograph. Frames that were synthesised from the catalog reference
    itself carry the reference pixels, not independent evidence about the
    bottle, so they are counted separately and never added to the same total.
    """
    by_key = {check.key: check for check in checks}
    photographic = {"sources": {}, "photographic_images": 0, "rendered_images": 0}
    for spec in config.sources(enabled_only=False):
        check = by_key.get(f"sources.{spec.name}.images_root")
        found = int((check.detail or {}).get("image_count", 0)) if check else 0
        photographic["sources"][spec.name] = {
            "kind": spec.kind,
            "photographic": spec.photographic,
            "enabled": spec.enabled,
            "images_found": found,
            "images_root": config.relative(spec.images_root),
            "present": bool(check and check.exists),
        }
        if spec.photographic:
            photographic["photographic_images"] += found
        else:
            photographic["rendered_images"] += found
    total = photographic["photographic_images"] + photographic["rendered_images"]
    photographic["photographic_share_percent"] = percentage(photographic["photographic_images"], total)
    return photographic


def run(config: Config, *, hash_small_files: bool = False) -> dict[str, Any]:
    checks = collect_checks(config, hash_small_files=hash_small_files)
    missing = [c for c in checks if not c.exists]
    blocking_missing = [c for c in missing if c.blocking]

    blocked_steps = sorted({step for check in blocking_missing for step in check.required_by})
    runnable_steps = sorted({"step1", "step2", "step5", "step8", "step9"} - set(blocked_steps))

    report = stage_record(
        "step1_inputs",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        config_path=str(config.config_path),
        checks=[c.as_dict() for c in checks],
        summary={
            "inputs_checked": len(checks),
            "present": len(checks) - len(missing),
            "missing": len(missing),
            "missing_blocking": len(blocking_missing),
            "blocked_steps": blocked_steps,
            "runnable_steps": runnable_steps,
        },
        missing_blocking_inputs=[
            {"key": c.key, "path": c.path, "purpose": c.purpose, "blocks": c.required_by}
            for c in blocking_missing
        ],
        missing_optional_inputs=[
            {"key": c.key, "path": c.path, "purpose": c.purpose} for c in missing if not c.blocking
        ],
        photographic_inventory=_photographic_inventory(config, checks),
    )

    destination = config.report_path("step1_inputs.json")
    write_json(destination, report)
    print(f"step1: {report['summary']['present']}/{len(checks)} inputs present, "
          f"{len(blocking_missing)} blocking gaps -> {config.relative(destination)}")
    for check in blocking_missing:
        print(f"  MISSING (blocking {', '.join(check.required_by)}): {check.key} = {check.path}")
    return report
