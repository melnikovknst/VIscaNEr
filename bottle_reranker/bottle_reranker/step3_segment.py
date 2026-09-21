"""Step 3 - segment every bottle in a frame and pick the right one.

A pretrained instance segmentation model produces a mask for every bottle in the
frame. Exactly one of them is the wine the label box belongs to, and the
selection rules never reward a bottle for being bigger or better segmented:

  1. Keep masks that contain the centre of the chosen label box.
  2. Among those, take the largest intersection with the label box; it must
     cover at least ``min_label_box_coverage`` of the box.
  3. If the runner-up is within ``ambiguous_coverage_margin`` of the winner, or
     nothing clears the floor, the frame is flagged - never guessed.

All geometry lives in the source frame defined in :mod:`geometry`: EXIF rotation
is applied once at decode, the model runs on a resized copy, and masks are
mapped back to full-resolution source pixels before anything is measured. Label
boxes declared in a different frame are rescaled explicitly and rejected if they
do not fit.

Masks are stored as run-length encoded PNG-free arrays inside a per-image .npz
so step 4 can re-render crops without re-running the model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import (
    Ledger,
    Progress,
    percentage,
    read_csv_rows,
    sha256_file,
    stage_record,
    summarise_counts,
    write_csv_rows,
    write_json,
)
from .config import Config, SourceSpec

CONFIG_FINGERPRINT_KEYS = (
    "segmentation.model",
    "segmentation.imgsz",
    "segmentation.conf",
    "segmentation.iou",
    "segmentation.keep_class_ids",
    "segmentation.retina_masks",
    "target_selection.require_label_centre_inside_mask",
    "target_selection.min_label_box_coverage",
    "target_selection.ambiguous_coverage_margin",
    "target_selection.min_mask_solidity",
    "labels.bbox_format",
)

STATUS_SELECTED = "selected"
STATUS_AMBIGUOUS = "ambiguous_target"
STATUS_NO_OWNER = "no_mask_owns_label"
STATUS_NO_MASKS = "no_bottle_detected"
STATUS_NO_LABEL = "no_label_box"
STATUS_BAD_GEOMETRY = "label_box_outside_image"
STATUS_ERROR = "processing_error"


@dataclass
class Candidate:
    """One segmented bottle, scored against the chosen label box."""

    index: int
    score: float                 # detector confidence, recorded but never decisive
    area_fraction: float
    contains_label_centre: bool
    label_box_coverage: float
    solidity: float
    bbox: tuple[int, int, int, int]


# ------------------------------------------------------------ label boxes
def load_label_boxes(config: Config) -> dict[str, dict[str, Any]]:
    """Read the chosen label box per source image from the crop metadata."""
    path = config.path("labels.crops_metadata_csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Chosen label boxes not found: {path}. Step 3 has no anchor for "
            "picking the target bottle without them."
        )
    key_column = config.get("labels.source_path_column")
    bbox_columns: Sequence[str] = config.get("labels.bbox_columns")
    status_column = config.get("labels.status_column")
    confidence_column = config.get("labels.confidence_column", None)
    accepted = set(config.get("labels.accepted_statuses"))
    review = set(config.get("labels.review_statuses", []))

    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError(f"Label metadata is empty: {path}")
    missing = [c for c in [key_column, status_column, *bbox_columns] if c not in rows[0]]
    if missing:
        raise RuntimeError(
            f"Label metadata {path} is missing columns {missing}. "
            f"Available columns: {sorted(rows[0])}. Fix `labels.*` in the config "
            "rather than guessing a format."
        )

    boxes: dict[str, dict[str, Any]] = {}
    for row in rows:
        status = (row.get(status_column) or "").strip()
        if status not in accepted and status not in review:
            continue
        try:
            values = [float(row[c]) for c in bbox_columns]
        except (TypeError, ValueError):
            continue
        key = Path((row.get(key_column) or "").strip()).as_posix()
        boxes[key] = {
            "values": values,
            "status": status,
            "needs_review": status in review,
            "confidence": float(row[confidence_column]) if confidence_column and row.get(confidence_column) else None,
            "reference_size": _declared_reference_size(config, row),
            "row": row,
        }
    return boxes


def _declared_reference_size(config: Config, row: dict[str, str]) -> tuple[int, int] | None:
    columns = config.get("labels.bbox_reference_size_columns", None)
    if not columns:
        return None
    try:
        width, height = (int(float(row[c])) for c in columns)
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _match_key(config: Config, image_path: Path, relative: str) -> list[str]:
    """Keys under which a frame may appear in the crop metadata.

    The existing metadata stores absolute paths in one build and manifest-
    relative paths in another, so both are tried; the basename is the last
    resort and is only accepted when it is unique.
    """
    return [
        Path(image_path).as_posix(),
        config.relative(image_path),
        Path(relative).as_posix(),
        Path(relative).name,
    ]


# ------------------------------------------------------------- the model
def _load_model(config: Config):
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "ultralytics is required for step 3. Install the ML extras: "
            "pip install -r bottle_reranker/requirements.txt"
        ) from error

    raw = str(config.get("segmentation.model"))
    candidate = Path(raw)
    weights_path = candidate if candidate.is_absolute() or candidate.parent != Path(".") else None
    if weights_path is not None:
        weights_path = config.resolve(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Segmentation weights not found: {weights_path}")
        model = YOLO(str(weights_path))
        digest = sha256_file(weights_path)
    else:
        # A bare name lets ultralytics resolve its own cached download.
        model = YOLO(raw)
        resolved = getattr(model, "ckpt_path", None)
        digest = sha256_file(resolved) if resolved and Path(resolved).is_file() else None
    return model, {"model": raw, "weights_sha256": digest}


def _device(config: Config) -> str | int:
    declared = str(config.get("segmentation.device", "auto"))
    if declared != "auto":
        return declared
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -------------------------------------------------------- target selection
def select_target(
    candidates: Sequence[Candidate],
    *,
    require_centre: bool,
    min_coverage: float,
    ambiguous_margin: float,
    min_solidity: float,
) -> tuple[Candidate | None, str, dict[str, Any]]:
    """Apply the selection rules in order and explain the outcome.

    Returns ``(winner, status, detail)``. ``winner`` is None for every status
    other than ``selected``; the frame is then flagged for review instead of
    being attached to an arbitrary bottle.
    """
    if not candidates:
        return None, STATUS_NO_MASKS, {"candidates": 0}

    usable = [c for c in candidates if c.solidity >= min_solidity]
    owners = [c for c in usable if c.contains_label_centre] if require_centre else list(usable)

    detail: dict[str, Any] = {
        "candidates": len(candidates),
        "candidates_after_solidity": len(usable),
        "candidates_containing_label_centre": len(owners),
    }

    pool = owners or ([c for c in usable if c.label_box_coverage >= min_coverage] if not require_centre else [])
    if not pool:
        # Nothing owns the label. Report the best coverage seen so the operator
        # can tell "no bottle found" from "the label sits between two bottles".
        best = max(usable, key=lambda c: c.label_box_coverage, default=None)
        detail["best_coverage_seen"] = round(best.label_box_coverage, 4) if best else 0.0
        return None, STATUS_NO_OWNER, detail

    ranked = sorted(pool, key=lambda c: c.label_box_coverage, reverse=True)
    winner = ranked[0]
    detail["winner_coverage"] = round(winner.label_box_coverage, 4)
    detail["winner_area_fraction"] = round(winner.area_fraction, 5)
    detail["winner_score"] = round(winner.score, 4)

    if winner.label_box_coverage < min_coverage:
        detail["reason"] = f"coverage_{winner.label_box_coverage:.3f}_below_{min_coverage}"
        return None, STATUS_NO_OWNER, detail

    if len(ranked) > 1:
        runner_up = ranked[1]
        detail["runner_up_coverage"] = round(runner_up.label_box_coverage, 4)
        if winner.label_box_coverage - runner_up.label_box_coverage < ambiguous_margin:
            detail["reason"] = "two_masks_own_the_label_almost_equally"
            return None, STATUS_AMBIGUOUS, detail

    return winner, STATUS_SELECTED, detail


# ------------------------------------------------------------------- run
def _iter_targets(config: Config, sources: Iterable[SourceSpec]) -> list[dict[str, Any]]:
    """Clean correspondences from step 2, restricted to frames that exist."""
    audit_csv = config.output_dir("audit", create=False) / "correspondences.csv"
    if not audit_csv.is_file():
        raise FileNotFoundError(
            f"{audit_csv} not found. Run step 2 before step 3 so every frame has "
            "a verified identity."
        )
    wanted = {spec.name for spec in sources}
    return [
        row for row in read_csv_rows(audit_csv)
        if row["source"] in wanted
        and row.get("clean", "").lower() in {"true", "1"}
        and row.get("image_exists", "").lower() in {"true", "1"}
    ]


def run(config: Config, *, sources: Sequence[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    import numpy as np

    from . import geometry as geo

    specs = [s for s in config.sources() if sources is None or s.name in sources]
    targets = _iter_targets(config, specs)
    if limit:
        targets = targets[:limit]

    boxes = load_label_boxes(config)
    box_format = str(config.get("labels.bbox_format"))
    require_centre = bool(config.get("target_selection.require_label_centre_inside_mask"))
    min_coverage = float(config.get("target_selection.min_label_box_coverage"))
    ambiguous_margin = float(config.get("target_selection.ambiguous_coverage_margin"))
    min_solidity = float(config.get("target_selection.min_mask_solidity"))
    min_area = float(config.get("target_selection.min_mask_area_fraction"))
    max_area = float(config.get("target_selection.max_mask_area_fraction"))
    keep_classes = config.get("segmentation.keep_class_ids", None)
    keep_classes = set(keep_classes) if keep_classes else None

    model, model_record = _load_model(config)
    device = _device(config)
    imgsz = int(config.get("segmentation.imgsz"))
    conf = float(config.get("segmentation.conf"))
    iou = float(config.get("segmentation.iou"))
    max_det = int(config.get("segmentation.max_det"))
    retina = bool(config.get("segmentation.retina_masks"))

    masks_dir = config.output_dir("masks_npz")
    rows_path = config.output_dir("audit") / "segmentation.csv"
    ledger = Ledger(
        config.output_dir("audit") / f"step3{config.get('runtime.ledger_name')}",
        enabled=bool(config.get("runtime.resume", True)),
    )

    existing: dict[str, dict[str, Any]] = {}
    if rows_path.is_file() and ledger.enabled:
        existing = {r["image_relative_path"]: r for r in read_csv_rows(rows_path)}

    results: list[dict[str, Any]] = []
    progress = Progress(len(targets), "segment", every=int(config.get("runtime.log_every", 200)))
    fail_fast = bool(config.get("runtime.fail_fast", False))

    with ledger:
        for target in targets:
            key = f"{target['source']}:{target['image_relative_path']}"
            if key in ledger and target["image_relative_path"] in existing:
                results.append(existing[target["image_relative_path"]])
                progress.step()
                continue

            image_path = config.resolve(target["image_path"])
            record: dict[str, Any] = {
                "source": target["source"],
                "image_relative_path": target["image_relative_path"],
                "image_path": target["image_path"],
                "wine_slug": target["wine_slug"],
                "status": STATUS_ERROR,
                "mask_npz": "",
            }
            try:
                loaded = geo.load_source_image(image_path)
                record.update(loaded.record())

                entry = None
                for candidate_key in _match_key(config, image_path, target["image_relative_path"]):
                    if candidate_key in boxes:
                        entry = boxes[candidate_key]
                        break
                if entry is None:
                    record["status"] = STATUS_NO_LABEL
                    results.append(record)
                    ledger.mark(key, status=record["status"])
                    progress.step()
                    continue

                reference_size = entry["reference_size"]
                box = geo.parse_box(entry["values"], fmt=box_format, reference_size=reference_size or loaded.source_size)
                if reference_size and reference_size != loaded.source_size:
                    box = geo.rescale_box(box, from_size=reference_size, to_size=loaded.source_size)
                    record["label_box_rescaled_from"] = f"{reference_size[0]}x{reference_size[1]}"
                if not geo.box_inside_image(box, loaded.source_size):
                    # Most often this means the box was measured before EXIF
                    # rotation, or against the stored size. Say so; do not
                    # silently clamp a box into a frame it does not belong to.
                    fitted = geo.infer_box_reference_size(box, [loaded.stored_size, loaded.source_size])
                    record["status"] = STATUS_BAD_GEOMETRY
                    record["label_box_fits_size"] = f"{fitted[0]}x{fitted[1]}" if fitted else "none"
                    results.append(record)
                    ledger.mark(key, status=record["status"])
                    progress.step()
                    continue

                record.update({
                    "label_x1": round(box[0], 2), "label_y1": round(box[1], 2),
                    "label_x2": round(box[2], 2), "label_y2": round(box[3], 2),
                    "label_status": entry["status"],
                    "label_needs_review": entry["needs_review"],
                    "label_confidence": entry["confidence"],
                })

                prediction = model.predict(
                    source=loaded.rgb[:, :, ::-1],   # ultralytics expects BGR arrays
                    imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                    retina_masks=retina, device=device, verbose=False,
                )[0]

                if prediction.masks is None or len(prediction.masks) == 0:
                    record["status"] = STATUS_NO_MASKS
                    record["masks_found"] = 0
                    results.append(record)
                    ledger.mark(key, status=record["status"])
                    progress.step()
                    continue

                mask_data = prediction.masks.data.cpu().numpy()
                classes = prediction.boxes.cls.cpu().numpy().astype(int)
                scores = prediction.boxes.conf.cpu().numpy()

                height, width = loaded.rgb.shape[:2]
                frame_area = float(height * width)
                candidates: list[Candidate] = []
                kept_masks: list[np.ndarray] = []

                for index, raw_mask in enumerate(mask_data):
                    if keep_classes is not None and int(classes[index]) not in keep_classes:
                        continue
                    mask = raw_mask
                    if mask.shape != (height, width):
                        import cv2

                        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
                    binary = mask > 0.5
                    area_fraction = float(binary.sum()) / frame_area
                    if not (min_area <= area_fraction <= max_area):
                        continue
                    bbox = geo.mask_bbox(binary)
                    if bbox is None:
                        continue
                    kept_masks.append(binary)
                    candidates.append(Candidate(
                        index=len(kept_masks) - 1,
                        score=float(scores[index]),
                        area_fraction=area_fraction,
                        contains_label_centre=geo.mask_contains_point(binary, geo.box_centre(box)),
                        label_box_coverage=geo.mask_box_coverage(binary, box),
                        solidity=geo.mask_solidity(binary),
                        bbox=bbox,
                    ))

                record["masks_found"] = len(mask_data)
                record["bottle_masks_kept"] = len(candidates)

                winner, status, detail = select_target(
                    candidates,
                    require_centre=require_centre,
                    min_coverage=min_coverage,
                    ambiguous_margin=ambiguous_margin,
                    min_solidity=min_solidity,
                )
                record["status"] = status
                record.update({f"selection_{k}": v for k, v in detail.items()})

                if winner is not None:
                    mask = kept_masks[winner.index]
                    npz_path = masks_dir / f"{target['source']}__{Path(target['image_relative_path']).as_posix().replace('/', '__')}.npz"
                    np.savez_compressed(
                        npz_path,
                        mask=np.packbits(mask, axis=None),
                        shape=np.array(mask.shape, dtype=np.int32),
                        label_box=np.array(box, dtype=np.float32),
                    )
                    record.update({
                        "mask_npz": config.relative(npz_path),
                        "mask_area_fraction": round(winner.area_fraction, 5),
                        "mask_solidity": round(winner.solidity, 4),
                        "mask_hole_fraction": round(geo.mask_hole_fraction(mask), 4),
                        "mask_x1": winner.bbox[0], "mask_y1": winner.bbox[1],
                        "mask_x2": winner.bbox[2], "mask_y2": winner.bbox[3],
                        **{f"mask_touches_{k}": v for k, v in geo.mask_touches_border(
                            mask, margin=int(config.get("crops.quality_flags.neck_border_margin_px", 2))
                        ).items()},
                    })
            except Exception as error:  # noqa: BLE001 - one bad frame must not kill a 45k run
                if fail_fast:
                    raise
                record["status"] = STATUS_ERROR
                record["error"] = f"{type(error).__name__}: {error}"

            results.append(record)
            ledger.mark(key, status=record["status"])
            progress.step()

    write_csv_rows(rows_path, results)
    statuses = summarise_counts(r["status"] for r in results)
    selected = statuses.get(STATUS_SELECTED, 0)

    report = stage_record(
        "step3_segment",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        segmentation_model=model_record,
        device=str(device),
        frames_processed=len(results),
        status_counts=statuses,
        selection_rate_percent=percentage(selected, len(results)),
        problem_mask_percent=percentage(len(results) - selected, len(results)),
        outputs={
            "segmentation_csv": config.relative(rows_path),
            "masks_dir": config.relative(masks_dir),
        },
    )
    write_json(config.report_path("step3_segment.json"), report)
    print(f"step3: {selected}/{len(results)} frames got a confident target mask "
          f"({report['selection_rate_percent']}%)")
    for status, count in statuses.items():
        if status != STATUS_SELECTED:
            print(f"  {status}: {count}")
    return report
