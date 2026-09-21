"""Step 4 - render the encoder inputs for queries and for catalog references.

Per selected bottle this writes four files and one manifest row:

  bottle/    the whole visible bottle, neck and capsule included, aspect kept
  mask/      its silhouette, same crop window, lossless
  normalized/ deskewed, composited on neutral grey, padded to square
  top/       capsule + neck + shoulders, cut at the shoulder line

Deliberate non-actions, because each of them would destroy the signal the
reranker is supposed to read:

* The bottle is never stretched to a fixed aspect ratio.
* Colour is never normalised: glass tint and capsule colour are evidence.
* Crops are never upscaled, and are saved near native resolution so the encoder
  input size stays a training-time choice.
* A missing, occluded or badly segmented neck is flagged, never painted in.
* A tilted bottle is only straightened when the mask axis is reliable, and it is
  only turned over when the narrow end is clearly the neck.

Catalog references go through the same two views so a query crop and a candidate
crop are directly comparable. References with a cut neck, poor resolution or an
unusable aspect are flagged rather than quietly used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .common import (
    Ledger,
    Progress,
    percentage,
    read_csv_rows,
    stage_record,
    summarise_counts,
    write_csv_rows,
    write_json,
)
from .config import Config

CONFIG_FINGERPRINT_KEYS = (
    "crops.pad_fraction",
    "crops.max_long_side",
    "crops.min_long_side",
    "crops.deskew",
    "crops.normalized",
    "crops.top_part",
    "crops.quality_flags",
    "reference_views",
)

VIEW_DIRS = ("bottle", "mask", "normalized", "top")


def _extension(fmt: str) -> str:
    """Canonical file extension for a configured format name."""
    return "png" if str(fmt).lower() in {"png"} else "jpg"


def _safe_stem(source: str, relative: str) -> str:
    return f"{source}__{Path(relative).as_posix().replace('/', '__').rsplit('.', 1)[0]}"


def _write_image(path: Path, array, *, fmt: str, quality: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 3:
        data = array[:, :, ::-1]          # RGB -> BGR for cv2.imencode
    else:
        data = array
    extension = ".png" if fmt == "png" else ".jpg"
    params = [] if extension == ".png" else [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buffer = cv2.imencode(extension, data, params)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    # imwrite cannot handle non-ASCII paths on Windows; write the buffer instead.
    path.write_bytes(buffer.tobytes())


def _render_views(
    rgb,
    mask,
    *,
    config: Config,
    geo,
    np,
    cv2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn one source frame plus its target mask into the four views."""
    pad_fraction = float(config.get("crops.pad_fraction"))
    max_long_side = int(config.get("crops.max_long_side"))
    min_long_side = int(config.get("crops.min_long_side"))
    quality_flags = config.get("crops.quality_flags")
    deskew_cfg = config.get("crops.deskew")
    normalized_cfg = config.get("crops.normalized")
    top_cfg = config.get("crops.top_part")

    flags: list[str] = []
    meta: dict[str, Any] = {}

    bbox = geo.mask_bbox(mask)
    if bbox is None:
        raise ValueError("empty target mask")
    x1, y1, x2, y2 = bbox
    long_side = max(x2 - x1, y2 - y1)
    pad = int(round(pad_fraction * long_side))
    height, width = mask.shape[:2]
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(width, x2 + pad), min(height, y2 + pad)
    meta["crop_window"] = [cx1, cy1, cx2, cy2]
    meta["requested_pad_px"] = pad
    # A truncated margin means the bottle runs off the frame edge on that side.
    meta["pad_applied"] = [x1 - cx1, y1 - cy1, cx2 - x2, cy2 - y2]

    crop_rgb = rgb[cy1:cy2, cx1:cx2].copy()
    crop_mask = mask[cy1:cy2, cx1:cx2].astype(np.uint8)

    borders = geo.mask_touches_border(mask, margin=int(quality_flags["neck_border_margin_px"]))
    if borders["top"]:
        flags.append("neck_truncated_by_frame")
    if borders["left"] or borders["right"]:
        flags.append("bottle_touches_side_border")

    hole_fraction = geo.mask_hole_fraction(mask)
    meta["mask_hole_fraction"] = round(hole_fraction, 4)
    if hole_fraction > float(quality_flags["max_hole_area_fraction"]):
        flags.append("silhouette_occluded")

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    sharp = geo.sharpness(gray[crop_mask.astype(bool)] if crop_mask.any() else gray)
    meta["sharpness_var_laplacian"] = round(sharp, 2)
    if sharp < float(quality_flags["min_sharpness_var_laplacian"]):
        flags.append("low_sharpness")

    if max(crop_rgb.shape[:2]) < min_long_side:
        flags.append("crop_too_small")

    # -- deskew -----------------------------------------------------------
    axis = geo.estimate_axis(
        crop_mask,
        min_elongation=float(deskew_cfg["min_elongation"]),
        max_abs_angle_deg=float(deskew_cfg["max_abs_angle_deg"]),
        min_end_width_ratio=float(deskew_cfg["min_end_width_ratio"]),
    )
    meta["axis"] = axis.record()
    upright_rgb, upright_mask = crop_rgb, crop_mask
    applied_angle = 0.0

    if bool(deskew_cfg.get("enabled", True)) and axis.reliable:
        applied_angle = axis.angle_deg
        if axis.flip_required:
            # The narrow end points down: the mask is upside down. The flip only
            # happens when the end-width and area-balance tests agree (see
            # geometry.estimate_axis); the bare principal axis cannot tell a
            # bottle from its own reflection.
            applied_angle += 180.0
            flags.append("orientation_flipped")
        upright_rgb = geo.rotate_about_centre(crop_rgb, applied_angle, border_value=(0, 0, 0))
        upright_mask = geo.rotate_about_centre(
            crop_mask, applied_angle, flags=cv2.INTER_NEAREST, border_value=0
        )
    else:
        flags.append(f"deskew_skipped_{axis.reason}")
        if not axis.orientation_confident:
            flags.append("orientation_uncertain")
    meta["deskew_applied_deg"] = round(applied_angle, 3)

    upright_mask = (upright_mask > 0).astype(np.uint8)

    # -- normalized view --------------------------------------------------
    normalized = None
    if bool(normalized_cfg.get("enabled", True)):
        background = np.array(normalized_cfg.get("background", [128, 128, 128]), dtype=np.uint8)
        if bool(normalized_cfg.get("keep_background_pixels", False)):
            composited = upright_rgb.copy()
        else:
            composited = np.where(
                upright_mask[..., None].astype(bool), upright_rgb, background[None, None, :]
            ).astype(np.uint8)
        tight = geo.mask_bbox(upright_mask)
        if tight is not None:
            tx1, ty1, tx2, ty2 = tight
            composited = composited[ty1:ty2, tx1:tx2]
        if bool(normalized_cfg.get("pad_to_square", True)):
            composited = geo.pad_to_square(composited, value=[int(v) for v in background])
        normalized = geo.limit_long_side(composited, max_long_side)

    # -- top part ---------------------------------------------------------
    shoulder = geo.find_shoulder(
        upright_mask,
        shoulder_width_ratio=float(top_cfg["shoulder_width_ratio"]),
        fallback_top_fraction=float(top_cfg["fallback_top_fraction"]),
        min_height_fraction=float(top_cfg["min_height_fraction"]),
    )
    meta["shoulder"] = shoulder.record()
    if not shoulder.confident:
        flags.append(f"top_part_estimated_{shoulder.reason}")

    mask_bounds = geo.mask_bbox(upright_mask)
    if mask_bounds is None:
        raise ValueError("target mask vanished after deskew")
    _, top_row, _, bottom_row = mask_bounds
    extent = max(1, bottom_row - top_row)
    margin = int(round(float(top_cfg["below_shoulder_margin"]) * extent))
    cut = min(upright_mask.shape[0], shoulder.shoulder_row + margin)
    cut = max(cut, top_row + int(float(top_cfg["min_height_fraction"]) * extent))

    top_rgb = upright_rgb[:cut]
    top_mask = upright_mask[:cut]
    top_fill = float(top_mask.sum()) / float(max(1, top_mask.size))
    meta["top_mask_fill"] = round(top_fill, 4)
    if top_fill < float(quality_flags["min_top_mask_fill"]):
        flags.append("top_part_poorly_covered")
    if not top_mask.any():
        flags.append("top_part_empty")

    return (
        {
            "bottle": geo.limit_long_side(upright_rgb, max_long_side),
            "mask": geo.limit_long_side(upright_mask * 255, max_long_side),
            "normalized": normalized,
            "top": geo.limit_long_side(top_rgb, max_long_side),
        },
        {**meta, "quality_flags": sorted(set(flags))},
    )


# ------------------------------------------------------------- query crops
def run_queries(config: Config, *, limit: int | None = None) -> dict[str, Any]:
    import cv2
    import numpy as np

    from . import geometry as geo

    segmentation_csv = config.output_dir("audit", create=False) / "segmentation.csv"
    if not segmentation_csv.is_file():
        raise FileNotFoundError(f"{segmentation_csv} not found. Run step 3 first.")

    rows = [r for r in read_csv_rows(segmentation_csv) if r.get("status") == "selected" and r.get("mask_npz")]
    if limit:
        rows = rows[:limit]

    fmt = str(config.get("crops.save_format"))
    quality = int(config.get("crops.jpeg_quality"))
    roots = {name: config.output_dir("queries", name) for name in VIEW_DIRS}
    manifest_rows: list[dict[str, Any]] = []
    ledger = Ledger(
        config.output_dir("audit") / f"step4q{config.get('runtime.ledger_name')}",
        enabled=bool(config.get("runtime.resume", True)),
    )
    crops_csv = config.output_dir("audit") / "query_crops.csv"
    existing = {r["stem"]: r for r in read_csv_rows(crops_csv)} if crops_csv.is_file() and ledger.enabled else {}

    progress = Progress(len(rows), "query crops", every=int(config.get("runtime.log_every", 200)))
    fail_fast = bool(config.get("runtime.fail_fast", False))

    with ledger:
        for row in rows:
            stem = _safe_stem(row["source"], row["image_relative_path"])
            if stem in ledger and stem in existing:
                manifest_rows.append(existing[stem])
                progress.step()
                continue

            record: dict[str, Any] = {
                "stem": stem,
                "source": row["source"],
                "image_relative_path": row["image_relative_path"],
                "image_path": row["image_path"],
                "wine_slug": row["wine_slug"],
                "status": "error",
            }
            try:
                loaded = geo.load_source_image(config.resolve(row["image_path"]))
                payload = np.load(config.resolve(row["mask_npz"]))
                shape = tuple(int(v) for v in payload["shape"])
                mask = np.unpackbits(payload["mask"])[: shape[0] * shape[1]].reshape(shape).astype(np.uint8)
                if mask.shape != loaded.rgb.shape[:2]:
                    raise ValueError(
                        f"mask {mask.shape} does not match the source frame "
                        f"{loaded.rgb.shape[:2]}; the coordinate frames disagree"
                    )

                views, meta = _render_views(loaded.rgb, mask, config=config, geo=geo, np=np, cv2=cv2)
                paths: dict[str, str] = {}
                for name, array in views.items():
                    if array is None:
                        continue
                    # A lossy mask is a corrupted mask: masks stay PNG.
                    extension = "png" if name == "mask" else _extension(fmt)
                    destination = roots[name] / f"{stem}.{extension}"
                    _write_image(destination, array, fmt=extension, quality=quality)
                    paths[f"{name}_path"] = (
                        str(destination.relative_to(config.output_root).as_posix())
                        if bool(config.get("output.relative_paths", True))
                        else str(destination)
                    )
                    paths[f"{name}_height"], paths[f"{name}_width"] = int(array.shape[0]), int(array.shape[1])

                record.update(paths)
                record.update({
                    "status": "ok",
                    "label_box": f"{row.get('label_x1')},{row.get('label_y1')},{row.get('label_x2')},{row.get('label_y2')}",
                    "source_width": loaded.source_size[0],
                    "source_height": loaded.source_size[1],
                    "exif_orientation": loaded.exif_orientation,
                    "crop_window": ",".join(str(v) for v in meta["crop_window"]),
                    "deskew_applied_deg": meta["deskew_applied_deg"],
                    "axis_angle_deg": meta["axis"]["angle_deg"],
                    "axis_elongation": meta["axis"]["elongation"],
                    "axis_reliable": meta["axis"]["reliable"],
                    "orientation_confident": meta["axis"]["orientation_confident"],
                    "shoulder_row": meta["shoulder"]["shoulder_row"],
                    "shoulder_method": meta["shoulder"]["method"],
                    "top_mask_fill": meta["top_mask_fill"],
                    "mask_hole_fraction": meta["mask_hole_fraction"],
                    "sharpness_var_laplacian": meta["sharpness_var_laplacian"],
                    "quality_flags": ";".join(meta["quality_flags"]),
                })
            except Exception as error:  # noqa: BLE001
                if fail_fast:
                    raise
                record["error"] = f"{type(error).__name__}: {error}"

            manifest_rows.append(record)
            ledger.mark(stem, status=record["status"])
            progress.step()

    write_csv_rows(crops_csv, manifest_rows)
    ok = sum(1 for r in manifest_rows if r["status"] == "ok")
    flag_counts = summarise_counts(
        flag for r in manifest_rows for flag in (r.get("quality_flags") or "").split(";") if flag
    )
    report = stage_record(
        "step4_query_crops",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        crops_written=ok,
        crops_failed=len(manifest_rows) - ok,
        quality_flag_counts=flag_counts,
        flagged_percent=percentage(
            sum(1 for r in manifest_rows if r.get("quality_flags")), max(1, len(manifest_rows))
        ),
        outputs={"query_crops_csv": config.relative(crops_csv),
                 "views_root": config.relative(config.output_dir("queries", create=False))},
    )
    write_json(config.report_path("step4_query_crops.json"), report)
    print(f"step4 (queries): {ok}/{len(manifest_rows)} crops written")
    return report


# --------------------------------------------------------- reference crops
def run_references(config: Config, *, limit: int | None = None) -> dict[str, Any]:
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    from . import geometry as geo

    source = str(config.get("reference_views.source", "rgba"))
    root = config.path(f"catalog.refs_{source}_root")
    if not root.is_dir():
        raise FileNotFoundError(f"Reference root not found: {root}")

    alpha_threshold = int(config.get("reference_views.alpha_threshold"))
    min_long_side = int(config.get("reference_views.min_long_side"))
    aspect_low, aspect_high = (float(v) for v in config.get("reference_views.aspect_range"))
    fmt = str(config.get("crops.save_format"))
    quality = int(config.get("crops.jpeg_quality"))
    roots = {name: config.output_dir("references", name) for name in VIEW_DIRS}

    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg"})
    if limit:
        files = files[:limit]

    ledger = Ledger(
        config.output_dir("audit") / f"step4r{config.get('runtime.ledger_name')}",
        enabled=bool(config.get("runtime.resume", True)),
    )
    refs_csv = config.output_dir("audit") / "reference_crops.csv"
    existing = {r["wine_slug"]: r for r in read_csv_rows(refs_csv)} if refs_csv.is_file() and ledger.enabled else {}

    manifest_rows: list[dict[str, Any]] = []
    progress = Progress(len(files), "reference crops", every=int(config.get("runtime.log_every", 200)))
    fail_fast = bool(config.get("runtime.fail_fast", False))

    with ledger:
        for path in files:
            slug = path.stem
            if slug in ledger and slug in existing:
                manifest_rows.append(existing[slug])
                progress.step()
                continue

            record: dict[str, Any] = {"wine_slug": slug, "reference_file": config.relative(path), "status": "error"}
            try:
                with Image.open(path) as handle:
                    upright = ImageOps.exif_transpose(handle)
                    rgba = np.asarray(upright.convert("RGBA"))
                rgb = rgba[:, :, :3]
                alpha = rgba[:, :, 3]

                if (alpha < 250).mean() < 0.01:
                    # A reference stored without real transparency: the silhouette
                    # has to come from the background instead, and that is worth
                    # flagging because it is less exact than a real alpha channel.
                    record.setdefault("flags", []).append("alpha_channel_absent")
                    mask = (np.abs(rgb.astype(np.int16) - 255).max(axis=2) > 12).astype(np.uint8)
                else:
                    mask = (alpha > alpha_threshold).astype(np.uint8)

                flags: list[str] = list(record.pop("flags", []))
                if max(rgb.shape[:2]) < min_long_side:
                    flags.append("low_resolution")

                bounds = geo.mask_bbox(mask)
                if bounds is None:
                    raise ValueError("reference silhouette is empty")
                bx1, by1, bx2, by2 = bounds
                silhouette_h, silhouette_w = by2 - by1, bx2 - bx1
                aspect = silhouette_h / max(1, silhouette_w)
                record["silhouette_aspect"] = round(aspect, 3)
                if bool(config.get("reference_views.flag_unusual_aspect", True)) and not (aspect_low <= aspect <= aspect_high):
                    # Out-of-range aspect means a crop that is not a whole bottle:
                    # a label-only shot, a box, or a group photo.
                    flags.append("unusual_aspect")

                if bool(config.get("reference_views.flag_truncated_neck", True)):
                    borders = geo.mask_touches_border(mask, margin=2)
                    if borders["top"]:
                        flags.append("neck_truncated")
                    if borders["left"] or borders["right"]:
                        flags.append("touches_side_border")

                views, meta = _render_views(rgb, mask, config=config, geo=geo, np=np, cv2=cv2)
                flags.extend(meta["quality_flags"])

                paths: dict[str, Any] = {}
                for name, array in views.items():
                    if array is None:
                        continue
                    # A lossy mask is a corrupted mask: masks stay PNG.
                    extension = "png" if name == "mask" else _extension(fmt)
                    destination = roots[name] / f"{slug}.{extension}"
                    _write_image(destination, array, fmt=extension, quality=quality)
                    paths[f"{name}_path"] = (
                        str(destination.relative_to(config.output_root).as_posix())
                        if bool(config.get("output.relative_paths", True))
                        else str(destination)
                    )
                record.update(paths)
                record.update({
                    "status": "ok",
                    "reference_width": int(rgb.shape[1]),
                    "reference_height": int(rgb.shape[0]),
                    "shoulder_method": meta["shoulder"]["method"],
                    "top_mask_fill": meta["top_mask_fill"],
                    "usable_for_bottle_comparison": not ({"neck_truncated", "low_resolution", "unusual_aspect"} & set(flags)),
                    "quality_flags": ";".join(sorted(set(flags))),
                })
            except Exception as error:  # noqa: BLE001
                if fail_fast:
                    raise
                record["error"] = f"{type(error).__name__}: {error}"

            manifest_rows.append(record)
            ledger.mark(slug, status=record["status"])
            progress.step()

    write_csv_rows(refs_csv, manifest_rows)
    ok = sum(1 for r in manifest_rows if r["status"] == "ok")
    usable = sum(1 for r in manifest_rows if str(r.get("usable_for_bottle_comparison")).lower() == "true")
    report = stage_record(
        "step4_reference_crops",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        references_processed=len(manifest_rows),
        references_rendered=ok,
        references_usable_for_bottle_comparison=usable,
        references_flagged_percent=percentage(ok - usable, max(1, ok)),
        quality_flag_counts=summarise_counts(
            flag for r in manifest_rows for flag in (r.get("quality_flags") or "").split(";") if flag
        ),
        outputs={"reference_crops_csv": config.relative(refs_csv),
                 "views_root": config.relative(config.output_dir("references", create=False))},
    )
    write_json(config.report_path("step4_reference_crops.json"), report)
    print(f"step4 (references): {ok}/{len(manifest_rows)} rendered, {usable} usable for bottle comparison")
    return report


def run(config: Config, *, limit: int | None = None, parts: Sequence[str] = ("queries", "references")) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "references" in parts:
        result["references"] = run_references(config, limit=limit)
    if "queries" in parts:
        result["queries"] = run_queries(config, limit=limit)
    return result
