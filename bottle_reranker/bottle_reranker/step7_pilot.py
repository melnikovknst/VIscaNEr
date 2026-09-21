"""Step 7 - pilot: is there a usable extra signal at all?

Before 45k frames are processed, a stratified pilot of 300-500 diverse cases is
rendered for human inspection. Each case is one HTML row showing, side by side:

  the original frame with the label box and the target mask drawn on it,
  the normalized bottle, its top part,
  both candidate references, and the correct answer.

The reviewer fills one column: which category the difference falls into.

  shape_neck_capsule_glass        a whole-bottle encoder can see this
  label_text_or_vintage           needs the label or OCR, not the silhouette
  no_visible_difference           nothing separates them in any photograph
  annotation_or_selection_error   the data is wrong, not the model
  insufficient_photo_quality      the frame cannot support any decision

The stage deliberately does NOT assume a whole-bottle encoder will solve the
second and third categories. Cases marked ``label_text_or_vintage`` are counted
separately as evidence for or against adding an OCR or label-zone check, which
is a different component from the one this dataset feeds.

After the pilot is reviewed and systematic errors are fixed, ``--full`` runs the
same rendering over everything, resumably.
"""

from __future__ import annotations

import html
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    percentage,
    read_csv_rows,
    stage_record,
    summarise_counts,
    write_csv_rows,
    write_json,
    write_text,
)
from .config import Config

CONFIG_FINGERPRINT_KEYS = ("pilot.size", "pilot.seed", "pilot.strata", "pilot.difference_categories")

STRATUM_HARD = "hard_pair"
STRATUM_MULTI = "multi_bottle"
STRATUM_TILTED = "tilted"
STRATUM_OCCLUDED = "occluded"
STRATUM_POOR_LIGHT = "poor_light"
STRATUM_RARE = "rare_identity"
STRATUM_CONFIDENT = "confident_correct"


def _stratum(row: dict[str, str], crop: dict[str, str] | None, rare: set[str]) -> str:
    flags = (crop or {}).get("quality_flags", "")
    case = row.get("case", "")
    if case in {"wrong_top1_correct_top2", "correct_top1_close", "correct_not_in_top2"}:
        stratum = STRATUM_HARD
    elif case == "confident_correct":
        stratum = STRATUM_CONFIDENT
    else:
        stratum = STRATUM_HARD
    if row.get("wine_slug") in rare:
        return STRATUM_RARE
    if "silhouette_occluded" in flags:
        return STRATUM_OCCLUDED
    if "low_sharpness" in flags:
        return STRATUM_POOR_LIGHT
    if crop and abs(float(crop.get("axis_angle_deg") or 0.0)) >= 6.0:
        return STRATUM_TILTED
    if str(crop.get("multi_bottle_expected", "")).lower() == "true" if crop else False:
        return STRATUM_MULTI
    return stratum


def _sample(rows: list[dict[str, Any]], strata: dict[str, float], size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row["_stratum"], []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked: list[dict[str, Any]] = []
    for name, share in strata.items():
        want = int(round(share * size))
        picked.extend(buckets.get(name, [])[:want])
    # Top up from whatever is left so a thin stratum does not shrink the pilot.
    if len(picked) < size:
        chosen = {id(r) for r in picked}
        leftovers = [r for r in rows if id(r) not in chosen]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: size - len(picked)])
    return picked[:size]


def _img(path: Path | str | None, root: Path, *, width: int = 150) -> str:
    if not path:
        return '<td class="missing">-</td>'
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    if not target.is_file():
        return f'<td class="missing">missing<br><small>{html.escape(target.name)}</small></td>'
    return f'<td><img src="{html.escape(target.as_uri())}" width="{width}"></td>'


def _render_html(cases: list[dict[str, Any]], config: Config, title: str) -> str:
    categories = list(config.get("pilot.difference_categories"))
    root = config.output_root
    refs_root = config.output_dir("references", create=False)

    head = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>
 body{{font:13px/1.45 system-ui,sans-serif;margin:24px;color:#1b1b1b;background:#fff}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #d8d8d8;padding:6px;vertical-align:top;text-align:left}}
 th{{background:#f3f3f3;position:sticky;top:0}}
 img{{display:block;max-height:260px;width:auto;background:#eee}}
 .missing{{color:#a00;font-size:11px}}
 .meta{{font-size:11px;color:#555;white-space:nowrap}}
 .truth{{font-weight:600}}
 code{{font-size:11px}}
</style>
<h1>{title}</h1>
<p>Cases: {count}. Fill the last column with one of:
 <code>{cats}</code>. Cosine similarities are raw similarities, not probabilities.</p>
<table><thead><tr>
 <th>#</th><th>original + box + mask</th><th>normalized bottle</th><th>top part</th>
 <th>candidate A</th><th>candidate B</th><th>facts</th><th>verdict</th>
</tr></thead><tbody>
""".format(title=html.escape(title), count=len(cases), cats=html.escape(" | ".join(categories)))

    body: list[str] = []
    for number, case in enumerate(cases, start=1):
        facts = (
            f'<div class="meta">slug <span class="truth">{html.escape(case["wine_slug"])}</span></div>'
            f'<div class="meta">case {html.escape(case.get("case", ""))}</div>'
            f'<div class="meta">split {html.escape(case.get("split", ""))} / exposure {html.escape(case.get("dino_exposure", ""))}</div>'
            f'<div class="meta">cos top1 {case.get("top1_cosine_similarity", "")} top2 {case.get("top2_cosine_similarity", "")}</div>'
            f'<div class="meta">margin {case.get("top1_minus_top2_cosine", "")}</div>'
            f'<div class="meta">correct rank {case.get("correct_rank", "-") or "-"}</div>'
            f'<div class="meta">flags {html.escape(case.get("quality_flags", "") or "-")}</div>'
        )
        candidate_a = case.get("cand1_slug", "")
        candidate_b = case.get("cand2_slug", "")
        body.append(
            "<tr>"
            f"<td>{number}</td>"
            + _img(case.get("overlay_path"), root, width=200)
            + _img(case.get("normalized_path"), root)
            + _img(case.get("top_path"), root, width=120)
            + _img(f"bottle/{candidate_a}.png" if candidate_a else None, refs_root, width=110)
            + _img(f"bottle/{candidate_b}.png" if candidate_b else None, refs_root, width=110)
            + f"<td>{facts}</td>"
            + '<td><input style="width:99%" placeholder="category"></td>'
            "</tr>"
        )
    return head + "\n".join(body) + "\n</tbody></table>\n"


def _render_overlays(config: Config, cases: list[dict[str, Any]]) -> None:
    """Draw the label box and target mask on a copy of each original frame."""
    try:
        import cv2
        import numpy as np

        from . import geometry as geo
    except ImportError:
        print("  overlays skipped: numpy/OpenCV not available")
        return

    overlays = config.output_dir("pilot", "overlays")
    for case in cases:
        source = config.resolve(case.get("image_path", ""))
        if not source.is_file():
            continue
        try:
            loaded = geo.load_source_image(source)
            canvas = loaded.rgb.copy()
            mask_npz = case.get("mask_npz")
            if mask_npz:
                payload = np.load(config.resolve(mask_npz))
                shape = tuple(int(v) for v in payload["shape"])
                mask = np.unpackbits(payload["mask"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
                tint = canvas.copy()
                tint[mask] = (0.45 * tint[mask] + 0.55 * np.array([0, 190, 120])).astype(np.uint8)
                canvas = tint
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, (0, 120, 255), 3)
            for key in ("label_x1", "label_y1", "label_x2", "label_y2"):
                if not case.get(key):
                    break
            else:
                x1, y1, x2, y2 = (int(float(case[f"label_{k}"])) for k in ("x1", "y1", "x2", "y2"))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 40, 40), 3)
            small = geo.limit_long_side(canvas, 700)
            destination = overlays / f"{case['source']}__{Path(case['image_relative_path']).name}.jpg"
            ok, buffer = cv2.imencode(".jpg", small[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if ok:
                destination.write_bytes(buffer.tobytes())
                case["overlay_path"] = str(destination.relative_to(config.output_root).as_posix())
        except Exception as error:  # noqa: BLE001
            case["overlay_error"] = f"{type(error).__name__}: {error}"


def run(config: Config, *, full: bool = False) -> dict[str, Any]:
    audit = config.output_dir("audit", create=False)
    candidates_csv = audit / "dino_candidates.csv"
    crops_csv = audit / "query_crops.csv"
    segmentation_csv = audit / "segmentation.csv"
    if not candidates_csv.is_file():
        raise FileNotFoundError(f"{candidates_csv} not found. Run step 6a first.")

    crops = {f"{r['source']}:{r['image_relative_path']}": r for r in read_csv_rows(crops_csv)} if crops_csv.is_file() else {}
    segments = {f"{r['source']}:{r['image_relative_path']}": r for r in read_csv_rows(segmentation_csv)} if segmentation_csv.is_file() else {}

    rows = read_csv_rows(candidates_csv)
    frequency = Counter(r["wine_slug"] for r in rows)
    rare = {slug for slug, count in frequency.items() if count <= 2}

    merged: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row['source']}:{row['image_relative_path']}"
        crop = crops.get(key, {})
        segment = segments.get(key, {})
        record = {**row, **{k: v for k, v in crop.items() if k not in row},
                  **{k: v for k, v in segment.items() if k not in row and k not in crop}}
        record["_stratum"] = _stratum(row, crop or None, rare)
        merged.append(record)

    selected = merged if full else _sample(
        merged,
        {k: float(v) for k, v in config.get("pilot.strata").items()},
        int(config.get("pilot.size")),
        int(config.get("pilot.seed")),
    )

    if bool(config.get("pilot.render_originals_with_overlay", True)):
        _render_overlays(config, selected)

    name = "full_review" if full else "pilot"
    sheet_csv = config.output_dir("pilot") / f"{name}_cases.csv"
    for case in selected:
        case["difference_category"] = ""     # filled in by the reviewer
        case["reviewer_note"] = ""
    write_csv_rows(sheet_csv, selected)

    html_path = config.report_path(f"{name}_review.html")
    write_text(html_path, _render_html(selected, config, f"VIscaNEr bottle reranker - {name} review"))

    report = stage_record(
        "step7_pilot" if not full else "step7_full_review",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        cases=len(selected),
        stratum_counts=summarise_counts(c["_stratum"] for c in selected),
        case_counts=summarise_counts(c.get("case", "") for c in selected),
        exposure_counts=summarise_counts(c.get("dino_exposure", "") for c in selected),
        quality_flag_counts=summarise_counts(
            flag for c in selected for flag in (c.get("quality_flags") or "").split(";") if flag
        ),
        overlays_rendered=sum(1 for c in selected if c.get("overlay_path")),
        review_categories=config.get("pilot.difference_categories"),
        instructions=(
            "Open the HTML sheet, judge each row, and write the category into "
            "difference_category in the CSV. Then run `viscaner-reranker pilot "
            "--summarise` to fold the verdicts into the headroom estimate. Do "
            "not assume a whole-bottle encoder solves label_text_or_vintage or "
            "no_visible_difference."
        ),
        outputs={"cases_csv": config.relative(sheet_csv), "review_html": config.relative(html_path)},
    )
    write_json(config.report_path(f"step7_{name}.json"), report)
    print(f"step7: {len(selected)} cases -> {config.relative(html_path)}")
    return report


def summarise(config: Config) -> dict[str, Any]:
    """Fold reviewed verdicts into a headroom estimate."""
    sheet = config.output_dir("pilot", create=False) / "pilot_cases.csv"
    if not sheet.is_file():
        raise FileNotFoundError(f"{sheet} not found. Run the pilot first.")
    rows = read_csv_rows(sheet)
    reviewed = [r for r in rows if (r.get("difference_category") or "").strip()]
    verdicts = summarise_counts(r["difference_category"].strip() for r in reviewed)

    recoverable = [
        r for r in reviewed
        if r["difference_category"].strip() == "shape_neck_capsule_glass"
        and r.get("case") in {"wrong_top1_correct_top2", "correct_top1_close"}
    ]
    fixable_cases = [r for r in reviewed if r.get("case") in {"wrong_top1_correct_top2", "correct_top1_close"}]

    report = stage_record(
        "step7_summary",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        cases_total=len(rows),
        cases_reviewed=len(reviewed),
        review_coverage_percent=percentage(len(reviewed), max(1, len(rows))),
        verdict_counts=verdicts,
        addressable_by_whole_bottle={
            "reviewed_error_or_close_cases": len(fixable_cases),
            "with_a_visible_bottle_difference": len(recoverable),
            "share_percent": percentage(len(recoverable), max(1, len(fixable_cases))),
            "note": (
                "Share of reviewed close/wrong cases where a reviewer could see a "
                "difference in shape, neck, capsule or glass. This is an upper "
                "bound on what a whole-bottle encoder could fix, measured before "
                "any training. It is not a predicted improvement."
            ),
        },
        needs_ocr_or_label_zone=verdicts.get("label_text_or_vintage", 0),
        data_problems=verdicts.get("annotation_or_selection_error", 0)
        + verdicts.get("insufficient_photo_quality", 0),
        unseparable=verdicts.get("no_visible_difference", 0),
    )
    write_json(config.report_path("step7_summary.json"), report)
    print(f"step7 summary: {len(reviewed)}/{len(rows)} reviewed; "
          f"{report['addressable_by_whole_bottle']['share_percent']}% of hard cases "
          "show a visible bottle-level difference")
    return report
