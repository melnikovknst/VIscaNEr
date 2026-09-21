"""Roll the per-stage JSON reports into one readable coverage report.

Reports what happened, including what did not. A stage that could not run
appears as a named gap with the input it was waiting for, never as an absence.
"""

from __future__ import annotations

from typing import Any

from .common import percentage, read_json, write_json, write_text
from .config import Config

STAGES = [
    ("step1_inputs.json", "1. Inputs"),
    ("step2_identities.json", "2. Identities and correspondences"),
    ("step3_segment.json", "3. Bottle segmentation and target selection"),
    ("step4_reference_crops.json", "4a. Catalog reference views"),
    ("step4_query_crops.json", "4b. Query bottle and top crops"),
    ("step5_splits.json", "5. Leakage-free splits"),
    ("step6_candidates.json", "6a. DINO candidates"),
    ("step6_pairs.json", "6b. Training pairs"),
    ("step7_pilot.json", "7. Pilot review"),
    ("step7_summary.json", "7b. Pilot verdicts"),
    ("step8_gap_plan.json", "8. Collection plan"),
    ("step9_package.json", "9. Package"),
]


def _line(key: str, value: Any) -> str:
    return f"- **{key}**: {value}"


def run(config: Config) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    lines: list[str] = ["# Bottle reranker dataset - coverage report", ""]

    step1 = read_json(config.report_path("step1_inputs.json", create_parent=False))
    if step1:
        missing = step1.get("missing_blocking_inputs", [])
        lines += ["## Status", ""]
        if missing:
            lines.append(
                f"**Not a finished dataset.** {len(missing)} required inputs are absent, "
                "so the stages that depend on them did not run. They are listed below "
                "with the exact path that was checked."
            )
        else:
            lines.append("All declared inputs were found.")
        lines.append("")

    for filename, title in STAGES:
        payload = read_json(config.report_path(filename, create_parent=False))
        key = filename.removesuffix(".json")
        collected[key] = payload
        lines += [f"## {title}", ""]
        if payload is None:
            lines += ["_Did not run._", ""]
            continue

        if key == "step1_inputs":
            summary = payload["summary"]
            lines += [
                _line("inputs present", f"{summary['present']}/{summary['inputs_checked']}"),
                _line("blocking gaps", summary["missing_blocking"]),
                _line("blocked steps", ", ".join(summary["blocked_steps"]) or "none"),
                "",
            ]
            inventory = payload.get("photographic_inventory", {})
            lines += [
                _line("photographic images found", inventory.get("photographic_images", 0)),
                _line("rendered images found", inventory.get("rendered_images", 0)),
                "",
            ]
            if payload.get("missing_blocking_inputs"):
                lines.append("Missing, blocking:")
                lines += [f"  - `{m['key']}` -> `{m['path']}` (blocks {', '.join(m['blocks'])})"
                          for m in payload["missing_blocking_inputs"]]
                lines.append("")

        elif key == "step2_identities":
            catalog = payload["catalog"]
            totals = payload["totals"]
            lines += [
                _line("identities", catalog["identities"]),
                _line("identities with a reference", catalog["identities_with_reference"]),
                _line("distinct reference photographs", catalog["distinct_reference_sha256"]),
                _line("identities sharing one photograph",
                      f"{catalog['identical_reference_photo']['identities']} "
                      f"in {catalog['identical_reference_photo']['groups']} groups"),
                _line("near-duplicate label series",
                      f"{catalog['near_duplicate_label_series']['identities']} identities "
                      f"in {catalog['near_duplicate_label_series']['groups']} groups"),
                _line("clean correspondences", totals["clean_rows"]),
                _line("queued for review", totals["review_rows"]),
                _line("identities with a real photograph",
                      f"{totals['identities_with_a_real_photograph']} "
                      f"({totals['identities_with_a_real_photograph_percent']}%)"),
                "",
            ]

        elif key == "step3_segment":
            lines += [
                _line("frames processed", payload["frames_processed"]),
                _line("confident target masks", f"{payload['selection_rate_percent']}%"),
                _line("problem masks", f"{payload['problem_mask_percent']}%"),
                _line("status breakdown", payload["status_counts"]),
                "",
            ]

        elif key in {"step4_query_crops", "step4_reference_crops"}:
            for field in ("crops_written", "crops_failed", "references_rendered",
                          "references_usable_for_bottle_comparison", "flagged_percent",
                          "references_flagged_percent"):
                if field in payload:
                    lines.append(_line(field.replace("_", " "), payload[field]))
            if payload.get("quality_flag_counts"):
                lines.append(_line("quality flags", payload["quality_flag_counts"]))
            lines.append("")

        elif key == "step5_splits":
            lines += [
                _line("split counts", payload["split_counts"]),
                _line("identities per split", payload["identities_per_split"]),
                _line("real photographs per split", payload["photographic_images_per_split"]),
                _line("provenance groups", payload["grouping"]["groups"]),
                _line("reserved identities leaking into training",
                      len(payload["integrity"]["reserved_identities_leaking_into_training"])),
                _line("seen identities pulled into the holdout by grouping",
                      payload["integrity"]["seen_identities_pulled_into_holdout_by_grouping"]),
                _line("held-out photographic test",
                      f"{payload['evaluation_status']['held_out_photographic_test_images']} images, "
                      f"{payload['evaluation_status']['held_out_photographic_test_identities']} identities"),
                "",
                f"> {payload['evaluation_status']['note']}",
                "",
            ]

        elif key == "step6_candidates":
            lines += [
                _line("queries scored", payload["queries"]),
                _line("case counts", payload["case_counts"]),
                _line("recall over all queries", payload["recall_all"]),
                _line("recall on DINO-held-out images", payload["recall_on_dino_held_out"]),
                _line("correct answer sitting at rank 2",
                      f"{payload['headroom']['correct_answer_at_rank_2']} "
                      f"({payload['headroom']['correct_answer_at_rank_2_percent']}%)"),
                "",
                f"> {payload['score_semantics']}",
                "",
            ]

        elif key == "step6_pairs":
            lines += [
                _line("organic pairs", payload["organic_pairs"]),
                _line("injected triplets", payload["injected_triplets"]),
                _line("label balance (organic)", payload["organic_label_counts"]),
                _line("correct answer in the first slot", f"{payload['first_slot_is_correct_percent']}%"),
                _line("confident-correct share", f"{payload['confident_correct_share_percent']}%"),
                _line("pairs with no possible visual difference", payload["unseparable_pairs"]),
                _line("dropped", payload["dropped"]),
                "",
            ]
            for warning in payload.get("warnings", []):
                lines += [f"> {warning}", ""]

        elif key == "step7_summary":
            addressable = payload["addressable_by_whole_bottle"]
            lines += [
                _line("cases reviewed", f"{payload['cases_reviewed']}/{payload['cases_total']}"),
                _line("verdicts", payload["verdict_counts"]),
                _line("hard cases with a visible bottle difference",
                      f"{addressable['with_a_visible_bottle_difference']}/"
                      f"{addressable['reviewed_error_or_close_cases']} "
                      f"({addressable['share_percent']}%)"),
                _line("cases needing OCR or a label-zone check", payload["needs_ocr_or_label_zone"]),
                _line("cases with no visible difference", payload["unseparable"]),
                "",
                f"> {addressable['note']}",
                "",
            ]

        elif key == "step8_gap_plan":
            lines += [
                _line("confusion pairs ranked", payload["confusion_pairs"]),
                _line("decidable from the bottle", payload["pairs_decidable_from_the_bottle"]),
                _line("needing label text or OCR", payload["pairs_needing_label_or_ocr"]),
                _line("no possible visual difference", payload["pairs_with_no_possible_visual_difference"]),
                _line("coverage", payload["coverage"]),
                _line("evidence source", payload["evidence_source"]),
                "",
            ]

        elif key == "step9_package":
            manifest = payload["manifest"]
            lines += [
                _line("manifest rows", manifest["rows"]),
                _line("by role", manifest["by_role"]),
                _line("by split", manifest["by_split"]),
                _line("usable for training", f"{manifest['usable_for_training']} ({manifest['usable_percent']}%)"),
                _line("kaggle dataset", payload["kaggle"]["dataset_id"]),
                "",
            ]
        else:
            lines += [_line("cases", payload.get("cases", "-")), ""]

    lines += [
        "## What this report does not claim",
        "",
        "- No improvement is predicted. Nothing has been trained yet, and a gain "
        "claimed before training and an independent test would be a guess.",
        "- Cosine similarities are not probabilities of being correct.",
        "- A figure computed on the DINO validation split is internal validation. "
        "The split was already used to pick the checkpoint.",
        "",
    ]

    destination = config.report_path("COVERAGE.md")
    write_text(destination, "\n".join(lines))
    write_json(config.report_path("coverage.json"), collected)
    print(f"report: {config.relative(destination)}")
    return collected
