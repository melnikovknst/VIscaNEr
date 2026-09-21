"""Step 8 - what to photograph next, and why.

Ranks confusion pairs by how much damage they do and how badly they are covered,
then says for each one what a photographer would have to bring back.

Two rules shape the plan:

* A burst of near-identical frames from one session is one observation, not
  twenty. Coverage is counted in independent shooting sessions (provenance
  groups), not in files.
* Photographs for the final test are collected as separate sessions and never
  enter training. A wine photographed once, with both halves of the shoot split
  between train and test, gives a flattering test number.

When DINO candidate data exists the ranking uses real confusion counts. Without
it the stage still ranks the *structurally* risky pairs - identities that share
a near-duplicate label series or an identical catalog photograph - so collection
can start before the model is available.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .common import (
    percentage,
    read_csv_rows,
    stage_record,
    write_csv_rows,
    write_json,
)
from .config import Config

CONFIG_FINGERPRINT_KEYS = ("catalog.catalog_csv", "splits.test_sources", "dino.top_k")

# What has to be visible to tell the two apart, inferred from what the catalog
# says the two identities differ in.
NEED_LABEL_TEXT = "label text or vintage must be legible"
NEED_BOTTLE = "bottle shape, shoulders, neck, capsule, glass colour"
NEED_COLOUR = "wine colour through the glass"
NEED_NOTHING = "no visual difference exists - fix the catalog, not the camera"


def _required_evidence(left: dict[str, str], right: dict[str, str]) -> str:
    if left.get("image_sha256") and left["image_sha256"] == right.get("image_sha256"):
        return NEED_NOTHING
    if left.get("color") != right.get("color") or left.get("category") != right.get("category"):
        return NEED_COLOUR
    from .step2_identities import _strip_variant

    if _strip_variant(left["slug"]) == _strip_variant(right["slug"]):
        return NEED_LABEL_TEXT
    return NEED_BOTTLE


def _missing_conditions(sessions: int, images: int) -> list[str]:
    gaps: list[str] = []
    if sessions == 0:
        gaps.append("no real photograph at all")
        return gaps
    if sessions == 1:
        gaps.append("only one shooting session - no independent angle, light or background")
    if sessions < 3:
        gaps.append("fewer than three independent sessions")
    if images < 4:
        gaps.append("too few frames for a train/test session split")
    gaps.append("needed: close and mid distance, front and 30-45 degree rotation, shelf and plain background, warm shop light and daylight")
    return gaps


def run(config: Config) -> dict[str, Any]:
    catalog = {r["slug"]: r for r in read_csv_rows(config.path("catalog.catalog_csv"))}

    splits_csv = config.output_dir("audit", create=False) / "splits.csv"
    sessions_by_slug: dict[str, set[str]] = defaultdict(set)
    photos_by_slug: Counter[str] = Counter()
    if splits_csv.is_file():
        for row in read_csv_rows(splits_csv):
            if str(row.get("photographic", "")).lower() != "true":
                continue
            sessions_by_slug[row["wine_slug"]].add(row["provenance_group"])
            photos_by_slug[row["wine_slug"]] += 1

    # ---- confusion pairs ------------------------------------------------
    candidates_csv = config.output_dir("audit", create=False) / "dino_candidates.csv"
    confusion: Counter[tuple[str, str]] = Counter()
    observed = candidates_csv.is_file()
    if observed:
        top_k = int(config.get("dino.top_k"))
        for row in read_csv_rows(candidates_csv):
            truth = row["wine_slug"]
            for rank in range(1, min(3, top_k) + 1):
                rival = (row.get(f"cand{rank}_slug") or "").strip()
                if rival and rival != truth:
                    confusion[tuple(sorted((truth, rival)))] += 1

    if not confusion:
        # Structural fallback: the catalog already knows which identities look
        # alike. Collection can start on those before the model exists.
        groups: dict[str, list[str]] = defaultdict(list)
        for slug, row in catalog.items():
            if row.get("near_dup_group"):
                groups[row["near_dup_group"]].append(slug)
            if row.get("same_image_group"):
                groups[f"img:{row['same_image_group']}"].append(slug)
        for members in groups.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    confusion[tuple(sorted((members[i], members[j])))] += 1

    rows: list[dict[str, Any]] = []
    for (left_slug, right_slug), weight in confusion.most_common():
        left, right = catalog.get(left_slug), catalog.get(right_slug)
        if not left or not right:
            continue
        left_sessions, right_sessions = len(sessions_by_slug[left_slug]), len(sessions_by_slug[right_slug])
        evidence = _required_evidence(left, right)
        rows.append({
            "slug_a": left_slug,
            "slug_b": right_slug,
            "name_a": left.get("name", ""),
            "name_b": right.get("name", ""),
            "winery_a": left.get("winery", ""),
            "winery_b": right.get("winery", ""),
            "confusion_weight": weight,
            "evidence_source": "dino_top3" if observed else "catalog_structure",
            "independent_sessions_a": left_sessions,
            "independent_sessions_b": right_sessions,
            "photos_a": photos_by_slug[left_slug],
            "photos_b": photos_by_slug[right_slug],
            "min_independent_sessions": min(left_sessions, right_sessions),
            "distinguishing_evidence_required": evidence,
            "whole_bottle_can_decide": evidence == NEED_BOTTLE,
            "share_one_catalog_photo": left.get("image_sha256") == right.get("image_sha256"),
            "missing_a": "; ".join(_missing_conditions(left_sessions, photos_by_slug[left_slug])),
            "missing_b": "; ".join(_missing_conditions(right_sessions, photos_by_slug[right_slug])),
            "action": (
                "fix the catalog: two identities share one photograph"
                if evidence == NEED_NOTHING else
                "reshoot both wines in separate sessions; keep one session aside for the final test"
            ),
        })

    # Priority: frequently confused, poorly covered, and actually decidable.
    for row in rows:
        row["priority_score"] = round(
            row["confusion_weight"] * (1.0 / (1 + row["min_independent_sessions"]))
            * (1.25 if row["whole_bottle_can_decide"] else 0.6),
            3,
        )
    rows.sort(key=lambda r: r["priority_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["priority_rank"] = rank

    plan_csv = config.output_dir("plan") / "confusion_pairs_to_collect.csv"
    write_csv_rows(plan_csv, rows)

    never_photographed = sorted(s for s in catalog if not sessions_by_slug.get(s))
    single_session = sorted(s for s, g in sessions_by_slug.items() if len(g) == 1)

    report = stage_record(
        "step8_gap_plan",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        evidence_source="dino_candidates" if observed else "catalog_structure_only",
        confusion_pairs=len(rows),
        pairs_decidable_from_the_bottle=sum(1 for r in rows if r["whole_bottle_can_decide"]),
        pairs_needing_label_or_ocr=sum(
            1 for r in rows if r["distinguishing_evidence_required"] == NEED_LABEL_TEXT
        ),
        pairs_with_no_possible_visual_difference=sum(
            1 for r in rows if r["distinguishing_evidence_required"] == NEED_NOTHING
        ),
        coverage={
            "identities_in_catalog": len(catalog),
            "identities_never_photographed": len(never_photographed),
            "identities_never_photographed_percent": percentage(len(never_photographed), len(catalog)),
            "identities_with_one_session_only": len(single_session),
            "identities_with_three_or_more_sessions": sum(1 for g in sessions_by_slug.values() if len(g) >= 3),
        },
        collection_rules=[
            "One shooting session is one observation. A burst of nearly identical "
            "frames does not count as variety.",
            "Vary angle, distance, light and background across sessions, not inside one.",
            "Photograph both members of a confusion pair, not only the one that loses.",
            "Shoot the final-test material as separate sessions and keep it out of "
            "training entirely.",
            "For pairs marked 'no visual difference exists', no amount of "
            "photography helps; the catalog entry has to be corrected.",
        ],
        top_20=rows[:20],
        outputs={"plan_csv": config.relative(plan_csv)},
    )
    write_json(config.report_path("step8_gap_plan.json"), report)
    print(f"step8: {len(rows)} confusion pairs ranked "
          f"({report['pairs_decidable_from_the_bottle']} decidable from the bottle, "
          f"{report['pairs_with_no_possible_visual_difference']} impossible) "
          f"-> {config.relative(plan_csv)}")
    return report
