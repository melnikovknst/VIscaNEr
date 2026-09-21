"""Step 2 - verify identities and image-to-identity correspondences.

For every query image this stage establishes: the source file, the wine_slug,
the catalog reference that slug points at, and whether a chosen label box
exists for it. It then asks the questions that decide whether the labelling can
support a bottle-level reranker at all:

* Does the labelling separate vintages, volumes and packaging variants, or does
  it merge them? Similar classes are never merged automatically here - the
  stage only reports what the catalog already distinguishes.
* Which identities are visually indistinguishable *by construction*, because
  two slugs share one catalog photograph? No bottle encoder can separate those.
* Which frames contain several bottles, so the target selection in step 3 has to
  be checked rather than assumed?

Anything doubtful goes to a review queue instead of into the clean training
pool. Nothing is silently repaired.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import (
    percentage,
    read_csv_rows,
    stable_unit_interval,
    stage_record,
    summarise_counts,
    write_csv_rows,
    write_json,
)
from .config import Config, SourceSpec

CONFIG_FINGERPRINT_KEYS = (
    "catalog.catalog_csv",
    "splits.seed",
    "splits.unseen_identity_fraction",
    "labels.crops_metadata_csv",
)

REFERENCE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Tokens that separate two bottlings of the same wine. Presence of these in a
# slug is evidence that the labelling *does* distinguish the variant.
YEAR_PATTERN = re.compile(r"(?:^|[-_])((?:19|20)\d{2})(?:$|[-_])")
VOLUME_PATTERN = re.compile(r"(?:^|[-_])(0[-_]?\d{1,2}|\d[-_]?\d{1,2}l|\d{3,4}ml)(?:$|[-_])")
ABV_PATTERN = re.compile(r"(?:^|[-_])(\d{2,4})$")
PACKAGING_TOKENS = ("beg-in-boks", "bag-in-box", "bib", "magnum", "tetra", "banka", "can", "keg")
COLOUR_TOKENS = ("beloe", "krasnoe", "rozovoe", "oranzhevoe")
SWEETNESS_TOKENS = ("suhoe", "polusuhoe", "polusladkoe", "sladkoe", "bryut", "ekstra-bryut")


# --------------------------------------------------------------- utilities
def _reference_lookup(*roots: Path) -> dict[str, dict[str, str]]:
    """slug -> {variant: path}. Mirrors dinov3_retrieval._reference_lookup."""
    lookup: dict[str, dict[str, str]] = defaultdict(dict)
    for root in roots:
        if not root.is_dir():
            continue
        variant = root.name
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in REFERENCE_EXTENSIONS:
                lookup[path.stem][variant] = str(path)
    return dict(lookup)


def _variant_tokens(slug: str) -> dict[str, Any]:
    """What a slug says about vintage, volume, packaging, colour and sweetness."""
    year = YEAR_PATTERN.search(slug)
    volume = VOLUME_PATTERN.search(slug)
    abv = ABV_PATTERN.search(slug)
    return {
        "year": year.group(1) if year else None,
        "volume_token": volume.group(1) if volume else None,
        "trailing_numeric": abv.group(1) if abv else None,
        "packaging": next((t for t in PACKAGING_TOKENS if t in slug), None),
        "colour": next((t for t in COLOUR_TOKENS if t in slug), None),
        "sweetness": next((t for t in SWEETNESS_TOKENS if slug.endswith(t) or f"-{t}-" in slug), None),
    }


def _strip_variant(slug: str) -> str:
    """The slug with vintage, volume and trailing ABV removed.

    Two identities that collapse to the same stripped slug differ *only* in a
    vintage, a volume or an alcohol figure - exactly the differences a bottle
    silhouette cannot show.
    """
    stripped = YEAR_PATTERN.sub("-", slug)
    stripped = VOLUME_PATTERN.sub("-", stripped)
    stripped = ABV_PATTERN.sub("", stripped)
    return re.sub(r"-+", "-", stripped).strip("-")


def _neighbour_count(params_json: str | None) -> int | None:
    """Neighbouring bottles recorded by the synthetic scene generator."""
    if not params_json:
        return None
    try:
        params = json.loads(params_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(params, dict):
        return None
    if params.get("background") == "shelf":
        # A shelf background always draws a row of bottles even when the
        # generator recorded no explicit neighbours.
        return max(int(params.get("neighbors", 0) or 0), 1)
    return int(params.get("neighbors", 0) or 0)


# ------------------------------------------------------------ catalog audit
def audit_catalog(config: Config) -> dict[str, Any]:
    catalog_path = config.path("catalog.catalog_csv")
    rows = read_csv_rows(catalog_path)
    if not rows:
        raise RuntimeError(f"Catalog is empty: {catalog_path}")

    refs = _reference_lookup(config.path("catalog.refs_rgb_root"), config.path("catalog.refs_rgba_root"))
    seed = int(config.get("splits.seed"))
    fraction = float(config.get("splits.unseen_identity_fraction"))

    by_slug: dict[str, dict[str, Any]] = {}
    same_image_groups: dict[str, list[str]] = defaultdict(list)
    near_dup_groups: dict[str, list[str]] = defaultdict(list)
    sha_groups: dict[str, list[str]] = defaultdict(list)
    stripped_groups: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        slug = row["slug"]
        tokens = _variant_tokens(slug)
        available = refs.get(slug, {})
        by_slug[slug] = {
            "slug": slug,
            "name": row.get("name", ""),
            "winery": row.get("winery", ""),
            "colour": row.get("color", ""),
            "category": row.get("category", ""),
            "image_sha256": row.get("image_sha256", ""),
            "near_dup_group": row.get("near_dup_group", ""),
            "same_image_group": row.get("same_image_group", ""),
            "reference_rgb": available.get("rgb"),
            "reference_rgba": available.get("rgba"),
            "has_reference": bool(available),
            "dino_unseen": stable_unit_interval(slug, seed) < fraction,
            **tokens,
        }
        if row.get("same_image_group"):
            same_image_groups[row["same_image_group"]].append(slug)
        if row.get("near_dup_group"):
            near_dup_groups[row["near_dup_group"]].append(slug)
        if row.get("image_sha256"):
            sha_groups[row["image_sha256"]].append(slug)
        stripped_groups[_strip_variant(slug)].append(slug)

    unseen = {s for s, r in by_slug.items() if r["dino_unseen"]}
    identical_reference = {g: s for g, s in same_image_groups.items() if len(s) > 1}
    identical_by_sha = {h: s for h, s in sha_groups.items() if len(s) > 1}
    variant_only = {k: v for k, v in stripped_groups.items() if len(v) > 1}

    # A group that has members on both sides of the DINO holdout boundary cannot
    # supply its hard negative without touching a reserved identity.
    straddling_near_dup = {
        g: s for g, s in near_dup_groups.items()
        if len(s) > 1 and (set(s) & unseen) and (set(s) - unseen)
    }
    straddling_same_image = {
        g: s for g, s in identical_reference.items()
        if (set(s) & unseen) and (set(s) - unseen)
    }

    return {
        "catalog_csv": config.relative(catalog_path),
        "identities": len(by_slug),
        "identities_with_reference": sum(1 for r in by_slug.values() if r["has_reference"]),
        "identities_without_reference": sorted(s for s, r in by_slug.items() if not r["has_reference"]),
        "distinct_reference_sha256": len(sha_groups),
        "labelling_distinguishes": {
            "vintage": sum(1 for r in by_slug.values() if r["year"]),
            "volume_token": sum(1 for r in by_slug.values() if r["volume_token"]),
            "packaging_variant": sum(1 for r in by_slug.values() if r["packaging"]),
            "colour_token": sum(1 for r in by_slug.values() if r["colour"]),
            "sweetness_token": sum(1 for r in by_slug.values() if r["sweetness"]),
            "note": (
                "Counts of identities whose slug carries the token. The catalog "
                "keeps these as separate identities; this stage does not merge them."
            ),
        },
        "variant_only_identity_groups": {
            "groups": len(variant_only),
            "identities": sum(len(v) for v in variant_only.values()),
            "note": (
                "Identities that become identical once vintage, volume and the "
                "trailing alcohol figure are stripped. A bottle silhouette "
                "cannot separate these; the label text or a vintage reader can."
            ),
            "examples": [sorted(v) for v in list(variant_only.values())[:10]],
        },
        "identical_reference_photo": {
            "groups": len(identical_reference),
            "identities": sum(len(v) for v in identical_reference.values()),
            "by_sha256_groups": len(identical_by_sha),
            "note": (
                "Two or more identities share one catalog photograph byte-for-byte. "
                "Neither a whole-bottle nor a label encoder can separate them, "
                "because the gallery holds the same image under both slugs."
            ),
            "members": {g: sorted(v) for g, v in sorted(identical_reference.items())},
        },
        "near_duplicate_label_series": {
            "groups": len(near_dup_groups),
            "identities": sum(len(v) for v in near_dup_groups.values()),
            "size_distribution": summarise_counts(len(v) for v in near_dup_groups.values()),
        },
        "dino_identity_holdout": {
            "seed": seed,
            "fraction": fraction,
            "unseen_identities": len(unseen),
            "unseen_percent": percentage(len(unseen), len(by_slug)),
            "near_dup_groups_straddling_holdout": len(straddling_near_dup),
            "identities_in_straddling_near_dup_groups": sum(len(v) for v in straddling_near_dup.values()),
            "same_image_groups_straddling_holdout": len(straddling_same_image),
            "note": (
                "Straddling groups are the ones whose natural hard negative lies "
                "on the other side of the generalisation holdout. Step 6 drops "
                "those pairs rather than leaking a reserved identity."
            ),
        },
        "_by_slug": by_slug,
        "_unseen": unseen,
        "_near_dup_groups": dict(near_dup_groups),
        "_same_image_groups": dict(identical_reference),
    }


# -------------------------------------------------------- correspondences
def audit_source(config: Config, spec: SourceSpec, catalog: dict[str, Any]) -> dict[str, Any]:
    """Resolve every frame of one source to a file, a slug and a reference."""
    by_slug = catalog["_by_slug"]
    rows: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []

    if not spec.manifest_csv.is_file():
        return {
            "source": spec.name,
            "available": False,
            "reason": f"manifest not found: {config.relative(spec.manifest_csv)}",
            "rows": [],
            "review": [],
        }

    manifest = read_csv_rows(spec.manifest_csv)
    seen_paths: Counter[str] = Counter()
    duplicate_hashes: dict[str, list[str]] = defaultdict(list)

    for raw in manifest:
        relative = (raw.get(spec.path_column) or "").strip()
        slug = (raw.get(spec.slug_column) or "").strip()
        reasons: list[str] = []

        for column, expected in spec.require_columns.items():
            if (raw.get(column) or "").strip() != expected:
                reasons.append(f"{column}!={expected}")

        if not relative:
            reasons.append("missing_path")
        if not slug:
            reasons.append("missing_slug")
        elif slug not in by_slug:
            reasons.append("slug_not_in_catalog")
        elif not by_slug[slug]["has_reference"]:
            reasons.append("identity_has_no_reference")

        image_path = spec.images_root / relative if relative else None
        file_exists = bool(image_path and image_path.is_file())
        if relative and not file_exists:
            reasons.append("image_file_missing")

        seen_paths[relative] += 1
        if seen_paths[relative] > 1:
            reasons.append("duplicate_manifest_row")

        digest = (raw.get(spec.exact_duplicate_column) or "").strip() if spec.exact_duplicate_column else ""
        if digest:
            duplicate_hashes[digest].append(relative)

        group_key = "|".join((raw.get(column) or "").strip() for column in spec.group_columns) or slug
        neighbours = _neighbour_count(raw.get("params"))

        record = {
            "source": spec.name,
            "source_kind": spec.kind,
            "photographic": spec.photographic,
            "image_relative_path": relative,
            "image_path": config.relative(image_path) if image_path else "",
            "image_exists": file_exists,
            "wine_slug": slug,
            "reference_rgb": by_slug.get(slug, {}).get("reference_rgb"),
            "reference_rgba": by_slug.get(slug, {}).get("reference_rgba"),
            "provenance_group": f"{spec.name}:{group_key}",
            "exact_hash": digest,
            "dino_unseen_identity": bool(by_slug.get(slug, {}).get("dino_unseen")),
            "near_dup_group": by_slug.get(slug, {}).get("near_dup_group", ""),
            "same_image_group": by_slug.get(slug, {}).get("same_image_group", ""),
            "generator_neighbours": neighbours,
            "multi_bottle_expected": bool(neighbours) if neighbours is not None else None,
            "review_reasons": ";".join(sorted(set(reasons))),
            "clean": not reasons,
        }
        rows.append(record)
        if reasons:
            review.append(record)

    # Byte-identical frames inside one source: they must never be split apart.
    exact_duplicate_sets = {h: p for h, p in duplicate_hashes.items() if len(p) > 1}

    per_slug = Counter(r["wine_slug"] for r in rows if r["clean"])
    multi = [r for r in rows if r["multi_bottle_expected"]]

    return {
        "source": spec.name,
        "kind": spec.kind,
        "photographic": spec.photographic,
        "available": True,
        "manifest_csv": config.relative(spec.manifest_csv),
        "images_root": config.relative(spec.images_root),
        "rows_in_manifest": len(manifest),
        "images_present_on_disk": sum(1 for r in rows if r["image_exists"]),
        "clean_rows": sum(1 for r in rows if r["clean"]),
        "review_rows": len(review),
        "review_reason_counts": summarise_counts(
            reason for r in review for reason in r["review_reasons"].split(";") if reason
        ),
        "identities_covered": len(per_slug),
        "identities_covered_percent": percentage(len(per_slug), catalog["identities"]),
        "images_per_identity_distribution": summarise_counts(per_slug.values()),
        "provenance_groups": len({r["provenance_group"] for r in rows}),
        "exact_duplicate_groups": len(exact_duplicate_sets),
        "exact_duplicate_images": sum(len(v) for v in exact_duplicate_sets.values()),
        "frames_with_neighbouring_bottles": len(multi),
        "frames_with_neighbouring_bottles_percent": percentage(len(multi), len(rows)),
        "rows": rows,
        "review": review,
    }


def run(config: Config) -> dict[str, Any]:
    catalog = audit_catalog(config)
    per_source: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_review: list[dict[str, Any]] = []

    for spec in config.sources():
        audit = audit_source(config, spec, catalog)
        all_rows.extend(audit.pop("rows"))
        all_review.extend(audit.pop("review"))
        per_source.append(audit)

    correspondence_path = config.output_dir("audit") / "correspondences.csv"
    review_path = config.output_dir("audit") / "review_queue.csv"
    write_csv_rows(correspondence_path, all_rows)
    write_csv_rows(review_path, all_review)

    photographic_rows = [r for r in all_rows if r["photographic"] and r["clean"]]
    rendered_rows = [r for r in all_rows if not r["photographic"] and r["clean"]]
    photographic_identities = {r["wine_slug"] for r in photographic_rows}

    public_catalog = {k: v for k, v in catalog.items() if not k.startswith("_")}
    report = stage_record(
        "step2_identities",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        catalog=public_catalog,
        sources=per_source,
        totals={
            "manifest_rows": len(all_rows),
            "clean_rows": sum(1 for r in all_rows if r["clean"]),
            "review_rows": len(all_review),
            "photographic_clean_rows": len(photographic_rows),
            "rendered_clean_rows": len(rendered_rows),
            "identities_with_a_real_photograph": len(photographic_identities),
            "identities_with_a_real_photograph_percent": percentage(
                len(photographic_identities), catalog["identities"]
            ),
        },
        outputs={
            "correspondences_csv": config.relative(correspondence_path),
            "review_queue_csv": config.relative(review_path),
        },
    )

    destination = config.report_path("step2_identities.json")
    write_json(destination, report)
    print(
        f"step2: {report['totals']['clean_rows']} clean rows, "
        f"{report['totals']['review_rows']} queued for review; "
        f"{report['totals']['identities_with_a_real_photograph']} of "
        f"{catalog['identities']} identities have a real photograph "
        f"-> {config.relative(destination)}"
    )
    return report
