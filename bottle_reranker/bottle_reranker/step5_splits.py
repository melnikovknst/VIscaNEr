"""Step 5 - split the data without leaks.

Three separate concerns, kept separate on purpose:

1. **Identity holdout.** The DINO retrieval pipeline reserves a deterministic
   slice of identities for generalisation. That decision is reproduced here
   bit-for-bit (``sha256("{seed}:{slug}")``) so an identity reserved there stays
   reserved here - as anchor, as positive, and as negative. Step 6 drops any
   pair that would use one.

2. **Provenance groups.** Exact duplicates, near-identical frames and frames
   from one shooting session travel together. For real photos the session is
   the source post; for renders every frame of a wine descends from that wine's
   single reference, so the identity is the group. Grouped images always land in
   the same part.

3. **Selection vs final test.** train/dev come from the seen identities. The
   final test is a separate pool that is never consulted when choosing an
   architecture, a threshold or an epoch count.

The stage is explicit about what the resulting numbers are worth. The DINO
validation that picked the current checkpoint has already been used for
development, so any figure computed on it is internal validation, not an
independent test, and is labelled that way in the output.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .common import (
    percentage,
    read_csv_rows,
    stable_unit_interval,
    stage_record,
    summarise_counts,
    write_csv_rows,
    write_json,
)
from .config import Config

CONFIG_FINGERPRINT_KEYS = (
    "splits.seed",
    "splits.unseen_identity_fraction",
    "splits.inherit_dino_holdout",
    "splits.train_fraction",
    "splits.dev_fraction",
    "splits.test_sources",
    "splits.near_duplicate_detection",
)

SPLIT_TRAIN = "train"
SPLIT_DEV = "dev"
SPLIT_TEST = "test"
SPLIT_UNSEEN = "holdout_unseen_identity"


class DisjointSet:
    """Union-find over provenance keys, so transitive links merge correctly."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:     # path compression
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self._parent[b] = a

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for key in self._parent:
            out[self.find(key)].append(key)
        return dict(out)


def _phash(path, *, hash_size: int = 8):
    """Perceptual hash as an integer, or None when imaging libs are absent.

    A small DCT-based hash: robust to JPEG level and mild resizing, which is
    exactly the "almost the same photo" case that must not straddle a split.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as handle:
            gray = handle.convert("L").resize((hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
    except Exception:  # noqa: BLE001 - an unreadable frame is reported elsewhere
        return None
    # 2-D DCT-II via the orthogonal basis; scipy is not a dependency here.
    size = pixels.shape[0]
    grid = np.arange(size)
    basis = np.cos(np.pi * (2 * grid[:, None] + 1) * grid[None, :] / (2 * size))
    transformed = basis.T @ pixels @ basis
    low = transformed[:hash_size, :hash_size].flatten()
    median = np.median(low[1:])
    bits = (low > median).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def build_groups(
    rows: list[dict[str, Any]],
    *,
    config: Config,
    compute_phash: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Merge every image into one provenance group per connected component."""
    detection = config.get("splits.near_duplicate_detection")
    threshold = int(detection.get("phash_hamming_threshold", 6))
    group_near = bool(detection.get("group_near_duplicates", True))
    # Near-duplicate links are confined to one identity by default. On the
    # synthetic renders a 64-bit perceptual hash is dominated by the shared
    # shelf and photo backgrounds rather than by the bottle, so unrestricted
    # linking chains unrelated wines together: an early run merged 953 images
    # spanning dozens of identities into a single group. Images of two
    # different wines that genuinely belong together - a shelf photo covering
    # both - are already joined by their provenance group (the source post).
    same_identity_only = bool(detection.get("restrict_to_same_identity", True))

    union = DisjointSet()
    by_hash: dict[str, list[str]] = defaultdict(list)
    keys: list[str] = []

    for row in rows:
        key = f"{row['source']}:{row['image_relative_path']}"
        keys.append(key)
        union.add(key)
        union.union(key, row["provenance_group"])
        digest = (row.get("exact_hash") or "").strip()
        if digest:
            by_hash[digest].append(key)

    exact_groups = 0
    for members in by_hash.values():
        if len(members) > 1:
            exact_groups += 1
            for other in members[1:]:
                union.union(members[0], other)

    near_pairs = 0
    skipped_cross_identity = 0
    if group_near and compute_phash:
        hashes: list[tuple[str, int, str]] = []
        for row, key in zip(rows, keys):
            value = _phash(config.resolve(row["image_path"]))
            if value is not None:
                hashes.append((key, value, row["wine_slug"]))

        # Comparisons are bucketed so the sweep stays near-linear instead of
        # quadratic over 45k frames. Buckets are keyed by the identity when
        # cross-identity linking is off, which is both cheaper and exact;
        # otherwise by the top bits of the hash, which is approximate.
        buckets: dict[Any, list[tuple[str, int, str]]] = defaultdict(list)
        for entry in hashes:
            buckets[entry[2] if same_identity_only else (entry[1] >> 48)].append(entry)

        for bucket in buckets.values():
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    if _hamming(bucket[i][1], bucket[j][1]) > threshold:
                        continue
                    if same_identity_only and bucket[i][2] != bucket[j][2]:
                        skipped_cross_identity += 1
                        continue
                    union.union(bucket[i][0], bucket[j][0])
                    near_pairs += 1

    assignment = {key: union.find(key) for key in keys}
    stats = {
        "images": len(keys),
        "groups": len(set(assignment.values())),
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_links": near_pairs,
        "near_duplicate_links_rejected_cross_identity": skipped_cross_identity,
        "near_duplicate_restricted_to_same_identity": same_identity_only,
        "perceptual_hash_computed": bool(compute_phash),
    }
    return assignment, stats


def run(config: Config, *, compute_phash: bool = False) -> dict[str, Any]:
    audit_csv = config.output_dir("audit", create=False) / "correspondences.csv"
    if not audit_csv.is_file():
        raise FileNotFoundError(f"{audit_csv} not found. Run step 2 first.")

    rows = [r for r in read_csv_rows(audit_csv) if r.get("clean", "").lower() in {"true", "1"}]
    seed = int(config.get("splits.seed"))
    fraction = float(config.get("splits.unseen_identity_fraction"))
    inherit = bool(config.get("splits.inherit_dino_holdout", True))
    train_fraction = float(config.get("splits.train_fraction"))
    test_sources = set(config.get("splits.test_sources", []))

    assignment, group_stats = build_groups(rows, config=config, compute_phash=compute_phash)

    # A provenance group is assigned as a unit. Its deterministic draw comes
    # from the group key, so the split survives adding new images to other
    # groups and is reproducible from the seed alone.
    group_identities: dict[str, set[str]] = defaultdict(set)
    group_sources: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = assignment[f"{row['source']}:{row['image_relative_path']}"]
        group_identities[key].add(row["wine_slug"])
        group_sources[key].add(row["source"])

    unseen_identities = {
        slug for slug in {r["wine_slug"] for r in rows}
        if inherit and stable_unit_interval(slug, seed) < fraction
    }

    group_split: dict[str, str] = {}
    for key, identities in group_identities.items():
        if identities & unseen_identities:
            # One reserved identity contaminates the whole group: the group is
            # held out rather than partially used.
            group_split[key] = SPLIT_UNSEEN
        elif group_sources[key] & test_sources:
            group_split[key] = SPLIT_TEST
        else:
            group_split[key] = SPLIT_TRAIN if stable_unit_interval(key, seed) < train_fraction else SPLIT_DEV

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        key = assignment[f"{row['source']}:{row['image_relative_path']}"]
        out_rows.append({
            "source": row["source"],
            "image_relative_path": row["image_relative_path"],
            "image_path": row["image_path"],
            "wine_slug": row["wine_slug"],
            "photographic": row["photographic"],
            "provenance_group": key,
            "dino_unseen_identity": row["dino_unseen_identity"],
            "split": group_split[key],
            "usable_for_reranker_training": group_split[key] in {SPLIT_TRAIN, SPLIT_DEV},
        })

    splits_csv = config.output_dir("audit") / "splits.csv"
    write_csv_rows(splits_csv, out_rows)

    counts = Counter(r["split"] for r in out_rows)
    identities_by_split = {
        split: len({r["wine_slug"] for r in out_rows if r["split"] == split})
        for split in (SPLIT_TRAIN, SPLIT_DEV, SPLIT_TEST, SPLIT_UNSEEN)
    }
    photographic_by_split = {
        split: sum(1 for r in out_rows if r["split"] == split and str(r["photographic"]).lower() == "true")
        for split in (SPLIT_TRAIN, SPLIT_DEV, SPLIT_TEST, SPLIT_UNSEEN)
    }

    # Cross-split identity overlap is legitimate for train/dev (the catalog is
    # closed, the same wine may appear in both). What must never happen is a
    # RESERVED identity reaching training. A merely seen identity that got
    # pulled into the holdout because it shared a provenance group with a
    # reserved one is a cost, not a leak, and is counted separately.
    training_slugs = {r["wine_slug"] for r in out_rows if r["split"] in {SPLIT_TRAIN, SPLIT_DEV}}
    held_out_slugs = {r["wine_slug"] for r in out_rows if r["split"] == SPLIT_UNSEEN}
    leaked = sorted(unseen_identities & training_slugs)
    pulled_in = sorted((held_out_slugs - unseen_identities) & training_slugs)

    group_leaks = [
        key for key, split in group_split.items()
        if len({group_split[assignment[f"{r['source']}:{r['image_relative_path']}"]] for r in rows
                if assignment[f"{r['source']}:{r['image_relative_path']}"] == key}) > 1
    ]

    report = stage_record(
        "step5_splits",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        grouping=group_stats,
        split_counts=dict(counts),
        split_percent={k: percentage(v, len(out_rows)) for k, v in counts.items()},
        identities_per_split=identities_by_split,
        photographic_images_per_split=photographic_by_split,
        photographic_share_per_split={
            split: percentage(photographic_by_split[split], counts.get(split, 0))
            for split in photographic_by_split
        },
        images_per_source_and_split=summarise_counts(f"{r['source']}/{r['split']}" for r in out_rows),
        integrity={
            "reserved_identities_leaking_into_training": leaked,
            "seen_identities_pulled_into_holdout_by_grouping": len(pulled_in),
            "seen_identities_pulled_into_holdout_note": (
                "These identities are not reserved, but some of their images "
                "shared a provenance group with a reserved identity, so those "
                "images were held out with it. They still contribute their other "
                "images to training. This is the intended cost of grouping."
            ),
            "provenance_groups_split_across_parts": len(group_leaks),
            "holdout_inherited_from_dino": inherit,
            "holdout_hash_scheme": config.get("splits.dino_hash_scheme"),
        },
        evaluation_status={
            "held_out_photographic_test_images": sum(
                1 for r in out_rows
                if r["split"] == SPLIT_TEST and str(r["photographic"]).lower() == "true"
            ),
            "held_out_photographic_test_identities": len({
                r["wine_slug"] for r in out_rows
                if r["split"] == SPLIT_TEST and str(r["photographic"]).lower() == "true"
            }),
            "internal_validation_only": [
                "Any figure computed on the DINO synthetic validation split. That "
                "split already selected the current checkpoint, so it is internal "
                "validation, not an independent test."
            ],
            "note": (
                "The test pool is held out from reranker training and from every "
                "selection decision. Two limits keep it from being a general "
                "verdict: it covers only the identities that happen to have been "
                "photographed, and it is the project-wide scanner check set, so "
                "each further use erodes its independence. Fresh sessions "
                "collected per the step 8 plan are the durable answer."
            ),
            "test_sources": sorted(test_sources),
            "test_is_holdout_only": bool(config.get("splits.test_is_holdout_only", True)),
        },
        outputs={"splits_csv": config.relative(splits_csv)},
    )
    write_json(config.report_path("step5_splits.json"), report)
    print(f"step5: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if leaked:
        print(f"  WARNING: {len(leaked)} RESERVED identities reached the training pool")
    if pulled_in:
        print(f"  note: {len(pulled_in)} seen identities had some images held out "
              "because they shared a provenance group with a reserved identity")
    return report
