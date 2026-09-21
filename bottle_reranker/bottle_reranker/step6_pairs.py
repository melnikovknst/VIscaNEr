"""Step 6b - turn DINO candidate lists into training pairs and triplets.

Each example is ``(query bottle view, reference A, reference B, label)``. The
rules that keep the labels honest:

* **Order carries no information.** A and B are swapped by a seeded coin flip,
  so "first" never means "correct".
* **Hard negatives are real rivals.** The main negatives are the identities the
  current DINO actually ranked against the truth, not random wines. Moderate
  (same winery or same near-duplicate series) and easy (random) negatives are
  mixed in at a configurable ratio - a starting point for an ablation, not a
  claim that the ratio is optimal.
* **Correct top-1 cases are included on purpose.** A reranker trained only on
  DINO failures learns to flip answers. It also has to learn to leave a correct
  answer alone, so confident-correct examples are a required share of the mix.
* **``neither_correct`` is a label, not a problem to hide.** When the truth is
  in neither slot, the example says so. One of the two is never promoted.
* **Injected triplets and organic pairs stay separate.** A triplet built by
  adding the true reference next to a rival is a training convenience; an
  organic (top-1, top-2) pair is what the model meets at inference. Mixing them
  in one pool would make the evaluation optimistic.
* **The identity holdout is respected on both sides.** A pair is dropped, not
  relabelled, when either candidate is a reserved identity.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
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
from .config import Config
from .step6_candidates import (
    CASE_CONFIDENT,
    CASE_CORRECT_CLOSE,
    CASE_NOT_IN_TOP2,
    CASE_WRONG_TOP1,
)

CONFIG_FINGERPRINT_KEYS = (
    "pairs.negatives_per_positive",
    "pairs.shuffle_candidate_order",
    "pairs.shuffle_seed",
    "pairs.min_confident_correct_fraction",
    "pairs.emit_neither_correct",
    "pairs.separate_injected_and_organic",
    "splits.drop_pairs_touching_unseen",
)

LABEL_A = "a_correct"
LABEL_B = "b_correct"
LABEL_NEITHER = "neither_correct"

KIND_ORGANIC = "organic_top1_top2"
KIND_INJECTED = "injected_truth_vs_rival"

NEGATIVE_DINO = "dino_competitor"
NEGATIVE_MODERATE = "moderate"
NEGATIVE_EASY = "easy"


def _candidate_slugs(row: dict[str, str], top_k: int) -> list[str]:
    out: list[str] = []
    for rank in range(1, top_k + 1):
        slug = (row.get(f"cand{rank}_slug") or "").strip()
        if slug:
            out.append(slug)
    return out


def _emit(
    rows: list[dict[str, Any]],
    *,
    query: dict[str, str],
    correct: str,
    other: str,
    kind: str,
    negative_kind: str,
    rng: random.Random,
    shuffle: bool,
    truth_present: bool,
    extra: dict[str, Any],
) -> None:
    first, second = (correct, other)
    if shuffle and rng.random() < 0.5:
        first, second = second, first
    if not truth_present:
        label = LABEL_NEITHER
    else:
        label = LABEL_A if first == correct else LABEL_B
    rows.append({
        "source": query["source"],
        "image_relative_path": query["image_relative_path"],
        "query_path": query.get("query_path", ""),
        "query_view": query.get("query_view", ""),
        "wine_slug": query["wine_slug"],
        "split": query["split"],
        "dino_exposure": query.get("dino_exposure", ""),
        "dino_case": query.get("case", ""),
        "candidate_a": first,
        "candidate_b": second,
        "label": label,
        "pair_kind": kind,
        "negative_kind": negative_kind,
        **extra,
    })


def run(config: Config) -> dict[str, Any]:
    candidates_csv = config.output_dir("audit", create=False) / "dino_candidates.csv"
    if not candidates_csv.is_file():
        raise FileNotFoundError(f"{candidates_csv} not found. Run step 6a first.")

    catalog_rows = read_csv_rows(config.path("catalog.catalog_csv"))
    winery_by_slug = {r["slug"]: r.get("winery", "") for r in catalog_rows}
    series_by_slug = {r["slug"]: r.get("near_dup_group", "") for r in catalog_rows}
    identical_image_group = {r["slug"]: r.get("same_image_group", "") for r in catalog_rows}
    by_winery: dict[str, list[str]] = defaultdict(list)
    by_series: dict[str, list[str]] = defaultdict(list)
    for slug, winery in winery_by_slug.items():
        by_winery[winery].append(slug)
    for slug, series in series_by_slug.items():
        if series:
            by_series[series].append(slug)
    all_slugs = sorted(winery_by_slug)

    seed = int(config.get("splits.seed"))
    fraction = float(config.get("splits.unseen_identity_fraction"))
    unseen = {s for s in all_slugs if stable_unit_interval(s, seed) < fraction}
    drop_unseen = bool(config.get("splits.drop_pairs_touching_unseen", True))

    ratios = config.get("pairs.negatives_per_positive")
    shuffle = bool(config.get("pairs.shuffle_candidate_order", True))
    emit_neither = bool(config.get("pairs.emit_neither_correct", True))
    top_k = int(config.get("dino.top_k"))
    rng = random.Random(int(config.get("pairs.shuffle_seed")))

    queries = read_csv_rows(candidates_csv)
    organic: list[dict[str, Any]] = []
    injected: list[dict[str, Any]] = []
    dropped = Counter()

    for query in queries:
        truth = query["wine_slug"]
        if query["split"] not in {"train", "dev"}:
            dropped["query_not_in_train_or_dev"] += 1
            continue
        if drop_unseen and truth in unseen:
            dropped["query_identity_is_reserved"] += 1
            continue

        ranked = _candidate_slugs(query, top_k)
        if len(ranked) < 2:
            dropped["fewer_than_two_candidates"] += 1
            continue

        scores = {
            slug: float(query.get(f"cand{rank}_cosine_similarity") or 0.0)
            for rank, slug in enumerate(ranked, start=1)
        }
        shared_photo = identical_image_group.get(truth, "")

        # ---- organic pair: exactly what the model meets at inference --------
        top1, top2 = ranked[0], ranked[1]
        if drop_unseen and ({top1, top2} & unseen):
            dropped["organic_pair_touches_reserved_identity"] += 1
        else:
            truth_present = truth in {top1, top2}
            if truth_present or emit_neither:
                _emit(
                    organic,
                    query=query, correct=truth if truth_present else top1,
                    other=top2 if truth_present and truth == top1 else top1 if truth_present else top2,
                    kind=KIND_ORGANIC, negative_kind=NEGATIVE_DINO,
                    rng=rng, shuffle=shuffle, truth_present=truth_present,
                    extra={
                        "cosine_a": scores.get(top1), "cosine_b": scores.get(top2),
                        "dino_margin": round(scores.get(top1, 0.0) - scores.get(top2, 0.0), 6),
                        "candidates_share_one_catalog_photo": bool(
                            shared_photo and identical_image_group.get(top2 if truth == top1 else top1) == shared_photo
                        ),
                    },
                )
            else:
                dropped["truth_not_in_top2_and_neither_correct_disabled"] += 1

        # ---- injected triplets: truth against progressively easier rivals ---
        rivals = [s for s in ranked if s != truth]
        pool: list[tuple[str, str]] = [(s, NEGATIVE_DINO) for s in rivals[: int(ratios[NEGATIVE_DINO])]]

        moderate_pool = [
            s for s in set(by_winery.get(winery_by_slug.get(truth, ""), []) + by_series.get(series_by_slug.get(truth, ""), []))
            if s != truth and s not in rivals
        ]
        rng.shuffle(moderate_pool)
        pool += [(s, NEGATIVE_MODERATE) for s in moderate_pool[: int(ratios[NEGATIVE_MODERATE])]]

        easy_wanted = int(ratios[NEGATIVE_EASY])
        guard = 0
        while easy_wanted > 0 and guard < 50:
            guard += 1
            pick = all_slugs[rng.randrange(len(all_slugs))]
            if pick == truth or pick in rivals or pick in moderate_pool:
                continue
            pool.append((pick, NEGATIVE_EASY))
            easy_wanted -= 1

        for rival, negative_kind in pool:
            if drop_unseen and rival in unseen:
                dropped[f"injected_{negative_kind}_touches_reserved_identity"] += 1
                continue
            _emit(
                injected,
                query=query, correct=truth, other=rival,
                kind=KIND_INJECTED, negative_kind=negative_kind,
                rng=rng, shuffle=shuffle, truth_present=True,
                extra={
                    "cosine_a": scores.get(truth), "cosine_b": scores.get(rival),
                    "dino_margin": "",
                    "candidates_share_one_catalog_photo": bool(
                        shared_photo and identical_image_group.get(rival) == shared_photo
                    ),
                },
            )

    organic_csv = config.output_dir("pairs") / "organic_pairs.csv"
    injected_csv = config.output_dir("pairs") / "injected_triplets.csv"
    write_csv_rows(organic_csv, organic)
    write_csv_rows(injected_csv, injected)

    confident = sum(1 for r in organic if r["dino_case"] == CASE_CONFIDENT)
    min_fraction = float(config.get("pairs.min_confident_correct_fraction"))
    confident_share = confident / len(organic) if organic else 0.0

    warnings: list[str] = []
    if organic and confident_share < min_fraction:
        warnings.append(
            f"Confident-correct examples are {confident_share:.1%} of the organic "
            f"pairs, below the configured floor of {min_fraction:.0%}. A reranker "
            "trained on this mix is biased towards overturning correct answers."
        )
    impossible = sum(1 for r in organic + injected if str(r.get("candidates_share_one_catalog_photo")).lower() == "true")
    if impossible:
        warnings.append(
            f"{impossible} pairs put two identities that share one catalog "
            "photograph against each other. No image encoder can separate those; "
            "they are kept and marked so they can be excluded from any metric."
        )

    report = stage_record(
        "step6_pairs",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        organic_pairs=len(organic),
        injected_triplets=len(injected),
        organic_label_counts=summarise_counts(r["label"] for r in organic),
        injected_label_counts=summarise_counts(r["label"] for r in injected),
        organic_case_counts=summarise_counts(r["dino_case"] for r in organic),
        negative_kind_counts=summarise_counts(r["negative_kind"] for r in organic + injected),
        first_slot_is_correct_percent=percentage(
            sum(1 for r in organic + injected if r["label"] == "a_correct"),
            max(1, len(organic) + len(injected)),
        ),
        confident_correct_share_percent=round(100 * confident_share, 2),
        unseparable_pairs=impossible,
        dropped=dict(dropped),
        warnings=warnings,
        split_counts=summarise_counts(r["split"] for r in organic + injected),
        outputs={
            "organic_pairs_csv": config.relative(organic_csv),
            "injected_triplets_csv": config.relative(injected_csv),
        },
        note=(
            "organic_pairs.csv is the distribution the reranker sees in "
            "production. injected_triplets.csv is a training aid. Reporting a "
            "single metric over both would flatter the model."
        ),
    )
    write_json(config.report_path("step6_pairs.json"), report)
    print(f"step6b: {len(organic)} organic pairs, {len(injected)} injected triplets")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    return report
