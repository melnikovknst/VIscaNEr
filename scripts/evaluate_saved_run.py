"""Score an infer_wine JSON on a relabeled CSV at fixed site thresholds.

No thresholds are fitted on this dataset. Unknown wines offered as choices are
reported separately from both confident false matches and explicit not_found.

python -m scripts.evaluate_saved_run --truth evaluation/store_shelves_v1_labels.csv \
    --bottle-run runs/inference/store_shelves_v1_bottle_pipeline.json \
    --out evaluation/store_shelves_v1_scores.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from backend.config import Settings
from scripts.evaluate_relabeled import ranking_scores


def served_status(ranked: list[tuple[str, float]], settings: Settings) -> str:
    """Local infer_wine decision, matching backend.main.resolve_prediction."""
    if not ranked:
        return "not_found"
    top_sim = ranked[0][1]
    margin = top_sim - ranked[1][1] if len(ranked) > 1 else None
    if top_sim >= settings.min_similarity:
        return "matched" if margin is not None and margin >= settings.min_margin else "uncertain"
    if settings.min_suggest_similarity is not None and top_sim >= settings.min_suggest_similarity:
        return "uncertain"
    return "not_found"


def predictions_by_id(run: dict) -> dict[str, list[tuple[str, float]]]:
    predictions = {}
    for result in run["results"]:
        qid = Path(result["source"].replace("\\", "/")).stem
        if qid in predictions:
            raise ValueError(f"Duplicate prediction: {qid}")
        ranked = [(p["wine_slug"], float(p["similarity"])) for p in result["predictions"]]
        if any(not math.isfinite(score) for _, score in ranked):
            raise ValueError(f"Non-finite similarity: {qid}")
        if len({slug for slug, _ in ranked}) != len(ranked):
            raise ValueError(f"Duplicate candidate: {qid}")
        if any(a[1] < b[1] for a, b in zip(ranked, ranked[1:])):
            raise ValueError(f"Unsorted predictions: {qid}")
        predictions[qid] = ranked
    return predictions


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def evaluate(truth: list[dict], preds: dict, settings: Settings, exclude=()) -> dict:
    if len({r["query_id"] for r in truth}) != len(truth):
        raise ValueError("Duplicate query_id in truth")
    scored = [r for r in truth if r["scored"] == "True" and r["query_id"] not in exclude]
    if not scored:
        raise ValueError("No scored rows")
    missing = [r["query_id"] for r in scored if r["query_id"] not in preds]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} scored rows: {missing}")
    rows, details = [], []
    for row in scored:
        qid = row["query_id"]
        if row["in_catalog"] not in {"True", "False"}:
            raise ValueError(f"Invalid in_catalog: {qid}")
        known = row["in_catalog"] == "True"
        answers = [s for s in row["accepted_slugs"].split(";") if s]
        if not known and answers:
            raise ValueError(f"Out-of-catalog row has accepted answers: {qid}")
        ranked = preds[qid]
        served = served_status(ranked, settings)
        top = ranked[0][0] if ranked else None
        correct = known and top in answers
        rows.append({"qid": qid, "in_catalog": known, "answers": answers})
        details.append({
            "query_id": qid, "source": row.get("source"), "in_catalog": known,
            "accepted_slugs": answers, "served": served, "top1": top,
            "similarity": ranked[0][1] if ranked else None,
            "margin": ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None,
            "top1_correct": correct,
            "confident_wrong": served == "matched" and not correct,
        })

    known = [r for r in details if r["in_catalog"]]
    unknown = [r for r in details if not r["in_catalog"]]
    matched = [r for r in details if r["served"] == "matched"]
    tp = sum(r["top1_correct"] for r in matched)
    fp = len(matched) - tp
    fn = len(known) - tp
    unknown_counts = Counter(r["served"] for r in unknown)
    ranking = ranking_scores(rows, preds)
    ranking["hits"] = {f"top{k}": sum(any(s in r["answers"] for s, _ in preds[r["qid"]][:k])
                                      for r in rows if r["in_catalog"]) for k in (1, 2, 3, 5)}
    return {
        "samples": len(details), "excluded": len(truth) - len(details),
        "in_catalog": len(known), "out_of_catalog": len(unknown),
        "source_photos": len({r["source"] for r in details if r["source"]}),
        "ranking": ranking,
        "served": {
            "status_counts": {s: sum(r["served"] == s for r in details)
                              for s in ("matched", "uncertain", "not_found")},
            "correct_cards": tp, "wrong_cards": fp,
            "wrong_cards_in_catalog": sum(r["confident_wrong"] for r in known),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, len(known)),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "coverage": ratio(len(matched), len(details)),
        },
        "out_of_catalog_behavior": {
            "n": len(unknown),
            **{s: {"count": unknown_counts[s], "rate": ratio(unknown_counts[s], len(unknown))}
               for s in ("matched", "uncertain", "not_found")},
            "false_match_query_ids": [r["query_id"] for r in unknown if r["confident_wrong"]],
        },
        "records": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--bottle-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-query", action="append", default=[])
    args = parser.parse_args()
    settings = Settings()
    with args.truth.open(encoding="utf-8-sig", newline="") as handle:
        truth = list(csv.DictReader(handle))
    unknown_exclusions = set(args.exclude_query) - {r["query_id"] for r in truth}
    if unknown_exclusions:
        parser.error(f"Unknown excluded query IDs: {sorted(unknown_exclusions)}")
    run = json.loads(args.bottle_run.read_text(encoding="utf-8"))
    report = {
        "truth": args.truth.as_posix(), "bottle_run": args.bottle_run.as_posix(),
        "truth_sha256": hashlib.sha256(args.truth.read_bytes()).hexdigest(),
        "run_sha256": hashlib.sha256(args.bottle_run.read_bytes()).hexdigest(),
        "pipeline": run.get("pipeline"), "threshold_policy": "fixed_site_settings_no_fitting",
        "thresholds": settings.model_dump(include={"min_similarity", "min_margin", "min_suggest_similarity"}),
        "additional_exclusions": args.exclude_query,
        **evaluate(truth, predictions_by_id(run), settings, args.exclude_query),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=True, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
