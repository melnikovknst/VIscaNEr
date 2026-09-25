"""Evaluate a running /api/scan against real_photos_v4 labels.csv or its relabeled export.

The relabeled file (evaluation/real_photos_v4_relabeled.csv) may accept several slugs
for one photo (several bottles, duplicate catalog cards) and excludes unscorable photos.

Only filenames and identities enter the output; source URLs are never copied.
Run from the repository root: python -m scripts.evaluate_api --help.
"""
import argparse
import csv
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx


def is_correct(row: dict) -> bool:
    """Accepted answer is one of the true slugs, or an abstention on an unknown wine."""
    if row["predicted_slug"] is None:
        return not row["in_catalog"]
    return row["predicted_slug"] in row["true_slugs"]


def truth(record: dict) -> tuple[list[str], bool] | None:
    """(accepted slugs, in catalog) for one labels row; None = not scored."""
    if "accepted_slugs" in record:                       # relabeled export
        if record["scored"] != "True":
            return None
        return [s for s in record["accepted_slugs"].split(";") if s], record["in_catalog"] == "True"
    known = record.get("in_catalog", "yes").lower() == "yes"
    return ([record["slug"]] if known else []), known


def split(source_post: str) -> str:
    # Same split as scripts/evaluate_relabeled.py: thresholds are chosen on
    # "selection", so only "report" gives an unbiased number.
    import hashlib
    return "selection" if hashlib.sha256(f"cascade-real:{source_post}".encode()).digest()[0] % 2 == 0 else "report"


def compute_metrics(rows: list[dict]) -> dict:
    successful = [r for r in rows if not r.get("error")]
    # Errors count as misses; never silently remove failed requests from accuracy.
    correct = sum(is_correct(r) and not r.get("error") for r in rows)
    known = [r for r in rows if r["in_catalog"]]
    top5_hits = sum(any(s in r["true_slugs"] for s in r["top5"]) for r in known)
    predicted_count = sum(len(r["top5"]) for r in rows)
    # Top-1 micro-F1 over catalogue identities; abstention on unknown is a true
    # negative, not an extra 'unknown wine' class. Wrong identities are FP + FN.
    hit = lambda r: r["predicted_slug"] is not None and r["predicted_slug"] in r["true_slugs"]
    tp = sum(r["in_catalog"] and hit(r) for r in successful)
    fp = sum(r["predicted_slug"] is not None and not hit(r) for r in rows)
    fn = sum(r["in_catalog"] and (not hit(r) or bool(r.get("error"))) for r in rows)
    latency = sorted(r["elapsed_ms"] for r in rows)
    return {
        "source": "evaluation", "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "samples": len(rows), "errors": len(rows) - len(successful),
        "model_versions": sorted({r["model_version"] for r in successful}),
        "f1_top1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "f1_top5": 2 * top5_hits / (predicted_count + len(known)) if predicted_count + len(known) else 0,
        "accuracy": correct / len(rows) if rows else 0,
        "recall_at_5": top5_hits / len(known) if known else 0,
        "coverage": sum(r["predicted_slug"] is not None for r in successful) / len(rows) if rows else 0,
        "latency_ms_median": statistics.median(latency) if latency else 0,
        "latency_ms_p95": latency[min(len(latency) - 1, int(len(latency) * .95))] if latency else 0,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "definitions": {
            "f1_top1": "Micro-F1 over catalogue identities after threshold abstention; wrong identity = FP and FN.",
            "f1_top5": "Set-retrieval micro-F1 = 2 * known top5 hits / (returned candidates + known queries). Not Recall@5.",
            "accuracy": "Accepted slug equals ground truth, including correct null on out-of-catalogue queries; HTTP errors are misses.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="CSV with image_path, slug, in_catalog (yes/no)")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing query images")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("backend/data/evaluation.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", choices=["all", "selection", "report"], default="all",
                        help="Score one half of real_photos_v4 (needs datasets/real_photos_v4/labels.csv for source posts)")
    args = parser.parse_args()
    labels = [r for r in csv.DictReader(args.labels.open(encoding="utf-8-sig")) if truth(r) is not None]
    if args.split != "all":
        posts = {r["query_id"]: r["source_post"] for r in
                 csv.DictReader(Path("datasets/real_photos_v4/labels.csv").open(encoding="utf-8-sig"))}
        labels = [r for r in labels if split(posts[r["query_id"]]) == args.split]
    if args.limit:
        labels = labels[:args.limit]
    if not labels:
        raise SystemExit("The labels file is empty")
    root = args.images.resolve()
    rows = []
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=120) as client:
        health = client.get("/api/health").raise_for_status().json()
        if health["provider"] == "demo" or not health["model_ready"]:
            raise SystemExit("Connect a real model before evaluation. Demo mode is never scored.")
        for index, record in enumerate(labels):
            image_path = (root / record["image_path"]).resolve()
            if not image_path.is_relative_to(root):
                raise SystemExit("Image path escapes the supplied images directory")
            true_slugs, in_catalog = truth(record)
            started = time.perf_counter()
            row = {"image": record["image_path"], "true_slugs": true_slugs, "in_catalog": in_catalog,
                   "predicted_slug": None,
                   "top5": [], "status": "error", "model_version": None}
            try:
                with image_path.open("rb") as image:
                    result = client.post("/api/scan", files={"file": (image_path.name, image)}).raise_for_status().json()
                row.update(predicted_slug=result["wine"]["slug"] if result["wine"] else None,
                    top5=[c["wine"]["slug"] for c in result["candidates"]],
                    status=result["status"], model_version=result["model_version"])
            except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
                row["error"] = type(exc).__name__
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            rows.append(row)
            print(f"{index + 1}/{len(labels)} {row['image']}: {row['status']} ({row['elapsed_ms']} ms)")
    report = compute_metrics(rows)
    report["labels"] = args.labels.as_posix()
    report["split"] = args.split
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
