"""Evaluate a running /api/scan against real_photos_v4 labels.csv.

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


def compute_metrics(rows: list[dict]) -> dict:
    successful = [r for r in rows if not r.get("error")]
    # Errors count as misses; never silently remove failed requests from accuracy.
    correct = sum(r["predicted_slug"] == r["true_slug"] and not r.get("error") for r in rows)
    known = [r for r in rows if r["true_slug"] is not None]
    top5_hits = sum(r["true_slug"] in r["top5"] for r in known)
    predicted_count = sum(len(r["top5"]) for r in rows)
    # Top-1 micro-F1 over catalogue identities; abstention on unknown is a true
    # negative, not an extra 'unknown wine' class. Wrong identities are FP + FN.
    tp = sum(r["true_slug"] is not None and r["predicted_slug"] == r["true_slug"] for r in successful)
    fp = sum(r["predicted_slug"] is not None and r["predicted_slug"] != r["true_slug"] for r in rows)
    fn = sum(r["true_slug"] is not None and (r["predicted_slug"] != r["true_slug"] or bool(r.get("error"))) for r in rows)
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
    args = parser.parse_args()
    labels = list(csv.DictReader(args.labels.open(encoding="utf-8-sig")))
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
            true_slug = record["slug"] if record.get("in_catalog", "yes").lower() == "yes" else None
            started = time.perf_counter()
            row = {"image": record["image_path"], "true_slug": true_slug, "predicted_slug": None,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
