"""Measure the served cascade on real photographs, without tuning on the report set.

The colleagues' metrics come from synthetic renders. A demo is judged on real
photos, so this runs the exact serving code (backend.cascade) over
real_photos_v4 and records every intermediate result: label box, stage-1
top-10 for two input framings, bottle box, and the stage-2 pair scores.

Leakage control: photos are split into two halves by Telegram post (one post is
one shooting session, so near-identical frames never straddle the halves).
  selection  - used to choose the framing, the stage-2 margin and thresholds
  report     - scored once, with the settings chosen on `selection`

    python -m scripts.evaluate_cascade_real run        # inference, cached
    python -m scripts.evaluate_cascade_real analyse    # sweep on selection, score report
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "datasets/real_photos_v4/labels.csv"
IMAGES = ROOT / "datasets/real_photos_v4/queries"
OUT = ROOT / "runs/cascade_real"
RAW = OUT / "raw.jsonl"


def split_of(source_post: str) -> str:
    digest = hashlib.sha256(f"cascade-real:{source_post}".encode()).digest()
    return "selection" if digest[0] % 2 == 0 else "report"


# ------------------------------------------------------------------ inference
def bottle_only(provider, image, label: dict | None) -> dict | None:
    """Diagnostic: the whole-bottle model ranking the full gallery on its own.

    Answers whether it should be a resolver or could lead. Uses exactly the
    served bottle input (backend.cascade._bottle_input).
    """
    from backend.cascade import _bottle_input

    if provider.resolver is None or provider.bottle_detector is None:
        return None
    bottle_input, record = _bottle_input(provider.bottle_detector, image, label)
    if bottle_input is None:
        return {"status": record["status"], "top5": []}
    query = provider._embed(provider.resolver, provider.resolver_transform, bottle_input)
    scores, indices = (query @ provider.bottle_gallery.T).squeeze(0).topk(5)
    return {"status": record["status"], "image_mode": record.get("image_mode"),
            "top5": [[provider.slugs[i], round(float(s), 5)] for s, i in zip(scores.tolist(), indices.tolist())]}


def run(limit: int | None) -> None:
    from PIL import Image, ImageOps

    from backend.cascade import CascadeProvider
    from backend.config import Settings

    # Stage 2 is scored for every photo (margin 1.0) so the sweep can decide
    # when to use it; the served margin is chosen afterwards.
    provider = CascadeProvider(Settings(model_provider="cascade", ambiguity_margin=1.0))
    detector = provider.label_detector
    rows = list(csv.DictReader(LABELS.open(encoding="utf-8-sig")))[: limit or None]
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with RAW.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            with Image.open(IMAGES / row["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)   # as backend.decode_image
            record = {"query_id": row["query_id"], "split": split_of(row["source_post"]),
                      "true_slug": row["slug"] if row["in_catalog"] == "yes" else None}
            for framing in ("detector", "whole_photo"):
                provider.label_detector = detector if framing == "detector" else None
                prediction = provider.predict(image)
                pipeline = prediction.pipeline
                if framing == "detector":
                    record["bottle_only"] = bottle_only(provider, image, pipeline.get("label"))
                # The raw stage-1 ranking, before any stage-2 swap.
                stage1 = [[c.slug, c.similarity] for c in prediction.candidates]
                if pipeline["resolver"].get("swapped"):
                    stage1[0], stage1[1] = stage1[1], stage1[0]
                record[framing] = {
                    "stage1": stage1,
                    "gap": pipeline["stage1_gap"],
                    "label_source": pipeline["label_source"],
                    "label_conf": (pipeline.get("label") or {}).get("confidence"),
                    "resolver": pipeline["resolver"],
                }
            provider.label_detector = detector
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 100 == 0:
                print(f"  {index}/{len(rows)}  {time.perf_counter() - started:.0f}s", flush=True)
    print(f"raw results -> {RAW.relative_to(ROOT)}")


# ------------------------------------------------------------------- analysis
def decide(rec: dict, framing: str, margin: float | None) -> tuple[str | None, str, float, float]:
    """Replay the served decision for one photo: (top1, basis, similarity, margin)."""
    view = rec[framing]
    ranked = [s for s, _ in view["stage1"]]
    similarity = view["stage1"][0][1]
    resolver = view["resolver"]
    if margin is not None and resolver.get("invoked") and view["gap"] <= margin:
        first, second = resolver["first"], resolver["second"]
        top = ranked[1] if second > first else ranked[0]
        return top, "resolver", similarity, abs(first - second)
    return ranked[0], "label", similarity, view["gap"]


def score(records: list[dict], framing: str, margin: float | None,
          min_sim: float, min_margin: float, min_resolver: float) -> dict:
    tp = fp = fn = correct = matched = 0
    for rec in records:
        top, basis, similarity, decision_margin = decide(rec, framing, margin)
        needed = min_resolver if basis == "resolver" else min_margin
        accepted = top if similarity >= min_sim and decision_margin >= needed else None
        truth = rec["true_slug"]
        matched += accepted is not None
        correct += accepted == truth
        tp += truth is not None and accepted == truth
        fp += accepted is not None and accepted != truth
        fn += truth is not None and accepted != truth
    n = len(records)
    return {"f1_top1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp),
            "coverage": matched / max(1, n), "accuracy": correct / max(1, n), "n": n}


def top1(records: list[dict], framing: str, margin: float | None) -> float:
    known = [r for r in records if r["true_slug"]]
    return sum(decide(r, framing, margin)[0] == r["true_slug"] for r in known) / max(1, len(known))


def topk(records: list[dict], framing: str, k: int) -> float:
    known = [r for r in records if r["true_slug"]]
    return sum(r["true_slug"] in [s for s, _ in r[framing]["stage1"][:k]] for r in known) / max(1, len(known))


def bottle_only_scores(records: list[dict]) -> dict:
    known = [r for r in records if r["true_slug"] and r.get("bottle_only") is not None]
    with_input = [r for r in known if r["bottle_only"]["top5"]]
    hit = lambda r, k: r["true_slug"] in [s for s, _ in r["bottle_only"]["top5"][:k]]
    n = max(1, len(known))
    return {"photos": len(known), "bottle_input_available": round(len(with_input) / n, 4),
            "top1_over_all_photos": round(sum(hit(r, 1) for r in with_input) / n, 4),
            "top5_over_all_photos": round(sum(hit(r, 5) for r in with_input) / n, 4),
            "top1_when_available": round(sum(hit(r, 1) for r in with_input) / max(1, len(with_input)), 4)}


def analyse() -> dict:
    records = [json.loads(line) for line in RAW.open(encoding="utf-8")]
    selection = [r for r in records if r["split"] == "selection"]
    report = [r for r in records if r["split"] == "report"]
    margins = [None, 0.005, 0.01, 0.01525, 0.02, 0.03, 0.05, 0.1, 1.0]

    # 1. Framing and stage-2 margin: raw top-1, no abstention, selection only.
    raw_grid = {f: {str(m): round(top1(selection, f, m), 4) for m in margins} for f in ("detector", "whole_photo")}
    framing, margin = max(((f, m) for f in raw_grid for m in margins), key=lambda fm: top1(selection, *fm))

    # 2. Thresholds. Rule, fixed in advance: among settings within 0.01 of the
    #    best selection F1, take the most precise one - F1 is flat near its
    #    optimum, and a trustworthy "found" matters more to the product.
    grid = []
    for min_sim in [x / 100 for x in range(30, 91, 2)]:
        for min_margin in [x / 1000 for x in range(0, 81, 4)]:
            for min_resolver in [x / 1000 for x in range(0, 61, 5)]:
                grid.append((score(selection, framing, margin, min_sim, min_margin, min_resolver),
                             min_sim, min_margin, min_resolver))
    best_f1 = max(g[0]["f1_top1"] for g in grid)
    chosen, min_sim, min_margin, min_resolver = max(
        (g for g in grid if g[0]["f1_top1"] >= best_f1 - 0.01),
        key=lambda g: (g[0]["precision"], g[0]["f1_top1"]))

    # 3. Report: scored once, with everything fixed above.
    known_report = [r for r in report if r["true_slug"]]
    resolver_stats = {"invoked": 0, "swapped": 0, "fixed": 0, "broke": 0}
    for r in known_report:
        view = r[framing]
        if margin is not None and view["resolver"].get("invoked") and view["gap"] <= margin:
            resolver_stats["invoked"] += 1
            before = view["stage1"][0][0]
            after = decide(r, framing, margin)[0]
            if after != before:
                resolver_stats["swapped"] += 1
                resolver_stats["fixed"] += after == r["true_slug"]
                resolver_stats["broke"] += before == r["true_slug"]
    result = {
        "photos": {"selection": len(selection), "report": len(report)},
        "selection_top1_grid": raw_grid,
        "chosen": {"framing": framing, "ambiguity_margin": margin, "min_similarity": min_sim,
                   "min_margin": min_margin, "min_resolver_margin": min_resolver},
        "selection_scores": {k: round(v, 4) if isinstance(v, float) else v for k, v in chosen.items()},
        "selection_best_f1": round(best_f1, 4),
        "report": {
            "stage1_only_top1": round(top1(report, framing, None), 4),
            "cascade_top1": round(top1(report, framing, margin), 4),
            "stage1_top2": round(topk(report, framing, 2), 4),
            "stage1_top5": round(topk(report, framing, 5), 4),
            "with_abstention": {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in score(report, framing, margin, min_sim, min_margin, min_resolver).items()},
            "resolver_on_report": resolver_stats,
            "label_found_percent": round(100 * sum(r["detector"]["label_source"] == "detector" for r in report) / len(report), 1),
        },
        "diagnostic_bottle_model_alone": {"selection": bottle_only_scores(selection), "report": bottle_only_scores(report)},
    }
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "analyse"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.limit) if args.command == "run" else analyse()


if __name__ == "__main__":
    main()
