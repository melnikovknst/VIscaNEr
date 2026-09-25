"""Score both pipelines on the relabeled real_photos_v4, without tuning on the report half.

Truth comes from evaluation/real_photos_v4_relabeled.csv (python -m scripts.relabel_tools
export). A photo counts as correct when the answer is any of its accepted slugs (several
for multi-bottle photos and for duplicate catalog cards of one wine). Photos marked
remove/unsure are excluded; not_in_catalog photos only count against a system that
answers instead of abstaining.

Systems, both read from saved runs so the numbers are reproducible:
  bottle - the colleagues' final pipeline (infer_wine.py: YOLO bottle -> DINOv3-B bottles),
           runs/inference/real_photos_v4_bottle_pipeline.json
  label  - DINOv3-B on the detected label crop (stage 1 of the earlier cascade),
           evaluation/relabel/predictions.json

Abstention thresholds (min similarity, min top1-top2 margin) are chosen on the selection
half - the most precise setting within 0.01 of the best F1 there - and scored once on the
report half.

    python -m scripts.evaluate_relabeled
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELABELED = ROOT / "evaluation/real_photos_v4_relabeled.csv"
ORIGINAL = ROOT / "datasets/real_photos_v4/labels.csv"
BOTTLE_RUN = ROOT / "runs/inference/real_photos_v4_bottle_pipeline.json"
LABEL_RUN = ROOT / "evaluation/relabel/predictions.json"
OUT = ROOT / "evaluation/relabeled_scores.json"


def split(source_post: str) -> str:
    digest = hashlib.sha256(f"cascade-real:{source_post}".encode()).digest()
    return "selection" if digest[0] % 2 == 0 else "report"


def load_systems() -> dict[str, dict[str, list[tuple[str, float]]]]:
    bottle = json.loads(BOTTLE_RUN.read_text(encoding="utf-8"))["results"]
    label = json.loads(LABEL_RUN.read_text(encoding="utf-8"))
    return {
        "bottle": {Path(r["source"]).stem: [(p["wine_slug"], p["similarity"]) for p in r["predictions"]]
                   for r in bottle},
        "label": {q: [(s["slug"], s["score"]) for s in v["top5"]] for q, v in label.items()},
    }


def load_truth(relabeled: bool) -> list[dict]:
    original = {r["query_id"]: r for r in csv.DictReader(ORIGINAL.open(encoding="utf-8-sig"))}
    rows = []
    if relabeled:
        for r in csv.DictReader(RELABELED.open(encoding="utf-8")):
            if r["scored"] != "True":
                continue
            answers = [s for s in r["accepted_slugs"].split(";") if s]
            rows.append({"qid": r["query_id"], "in_catalog": r["in_catalog"] == "True", "answers": answers,
                         "split": split(original[r["query_id"]]["source_post"])})
    else:
        for q, r in original.items():
            yes = r["in_catalog"] == "yes"
            rows.append({"qid": q, "in_catalog": yes, "answers": [r["slug"]] if yes else [],
                         "split": split(r["source_post"])})
    return rows


def ranking_scores(rows, preds) -> dict:
    known = [r for r in rows if r["in_catalog"]]
    hit = lambda r, k: any(s in r["answers"] for s, _ in preds[r["qid"]][:k])
    return {"photos_in_catalog": len(known),
            **{f"top{k}": round(sum(hit(r, k) for r in known) / max(1, len(known)), 4) for k in (1, 2, 3, 5)}}


def abstention_scores(rows, preds, min_sim: float, min_margin: float) -> dict:
    tp = fp = fn = answered = 0
    for r in rows:
        ranked = preds[r["qid"]]
        top, sim = ranked[0]
        margin = sim - ranked[1][1] if len(ranked) > 1 else 1.0
        answer = top if sim >= min_sim and margin >= min_margin else None
        answered += answer is not None
        good = answer is not None and answer in r["answers"]
        tp += good
        fp += answer is not None and not good
        fn += r["in_catalog"] and not good
    return {"f1": round(2 * tp / max(1, 2 * tp + fp + fn), 4), "precision": round(tp / max(1, tp + fp), 4),
            "coverage": round(answered / max(1, len(rows)), 4), "n": len(rows)}


def choose_thresholds(rows, preds) -> tuple[float, float]:
    grid = [(s / 100, m / 1000) for s in range(30, 91) for m in range(0, 81, 2)]
    scored = [(abstention_scores(rows, preds, s, m), s, m) for s, m in grid]
    best = max(x["f1"] for x, _, _ in scored)
    near = [t for t in scored if t[0]["f1"] >= best - 0.01]
    _, s, m = max(near, key=lambda t: (t[0]["precision"], t[0]["f1"]))
    return s, m


def main() -> None:
    systems = load_systems()
    report = {"truth": str(RELABELED.relative_to(ROOT)), "systems": {}}
    for name, preds in systems.items():
        entry = {}
        for relabeled in (False, True):
            rows = [r for r in load_truth(relabeled) if r["qid"] in preds]
            sel = [r for r in rows if r["split"] == "selection"]
            rep = [r for r in rows if r["split"] == "report"]
            s, m = choose_thresholds(sel, preds)
            entry["relabeled" if relabeled else "original_labels"] = {
                "ranking": {"selection": ranking_scores(sel, preds), "report": ranking_scores(rep, preds),
                            "all": ranking_scores(rows, preds)},
                "thresholds_from_selection": {"min_similarity": s, "min_margin": m},
                "with_abstention": {"selection": abstention_scores(sel, preds, s, m),
                                    "report": abstention_scores(rep, preds, s, m)},
            }
        report["systems"][name] = entry
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, entry in report["systems"].items():
        for kind, e in entry.items():
            a, rep = e["ranking"]["all"], e["with_abstention"]["report"]
            print(f"{name:6} {kind:15} n={a['photos_in_catalog']:3} top1={a['top1']:.3f} top2={a['top2']:.3f} "
                  f"top5={a['top5']:.3f} | report F1={rep['f1']:.3f} P={rep['precision']:.3f} "
                  f"cov={rep['coverage']:.3f} @ {e['thresholds_from_selection']}")
    print(OUT)


if __name__ == "__main__":
    main()
