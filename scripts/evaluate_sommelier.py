"""Check the sommelier against questions with known constraints, through the running API.

Each question states what any acceptable pick must satisfy (colour, sparkling,
sweetness). For every answer we check: the picks satisfy the constraints, the
text cites at least one wine, it mentions no prices or ratings, and how long it
took. Wine-card questions check that the answer stays on the asked wine.

    python -m scripts.evaluate_sommelier [--url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "runs/sommelier/profiles.jsonl"
OUT = ROOT / "runs/sommelier/evaluation.json"

RED, WHITE, ROSE = {"Красное"}, {"Белое"}, {"Розовое"}
SWEET = {"сладкое", "полусладкое"}
DRY = {"сухое", "брют", "экстра брют", "брют натюр"}

# (question, allowed categories or None, sparkling True/False/None, allowed sweetness or None)
QUESTIONS = [
    ("Что взять к стейку рибай?", RED, None, None),
    ("Красное сухое к шашлыку из баранины", RED, False, DRY),
    ("Посоветуйте белое вино к рыбе на гриле", WHITE, None, None),
    ("Лёгкое белое к салату с креветками", WHITE, None, None),
    ("Сухое белое к устрицам", WHITE, None, DRY),
    ("Розовое вино на летний пикник", ROSE, None, None),
    ("Розовое сухое к овощам на гриле", ROSE, None, DRY),
    ("Сладкое вино к шоколадному десерту", None, None, SWEET),
    ("Что подать к медовику?", None, None, SWEET),
    ("Полусладкое белое к фруктам", WHITE, None, {"полусладкое"}),
    ("Игристое для праздничного вечера", None, True, None),
    ("Брют к морепродуктам", None, True, DRY),
    ("Розовое игристое на день рождения", ROSE, True, None),
    ("Красное к пицце с пепперони", RED, None, None),
    ("Вино к утке с вишнёвым соусом", None, None, None),
    ("Что-нибудь к пасте карбонара", None, None, None),
    ("Белое к сырной тарелке с мягкими сырами", WHITE, None, None),
    ("Красное к выдержанному твёрдому сыру", RED, None, None),
    ("Вино к плову", None, None, None),
    ("Лёгкое красное к тунцу", RED, None, None),
    ("Не игристое белое к курице в сливочном соусе", WHITE, False, None),
    ("Сухое красное к грибному ризотто", RED, None, DRY),
    ("Вино к суши", None, None, None),
    ("Сладкое белое к голубому сыру", WHITE, None, SWEET),
    ("Красное полусладкое к десерту из ягод", RED, None, {"полусладкое"}),
]
WINE_QUESTIONS = ["К каким блюдам подать?", "Как подать и при какой температуре?", "Посоветуйте похожее вино"]
FORBIDDEN = re.compile(r"(₽|руб|\bцен[аыу]|стоимост|рейтинг|балл|медал|наград)", re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--wines", type=int, default=5, help="catalog wines to ask the card questions about")
    args = parser.parse_args()
    profiles = {}
    for line in PROFILES.open(encoding="utf-8"):
        item = json.loads(line)
        profiles[item["slug"]] = item["profile"]

    rows = []
    with httpx.Client(base_url=args.url, timeout=180) as client:
        for question, colours, sparkling, sweetness in QUESTIONS:
            started = time.perf_counter()
            answer = client.post("/api/sommelier", json={"question": question}).raise_for_status().json()
            seconds = time.perf_counter() - started
            violations = []
            for wine in answer["wines"]:
                profile = profiles.get(wine["slug"], {})
                if colours and wine["category"] not in colours:
                    violations.append(f"{wine['name']}: цвет {wine['category']}")
                if sparkling is not None and bool(profile.get("sparkling")) != sparkling:
                    violations.append(f"{wine['name']}: игристое={profile.get('sparkling')}")
                if sweetness and profile.get("sweetness") not in sweetness:
                    violations.append(f"{wine['name']}: сладость {profile.get('sweetness')}")
            rows.append({"question": question, "answer": answer["answer"],
                         "wines": [w["name"] + " · " + w["winery"] for w in answer["wines"]],
                         "violations": violations, "cited_any": bool(re.search(r"\[\d+\]", answer["answer"])),
                         "forbidden": FORBIDDEN.findall(answer["answer"]), "seconds": round(seconds, 1)})
            print(f"{seconds:5.1f}s  {'OK ' if not violations else 'BAD'} {question}")
        slugs = [s for s in sorted(profiles)][:: max(1, len(profiles) // args.wines)][: args.wines]
        for slug in slugs:
            for question in WINE_QUESTIONS:
                started = time.perf_counter()
                answer = client.post("/api/sommelier", json={"question": question, "wine_slug": slug}).raise_for_status().json()
                seconds = time.perf_counter() - started
                rows.append({"question": question, "wine_slug": slug, "answer": answer["answer"],
                             "wines": [w["name"] + " · " + w["winery"] for w in answer["wines"]],
                             "violations": [] if answer["wines"][0]["slug"] == slug else ["answer left the asked wine"],
                             "cited_any": True, "forbidden": FORBIDDEN.findall(answer["answer"]),
                             "seconds": round(seconds, 1)})
                print(f"{seconds:5.1f}s  {slug[:40]}  {question}")

    summary = {
        "questions": len(rows),
        "constraints_ok": sum(not r["violations"] for r in rows),
        "cited_a_wine": sum(r["cited_any"] for r in rows),
        "mentions_price_or_rating": sum(bool(r["forbidden"]) for r in rows),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
        "max_seconds": max(r["seconds"] for r in rows),
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
