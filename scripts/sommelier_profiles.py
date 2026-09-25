"""Extract a structured taste profile for every catalog wine with a local LLM.

The profile is what the sommelier searches and filters on: sweetness, body,
acidity, tannins, aromas, food pairings and serving temperature. It is read from
the wine's own catalog description, name and grapes - the model is told not to
add anything the text does not support - and every answer is validated against
fixed vocabularies. Sweetness and sparkle are also checked against the wine's
name, which states them for most wines.

    python -m scripts.sommelier_profiles --limit 100            # stratified sample
    python -m scripts.sommelier_profiles                        # the whole catalog
    python -m scripts.sommelier_profiles --report runs/sommelier/profiles.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "datasets/wine-scanner/data/catalog.csv"
MODEL = ROOT / "models/llm/yandexgpt5-lite-8b-instruct"
OUT = ROOT / "runs/sommelier/profiles.jsonl"

SWEETNESS = ["сухое", "полусухое", "полусладкое", "сладкое", "брют", "экстра брют", "брют натюр", "неизвестно"]
BODY = ["лёгкое", "среднее", "полнотелое", "неизвестно"]
ACIDITY = ["низкая", "средняя", "высокая", "неизвестно"]
TANNINS = ["нет", "мягкие", "средние", "выраженные", "неизвестно"]

SYSTEM = (
    "Ты сомелье. По карточке вина из каталога заполни его профиль. "
    "Ароматы, тело, кислотность и танины бери только из описания; если там этого нет, "
    "пиши «неизвестно» или оставляй список пустым. Сочетания с едой и температуру подачи "
    "подбери сам по стилю, цвету, сорту и сладости вина, как это сделал бы сомелье. "
    "Не упоминай цены и оценки. Отвечай одним JSON-объектом без пояснений."
)

TEMPLATE = """Вино: {name}
Винодельня: {winery}
Цвет (категория): {category}
Оттенок: {color}
Регион: {region}
Сорт: {grapes}
Сладость: {sweet_hint}
Игристое: {sparkling_hint}
Описание: {description}

Верни JSON с полями:
"sweetness": одно из {sweetness}
"sparkling": true или false
"body": одно из {body}
"acidity": одно из {acidity}
"tannins": одно из {tannins} (для белых и розовых обычно «нет»)
"aromas": до 6 коротких ароматов и вкусов из описания, например ["вишня", "ваниль"]
"pairing": 3–5 конкретных блюд или продуктов, к которым подойдёт это вино
"serve_temp": температура подачи, например "10–12 °C"
"summary": одно предложение о стиле вина"""


def load_catalog() -> list[dict]:
    return list(csv.DictReader(CATALOG.open(encoding="utf-8")))


def sample(rows: list[dict], limit: int, seed: int = 7) -> list[dict]:
    """Stratified by colour so a small sample still covers every style."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)
    picked: list[dict] = []
    for name, group in sorted(groups.items()):
        rng.shuffle(group)
        picked += group[: max(1, round(limit * len(group) / len(rows)))]
    rng.shuffle(picked)
    return picked[:limit]


def name_hints(row: dict) -> dict:
    """Sweetness and sparkle as stated in the name/slug, when they are."""
    text = f"{row['name']} {row['slug']}".casefold().replace("ё", "е")
    sweet = None
    for key, words in [("брют натюр", ["брют натюр", "brut nature", "zero dosage", "зеро дозаж"]),
                       ("экстра брют", ["экстра брют", "ekstra-bryut", "extra brut"]),
                       ("брют", ["брют", "bryut", "brut"]),
                       ("полусладкое", ["полусладк", "polusladk"]),
                       ("полусухое", ["полусух", "polusuh"]),
                       ("сладкое", ["сладк", "sladk"]),
                       ("сухое", ["сухое", "suhoe"])]:
        if any(w in text for w in words):
            sweet = key
            break
    sparkling = any(w in text for w in ["брют", "bryut", "brut", "игрист", "igrist", "петнат", "пет-нат",
                                        "pet-nat", "petnat", "шампанск", "spumante", "frizzante"])
    return {"sweetness": sweet, "sparkling": sparkling or None}


def parse(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate(profile: dict) -> list[str]:
    problems = []
    for key, allowed in [("sweetness", SWEETNESS), ("body", BODY), ("acidity", ACIDITY), ("tannins", TANNINS)]:
        if profile.get(key) not in allowed:
            problems.append(f"{key}={profile.get(key)!r}")
    if not isinstance(profile.get("sparkling"), bool):
        problems.append("sparkling")
    for key, most in [("aromas", 6), ("pairing", 5)]:
        value = profile.get(key)
        if not isinstance(value, list) or len(value) > most or not all(isinstance(v, str) for v in value):
            problems.append(key)
    for key in ("serve_temp", "summary"):
        if not isinstance(profile.get(key), str):
            problems.append(key)
    return problems


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True),
    )
    model.eval()
    return tokenizer, model


def prompt(tokenizer, row: dict) -> str:
    hints = name_hints(row)
    user = TEMPLATE.format(**{k: row.get(k) or "—" for k in ("name", "winery", "category", "color", "region",
                                                          "grapes", "description")},
                           sweetness=SWEETNESS, body=BODY, acidity=ACIDITY, tannins=TANNINS,
                           sweet_hint=hints["sweetness"] or "не указана",
                           sparkling_hint="да" if hints["sparkling"] else "не указано")
    # YandexGPT-5 Lite's chat template has no system role and silently drops
    # it, so the instruction opens the user turn instead.
    messages = [{"role": "user", "content": f"{SYSTEM}\n\n{user}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(tokenizer, model, rows: list[dict], batch: int) -> list[tuple[dict, str, float]]:
    import torch

    results = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        texts = [prompt(tokenizer, row) for row in chunk]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        began = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=320, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        seconds = time.perf_counter() - began
        for row, sequence in zip(chunk, output):
            answer = tokenizer.decode(sequence[encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append((row, answer, seconds / len(chunk)))
        print(f"{min(start + batch, len(rows))}/{len(rows)}  {seconds / len(chunk):.1f} s/wine", flush=True)
    return results


def report(path: Path) -> None:
    items = [json.loads(line) for line in path.open(encoding="utf-8")]
    ok = [i for i in items if not i["problems"]]
    print(f"{len(items)} wines, valid profiles {len(ok)} ({len(ok) / len(items):.0%})")
    print("problems:", Counter(p.split("=")[0] for i in items for p in i["problems"]).most_common())
    sweet = [i for i in ok if i["hints"]["sweetness"]]
    agree = sum(i["profile"]["sweetness"] == i["hints"]["sweetness"] for i in sweet)
    print(f"sweetness vs name: {agree}/{len(sweet)} agree")
    spark = [i for i in ok if i["hints"]["sparkling"]]
    print(f"sparkling vs name: {sum(i['profile']['sparkling'] for i in spark)}/{len(spark)} agree")
    for key in ("sweetness", "body", "acidity", "tannins"):
        print(f"  {key}: {dict(Counter(i['profile'][key] for i in ok))}")
    empty = sum(not i["profile"]["pairing"] or any("неизв" in x or "не указ" in x for x in i["profile"]["pairing"]) for i in ok)
    print(f"pairing missing: {empty}/{len(ok)}; serve_temp missing: "
          f"{sum('неизв' in i['profile']['serve_temp'] for i in ok)}/{len(ok)}")
    print(f"median time {sorted(i['seconds'] for i in items)[len(items) // 2]:.1f} s/wine")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="stratified sample size; 0 = whole catalog")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, help="only summarise an existing profiles file")
    args = parser.parse_args()
    if args.report:
        report(args.report)
        return
    rows = load_catalog()
    rows = sample(rows, args.limit) if args.limit else rows
    tokenizer, model = load_model()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row, answer, seconds in generate(tokenizer, model, rows, args.batch):
            profile = parse(answer) or {}
            # What the name states wins over the model's reading.
            hints = name_hints(row)
            if profile and hints["sweetness"]:
                profile["sweetness"] = hints["sweetness"]
            if profile and hints["sparkling"]:
                profile["sparkling"] = True
            if isinstance(profile.get("aromas"), list):
                profile["aromas"] = profile["aromas"][:6]
            handle.write(json.dumps({"slug": row["slug"], "profile": profile, "problems": validate(profile),
                                     "hints": name_hints(row), "raw": answer, "seconds": round(seconds, 2)},
                                    ensure_ascii=False) + "\n")
    report(args.out)


if __name__ == "__main__":
    main()
