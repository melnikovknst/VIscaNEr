"""Show, photo by photo, what the served model saw, what it answered and the truth.

Runs the exact served provider (backend.cascade, current .env settings) over
the held-out half of real_photos_v4 - the half every reported number comes
from - and captures the model inputs by intercepting the provider's own embed
call, so the page shows the pixels the network actually received, not a
reconstruction.

Output is a local page under runs/ (git-ignored). It is deliberately not
published: the photos belong to the Telegram authors (see DATASETS.md).

    python -m scripts.visualize_real_predictions
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from scripts.evaluate_cascade_real import IMAGES, LABELS, split_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/cascade_real/viewer"
IMG = OUT / "img"
REFS = ROOT / "datasets/wine-scanner/data/refs/rgb"


def save(image: Image.Image, name: str, size: int, quality: int = 82) -> str:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    image.save(IMG / name, quality=quality)
    return f"img/{name}"


def main() -> None:
    from backend.cascade import CascadeProvider
    from backend.config import Settings
    from dinov3_retrieval import build_transforms

    IMG.mkdir(parents=True, exist_ok=True)
    settings = Settings()                       # the served configuration (.env)
    provider = CascadeProvider(settings)
    pad_and_resize = build_transforms(provider.primary_info["image_size"])[1].transforms[:2]

    # Record every image handed to a network, in call order.
    captured: list[tuple[str, Image.Image]] = []
    original_embed = provider._embed

    def spy(model, transform, image):
        captured.append(("label" if model is provider.primary else "bottle", image))
        return original_embed(model, transform, image)

    provider._embed = spy

    catalog = {r["slug"]: r for r in csv.DictReader(
        (ROOT / "datasets/wine-scanner/data/catalog.csv").open(encoding="utf-8"))}
    ref_done: set[str] = set()

    def ref(slug: str) -> str:
        if slug not in ref_done:
            save(Image.open(REFS / f"{slug}.jpg"), f"ref_{slug}.jpg", 150)
            ref_done.add(slug)
        return f"img/ref_{slug}.jpg"

    def wine(slug: str | None) -> dict:
        if not slug:
            return {"slug": None, "name": "Нет в каталоге", "winery": "", "img": None}
        row = catalog[slug]
        return {"slug": slug, "name": row["name"], "winery": row["winery"], "img": ref(slug)}

    rows = [r for r in csv.DictReader(LABELS.open(encoding="utf-8-sig")) if split_of(r["source_post"]) == "report"]
    records = []
    for index, row in enumerate(rows, 1):
        with Image.open(IMAGES / row["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)   # exactly as backend.decode_image
        captured.clear()
        prediction = provider.predict(image)
        pipeline = prediction.pipeline
        qid = row["query_id"]

        # The original with the boxes the pipeline chose.
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        stroke = max(3, image.width // 180)
        if pipeline.get("bottle"):
            draw.rectangle(pipeline["bottle"]["box"], outline=(40, 140, 255), width=stroke)
        if pipeline.get("label"):
            draw.rectangle(pipeline["label"]["box"], outline=(255, 60, 60), width=stroke)
        original = save(marked, f"{qid}_orig.jpg", 420)

        # What the network received: the captured crop after the model's own
        # pad-to-square and resize - pixel for pixel its input.
        inputs = {}
        for kind, fed in captured:
            view = fed
            for step in pad_and_resize:
                view = step(view)
            inputs[kind] = save(view, f"{qid}_{kind}.jpg", 224, 90)

        truth = row["slug"] if row["in_catalog"] == "yes" else None
        ranked = [c.slug for c in prediction.candidates]
        scores = {c.slug: round(c.similarity, 3) for c in prediction.candidates}
        rank = ranked.index(truth) + 1 if truth in ranked else None
        records.append({
            "id": qid,
            "original": original,
            "label_input": inputs.get("label"),
            "bottle_input": inputs.get("bottle"),
            "label_source": pipeline["label_source"],
            "label_conf": (pipeline.get("label") or {}).get("confidence"),
            "resolver": pipeline["resolver"],
            "candidates": [{**wine(s), "score": scores[s], "correct": s == truth} for s in ranked[:5]],
            "truth": wine(truth),
            "rank": rank,
            "verdict": ("unknown" if truth is None else "top1" if rank == 1 else "top5" if rank else "miss"),
            "gap": pipeline["stage1_gap"],
        })
        if index % 100 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    (OUT / "data.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (OUT / "index.html").write_text(page(records, provider.version, settings), encoding="utf-8")
    counts = {k: sum(r["verdict"] == k for r in records) for k in ("top1", "top5", "miss", "unknown")}
    print(json.dumps({"photos": len(records), **counts, "page": str(OUT / "index.html")}, ensure_ascii=False))


def page(records: list[dict], version: str, settings) -> str:
    known = [r for r in records if r["truth"]["slug"]]
    top1 = sum(r["verdict"] == "top1" for r in known)
    top5 = sum(r["verdict"] in {"top1", "top5"} for r in known)

    def card(r: dict) -> str:
        badge = {"top1": ("ok", "Верно"), "top5": ("near", f"Правильное на {r['rank']}-м месте"),
                 "miss": ("bad", "Мимо — нет в топ-5"), "unknown": ("muted", "Вина нет в каталоге")}[r["verdict"]]
        pred = r["candidates"][0]
        others = "".join(
            f'<figure class="alt{" hit" if c["correct"] else ""}"><img src="{c["img"]}" loading="lazy">'
            f'<figcaption>{i}. {html.escape(c["name"])}<b>{c["score"]:.3f}</b></figcaption></figure>'
            for i, c in enumerate(r["candidates"][1:], start=2))
        truth = r["truth"]
        label_note = (f'детектор, уверенность {r["label_conf"]:.2f}' if r["label_source"] == "detector"
                      else "этикетка не найдена — в модель ушло всё фото")
        bottle = (f'<figure><img src="{r["bottle_input"]}" loading="lazy"><figcaption>Вход модели бутылки'
                  f'{" — поменяла ответ" if r["resolver"].get("swapped") else ""}</figcaption></figure>'
                  if r["bottle_input"] else "")
        return f"""
<article class="row {r['verdict']}" data-verdict="{r['verdict']}">
  <header><strong>{r['id']}</strong><span class="badge {badge[0]}">{badge[1]}</span>
    <small>разрыв top-1/top-2: {r['gap']:.3f}</small></header>
  <div class="cols">
    <figure class="orig"><img src="{r['original']}" loading="lazy">
      <figcaption>Фото. <i class="red">красная</i> рамка — выбранная этикетка{', <i class="blue">синяя</i> — бутылка' if r['bottle_input'] else ''}</figcaption></figure>
    <figure class="input"><img src="{r['label_input']}" loading="lazy"><figcaption>Что получила модель (224×224)<br>{label_note}</figcaption></figure>
    {bottle}
    <figure class="pred {'good' if pred['correct'] else 'wrong'}"><img src="{pred['img']}" loading="lazy">
      <figcaption>Ответ модели<br><b>{html.escape(pred['name'])}</b><br>{html.escape(pred['winery'])}<br>сходство {pred['score']:.3f}</figcaption></figure>
    <figure class="truth">{f'<img src="{truth["img"]}" loading="lazy">' if truth['img'] else '<div class="none">—</div>'}
      <figcaption>На самом деле<br><b>{html.escape(truth['name'])}</b><br>{html.escape(truth['winery'])}</figcaption></figure>
    <div class="alts">{others}</div>
  </div>
</article>"""

    cards = "".join(card(r) for r in records)
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Реальные фото — что видела модель</title>
<style>
:root {{ --ok:#2f7d4f; --near:#b07a1a; --bad:#b33a3a; --line:#e2e4dc; }}
* {{ box-sizing:border-box }}
body {{ margin:0; font:14px/1.45 system-ui, sans-serif; background:#f5f6f2; color:#1f2a24 }}
.top {{ position:sticky; top:0; z-index:5; background:#1c3d33; color:#eef3ea; padding:14px 22px }}
.top h1 {{ font-size:18px; margin:0 0 4px }}
.top p {{ margin:0 0 10px; color:#c9d7c6; font-size:13px }}
.filters button {{ border:1px solid #ffffff40; background:transparent; color:inherit; border-radius:20px;
  padding:6px 13px; margin:0 6px 6px 0; cursor:pointer; font:inherit }}
.filters button.on {{ background:#eef3ea; color:#1c3d33 }}
main {{ padding:16px 22px 60px; max-width:1500px; margin:auto }}
.row {{ background:#fff; border:1px solid var(--line); border-left:5px solid var(--line); border-radius:10px;
  padding:12px 14px; margin-bottom:14px }}
.row.top1 {{ border-left-color:var(--ok) }} .row.top5 {{ border-left-color:var(--near) }} .row.miss {{ border-left-color:var(--bad) }}
.row header {{ display:flex; gap:12px; align-items:center; margin-bottom:10px; flex-wrap:wrap }}
.row header small {{ color:#6b7466 }}
.badge {{ border-radius:20px; padding:2px 10px; font-weight:600; font-size:12px }}
.badge.ok {{ background:#e3f1e7; color:var(--ok) }} .badge.near {{ background:#f7ecd6; color:var(--near) }}
.badge.bad {{ background:#f6e1e1; color:var(--bad) }} .badge.muted {{ background:#eceee8; color:#6b7466 }}
.cols {{ display:flex; gap:14px; align-items:flex-start; flex-wrap:wrap }}
figure {{ margin:0; font-size:12px; color:#4b5647 }}
figure img {{ display:block; border-radius:6px; background:#eceee8 }}
.orig img {{ max-width:300px; max-height:300px }}
.input img {{ width:180px; height:180px }}
.pred img, .truth img, .none {{ width:150px; height:150px; object-fit:contain; background:#fff; border:1px solid var(--line) }}
.none {{ display:grid; place-items:center; border-radius:6px; font-size:28px; color:#aaa }}
.pred.good img {{ outline:3px solid var(--ok) }} .pred.wrong img {{ outline:3px solid var(--bad) }}
figcaption {{ max-width:190px; margin-top:5px }}
.alts {{ display:grid; grid-template-columns:repeat(2, 120px); gap:8px }}
.alt img {{ width:120px; height:84px; object-fit:contain; background:#fff; border:1px solid var(--line) }}
.alt figcaption {{ max-width:120px; font-size:11px }}
.alt figcaption b {{ display:block; font-weight:500; color:#8a9386 }}
.alt.hit img {{ outline:3px solid var(--near) }}
i.red {{ color:#d33; font-style:normal; font-weight:600 }} i.blue {{ color:#2a7be0; font-style:normal; font-weight:600 }}
</style></head><body>
<div class="top">
  <h1>Реальные фото: что получила модель, что ответила, что было на самом деле</h1>
  <p>Отложенная половина real_photos_v4 — {len(records)} фото, все подряд. Модель {html.escape(version)}.
   Вин из каталога {len(known)}: верно с первого ответа {top1} ({100*top1/len(known):.1f}%), правильное в топ-5 {top5} ({100*top5/len(known):.1f}%).
   Сходство — косинусное сходство эмбеддингов, не вероятность.</p>
  <div class="filters">
    <button class="on" data-f="all">Все ({len(records)})</button>
    <button data-f="top1">Верно ({top1})</button>
    <button data-f="top5">Правильное в топ-2…5 ({sum(r['verdict']=='top5' for r in records)})</button>
    <button data-f="miss">Мимо ({sum(r['verdict']=='miss' for r in records)})</button>
    <button data-f="unknown">Нет в каталоге ({sum(r['verdict']=='unknown' for r in records)})</button>
  </div>
</div>
<main>{cards}</main>
<script>
document.querySelectorAll('.filters button').forEach(b => b.onclick = () => {{
  document.querySelectorAll('.filters button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.row').forEach(r => r.style.display = (b.dataset.f === 'all' || r.dataset.verdict === b.dataset.f) ? '' : 'none');
}});
</script></body></html>"""


if __name__ == "__main__":
    main()
