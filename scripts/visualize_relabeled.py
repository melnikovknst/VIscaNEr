"""Photo-by-photo accuracy breakdown of the served pipeline against the relabeled truth.

For every real_photos_v4 photo: the photo with the YOLO box, the exact image DINO
received, the model's top-5 with reference images and similarities, what the site
would show (card / choice / not recognised) at the current .env thresholds, and the
relabeled truth with the reviewer's note - so both the model and the labels can be
checked by eye.

Reads saved runs only:
  runs/inference/real_photos_v4_bottle_pipeline.json   (python infer_wine.py ... --output)
  evaluation/real_photos_v4_relabeled.csv              (python -m scripts.relabel_tools export)

The page is written under runs/ (git-ignored) and is not to be published: the photos
belong to their Telegram authors (see DATASETS.md).

    python -m scripts.visualize_relabeled [--bottle-run other_run.json]
    python -m scripts.visualize_relabeled --truth evaluation/real_photos_extra_labels.csv         --images datasets/real_photos_extra/queries         --bottle-run runs/inference/real_photos_extra_bottle_pipeline.json --out runs/extra_viewer
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from backend.config import Settings
from scripts.evaluate_saved_run import served_status
from scripts.relabel_tools import IMAGES, LABELS, REFS, catalog

ROOT = Path(__file__).resolve().parents[1]
RELABELED = ROOT / "evaluation/real_photos_v4_relabeled.csv"
BOTTLE_RUN = ROOT / "runs/inference/real_photos_v4_bottle_pipeline.json"
OUT = ROOT / "runs/relabeled_viewer"


def split(source_post: str) -> str:
    digest = hashlib.sha256(f"cascade-real:{source_post}".encode()).digest()
    return "selection" if digest[0] % 2 == 0 else "report"


def thumb(image: Image.Image, path: Path, size: int) -> None:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    image.save(path, quality=82)


def outcome(row: dict, ranked: list[tuple[str, float]], served: str) -> str:
    if row["scored"] != "True":
        return "excluded"
    answers = [s for s in row["accepted_slugs"].split(";") if s]
    if row["in_catalog"] != "True":
        return {"matched": "notcat_wrong", "uncertain": "notcat_choice", "not_found": "notcat_ok"}[served]
    slugs = [s for s, _ in ranked]
    if slugs[:1] and slugs[0] in answers:
        return "top1"
    if any(s in answers for s in slugs[:5]):
        return "top5"
    return "miss"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bottle-run", type=Path, default=BOTTLE_RUN)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--truth", type=Path, default=RELABELED, help="relabeled export or an extra labels CSV")
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--review-notes", type=Path, help="Optional query_id -> {note, slugs} JSON for review")
    args = parser.parse_args()

    settings = Settings()
    reviews = json.loads(args.review_notes.read_text(encoding="utf-8")) if args.review_notes else {}
    cat = catalog()
    # real_photos_v4 has original labels and a selection/report split; an extra
    # set labelled from scratch has neither and is shown as one "extra" half.
    original = ({r["query_id"]: r for r in csv.DictReader(LABELS.open(encoding="utf-8-sig"))}
                if args.truth == RELABELED else {})
    truth = {r["query_id"]: r for r in csv.DictReader(args.truth.open(encoding="utf-8"))}
    run = json.loads(args.bottle_run.read_text(encoding="utf-8"))
    results = {Path(r["source"].replace("\\", "/")).stem: r for r in run["results"]}
    img_dir = args.out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    records, needed_refs = [], set()
    for index, (qid, row) in enumerate(sorted(truth.items())):
        result = results.get(qid) or results.get(Path(row["image_path"]).stem)
        if result is None:
            raise ValueError(f"Missing prediction: {qid}")
        ranked = [(p["wine_slug"], p["similarity"]) for p in result["predictions"]][:5]
        served = served_status(ranked, settings)
        with Image.open(args.images / row["image_path"]) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        # The photo with every box DINO looked at; the input itself next to it.
        marked = photo.copy()
        draw = ImageDraw.Draw(marked)
        inputs = []
        for n, item in enumerate(result.get("dino_inputs") or []):
            box = item.get("yolo_box")
            if box:
                colour = (40, 200, 90) if item["image_mode"] == "yolo_bottle_crop" else (240, 170, 30)
                draw.rectangle(box, outline=colour, width=max(4, photo.width // 120))
            model_input = (photo.crop(box) if item["image_mode"] == "yolo_bottle_crop" and box else photo)
            name = f"{qid}_in{n}.jpg"
            thumb(model_input, img_dir / name, 360)
            inputs.append({"src": f"img/{name}", "mode": item["image_mode"],
                           "confidence": item.get("yolo_confidence")})
        cx, cy = photo.width / 2, photo.height / 2
        r = max(8, photo.width // 60)
        draw.line((cx - r * 2, cy, cx + r * 2, cy), fill=(255, 255, 255), width=max(2, r // 3))
        draw.line((cx, cy - r * 2, cx, cy + r * 2), fill=(255, 255, 255), width=max(2, r // 3))
        thumb(marked, img_dir / f"{qid}.jpg", 720)

        answers = [s for s in row["accepted_slugs"].split(";") if s]
        review = reviews.get(qid, {})
        needed_refs.update(review.get("slugs", []))
        needed_refs.update(answers)
        needed_refs.update(s for s, _ in ranked)
        orig = original.get(qid)
        records.append({
            "qid": qid, "half": split(orig["source_post"]) if orig else "extra",
            "outcome": outcome(row, ranked, served), "served": served,
            "status": row["status"], "reviewer": row.get("reviewer", "claude"), "note": row["note"],
            "in_catalog": row["in_catalog"],
            "original": orig["slug"] if orig and orig["in_catalog"] == "yes" else None,
            "answers": answers,
            "review": review,
            "top5": [{"slug": s, "sim": round(v, 3)} for s, v in ranked],
            "photo": f"img/{qid}.jpg", "inputs": inputs,
            "pair": result.get("selection_mode") == "ambiguous_center_pair",
        })
        if index % 100 == 0:
            print(f"{index}/{len(truth)}")

    wines = {}
    for slug in sorted(needed_refs):
        path = REFS / f"{slug}.jpg"
        if path.is_file():
            with Image.open(path) as ref:
                thumb(ref, img_dir / f"ref_{slug}.jpg", 220)
        wines[slug] = {"name": cat[slug]["name"], "winery": cat[slug]["winery"],
                       "src": f"img/ref_{slug}.jpg"}

    data = {"records": records, "wines": wines, "model": run.get("pipeline"),
            "thresholds": {"min_similarity": settings.min_similarity, "min_margin": settings.min_margin,
                           "min_suggest_similarity": settings.min_suggest_similarity}}
    page = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(args.out / "index.html")


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разбор точности</title>
<style>
:root { --bg:#f6f7f3; --card:#fff; --ink:#1e2d27; --muted:#6a7568; --line:#e2e6dc;
  --ok:#1f8a4c; --ok-bg:#e3f4e8; --mid:#b7791f; --mid-bg:#fbf1dc; --bad:#c0392b; --bad-bg:#fbe5e2;
  --info:#2a5d9f; --info-bg:#e4edf8; --grey-bg:#eceee8; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#141a17; --card:#1d2521; --ink:#e7ece6; --muted:#9aa698; --line:#2e3833;
  --ok:#5fcf8c; --ok-bg:#1c3326; --mid:#e0b25a; --mid-bg:#352b17; --bad:#ef7f71; --bad-bg:#3a201d;
  --info:#8ab4ef; --info-bg:#1b2940; --grey-bg:#262e2a; } }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, "Segoe UI", sans-serif }
header { padding:24px 16px 8px; max-width:1400px; margin:auto }
h1 { margin:0 0 4px; font-size:26px }
.sub { color:var(--muted); font-size:13px }
.stats { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0 }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; cursor:pointer; text-align:left; color:inherit; font:inherit }
.stat.on { outline:2px solid var(--ink) }
.stat b { display:block; font-size:22px }
.stat span { font-size:12px; color:var(--muted) }
.bar { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line) }
.bar-inner { max-width:1400px; margin:auto; padding:10px 16px; display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center }
.bar label { font-size:13px; color:var(--muted) }
select, input[type=search] { font:inherit; padding:5px 8px; border:1px solid var(--line); border-radius:8px; background:var(--card); color:var(--ink) }
main { max-width:1400px; margin:auto; padding:12px 16px 60px }
.row { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; margin-bottom:14px;
  display:grid; grid-template-columns:minmax(0, 270px) minmax(0, 170px) minmax(0, 1fr); gap:14px }
.photo img { width:100%; border-radius:8px; display:block }
.inputs img { width:100%; border-radius:6px; display:block; margin-bottom:4px; background:var(--grey-bg) }
.cap { font-size:12px; color:var(--muted) }
.head { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:8px }
.qid { font-weight:700; margin-right:4px }
.tag { font-size:12px; padding:2px 8px; border-radius:99px; background:var(--grey-bg); color:var(--muted) }
.t-top1 { background:var(--ok-bg); color:var(--ok) } .t-top5 { background:var(--mid-bg); color:var(--mid) }
.t-miss, .t-notcat_wrong { background:var(--bad-bg); color:var(--bad) } .t-notcat_ok { background:var(--info-bg); color:var(--info) }
.t-notcat_choice { background:var(--mid-bg); color:var(--mid) }
.section-title { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:10px 0 6px }
.wines { display:flex; gap:8px; overflow-x:auto; padding-bottom:4px }
.wine { flex:0 0 112px; border:2px solid var(--line); border-radius:8px; padding:4px; background:var(--card); font-size:12px; line-height:1.3 }
.wine img { width:100%; height:110px; object-fit:contain; background:#fff; border-radius:4px; display:block }
.wine.hit { border-color:var(--ok) }
.wine .sim { font-weight:700 }
.note { font-size:13px; background:var(--grey-bg); border-radius:8px; padding:6px 10px; margin-top:8px }
.note b { font-weight:600 }
.empty { color:var(--muted); padding:40px; text-align:center }
@media (max-width: 760px) { .row { grid-template-columns:1fr 1fr } .detail { grid-column:1 / -1 } }
</style>
</head>
<body>
<header>
  <h1>Разбор точности на реальных фото</h1>
  <div class="sub" id="sub"></div>
  <div class="stats" id="stats"></div>
</header>
<div class="bar"><div class="bar-inner">
  <label>Половина <select id="half"><option value="all">обе</option><option value="report">отчётная</option><option value="selection">подборная</option><option value="extra">новый набор</option></select></label>
  <label>Разметка <select id="status"><option value="all">любая</option><option value="changed">исправлена</option><option value="keep">оставлена</option><option value="fix">fix</option><option value="multi">multi</option><option value="not_in_catalog">not_in_catalog</option><option value="wrong_unknown">wrong_unknown</option><option value="remove">remove</option><option value="unsure">unsure</option></select></label>
  <label>Сайт покажет <select id="served"><option value="all">что угодно</option><option value="matched">карточку</option><option value="uncertain">выбор</option><option value="not_found">«не узнали»</option></select></label>
  <label><input type="checkbox" id="reviewOnly"> повторная проверка</label>
  <label>Разметчик <select id="reviewer"><option value="all">любой</option><option value="user">вы</option><option value="claude">Claude</option></select></label>
  <input type="search" id="q" placeholder="s-149, название или примечание">
  <span class="cap" id="count"></span>
</div></div>
<main id="list"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const W = D.wines;
const OUT = {
  top1: ["Верно с первого раза", "t-top1"], top5: ["Верное в топ-5", "t-top5"], miss: ["Промах", "t-miss"],
  notcat_ok: ["Нет в каталоге, отказ", "t-notcat_ok"], notcat_wrong: ["Нет в каталоге, ответил зря", "t-notcat_wrong"],
  notcat_choice: ["Нет в каталоге, предложен выбор", "t-notcat_choice"],
  excluded: ["Исключено из оценки", ""],
};
const SERVED = { matched: "сайт: карточка", uncertain: "сайт: выбор из вариантов", not_found: "сайт: «не узнали»" };
const MODE = { yolo_bottle_crop: "кроп бутылки", original_low_confidence: "весь кадр (YOLO < 0.75)", original_no_detection: "весь кадр (бутылка не найдена)" };
let outcomeFilter = "all";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const wineName = (s) => W[s] ? `${esc(W[s].name)} · ${esc(W[s].winery)}` : esc(s);

function visible() {
  const half = $("half").value, st = $("status").value, sv = $("served").value, rv = $("reviewer").value;
  const q = $("q").value.trim().toLowerCase();
  return D.records.filter((r) =>
    (half === "all" || r.half === half) &&
    (outcomeFilter === "all" || r.outcome === outcomeFilter) &&
    (st === "all" || (st === "changed" ? !["keep", "unreviewed"].includes(r.status) : r.status === st)) &&
    (sv === "all" || r.served === sv) &&
    (rv === "all" || r.reviewer === rv) &&
    (!$("reviewOnly").checked || r.review?.note) &&
    (!q || r.qid.includes(q) || (r.note || "").toLowerCase().includes(q) || [...r.answers, ...r.top5.map((t) => t.slug), r.original || ""].some((s) => W[s] && (W[s].name + " " + W[s].winery).toLowerCase().includes(q))));
}

function stats() {
  const half = $("half").value;
  const rows = D.records.filter((r) => half === "all" || r.half === half);
  const known = rows.filter((r) => ["top1", "top5", "miss"].includes(r.outcome));
  const n = (o) => rows.filter((r) => r.outcome === o).length;
  const pct = (x) => known.length ? (100 * x / known.length).toFixed(1) + "%" : "–";
  const cards = [
    ["all", rows.length, "все фото"],
    ["top1", n("top1"), `верно с первого раза · ${pct(n("top1"))} от ${known.length}`],
    ["top5", n("top5"), `верное в топ-5 (не первым) · топ-5 всего ${pct(n("top1") + n("top5"))}`],
    ["miss", n("miss"), `промах · ${pct(n("miss"))}`],
    ["notcat_ok", n("notcat_ok"), "нет в каталоге, отказ"],
    ["notcat_wrong", n("notcat_wrong"), "нет в каталоге, ответил зря"],
    ["notcat_choice", n("notcat_choice"), "нет в каталоге, предложен выбор"],
    ["excluded", n("excluded"), "исключено (нет цели / не решить)"],
  ];
  $("stats").innerHTML = cards.map(([k, v, t]) =>
    `<button class="stat ${outcomeFilter === k ? "on" : ""}" data-k="${k}"><b>${v}</b><span>${t}</span></button>`).join("");
  $("stats").querySelectorAll(".stat").forEach((b) => b.onclick = () => { outcomeFilter = b.dataset.k; render(); });
}

function row(r) {
  const [label, cls] = OUT[r.outcome];
  const changed = r.half !== "extra" && !["keep", "unreviewed"].includes(r.status);
  const top = r.top5.map((t, i) => `<div class="wine ${r.answers.includes(t.slug) ? "hit" : ""}">
      <img loading="lazy" src="${W[t.slug]?.src}" alt=""><div><span class="sim">${i + 1}. ${t.sim.toFixed(3)}</span><br>${wineName(t.slug)}</div></div>`).join("");
  const truth = r.answers.length ? r.answers.map((s) => `<div class="wine hit">
      <img loading="lazy" src="${W[s]?.src}" alt=""><div>${wineName(s)}</div></div>`).join("")
    : `<div class="cap">${["notcat", "not_in_catalog"].includes(r.status) ? "Вина нет в каталоге — правильно отказаться"
        : r.status === "wrong_unknown" ? "Исходная метка неверна, настоящее вино не опознано — любой ответ считается промахом"
        : r.status === "remove" || r.status === "unsure" ? "Фото не оценивается" : "Нет в каталоге"}</div>`;
  const inputs = r.inputs.map((x) => `<img loading="lazy" src="${x.src}" alt=""><div class="cap">${MODE[x.mode] || x.mode}${x.confidence != null ? `, YOLO ${x.confidence.toFixed(2)}` : ""}</div>`).join("");
  const orig = changed ? `<div class="note"><b>Было в исходной разметке:</b> ${r.original ? wineName(r.original) : "нет в каталоге"}</div>` : "";
  const note = r.note ? `<div class="note"><b>${r.reviewer === "user" ? "Вы" : "Claude"} (${esc(r.status)}):</b> ${esc(r.note)}</div>` : "";
  const review = r.review?.note ? `<div class="note"><b>Повторная проверка (метка пока сохранена):</b> ${esc(r.review.note)}</div>
    <div class="section-title">Карточки для сравнения, не дополнительные правильные ответы</div><div class="wines">${(r.review.slugs || []).map((s) =>
      `<div class="wine"><img loading="lazy" src="${W[s]?.src}" alt=""><div>${wineName(s)}</div></div>`).join("")}</div>` : "";
  return `<article class="row" id="${esc(r.qid)}">
    <div class="photo"><img loading="lazy" src="${r.photo}" alt=""><div class="cap">+ центр кадра; рамка: зелёная — кроп, жёлтая — YOLO < 0.75</div></div>
    <div class="inputs"><div class="section-title">Вход модели</div>${inputs}${r.pair ? '<div class="cap">две бутылки, берётся максимум</div>' : ""}</div>
    <div class="detail">
      <div class="head"><span class="qid">${r.qid}</span><span class="tag ${cls}">${label}</span>
        <span class="tag">${SERVED[r.served]}</span><span class="tag">${{report: "отчётная половина", selection: "подборная половина", extra: "новый набор"}[r.half]}</span>
        ${changed ? `<span class="tag">разметка: ${esc(r.status)}</span>` : ""}</div>
      <div class="section-title">Ответ модели (топ-5, similarity)</div><div class="wines">${top}</div>
      <div class="section-title">Правильно (разметка)</div><div class="wines">${truth}</div>
      ${orig}${note}${review}
    </div></article>`;
}

let shown = 0, current = [];
function render() {
  stats();
  current = visible();
  shown = 0;
  $("list").innerHTML = current.length ? "" : '<div class="empty">Ничего не найдено</div>';
  $("count").textContent = `${current.length} фото`;
  more();
}
function more() {
  const next = current.slice(shown, shown + 40);
  $("list").insertAdjacentHTML("beforeend", next.map(row).join(""));
  shown += next.length;
}
window.addEventListener("scroll", () => {
  if (shown < current.length && innerHeight + scrollY > document.body.scrollHeight - 1500) more();
});
["half", "status", "served", "reviewer"].forEach((id) => $(id).onchange = render);
$("reviewOnly").onchange = render;
$("q").oninput = render;
for (const status of [...new Set(D.records.map((r) => r.status))]) {
  if (![...$("status").options].some((o) => o.value === status)) $("status").add(new Option(status, status));
}
if (location.hash) $("q").value = decodeURIComponent(location.hash.slice(1));
const t = D.thresholds;
$("sub").textContent = `Модель: ${D.model || "infer_wine"} · карточка при similarity ≥ ${t.min_similarity} и отрыве ≥ ${t.min_margin}, выбор при ≥ ${t.min_suggest_similarity}. Процент — от фото, где вино есть в каталоге и известно.`;
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
