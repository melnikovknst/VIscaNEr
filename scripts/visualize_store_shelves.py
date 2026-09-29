"""Browse the store-shelf annotation: every shelf photo with its numbered points,
and under it each frame with its label, the reviewer's note, the model's top-3 and
everything the bottle YOLO sees in the frame - every box with its confidence, the
box the pipeline picked, and whether DINO got that crop or the whole frame.

Two pipelines, one page each, YOLO run with that pipeline's own settings:
  bottle - infer_wine.py: bottle detector, imgsz 768, conf >= 0.05, max 60 boxes;
           the picked box >= 0.75 is cropped for DINO, below that DINO gets the frame
  labels - infer_wine_labels.py: label detector, imgsz 640, conf >= 0.25, max 30;
           the picked label box (+10 %) is always cropped for DINO
The picked box is read from the saved run of that pipeline.

Reads datasets/store_shelves_v*/{labels,points}.csv, review.json and, when present,
runs/inference/store_shelves_v*_bottle_pipeline.json. Writes a local page:

    python -m scripts.visualize_store_shelves                    # bottle pipeline
    python -m scripts.visualize_store_shelves --pipeline labels  # label pipeline
    python -m http.server 8768 --directory runs/store_shelves_viewer
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from backend.config import Settings
from scripts.relabel_tools import REFS, catalog
from scripts.visualize_relabeled import served_status

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/store_shelves_viewer"
DATASETS = ["store_shelves_v1", "store_shelves_v2"]
PIPELINES = {
    "bottle": {"weights": ROOT / "models/bottle_reranker/best_bottle_detector.pt", "imgsz": 768,
               "conf": 0.05, "max_det": 60, "high": 0.75, "run": "bottle_pipeline", "page": "index.html",
               "title": "YOLO бутылок (infer_wine.py)"},
    "labels": {"weights": ROOT / "models/yolo_label_detector/best.pt", "imgsz": 640,
               "conf": 0.25, "max_det": 30, "high": 0.50, "run": "label_pipeline", "page": "labels.html",
               "title": "YOLO этикеток (infer_wine_labels.py)"},
}


def detect(model, path: Path, cfg: dict) -> list[dict]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    result = model.predict(source=im, imgsz=cfg["imgsz"], conf=cfg["conf"], iou=0.70,
                           max_det=cfg["max_det"], verbose=False)[0]
    cx, cy = im.width / 2, im.height / 2
    boxes = [{"box": [round(v) for v in b], "conf": round(float(c), 3),
              "centre": b[0] <= cx <= b[2] and b[1] <= cy <= b[3]}
             for b, c, k in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
             if int(k) == 0]
    return sorted(boxes, key=lambda b: -b["conf"])


def is_picked(box: dict, picked: float | None) -> bool:
    # infer_wine stores the padded crop, not the raw box, so match by confidence.
    return picked is not None and abs(box["conf"] - picked) < 0.002


def overlay(src: Path, dst: Path, boxes: list[dict], picked: float | None, size: int, high: float) -> None:
    """The frame with every YOLO box: picked = green, >= high = yellow, lower = orange."""
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    scale = size / max(im.size)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial.ttf", 15)
    for b in sorted(boxes, key=lambda b: (is_picked(b, picked), b["conf"])):
        x1, y1, x2, y2 = (v * scale for v in b["box"])
        chosen = is_picked(b, picked)
        colour = (40, 220, 90) if chosen else (255, 215, 0) if b["conf"] >= high else (255, 140, 30)
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=4 if chosen else 2)
        label = f"{b['conf']:.2f}"
        tw = draw.textlength(label, font=font)
        ty = y1 - 18 if y1 > 18 else y1 + 1
        draw.rectangle((x1, ty, x1 + tw + 6, ty + 17), fill=colour)
        draw.text((x1 + 3, ty), label, fill=(0, 0, 0), font=font)
    cx, cy = im.width / 2, im.height / 2
    for width, colour in ((5, (0, 0, 0)), (2, (255, 255, 255))):
        draw.line((cx - 14, cy, cx + 14, cy), fill=colour, width=width)
        draw.line((cx, cy - 14, cx, cy + 14), fill=colour, width=width)
    im.save(dst, quality=85)


def thumb(src: Path, dst: Path, size: int) -> None:
    if dst.exists():
        return
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    im.save(dst, quality=82)


def main() -> None:
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="bottle")
    args = parser.parse_args()
    cfg = PIPELINES[args.pipeline]
    settings, cat = Settings(), catalog()
    yolo = YOLO(str(cfg["weights"]))
    img = OUT / f"img_{args.pipeline}"
    shared = OUT / "img"
    shared.mkdir(parents=True, exist_ok=True)
    img.mkdir(parents=True, exist_ok=True)
    photos, wines = [], {}
    for ds in DATASETS:
        base = ROOT / "datasets" / ds
        labels = list(csv.DictReader((base / "labels.csv").open(encoding="utf-8")))
        points = list(csv.DictReader((base / "points.csv").open(encoding="utf-8")))
        review = json.loads((base / "review.json").read_text(encoding="utf-8")) if (base / "review.json").is_file() else {}
        run_path = ROOT / f"runs/inference/{ds}_{cfg['run']}.json"
        run, inputs_by_qid = {}, {}
        if run_path.is_file():
            for r in json.loads(run_path.read_text(encoding="utf-8"))["results"]:
                stem = Path(r["source"].replace("\\", "/")).stem
                run[stem] = [(p["wine_slug"], p["similarity"]) for p in r["predictions"]]
                # The label pipeline has one input and keeps it at the top level.
                inputs_by_qid[stem] = r.get("dino_inputs") or [
                    {"image_mode": r["image_mode"], "yolo_confidence": r.get("yolo_confidence")}]
        by_source: dict[str, list] = {}
        for row, pt in zip(labels, points):
            qid = row["query_id"]
            answers = [s for s in row["accepted_slugs"].split(";") if s]
            ranked = run.get(qid, [])
            # Site thresholds were fitted for the bottle pipeline only.
            served = served_status(ranked, settings) if ranked and args.pipeline == "bottle" else None
            if row["scored"] != "True":
                verdict = "excluded"
            elif row["in_catalog"] == "True":
                verdict = ("top1" if ranked and ranked[0][0] in answers
                           else "top5" if any(s in answers for s, _ in ranked[:5]) else "miss")
            else:
                verdict = ("notcat_wrong" if served == "matched" else "notcat_ok") if served else "notcat"
            frame_path = base / "queries" / row["image_path"]
            boxes = detect(yolo, frame_path, cfg)
            inputs = inputs_by_qid.get(qid, [])
            chosen = inputs[0].get("yolo_confidence") if inputs else None
            overlay(frame_path, img / f"{qid}.jpg", boxes, chosen, 420, cfg["high"])
            pick = next((b for b in boxes if is_picked(b, chosen)), None)
            yolo_view = {"boxes": [{"conf": b["conf"], "picked": is_picked(b, chosen)} for b in boxes],
                         # Does the box the pipeline used cover the crosshair (the labelled bottle)?
                         "miss": pick is not None and not pick["centre"],
                         "centre_boxes": sum(b["centre"] for b in boxes),
                         "inputs": [{"mode": i["image_mode"], "conf": i.get("yolo_confidence")} for i in inputs]}
            for s in answers + [s for s, _ in ranked[:3]]:
                if s not in wines:
                    thumb(REFS / f"{s}.jpg", shared / f"ref_{s}.jpg", 180)
                    wines[s] = {"name": cat[s]["name"], "winery": cat[s]["winery"], "src": f"img/ref_{s}.jpg"}
            by_source.setdefault(row["source"], []).append({
                "qid": qid, "x": float(pt["x"]), "y": float(pt["y"]), "status": row["status"],
                "training_use": row["training_use"], "answers": answers, "note": row["note"],
                "review": review.get(qid, {}).get("note", ""), "verdict": verdict, "served": served,
                "top3": [{"slug": s, "sim": round(v, 3)} for s, v in ranked[:3]], "yolo": yolo_view})
        for source, items in by_source.items():
            name = f"{ds}_{Path(source).stem}.jpg"
            thumb(base / "source" / source, shared / name, 900)
            photos.append({"dataset": ds.replace("store_shelves_", ""), "source": source, "src": f"img/{name}", "items": items})
    meta = {"pipeline": args.pipeline, "title": cfg["title"], "high": cfg["high"],
            "img": f"img_{args.pipeline}",
            "other": {"href": PIPELINES["labels" if args.pipeline == "bottle" else "bottle"]["page"],
                      "title": PIPELINES["labels" if args.pipeline == "bottle" else "bottle"]["title"]}}
    data = json.dumps({"photos": photos, "wines": wines, "meta": meta}, ensure_ascii=False).replace("</", "<\\/")
    (OUT / cfg["page"]).write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
    print(OUT / cfg["page"], len(photos), "photos")


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка полок</title>
<style>
:root{--bg:#f6f7f3;--card:#fff;--ink:#1e2d27;--muted:#6a7568;--line:#e2e6dc;
--cat:#1f8a4c;--cat-bg:#e3f4e8;--out:#2a5d9f;--out-bg:#e4edf8;--rev:#b7791f;--rev-bg:#fbf1dc;--exc:#7b7f7a;--exc-bg:#eceee8;--bad:#c0392b;--bad-bg:#fbe5e2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141a17;--card:#1d2521;--ink:#e7ece6;--muted:#9aa698;--line:#2e3833;
--cat:#5fcf8c;--cat-bg:#1c3326;--out:#8ab4ef;--out-bg:#1b2940;--rev:#e0b25a;--rev-bg:#352b17;--exc:#a4aaa3;--exc-bg:#262e2a;--bad:#ef7f71;--bad-bg:#3a201d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,"Segoe UI",sans-serif}
header{max-width:1300px;margin:auto;padding:22px 16px 6px}h1{margin:0 0 6px;font-size:25px}.sub{color:var(--muted);font-size:13px}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px}.stat b{font-size:20px;margin-right:6px}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}.bar>div{max-width:1300px;margin:auto;padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;font-size:13px;color:var(--muted)}
select,input{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
main{max-width:1300px;margin:auto;padding:14px 16px 60px}
.shelf{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin-bottom:18px;display:grid;grid-template-columns:minmax(0,380px) minmax(0,1fr);gap:16px}
.shelf h2{margin:0 0 8px;font-size:17px}.pic{position:relative;align-self:start}.pic img{width:100%;border-radius:8px;display:block}
.dot{position:absolute;transform:translate(-50%,-50%);min-width:24px;height:24px;border-radius:99px;display:grid;place-items:center;font-size:11px;font-weight:700;color:#fff;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.5);cursor:pointer;padding:0 4px}
.dot.cat{background:#1f8a4c}.dot.out{background:#2a5d9f}.dot.rev{background:#b7791f}.dot.exc{background:#6b6f6a}.dot.hi{outline:3px solid #ffd400}
.items{display:grid;gap:10px}.item{display:grid;grid-template-columns:210px minmax(0,1fr);gap:12px;border:1px solid var(--line);border-radius:10px;padding:8px;scroll-margin-top:70px}
.item.hi{outline:2px solid #ffd400}.item>figure{margin:0}.item>figure img{width:210px;border-radius:6px;display:block;cursor:zoom-in}
.yolo{font-size:12px;color:var(--muted);margin-top:4px;line-height:1.4}.yolo b{color:var(--ink)}
.chip{display:inline-block;padding:0 5px;border-radius:4px;margin:1px 2px 0 0;font-variant-numeric:tabular-nums;color:#111;font-size:12px}
.c-pick{background:#28dc5a}.c-hi{background:#ffd700}.c-lo{background:#ff8c1e}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;place-items:center;z-index:20;cursor:zoom-out}#zoom img{max-width:94vw;max-height:94vh}
@media(max-width:560px){.item{grid-template-columns:1fr}.item>figure img{width:100%}}
.head{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:4px}.qid{font-weight:700}
.tag{font-size:12px;padding:1px 8px;border-radius:99px}.t-cat{background:var(--cat-bg);color:var(--cat)}.t-out{background:var(--out-bg);color:var(--out)}.t-rev{background:var(--rev-bg);color:var(--rev)}.t-exc{background:var(--exc-bg);color:var(--exc)}
.v-top1{background:var(--cat-bg);color:var(--cat)}.v-top5{background:var(--rev-bg);color:var(--rev)}.v-miss,.v-notcat_wrong{background:var(--bad-bg);color:var(--bad)}.v-notcat_ok{background:var(--out-bg);color:var(--out)}.v-excluded{background:var(--exc-bg);color:var(--exc)}
.note{font-size:13px;margin:2px 0 6px}.review{font-size:12px;background:var(--rev-bg);color:var(--ink);border-radius:6px;padding:4px 8px;margin-bottom:6px}
.refs{display:flex;gap:6px;flex-wrap:wrap}.ref{width:92px;font-size:11px;line-height:1.25;border:2px solid var(--line);border-radius:6px;padding:3px;background:var(--card)}
.ref img{width:100%;height:92px;object-fit:contain;background:#fff;border-radius:3px;display:block}.ref.ok{border-color:var(--cat)}.lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin:4px 0 3px}
.cols{display:flex;gap:16px;flex-wrap:wrap}
@media(max-width:800px){.shelf{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Разметка магазинных полок · <span id="ptitle"></span></h1>
<div class="sub"><a id="other" style="color:inherit"></a><br>На кадрах — все рамки YOLO с уверенностью: <span class="chip c-pick">выбрана пайплайном</span> <span class="chip c-hi" id="hi"></span> <span class="chip c-lo" id="lo"></span>; крест — центр кадра. <span id="rule"></span> Клик по кадру — увеличить.<br>Точка — центр этикетки; вокруг неё вырезан кадр 3:4. <span style="color:#1f8a4c">●</span> в каталоге <span style="color:#2a5d9f">●</span> нет в каталоге <span style="color:#b7791f">●</span> на проверку <span style="color:#6b6f6a">●</span> исключено. Нажмите на точку — подсветится кадр.</div>
<div class="stats" id="stats"></div></header>
<div class="bar"><div>
<label>Набор <select id="ds"><option value="">оба</option><option value="v1">v1 (фото 1–50)</option><option value="v2">v2 (фото 51–111)</option></select></label>
<label>Разметка <select id="st"><option value="">любая</option><option value="catalog">в каталоге</option><option value="out_of_catalog">нет в каталоге</option><option value="review">на проверку</option><option value="exclude">исключено</option></select></label>
<label>Модель <select id="vd"><option value="">любой исход</option><option value="top1">верно первым</option><option value="top5">верное в топ-5</option><option value="miss">промах</option><option value="notcat_wrong">чужая карточка вместо отказа</option><option value="notcat_ok">отказ/выбор на вино вне каталога</option></select></label>
<label>YOLO <select id="yo"><option value="">любые кадры</option><option value="miss">выбранная рамка не под крестом</option><option value="whole">DINO получил весь кадр</option></select></label>
<input id="q" type="search" placeholder="название, винодельня или s-012"><span id="count"></span></div></div>
<main id="list"></main><div id="zoom"><img alt=""></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById("data").textContent),W=D.wines,$=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const USE={catalog:["в каталоге","t-cat","cat"],out_of_catalog:["нет в каталоге","t-out","out"],review:["на проверку","t-rev","rev"],exclude:["исключено","t-exc","exc"]};
const VER={top1:"модель: верно первым",top5:"модель: верное в топ-5",miss:"модель: промах",notcat_wrong:"модель: чужая карточка",notcat_ok:"модель: не показала карточку",notcat:"вина нет в каталоге",excluded:"не оценивается"};
const M=D.meta;$("ptitle").textContent=M.title;$("other").href=M.other.href;$("other").textContent="→ то же для: "+M.other.title;
$("hi").textContent="≥ "+M.high;$("lo").textContent="< "+M.high;
$("rule").textContent=M.pipeline==="bottle"?"Если у выбранной рамки уверенность < 0.75, DINO получает весь кадр.":"DINO этикеток всегда получает вырез выбранной рамки (+10 %); если рамок нет — весь кадр.";
const MODE={yolo_label_crop:"DINO этикеток получил вырез выбранной рамки",yolo_bottle_crop:"DINO получил вырез выбранной рамки",original_low_confidence:"DINO получил весь кадр (выбранная рамка < 0.75)",original_no_detection:"DINO получил весь кадр (бутылок не найдено)"};
function yoloText(y){if(!y)return"";const pick=y.inputs[0];
 const chips=y.boxes.map(b=>`<span class="chip ${b.picked?"c-pick":b.conf>=M.high?"c-hi":"c-lo"}">${b.conf.toFixed(2)}</span>`).join("");
 const warn=y.miss?`<br><span style="color:var(--bad);font-weight:600">⚠ выбранная рамка не под крестом</span>${y.centre_boxes?"":" (под крестом рамок нет)"}`:"";
 return `<div class="yolo"><b>YOLO:</b> ${y.boxes.length} рамок<br>${chips||"—"}<br>${pick?(MODE[pick.mode]||pick.mode):"нет данных прогона"}${y.inputs.length>1?"; две бутылки в центре, берётся лучшая":""}${warn}</div>`}
const SERVED={matched:"карточка",uncertain:"выбор",not_found:"«не узнали»"};
const ref=(s,ok,extra="")=>W[s]?`<div class="ref ${ok?"ok":""}"><img loading="lazy" src="${W[s].src}" alt="">${extra}${esc(W[s].name)} · ${esc(W[s].winery)}</div>`:"";
function match(it,p){const q=$("q").value.trim().toLowerCase();
 return (!$("st").value||it.training_use===$("st").value)&&(!$("vd").value||it.verdict===$("vd").value)&&
 (!$("yo").value||($("yo").value==="miss"?it.yolo.miss:!/_crop$/.test((it.yolo.inputs[0]||{}).mode||"")))&&
 (!q||it.qid.includes(q)||it.note.toLowerCase().includes(q)||it.answers.some(s=>W[s]&&(W[s].name+" "+W[s].winery).toLowerCase().includes(q)));}
function render(){
 const photos=D.photos.filter(p=>!$("ds").value||p.dataset===$("ds").value);
 const all=photos.flatMap(p=>p.items),c=k=>all.filter(i=>i.training_use===k).length,v=k=>all.filter(i=>i.verdict===k).length;
 const known=v("top1")+v("top5")+v("miss");
 $("stats").innerHTML=[[photos.length,"фото"],[all.length,"кадров"],[c("catalog"),"в каталоге"],[c("out_of_catalog"),"нет в каталоге"],[c("review"),"на проверку"],[c("exclude"),"исключено"],
  [known?Math.round(100*v("top1")/known)+"%":"–","модель верно первым"],[known?Math.round(100*(v("top1")+v("top5"))/known)+"%":"–","верное в топ-5"],[all.filter(i=>i.yolo.miss).length,M.pipeline==="bottle"?"YOLO выбрал не ту бутылку":"выбранная этикетка не под крестом"]]
  .map(([n,t])=>`<div class="stat"><b>${n}</b>${t}</div>`).join("");
 let shown=0;
 $("list").innerHTML=photos.map(p=>{const items=p.items.filter(it=>match(it,p));if(!items.length)return"";shown+=items.length;
  const dots=p.items.map(it=>`<span class="dot ${USE[it.training_use][2]}" style="left:${it.x}%;top:${it.y}%" data-q="${it.qid}" title="${esc(it.qid)}">${it.qid.split("-")[1].replace(/^0+/,"")}</span>`).join("");
  const rows=items.map(it=>{const [u,uc]=USE[it.training_use];
   const truth=it.answers.length?it.answers.map(s=>ref(s,true)).join(""):`<div class="note" style="color:var(--muted)">${it.training_use==="exclude"?"не оценивается":"вина нет в каталоге"}</div>`;
   const top=it.top3.map((t,i)=>ref(t.slug,it.answers.includes(t.slug),`<b>${i+1}. ${t.sim.toFixed(3)}</b><br>`)).join("");
   return `<div class="item" id="i-${it.qid}"><figure><img loading="lazy" src="${M.img}/${it.qid}.jpg" alt="" data-zoom="1">${yoloText(it.yolo)}</figure>
   <div><div class="head"><span class="qid">${it.qid}</span><span class="tag ${uc}">${u}</span><span class="tag v-${it.verdict}">${VER[it.verdict]}</span>${it.served?`<span class="tag t-exc">сайт: ${SERVED[it.served]}</span>`:""}</div>
   <div class="note">${esc(it.note)}</div>${it.review?`<div class="review"><b>Проверка:</b> ${esc(it.review)}</div>`:""}
   <div class="cols"><div><div class="lbl">Правильно</div><div class="refs">${truth}</div></div>${top?`<div><div class="lbl">Модель, топ-3</div><div class="refs">${top}</div></div>`:""}</div></div></div>`}).join("");
  return `<section class="shelf"><div><h2>${p.dataset} · ${esc(p.source)}</h2><div class="pic"><img loading="lazy" src="${p.src}" alt="">${dots}</div></div><div class="items">${rows}</div></section>`}).join("")||'<p style="color:var(--muted)">Ничего не найдено</p>';
 $("count").textContent=shown+" кадров";
}
document.addEventListener("click",e=>{const z=e.target.closest("[data-zoom]");if(z){$("zoom").firstElementChild.src=z.src;$("zoom").style.display="grid";return}
 if(e.target.closest("#zoom")){$("zoom").style.display="none";return}
 const d=e.target.closest(".dot");if(!d)return;document.querySelectorAll(".hi").forEach(x=>x.classList.remove("hi"));
 d.classList.add("hi");const el=$("i-"+d.dataset.q);if(el){el.classList.add("hi");el.scrollIntoView({behavior:"smooth",block:"center"});}});
["ds","st","vd","yo"].forEach(id=>$(id).onchange=render);$("q").oninput=render;render();
</script></body></html>"""


if __name__ == "__main__":
    main()
