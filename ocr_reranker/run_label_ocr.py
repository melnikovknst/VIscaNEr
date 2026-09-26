#!/usr/bin/env python3
"""Read label-DINO JSON, OCR each saved label input and rerank its Top-5."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCRVL

from ocr_reranker.reranker import DEFAULT_ALPHA, DEFAULT_EVIDENCE_GATE, normalize, rerank_top5


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "paddleocr_vl" / "PaddleOCR-VL-1.6"
DEFAULT_REFS_ROOT = PROJECT_ROOT / "datasets" / "wine-scanner" / "data" / "refs" / "rgb"


def collect_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        cleaned = " ".join(value.split())
        if cleaned:
            strings.append(cleaned)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key not in {"input_path", "page_index", "model_settings"}:
                strings.extend(collect_strings(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            strings.extend(collect_strings(child))
    return strings


def extract_text(result: Any) -> str:
    markdown = result.markdown
    if isinstance(markdown, dict):
        strings = collect_strings(markdown.get("markdown_texts"))
        if strings:
            return "\n".join(dict.fromkeys(strings))
    return "\n".join(dict.fromkeys(collect_strings(result.json)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON from infer_wine_labels.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--refs-root", type=Path, default=DEFAULT_REFS_ROOT)
    parser.add_argument("--server-url", default="http://127.0.0.1:8111/")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--evidence-gate", type=float, default=DEFAULT_EVIDENCE_GATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.resolve().read_text(encoding="utf-8"))
    refs_root = args.refs_root.resolve()
    gallery_slugs = sorted(path.stem for path in refs_root.iterdir() if path.is_file())
    if not gallery_slugs:
        raise RuntimeError(f"Empty reference gallery: {refs_root}")
    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        use_layout_detection=False,
        vl_rec_backend="mlx-vlm-server",
        vl_rec_server_url=args.server_url,
        vl_rec_api_model_name=str(args.model_dir.resolve()),
    )
    output_rows: list[dict[str, Any]] = []
    rows = payload.get("results", [])
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        try:
            results = list(
                pipeline.predict(
                    row["ocr_input"],
                    use_layout_detection=False,
                    prompt_label="ocr",
                    temperature=0.0,
                    max_new_tokens=256,
                )
            )
            text = "\n".join(dict.fromkeys(filter(None, (extract_text(result) for result in results))))
            if normalize(text) in {"ocr", "text"}:
                text = ""
            ranked = rerank_top5(
                row["predictions"],
                text,
                gallery_slugs,
                alpha=args.alpha,
                evidence_gate=args.evidence_gate,
            )
            enriched = {
                **row,
                "ocr_status": "ok",
                "ocr_text": text,
                "ocr_predictions": ranked,
                "ocr_latency_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            enriched = {
                **row,
                "ocr_status": "error",
                "ocr_text": "",
                "ocr_error": f"{type(exc).__name__}: {exc}",
                "ocr_predictions": row["predictions"][:5],
                "ocr_latency_seconds": time.perf_counter() - started,
            }
        output_rows.append(enriched)
        print(
            f"OCR | {index:03d}/{len(rows):03d} | {enriched['ocr_status']} | "
            f"{enriched['ocr_latency_seconds']:.1f}s",
            flush=True,
        )
    result = {
        "pipeline": "label_yolo_to_dinov3_vitb16_to_paddleocr_vl_top5",
        "alpha": args.alpha,
        "evidence_gate": args.evidence_gate,
        "results": output_rows,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
