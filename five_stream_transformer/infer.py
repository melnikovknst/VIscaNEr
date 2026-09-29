#!/usr/bin/env python3
"""Deployable inference for the Stage-2C five-stream residual Transformer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

from five_stream_transformer.data import resolve_refs
from five_stream_transformer.dino import choose_device, eval_transform
from five_stream_transformer.features import extract_visual_streams
from five_stream_transformer.model import STREAM_NAMES, FiveStreamResidualTransformer
from five_stream_transformer.stage2c import load_stage2c_models
from five_stream_transformer.text import frozen_char_ngram_embedding
from joint_yolo.infer import (
    pair_bottle,
    padded_crop,
    predict_detections,
    select_bottle_candidates,
    select_label_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO = ROOT / "models" / "joint_yolo" / "best.pt"
DEFAULT_STAGE2C = ROOT / "models" / "stage2c" / "manual_stage2c_best.pt"
DEFAULT_TRANSFORMER = ROOT / "models" / "five_stream_transformer" / "residual_best.pt"
DEFAULT_OCR = ROOT / "models" / "easyocr_ru_en"
DEFAULT_GALLERIES_ZIP = ROOT / "data" / "inference_galleries.zip"
DEFAULT_GALLERIES_ROOT = ROOT / "data" / "inference_galleries"
DEFAULT_CACHE = ROOT / "models" / "five_stream_transformer" / "gallery_features.pt"
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(values: Iterable[str]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for raw in values:
        path = Path(raw).expanduser().resolve()
        candidates = [path] if path.is_file() else sorted(path.rglob("*")) if path.is_dir() else []
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED and candidate not in seen:
                seen.add(candidate)
                output.append(candidate)
    if not output:
        raise ValueError("No supported input images")
    return output


def ensure_galleries(root: Path, archive: Path) -> tuple[Path, Path]:
    label_root = root / "label_refs"
    bottle_root = root / "bottle_refs"
    if label_root.is_dir() and bottle_root.is_dir():
        return label_root, bottle_root
    if not archive.is_file():
        raise FileNotFoundError(f"Missing galleries and archive: {root}, {archive}")
    root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        members = [
            name for name in handle.namelist()
            if name.startswith("inference_galleries/") and not name.endswith("/")
        ]
        if not members:
            raise ValueError(f"No inference galleries in {archive}")
        for name in members:
            relative = Path(name)
            if ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {name}")
            destination = root.parent / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(name) as source, destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    return label_root, bottle_root


def load_foreign_checkpoint(path: Path) -> dict[str, Any]:
    """torch.load a checkpoint whose pickled args hold PosixPath (saved on Kaggle/macOS).

    Windows cannot instantiate PosixPath, so it is unpickled as PurePosixPath instead.
    """
    import pathlib

    original = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.PurePosixPath  # type: ignore[misc]
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original  # type: ignore[misc]


def crop_bottle(
    image: Image.Image,
    detection: dict[str, Any] | None,
    threshold: float,
) -> tuple[Image.Image, str]:
    if detection is None:
        return image.copy(), "original_no_bottle"
    if float(detection["confidence"]) < threshold:
        return image.copy(), "original_low_bottle_confidence"
    crop, _ = padded_crop(image, detection["box"], 0.06)
    return crop, "yolo_bottle_crop"


class FiveStreamInferencePipeline:
    """Load all frozen components once and rank catalogue candidates."""

    def __init__(
        self,
        *,
        joint_yolo: Path = DEFAULT_YOLO,
        stage2c_checkpoint: Path = DEFAULT_STAGE2C,
        transformer_checkpoint: Path = DEFAULT_TRANSFORMER,
        easyocr_model_dir: Path = DEFAULT_OCR,
        galleries_root: Path = DEFAULT_GALLERIES_ROOT,
        galleries_zip: Path = DEFAULT_GALLERIES_ZIP,
        gallery_cache: Path = DEFAULT_CACHE,
        device: str = "auto",
        image_size: int = 224,
        gallery_batch_size: int = 48,
        num_workers: int = 2,
        verify_stage2c_hash: bool = False,
    ) -> None:
        self.device = choose_device(device)
        self.image_size = int(image_size)
        self.transform = eval_transform(self.image_size)
        self.joint_yolo_path = Path(joint_yolo).resolve()
        self.stage2c_path = Path(stage2c_checkpoint).resolve()
        self.transformer_path = Path(transformer_checkpoint).resolve()
        self.easyocr_dir = Path(easyocr_model_dir).resolve()
        self.galleries_zip = Path(galleries_zip).resolve()
        for required in (
            self.joint_yolo_path, self.stage2c_path, self.transformer_path,
            self.easyocr_dir / "craft_mlt_25k.pth", self.easyocr_dir / "cyrillic_g2.pth",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)

        label_root, bottle_root = ensure_galleries(
            Path(galleries_root).resolve(), self.galleries_zip
        )
        self.slugs, self.label_refs = resolve_refs(label_root)
        bottle_slugs, self.bottle_refs = resolve_refs(bottle_root)
        if self.slugs != bottle_slugs:
            raise ValueError("Label and bottle galleries are not aligned")

        print(f"LOAD | device={self.device} | gallery={len(self.slugs)}", flush=True)
        self.label_model, self.bottle_model, self.stage2c_info = load_stage2c_models(
            self.stage2c_path, self.device, self.image_size,
            verify_sha256=verify_stage2c_hash,
        )
        self.transformer, self.transformer_info = self._load_transformer()
        self.gallery = self._load_or_build_gallery(
            Path(gallery_cache).resolve(), gallery_batch_size, num_workers
        )
        self.detector = YOLO(str(self.joint_yolo_path))

        import easyocr

        self.ocr = easyocr.Reader(
            ["ru", "en"], gpu=self.device.type == "cuda",
            model_storage_directory=str(self.easyocr_dir), download_enabled=False, verbose=False,
        )

    def _gallery_signature(self) -> dict[str, Any]:
        return {
            "stage2c_sha256": self.transformer_info["stage2c_source_sha256"],
            "galleries_zip_sha256": file_sha256(self.galleries_zip),
            "image_size": self.image_size,
            "slugs": self.slugs,
        }

    def _load_or_build_gallery(
        self, cache_path: Path, batch_size: int, num_workers: int
    ) -> dict[str, torch.Tensor]:
        signature = self._gallery_signature()
        if cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if payload.get("signature") == signature:
                print(f"GALLERY | cache={cache_path}", flush=True)
                return payload["streams"]
        label_raw, label_embedding = extract_visual_streams(
            self.label_model, self.label_refs, self.image_size, self.device,
            batch_size, num_workers, "label_gallery",
        )
        bottle_raw, bottle_embedding = extract_visual_streams(
            self.bottle_model, self.bottle_refs, self.image_size, self.device,
            batch_size, num_workers, "bottle_gallery",
        )
        streams = {
            "bottle_crop": bottle_raw, "label_crop": label_raw,
            "bottle_dino": bottle_embedding, "label_dino": label_embedding,
            "ocr_text": frozen_char_ngram_embedding(self.slugs).half(),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"signature": signature, "streams": streams}, cache_path)
        print(f"GALLERY | saved={cache_path}", flush=True)
        return streams

    def _load_transformer(self) -> tuple[FiveStreamResidualTransformer, dict[str, Any]]:
        checkpoint = load_foreign_checkpoint(self.transformer_path)
        if checkpoint.get("format") != "stage2c-five-stream-residual-v1":
            raise ValueError(f"Unsupported Transformer checkpoint: {checkpoint.get('format')}")
        source = checkpoint.get("stage2c_source", {})
        if source.get("epoch") != self.stage2c_info["epoch"]:
            raise ValueError("Transformer and Stage-2C checkpoint epochs do not match")
        architecture = checkpoint["architecture"]
        max_candidates = int(checkpoint["model_state_dict"]["rank_embedding.weight"].shape[0])
        model = FiveStreamResidualTransformer(
            model_dim=int(architecture["model_dim"]), num_heads=int(architecture["num_heads"]),
            num_layers=int(architecture["num_layers"]),
            feedforward_dim=int(architecture["feedforward_dim"]),
            dropout=float(architecture["dropout"]), max_candidates=max_candidates,
            base_temperature=float(architecture["base_temperature"]),
            max_residual=float(architecture["max_residual"]),
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        return model, {
            "format": checkpoint["format"], "epoch": int(checkpoint["epoch"]),
            "top_k_per_stream": int(checkpoint.get("args", {}).get("top_k", max_candidates // 3)),
            "max_candidates": max_candidates,
            "stage2c_source_sha256": source.get("sha256"),
        }

    @torch.inference_mode()
    def _visual(self, model: torch.nn.Module, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = self.transform(image).unsqueeze(0).to(self.device)
        with autocast(self.device):
            raw = model.backbone_features(pixels)
            embedding = torch.nn.functional.normalize(model.projection(raw), dim=1)
        return raw.float().cpu(), embedding.float().cpu()

    def _ocr_text(self, image: Image.Image) -> str:
        pieces = self.ocr.readtext(
            np.asarray(image), detail=0, paragraph=False, decoder="greedy",
            batch_size=1, workers=0,
        )
        return " ".join(str(piece).strip() for piece in pieces if str(piece).strip())

    def _prepare_views(self, image: Image.Image) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        yolo_device: str | int = (
            self.device.index or 0 if self.device.type == "cuda" else self.device.type
        )
        detections = predict_detections(self.detector, image, yolo_device, 0.05)
        bottles = [item for item in detections if item["class_id"] == 0]
        labels = [item for item in detections if item["class_id"] == 1]
        selected_bottles, bottle_ambiguous = select_bottle_candidates(bottles, image.width)
        selected_labels, label_ambiguous = select_label_candidates(
            labels, (image.width * 0.5, image.height * 0.5),
            math.hypot(image.width, image.height), 0.06, image_width=image.width,
        )
        selected_labels_or_fallback: list[dict[str, Any] | None] = selected_labels or [None]
        views: list[dict[str, Any]] = []
        for index, label in enumerate(selected_labels_or_fallback, start=1):
            if label is None:
                label_image = image.copy()
                bottle = selected_bottles[0] if selected_bottles else None
            else:
                label_image, _ = padded_crop(image, label["box"], 0.10)
                bottle = pair_bottle(label, bottles)
            bottle_image, bottle_mode = crop_bottle(image, bottle, 0.75)
            views.append({
                "view": index, "label_image": label_image, "bottle_image": bottle_image,
                "label_confidence": None if label is None else float(label["confidence"]),
                "bottle_confidence": None if bottle is None else float(bottle["confidence"]),
                "bottle_mode": bottle_mode,
            })
        diagnostics = {
            "bottle_detections": len(bottles), "label_detections": len(labels),
            "selected_views": len(views), "bottle_ambiguous": bottle_ambiguous,
            "label_ambiguous": label_ambiguous,
        }
        return views, diagnostics

    def _candidate_ids(
        self, label_embedding: torch.Tensor, bottle_embedding: torch.Tensor,
        ocr_embedding: torch.Tensor, ocr_available: bool,
    ) -> list[int]:
        top_k = self.transformer_info["top_k_per_stream"]
        label_scores = label_embedding @ self.gallery["label_dino"].float().T
        bottle_scores = bottle_embedding @ self.gallery["bottle_dino"].float().T
        ocr_scores = ocr_embedding @ self.gallery["ocr_text"].float().T
        pool = set(label_scores[0].topk(top_k).indices.tolist())
        pool |= set(bottle_scores[0].topk(top_k).indices.tolist())
        if ocr_available:
            pool |= set(ocr_scores[0].topk(top_k).indices.tolist())
        return sorted(
            pool,
            key=lambda candidate: max(
                float(label_scores[0, candidate]), float(bottle_scores[0, candidate]),
                float(ocr_scores[0, candidate]) if ocr_available else -1.0,
            ),
            reverse=True,
        )[: self.transformer_info["max_candidates"]]

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 5) -> dict[str, Any]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        views, diagnostics = self._prepare_views(image)
        fused: dict[int, dict[str, Any]] = {}
        view_metadata: list[dict[str, Any]] = []
        for view in views:
            ocr_text = self._ocr_text(view["label_image"])
            label_raw, label_embedding = self._visual(self.label_model, view["label_image"])
            bottle_raw, bottle_embedding = self._visual(self.bottle_model, view["bottle_image"])
            ocr_embedding = frozen_char_ngram_embedding([ocr_text])
            ocr_available = bool(ocr_text.strip())
            candidate_ids = self._candidate_ids(
                label_embedding, bottle_embedding, ocr_embedding, ocr_available
            )
            ids = torch.tensor(candidate_ids, dtype=torch.long)
            query = {
                "bottle_crop": bottle_raw.to(self.device),
                "label_crop": label_raw.to(self.device),
                "bottle_dino": bottle_embedding.to(self.device),
                "label_dino": label_embedding.to(self.device),
                "ocr_text": ocr_embedding.to(self.device),
            }
            references = {
                name: self.gallery[name][ids][None].float().to(self.device)
                for name in STREAM_NAMES
            }
            mask = torch.ones(1, len(candidate_ids), dtype=torch.bool, device=self.device)
            available = torch.ones(1, len(STREAM_NAMES), dtype=torch.bool, device=self.device)
            available[:, STREAM_NAMES.index("ocr_text")] = ocr_available
            with autocast(self.device):
                final, base, residual = self.transformer(query, references, mask, available)
            label_scores = label_embedding @ self.gallery["label_dino"].float().T
            bottle_scores = bottle_embedding @ self.gallery["bottle_dino"].float().T
            ocr_scores = ocr_embedding @ self.gallery["ocr_text"].float().T
            for position, candidate in enumerate(candidate_ids):
                record = {
                    "wine_slug": self.slugs[candidate],
                    "score": float(final[0, position]),
                    "base_logit": float(base[0, position]),
                    "residual": float(residual[0, position]),
                    "bottle_similarity": float(bottle_scores[0, candidate]),
                    "label_similarity": float(label_scores[0, candidate]),
                    "ocr_similarity": float(ocr_scores[0, candidate]) if ocr_available else None,
                    "source_view": int(view["view"]),
                }
                if candidate not in fused or record["score"] > fused[candidate]["score"]:
                    fused[candidate] = record
            view_metadata.append({
                "view": int(view["view"]), "ocr_text": ocr_text,
                "label_confidence": view["label_confidence"],
                "bottle_confidence": view["bottle_confidence"],
                "bottle_mode": view["bottle_mode"], "candidate_count": len(candidate_ids),
            })
        ranked = sorted(fused.values(), key=lambda row: row["score"], reverse=True)
        for rank, candidate in enumerate(ranked[:top_k], start=1):
            candidate["rank"] = rank
        return {
            "pipeline": "stage2c-five-stream-residual-v1",
            "score_semantics": "ranking logit; not probability and not calibrated confidence",
            "stage2c_epoch": self.stage2c_info["epoch"],
            "transformer_epoch": self.transformer_info["epoch"],
            "diagnostics": diagnostics,
            "views": view_metadata,
            "candidates": ranked[:top_k],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Image files or directories")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--joint-yolo", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--stage2c-checkpoint", type=Path, default=DEFAULT_STAGE2C)
    parser.add_argument("--transformer-checkpoint", type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument("--easyocr-model-dir", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--galleries-root", type=Path, default=DEFAULT_GALLERIES_ROOT)
    parser.add_argument("--galleries-zip", type=Path, default=DEFAULT_GALLERIES_ZIP)
    parser.add_argument("--gallery-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--verify-stage2c-hash", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    return args


def main() -> None:
    args = parse_args()
    pipeline = FiveStreamInferencePipeline(
        joint_yolo=args.joint_yolo, stage2c_checkpoint=args.stage2c_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        easyocr_model_dir=args.easyocr_model_dir, galleries_root=args.galleries_root,
        galleries_zip=args.galleries_zip, gallery_cache=args.gallery_cache,
        device=args.device, verify_stage2c_hash=args.verify_stage2c_hash,
    )
    results = []
    for path in collect_images(args.inputs):
        result = pipeline.predict(open_rgb(path), top_k=args.top_k)
        result["source"] = str(path)
        results.append(result)
        print(
            f"PREDICT | {path.name} | "
            + " | ".join(
                f"#{item['rank']} {item['wine_slug']} score={item['score']:.4f}"
                for item in result["candidates"]
            ), flush=True,
        )
    payload = {"results": results}
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
