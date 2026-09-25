"""Serve the colleagues' two-stage DINO cascade behind the web API.

Stage 1  DINOv3-B/16 on a *label crop* ranks the whole 2,103-wine gallery.
Stage 2  A whole-bottle DINOv3 model re-orders only the top two, and only when
         stage 1 is close to a tie (BOTTLE_CLASSIFIER.md: "the label DINO runs
         first; a whole-bottle classifier is consulted only when the label
         ranking is ambiguous").

Both models were trained on detector crops, not raw photos, so the provider
reproduces that framing with the same code that built the training data:

  label   YOLO label detector, 10% padding (train_yolo_label_detector.ipynb)
  bottle  build_bottle_classifier_dataset.select_target_bottle picks the bottle
          that owns the label; choose_output_policy then feeds the padded crop
          (confidence >= 0.75) or the whole photo (below), exactly as the
          training inputs were saved. Ambiguous or cut-off bottles, which were
          never training positives, skip stage 2.

Model loading, gallery embedding and the eval transform are imported from the
training code too, so serving and evaluation cannot drift apart. Every
response records which path it took.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from PIL import Image

from backend.config import ROOT, Settings
from backend.schemas import Candidate, Prediction

logger = logging.getLogger(__name__)

LABEL_PADDING = 0.10      # train_yolo_label_detector.ipynb, crop_labels_from_directory
DETECTOR_CONF = 0.25      # label detector low-confidence floor
TOP_K = 10

# build_bottle_classifier_dataset.py CLI defaults - the policy the whole-bottle
# model's training inputs were produced with.
BOTTLE_POLICY = {
    "imgsz": 768, "minimum_confidence": 0.05, "nms_iou": 0.70, "max_det": 60,
    "minimum_label_coverage": 0.01, "ambiguity_margin": 0.12, "duplicate_iou": 0.80,
    "crop_confidence_threshold": 0.75, "padding": 0.06,
}


def _padded(box, width: int, height: int, fraction: float) -> tuple[int, int, int, int]:
    """Pad by a fraction of each side, clipped to the frame (training convention)."""
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * fraction, (y2 - y1) * fraction
    return (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x + 0.999)), min(height, int(y2 + pad_y + 0.999)))


def _detections(detector, image: Image.Image, imgsz: int) -> list[dict[str, Any]]:
    result = detector.predict(image, imgsz=imgsz, conf=DETECTOR_CONF, verbose=False)[0]
    if result.boxes is None:
        return []
    return [{"box": [float(v) for v in b.xyxy[0].tolist()], "confidence": float(b.conf.item())}
            for b in result.boxes]


def _pick_label(labels: list[dict[str, Any]], width: int, height: int) -> dict[str, Any] | None:
    """The scanner is aimed at one bottle: prefer the confident label nearest the centre.

    Distance is normalised by the frame so a slightly off-centre but clearly
    detected label is not beaten by a faint one that happens to sit mid-frame.
    """
    if not labels:
        return None

    def cost(item):
        x1, y1, x2, y2 = item["box"]
        dx = ((x1 + x2) / 2 - width / 2) / width
        dy = ((y1 + y2) / 2 - height / 2) / height
        return (dx * dx + dy * dy) ** 0.5 - 0.25 * item["confidence"]

    return min(labels, key=cost)


def _bottle_input(detector, image: Image.Image, label: dict[str, Any] | None):
    """Whole-bottle model input, chosen by the training data's own rules.

    Returns (input image or None, record). None means stage 2 must not run:
    no label to anchor on, no owning bottle, an ambiguous owner, or a
    confidently detected bottle that is cut off - none of which were ever
    training positives.
    """
    import numpy as np

    from build_bottle_classifier_dataset import (
        choose_output_policy, padded_box, select_target_bottle,
    )

    if label is None:
        return None, {"status": "no_label_anchor"}
    width, height = image.size
    result = detector.predict(
        np.asarray(image)[:, :, ::-1], imgsz=BOTTLE_POLICY["imgsz"],
        conf=BOTTLE_POLICY["minimum_confidence"], iou=BOTTLE_POLICY["nms_iou"],
        max_det=BOTTLE_POLICY["max_det"], verbose=False,
    )[0]
    selected, info = select_target_bottle(
        result, tuple(label["box"]),
        minimum_confidence=BOTTLE_POLICY["minimum_confidence"],
        minimum_label_coverage=BOTTLE_POLICY["minimum_label_coverage"],
        ambiguity_margin=BOTTLE_POLICY["ambiguity_margin"],
        duplicate_iou=BOTTLE_POLICY["duplicate_iou"],
    )
    if selected is None:
        return None, {"status": "no_owner", "detections": info.get("detections", 0)}
    x1, y1, x2, y2 = selected["box"]
    status, mode = choose_output_policy(
        selected["confidence"], ambiguous=bool(info.get("ambiguous")),
        vertically_truncated=y1 <= 1.0 or y2 >= height - 1.0,
        crop_confidence_threshold=BOTTLE_POLICY["crop_confidence_threshold"],
    )
    record = {"status": status, "image_mode": mode,
              "box": [round(v, 1) for v in selected["box"]],
              "confidence": round(selected["confidence"], 3)}
    if status in {"ambiguous", "partial"}:
        return None, record
    if mode == "original_image":
        return image, record
    return image.crop(padded_box(selected["box"], width, height, BOTTLE_POLICY["padding"])), record


class CascadeProvider:
    """DINO-B label retrieval with a conditional whole-bottle resolver."""

    def __init__(self, settings: Settings):
        import torch
        from torch.nn import functional as F

        from cascade_resolver.modeling import choose_device, embed_paths, load_retrieval_model
        from dinov3_retrieval import build_transforms

        self.torch, self.F = torch, F
        self.settings = settings
        self.device = choose_device(settings.device)
        started = time.perf_counter()

        for path in (settings.primary_checkpoint, settings.label_refs_root):
            if not Path(path).exists():
                raise FileNotFoundError(f"Не найден файл модели или галереи: {path}")

        cache_dir = ROOT / "runs" / "web_gallery"
        workers = 0  # the API process must not fork a DataLoader pool on Windows

        # ---- stage 1: DINO-B on label crops ---------------------------------
        self.primary, self.primary_info = load_retrieval_model(settings.primary_checkpoint, "vitb16", self.device)
        _, self.primary_transform = build_transforms(self.primary_info["image_size"])
        self.slugs, label_paths = self._gallery(settings.label_refs_root)
        embeddings, valid, _, _ = embed_paths(
            self.primary, label_paths, self.device, self.primary_info["image_size"],
            batch_size=48, num_workers=workers, description="web_label_gallery",
            cache_dir=cache_dir, checkpoint_path=settings.primary_checkpoint,
        )
        if not bool(valid.all()):
            raise ValueError("Часть эталонов этикеток не прочиталась - галерея неполная")
        self.label_gallery = F.normalize(embeddings.float(), dim=1).to(self.device)

        # ---- stage 2: whole-bottle model (optional) --------------------------
        self.resolver = None
        self.bottle_gallery = None
        if settings.resolver_checkpoint and Path(settings.resolver_checkpoint).is_file() \
                and settings.bottle_refs_root and Path(settings.bottle_refs_root).is_dir():
            self.resolver, self.resolver_info = self._load_resolver(settings)
            _, self.resolver_transform = build_transforms(self.resolver_info["image_size"])
            bottle_slugs, bottle_paths = self._gallery(settings.bottle_refs_root)
            if bottle_slugs != self.slugs:
                raise ValueError("Галереи этикеток и бутылок описывают разные наборы вин")
            embeddings, valid, _, _ = embed_paths(
                self.resolver, bottle_paths, self.device, self.resolver_info["image_size"],
                batch_size=64, num_workers=workers, description="web_bottle_gallery",
                cache_dir=cache_dir, checkpoint_path=settings.resolver_checkpoint,
            )
            self.bottle_gallery = F.normalize(embeddings.float(), dim=1).to(self.device)
            self.bottle_gallery_valid = valid.to(self.device)

        # ---- detectors (optional) -------------------------------------------
        from ultralytics import YOLO

        self.label_detector = YOLO(str(settings.label_detector_path)) \
            if settings.label_detector_path and Path(settings.label_detector_path).is_file() else None
        self.bottle_detector = YOLO(str(settings.bottle_detector_path)) \
            if settings.bottle_detector_path and Path(settings.bottle_detector_path).is_file() else None

        # 0 disables stage 2. On real photos the bottle resolver's effect is
        # within noise at any margin (runs/cascade_real/summary.json), so the
        # served .env limits it to near-exact ties.
        self.ambiguity_margin = float(settings.ambiguity_margin)
        stages = ["dinov3-b16-labels"]
        if self.resolver is not None and self.bottle_detector is not None and self.ambiguity_margin > 0:
            stages.append(f"dinov3-{settings.resolver_variant.removeprefix('vit')}-bottles")
        self.version = "+".join(stages) + f"@e{self.primary_info['epoch']}"

        self._warm_up()
        logger.info("Cascade ready in %.1fs on %s: %s | label detector=%s | bottle detector=%s",
                    time.perf_counter() - started, self.device, self.version,
                    bool(self.label_detector), bool(self.bottle_detector))

    def _load_resolver(self, settings: Settings):
        """Build the whole-bottle model the way it was trained.

        vitb16: deeptune_backbones loads the pinned DINOv3-B backbone
        (models/dinov3/model.safetensors, SHA-256 checked) and the fine-tuned
        checkpoint is loaded strictly on top - the same construction as
        training. The checkpoint holds the full fine-tuned backbone, so every
        served weight comes from it; the pinned base makes a wrong or corrupted
        backbone file fail loudly instead of quietly.
        """
        from cascade_resolver.modeling import load_retrieval_model

        if settings.resolver_variant == "vits16":
            return load_retrieval_model(settings.resolver_checkpoint, "vits16", self.device)

        import deeptune_backbones
        from dinov3_retrieval import DINOv3RetrievalModel, normalize_retrieval_checkpoint_state_dict

        if not settings.resolver_backbone_path or not Path(settings.resolver_backbone_path).is_file():
            raise FileNotFoundError(f"Не найден backbone DINOv3-B: {settings.resolver_backbone_path}")
        payload = self.torch.load(settings.resolver_checkpoint, map_location="cpu", weights_only=False)
        config, state = payload["config"], payload["model_state_dict"]
        image_size = int(config.get("image_size", 224))
        backbone = deeptune_backbones.make_loader("vitb16")(settings.resolver_backbone_path, image_size)
        model = DINOv3RetrievalModel(
            backbone=backbone,
            num_classes=int(state["classifier.weight"].shape[0]),
            embedding_dim=int(config.get("embedding_dim", 256)),
            projection_hidden_dim=int(config.get("projection_hidden_dim", 512)),
            ce_temperature=float(config.get("ce_temperature", 0.07)),
        )
        result = model.load_state_dict(normalize_retrieval_checkpoint_state_dict(model, state), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Чекпойнт бутылок не совпадает с архитектурой: {result}")
        model.eval().to(self.device)
        return model, {"image_size": image_size, "epoch": int(payload.get("epoch", -1)),
                       "variant": "vitb16", "monitor_value": payload.get("monitor_value")}

    @staticmethod
    def _gallery(root: Path) -> tuple[list[str], list[str]]:
        suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        files = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in suffixes)
        stems = [p.stem for p in files]
        if len(stems) != len(set(stems)):
            raise ValueError(f"В галерее {root} есть повторяющиеся эталоны")
        return stems, [str(p) for p in files]

    def _warm_up(self) -> None:
        # The first CUDA call compiles kernels; do it at start-up, not on the
        # first visitor's request.
        blank = Image.new("RGB", (320, 480), (200, 200, 200))
        self.predict(blank)

    def _embed(self, model, transform, image: Image.Image):
        with self.torch.inference_mode():
            embedding, _ = model(transform(image).unsqueeze(0).to(self.device))
            return self.F.normalize(embedding.float(), dim=1)

    def predict(self, image: Image.Image) -> Prediction:
        width, height = image.size
        pipeline: dict[str, Any] = {"image_width": width, "image_height": height}

        # ---- label crop ----------------------------------------------------
        label = None
        if self.label_detector is not None:
            labels = _detections(self.label_detector, image, 640)
            label = _pick_label(labels, width, height)
            pipeline["labels_detected"] = len(labels)
        if label is not None:
            crop_box = _padded(label["box"], width, height, LABEL_PADDING)
            label_crop = image.crop(crop_box)
            pipeline["label"] = {"box": [round(v, 1) for v in label["box"]],
                                 "confidence": round(label["confidence"], 3)}
            pipeline["label_source"] = "detector"
        else:
            # A close-up where the label fills the frame is the documented way
            # to use the scanner without a detector.
            label_crop = image
            pipeline["label_source"] = "whole_photo"

        query = self._embed(self.primary, self.primary_transform, label_crop)
        scores, indices = (query @ self.label_gallery.T).squeeze(0).topk(min(TOP_K, len(self.slugs)))
        ranked = [(self.slugs[i], float(s)) for s, i in zip(scores.tolist(), indices.tolist())]
        gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
        pipeline["stage1_gap"] = round(gap, 5)

        # ---- stage 2: only for a near tie ----------------------------------
        resolver = {"available": self.resolver is not None and self.bottle_detector is not None
                                 and self.ambiguity_margin > 0,
                    "ambiguous": gap <= self.ambiguity_margin, "invoked": False, "swapped": False}
        if resolver["available"] and resolver["ambiguous"]:
            bottle_input, record = _bottle_input(self.bottle_detector, image, label)
            resolver["bottle_status"] = record["status"]
            if "box" in record:
                pipeline["bottle"] = {k: record[k] for k in ("box", "confidence", "image_mode")}
            if bottle_input is not None:
                bottle_query = self._embed(self.resolver, self.resolver_transform, bottle_input)
                top_two = [self.slugs.index(ranked[0][0]), self.slugs.index(ranked[1][0])]
                if bool(self.bottle_gallery_valid[top_two].all()):
                    pair = (bottle_query @ self.bottle_gallery[top_two].T).squeeze(0).tolist()
                    resolver.update(invoked=True, first=round(pair[0], 5), second=round(pair[1], 5))
                    # Exactly resolve_top_two(): swap only when the bottle
                    # model prefers the runner-up.
                    if pair[1] > pair[0]:
                        ranked[0], ranked[1] = ranked[1], ranked[0]
                        resolver["swapped"] = True
        pipeline["resolver"] = resolver

        # Candidates go out in the final order with their true stage-1 scores;
        # `ranked=True` tells the API not to re-sort them by score, which would
        # silently undo a resolver swap. The confidence that matters is the one
        # of the stage that made the call: after the resolver, stage 1 was a
        # near tie by definition, so its margin says nothing about the answer.
        if resolver["invoked"]:
            basis, margin = "resolver", abs(resolver["first"] - resolver["second"])
        else:
            basis, margin = "label", gap
        return Prediction(
            model_version=self.version,
            candidates=[Candidate(slug=s, similarity=max(-1.0, min(1.0, v))) for s, v in ranked[:5]],
            ranked=True, decision_basis=basis, decision_margin=round(margin, 5), pipeline=pipeline,
        )
