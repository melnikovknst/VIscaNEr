"""Model adapters. Training dependencies are imported only for local inference."""
import io
import logging
from typing import Protocol
from urllib.parse import urlparse

import httpx
from PIL import Image

from backend.config import Settings
from backend.schemas import Candidate, Prediction

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    pass


class Provider(Protocol):
    def predict(self, image: Image.Image) -> Prediction: ...


class DemoProvider:
    def predict(self, image: Image.Image) -> Prediction:
        raise ModelUnavailable("Модель ещё не подключена. Пока можно открыть пример результата или каталог.")


class RemoteProvider:
    def __init__(self, settings: Settings):
        if urlparse(settings.remote_url).scheme not in {"http", "https"}:
            raise ValueError("VISCANER_REMOTE_URL должен быть полным HTTP(S) URL обработчика модели")
        self.settings = settings

    def predict(self, image: Image.Image) -> Prediction:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        headers = {}
        if self.settings.remote_api_key.get_secret_value():
            headers["Authorization"] = "Bearer " + self.settings.remote_api_key.get_secret_value()
        try:
            with httpx.Client(timeout=self.settings.timeout_seconds, follow_redirects=False) as client:
                response = client.post(self.settings.remote_url, headers=headers,
                    files={"file": ("label.jpg", buffer.getvalue(), "image/jpeg")}, data={"top_k": "5"})
                response.raise_for_status()
                return Prediction.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Remote inference failed: %s", type(exc).__name__)
            raise ModelUnavailable("Сервис распознавания не ответил или вернул неверный результат. Попробуйте ещё раз.") from exc


class LocalProvider:
    def __init__(self, settings: Settings):
        if not settings.checkpoint_path.is_file():
            raise FileNotFoundError(f"Файл модели не найден: {settings.checkpoint_path}")
        # Reuse exactly the architecture, state-key normalization and evaluation
        # transforms used by colleagues. A trained checkpoint includes the backbone;
        # no original safetensors or training-machine paths are needed at serving time.
        import torch
        from transformers import DINOv3ViTModel
        from dinov3_retrieval import (
            DINOv3RetrievalModel, build_transforms, choose_device,
            make_dinov3_vitb16_config, normalize_retrieval_checkpoint_state_dict,
        )

        self.torch = torch
        self.device = choose_device(settings.device)
        checkpoint = torch.load(settings.checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        state = checkpoint["model_state_dict"]
        size = int(config.get("image_size", 224))
        backbone = DINOv3ViTModel(make_dinov3_vitb16_config(size))
        self.model = DINOv3RetrievalModel(backbone, int(state["classifier.weight"].shape[0]),
            int(config.get("embedding_dim", 256)), int(config.get("projection_hidden_dim", 512)),
            float(config.get("ce_temperature", 0.07)))
        self.model.load_state_dict(normalize_retrieval_checkpoint_state_dict(self.model, state), strict=True)
        self.model.to(self.device).eval()
        _, self.transform = build_transforms(size)
        gallery = (torch.load(settings.gallery_path, map_location="cpu", weights_only=True)
                   if settings.gallery_path.is_file() else None)
        if gallery is None or "signature" in gallery:
            # infer_wine.py's cache: signed with the checkpoint, backbone and
            # reference files. Let infer_wine itself load it, or rebuild it when
            # the weights changed - a stale gallery would silently match the new
            # model's queries against the old model's embeddings.
            from infer_wine import (DEFAULT_DATASET_ZIP, DEFAULT_DINO_WEIGHTS, DEFAULT_REFS_ROOT,
                                    build_or_load_gallery, ensure_references)
            embeddings, self.slugs = build_or_load_gallery(
                model=self.model, transform=self.transform,
                refs=ensure_references(DEFAULT_REFS_ROOT, DEFAULT_DATASET_ZIP),
                checkpoint=settings.checkpoint_path.resolve(), weights=DEFAULT_DINO_WEIGHTS,
                cache_path=settings.gallery_path.resolve(), device=self.device, image_size=size,
                batch_size=32, rebuild=False)
            embeddings = embeddings.float().cpu()
        else:
            # A precomputed gallery file (README, "Вариант A"): used as is.
            embeddings = gallery["embeddings"].float()
            self.slugs = list(gallery["wine_slugs"])
        if (embeddings.ndim != 2 or len(embeddings) != len(self.slugs) or not len(self.slugs)
                or len(set(self.slugs)) != len(self.slugs)
                or embeddings.shape[1] != int(config.get("embedding_dim", 256))
                or not torch.isfinite(embeddings).all() or (embeddings.norm(dim=1) == 0).any()):
            raise ValueError("Галерея несовместима с моделью или содержит неверные эмбеддинги")
        self.embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to(self.device)
        self.version = f"dinov3-epoch-{checkpoint.get('epoch', 'unknown')}"
        self.detector = None
        if settings.detector_path:
            if not settings.detector_path.is_file():
                raise FileNotFoundError(settings.detector_path)
            from ultralytics import YOLO
            self.detector = YOLO(str(settings.detector_path))
        with torch.inference_mode():
            self.model(self.transform(Image.new("RGB", (size, size))).unsqueeze(0).to(self.device))

    def predict(self, image: Image.Image) -> Prediction:
        if self.detector is not None:
            from infer_wine import crop_with_policy, select_target_detections

            detector_device = (
                self.device.index if self.device.type == "cuda" and self.device.index is not None
                else 0 if self.device.type == "cuda"
                else self.device.type
            )
            detection = self.detector.predict(
                image,
                verbose=False,
                conf=0.05,
                imgsz=768,
                device=detector_device,
            )[0]
            selected, _, _ = select_target_detections(
                detection,
                image_width=image.width,
                image_height=image.height,
                candidate_confidence=0.05,
                crosshair_x=0.5,
                crosshair_y=0.5,
                ambiguity_confidence=0.75,
            )
            selected_for_inference = selected or [None]
            model_inputs = [
                crop_with_policy(
                    image,
                    candidate,
                    confidence_threshold=0.75,
                    padding=0.06,
                )[0]
                for candidate in selected_for_inference
            ]
        else:
            model_inputs = [image]
        with self.torch.inference_mode():
            pixels = self.torch.stack([self.transform(item) for item in model_inputs]).to(self.device)
            embeddings, _ = self.model(pixels)
            similarities_by_bottle = embeddings @ self.embeddings.T
            similarities, _ = similarities_by_bottle.max(dim=0)
            scores, indices = similarities.topk(min(5, len(self.slugs)))
        return Prediction(model_version=self.version, candidates=[
            Candidate(slug=self.slugs[i], similarity=max(-1.0, min(1.0, float(score))))
            for score, i in zip(scores.cpu().tolist(), indices.cpu().tolist())])


class FiveStreamProvider:
    """Joint YOLO -> two Stage-2C DINO-B -> EasyOCR -> residual Transformer."""

    # The confidence thresholds were calibrated on the softmax of exactly this many logits.
    SOFTMAX_POOL = 10

    def __init__(self, settings: Settings):
        from five_stream_transformer.infer import FiveStreamInferencePipeline

        self.pipeline = FiveStreamInferencePipeline(device=settings.device)
        self.slugs = self.pipeline.slugs
        info = self.pipeline.transformer_info
        self.version = f"five-stream-e{info['epoch']}-stage2c-e{self.pipeline.stage2c_info['epoch']}"
        # Warm up CUDA kernels and EasyOCR so the first visitor does not wait for them.
        self.pipeline.predict(Image.new("RGB", (640, 640), "white"), top_k=1)

    def predict(self, image: Image.Image) -> Prediction:
        import torch

        result = self.pipeline.predict(image, top_k=self.SOFTMAX_POOL)
        rows = result["candidates"]
        if not rows:
            return Prediction(model_version=self.version, abstain=True, ranked=True)
        probs = torch.tensor([row["score"] for row in rows]).softmax(dim=0).tolist()
        margin = probs[0] - probs[1] if len(probs) > 1 else None
        return Prediction(
            model_version=self.version,
            ranked=True,
            decision_margin=margin,
            candidates=[Candidate(slug=row["wine_slug"], similarity=p) for row, p in zip(rows[:5], probs)],
            pipeline={
                "confidence": "softmax over the top-10 Transformer logits",
                "diagnostics": result["diagnostics"],
                "views": result["views"],
                "logits": [round(row["score"], 4) for row in rows[:5]],
            },
        )


def create_provider(settings: Settings) -> Provider:
    if settings.model_provider == "five_stream":
        return FiveStreamProvider(settings)
    if settings.model_provider == "cascade":
        from backend.cascade import CascadeProvider

        return CascadeProvider(settings)
    if settings.model_provider == "local":
        return LocalProvider(settings)
    if settings.model_provider == "remote":
        return RemoteProvider(settings)
    return DemoProvider()
