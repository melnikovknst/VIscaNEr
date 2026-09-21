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
        for path in (settings.checkpoint_path, settings.gallery_path):
            if not path.is_file():
                raise FileNotFoundError(f"Файл модели не найден: {path}")
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
        gallery = torch.load(settings.gallery_path, map_location="cpu", weights_only=True)
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
            detection = self.detector.predict(image, verbose=False, conf=0.25)[0]
            boxes = [b for b in detection.boxes if detection.names[int(b.cls.item())] == "label"]
            if not boxes:
                return Prediction(model_version=self.version, abstain=True)
            # The scanner asks for a single, centered label; select the closest
            # detected label to image center if shelf neighbours are visible.
            def center_distance(box):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                return ((x1 + x2) / 2 - image.width / 2) ** 2 + ((y1 + y2) / 2 - image.height / 2) ** 2
            x1, y1, x2, y2 = min(boxes, key=center_distance).xyxy[0].tolist()
            image = image.crop((max(0, int(x1)), max(0, int(y1)), min(image.width, int(x2)), min(image.height, int(y2))))
        with self.torch.inference_mode():
            embedding, _ = self.model(self.transform(image).unsqueeze(0).to(self.device))
            scores, indices = (embedding @ self.embeddings.T).squeeze(0).topk(min(5, len(self.slugs)))
        return Prediction(model_version=self.version, candidates=[
            Candidate(slug=self.slugs[i], similarity=max(-1.0, min(1.0, float(score))))
            for score, i in zip(scores.cpu().tolist(), indices.cpu().tolist())])


def create_provider(settings: Settings) -> Provider:
    if settings.model_provider == "local":
        return LocalProvider(settings)
    if settings.model_provider == "remote":
        return RemoteProvider(settings)
    return DemoProvider()
