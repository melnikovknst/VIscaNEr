"""Recognition providers. Heavy ML imports happen only when the pipeline is loaded."""
from typing import Protocol

from PIL import Image

from backend.config import Settings
from backend.schemas import Candidate, Prediction


class ModelUnavailable(RuntimeError):
    pass


class Provider(Protocol):
    def predict(self, image: Image.Image) -> Prediction: ...


class DemoProvider:
    def predict(self, image: Image.Image) -> Prediction:
        raise ModelUnavailable("Модель ещё не подключена. Пока можно открыть пример результата или каталог.")


class FiveStreamProvider:
    """Joint YOLO -> two Stage-2C DINOv3-B -> EasyOCR -> residual Transformer."""

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
            return Prediction(model_version=self.version, abstain=True)
        probs = torch.tensor([row["score"] for row in rows]).softmax(dim=0).tolist()
        return Prediction(
            model_version=self.version,
            candidates=[Candidate(slug=row["wine_slug"], confidence=p) for row, p in zip(rows[:5], probs)],
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
    return DemoProvider()
