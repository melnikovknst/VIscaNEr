"""Model construction, robust image loading, embedding, and cache helpers."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import DINOv3ViTModel

from bottle_classifier.local_backbone import make_config as make_small_config
from dinov3_retrieval import (
    DINOv3RetrievalModel,
    build_transforms,
    make_dinov3_vitb16_config,
    normalize_retrieval_checkpoint_state_dict,
)


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment_summary(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def release_accelerator_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _checkpoint_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "config", "epoch", "stage"}
    if missing := required.difference(payload):
        raise ValueError(f"Checkpoint {path} is missing keys: {sorted(missing)}")
    return payload


def load_retrieval_model(
    checkpoint_path: str | Path,
    variant: str,
    device: torch.device,
) -> tuple[DINOv3RetrievalModel, dict[str, Any]]:
    payload = _checkpoint_payload(checkpoint_path)
    train_cfg = payload["config"]
    image_size = int(train_cfg.get("image_size", 224))
    embedding_dim = int(train_cfg.get("embedding_dim", 256))
    projection_hidden_dim = int(train_cfg.get("projection_hidden_dim", 512))
    state = payload["model_state_dict"]
    classifier_key = "classifier.weight"
    if classifier_key not in state:
        raise ValueError(f"Checkpoint {checkpoint_path} has no {classifier_key}")
    num_classes = int(state[classifier_key].shape[0])

    if variant == "vitb16":
        backbone = DINOv3ViTModel(make_dinov3_vitb16_config(image_size))
    elif variant == "vits16":
        backbone = DINOv3ViTModel(make_small_config("vits16", image_size))
    else:
        raise ValueError(f"Unsupported checkpoint variant: {variant}")
    model = DINOv3RetrievalModel(
        backbone=backbone,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        projection_hidden_dim=projection_hidden_dim,
        ce_temperature=float(train_cfg.get("ce_temperature", 0.07)),
    )
    normalized = normalize_retrieval_checkpoint_state_dict(model, state)
    result = model.load_state_dict(normalized, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")
    model.eval().to(device)
    info = {
        "path": str(Path(checkpoint_path).resolve()),
        "variant": variant,
        "image_size": image_size,
        "embedding_dim": embedding_dim,
        "num_classes": num_classes,
        "epoch": int(payload["epoch"]),
        "stage": int(payload["stage"]),
        "monitor_split": payload.get("monitor_split"),
        "monitor_metric": payload.get("monitor_metric"),
        "monitor_value": payload.get("monitor_value"),
    }
    return model, info


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[str], image_size: int) -> None:
        self.paths = list(paths)
        _, self.transform = build_transforms(image_size)
        self.blank = torch.zeros(3, image_size, image_size, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
            return {"pixel_values": tensor, "index": index, "valid": True, "error": ""}
        except Exception as exc:  # one corrupt image must not abort 45k evaluation
            return {
                "pixel_values": self.blank.clone(),
                "index": index,
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }


def _cache_key(checkpoint: str | Path, paths: Sequence[str], image_size: int) -> str:
    checkpoint = Path(checkpoint)
    stat = checkpoint.stat()
    digest = hashlib.sha256()
    digest.update(f"{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{image_size}".encode())
    for path in paths:
        digest.update(b"\0")
        digest.update(str(path).encode("utf-8"))
    return digest.hexdigest()[:20]


@torch.inference_mode()
def embed_paths(
    model: DINOv3RetrievalModel,
    paths: Sequence[str],
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
    description: str,
    cache_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    force: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict[str, Any]]:
    if not paths:
        width = int(model.classifier.weight.shape[1])
        return torch.empty(0, width), torch.empty(0, dtype=torch.bool), [], {"seconds": 0.0}
    cache_path: Path | None = None
    if cache_dir is not None and checkpoint_path is not None:
        cache_path = Path(cache_dir) / f"{description}_{_cache_key(checkpoint_path, paths, image_size)}.pt"
        if cache_path.is_file() and not force:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"  CACHE       | {description} | {len(paths)} embeddings", flush=True)
            return cached["embeddings"], cached["valid"], cached["errors"], cached["timing"]

    dataset = ImagePathDataset(paths, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    outputs: list[torch.Tensor] = []
    valid_chunks: list[torch.Tensor] = []
    errors = [""] * len(paths)
    total_batches = len(loader)
    interval = max(1, total_batches // 10)
    started = time.perf_counter()
    print(f"  {description.upper():<11} | 0/{total_batches} batches", flush=True)
    for batch_number, batch in enumerate(loader, start=1):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        embeddings, _ = model(pixels)
        embeddings = F.normalize(embeddings.float(), dim=1).cpu()
        valid = batch["valid"].bool()
        embeddings[~valid] = 0
        outputs.append(embeddings)
        valid_chunks.append(valid)
        for index, error, is_valid in zip(batch["index"].tolist(), batch["error"], valid.tolist(), strict=True):
            if not is_valid:
                errors[index] = str(error)
        if batch_number % interval == 0 or batch_number == total_batches:
            print(
                f"  {description.upper():<11} | {batch_number}/{total_batches} batches "
                f"({100.0 * batch_number / total_batches:5.1f}%)",
                flush=True,
            )
    synchronize(device)
    timing = {"seconds": time.perf_counter() - started, "images": len(paths)}
    result_embeddings = torch.cat(outputs)
    result_valid = torch.cat(valid_chunks)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": result_embeddings,
                "valid": result_valid,
                "errors": errors,
                "timing": timing,
            },
            cache_path,
        )
    return result_embeddings, result_valid, errors, timing


def checkpoint_fingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def save_json(path: str | Path, payload: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if hasattr(value, "item"):
            try:
                return clean(value.item())
            except (ValueError, TypeError):
                pass
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
