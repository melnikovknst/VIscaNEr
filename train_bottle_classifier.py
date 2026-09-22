#!/usr/bin/env python3
"""Train or validate one whole-bottle DINOv3 classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import dinov3_retrieval as retrieval
from bottle_classifier import MODEL_VARIANTS, install_local_backbone_loader


def load_config(path: str | Path, overrides: dict[str, Any]) -> retrieval.PipelineConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    valid = {field.name for field in fields(retrieval.PipelineConfig)}
    if unknown := set(payload).difference(valid):
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return retrieval.PipelineConfig(**payload).resolved()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "prepare", "evaluate", "benchmark"))
    parser.add_argument("--variant", choices=tuple(MODEL_VARIANTS), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root")
    parser.add_argument("--weights-path")
    parser.add_argument("--crops-metadata-path")
    parser.add_argument("--bottle-manifest-path")
    parser.add_argument("--crops-root")
    parser.add_argument("--refs-root")
    parser.add_argument("--index-path")
    parser.add_argument("--models-dir")
    parser.add_argument("--runs-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--quick-smoke", action="store_true")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_spec = install_local_backbone_loader(args.variant)
    overrides = {
        key: getattr(args, key)
        for key in (
            "project_root",
            "weights_path",
            "crops_metadata_path",
            "bottle_manifest_path",
            "crops_root",
            "refs_root",
            "index_path",
            "models_dir",
            "runs_dir",
            "run_name",
            "device",
            "num_workers",
            "eval_batch_size",
        )
    }
    cfg = load_config(args.config, overrides)

    prepared_summary: dict[str, Any] | None = None
    if args.command == "prepare":
        _, prepared_summary = retrieval.prepare_retrieval_index(cfg, validate_files=True)
        print(json.dumps(prepared_summary, ensure_ascii=False, indent=2), flush=True)

    if args.command == "all":
        result = retrieval.train_pipeline(
            cfg,
            quick_smoke=args.quick_smoke,
            resume_checkpoint=args.resume_checkpoint,
        )
        result["variant"] = args.variant
        result["model_name"] = model_spec["display_name"]
        result["pretrained_weights"] = Path(cfg.weights_path).name
        result["pretrained_weights_sha256"] = model_spec["sha256"]
        run_summary = Path(cfg.runs_dir) / cfg.run_name / "run_summary.json"
        run_summary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    if args.command in {"evaluate", "benchmark"}:
        index = pd.read_csv(cfg.index_path)
        device = retrieval.choose_device(cfg.device)
        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg.models_dir) / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, payload = retrieval.load_trained_model(checkpoint, cfg.weights_path, device)
        run_dir = Path(cfg.runs_dir) / cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "evaluate":
            metrics, details = retrieval.evaluate_splits(model, index, cfg, device)
            audit = retrieval.save_retrieval_audit(details, run_dir / "evaluation_audit", n=12)
            retrieval.plot_retrieval_failures(details, run_dir / "evaluation_failures.png")
            result = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_metrics": payload.get("metrics"),
                "current_metrics": metrics,
                "audit": audit,
            }
        else:
            gallery = index[index["split"].eq("gallery")].head(max(cfg.eval_batch_size, 128))
            loader = retrieval.create_eval_loader(gallery, cfg)
            result = retrieval.benchmark_model(model, loader, device)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
