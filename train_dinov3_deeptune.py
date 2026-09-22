#!/usr/bin/env python3
"""Train DINOv3 ViT-B/16 or ConvNeXt-B on label or bottle crops."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

import dinov3_retrieval as retrieval
from deeptune_backbones import MODEL_VARIANTS, install_local_backbone_loader


def load_config(path: str | Path, overrides: dict[str, Any]) -> retrieval.PipelineConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    valid = {field.name for field in fields(retrieval.PipelineConfig)}
    if unknown := set(payload).difference(valid):
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return retrieval.PipelineConfig(**payload).resolved()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "prepare"))
    parser.add_argument("--backbone", choices=tuple(MODEL_VARIANTS), required=True)
    parser.add_argument("--dataset-kind", choices=("labels", "bottles"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--quick-smoke", action="store_true")
    parser.add_argument("--resume-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_spec = install_local_backbone_loader(args.backbone)
    cfg = load_config(args.config, {})
    print(
        "PIPELINE | "
        f"backbone={args.backbone} ({model_spec['display_name']}) | "
        f"dataset={args.dataset_kind} | weights={cfg.weights_path}",
        flush=True,
    )
    if args.command == "prepare":
        _, summary = retrieval.prepare_retrieval_index(cfg, validate_files=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    result = retrieval.train_pipeline(
        cfg,
        quick_smoke=args.quick_smoke,
        resume_checkpoint=args.resume_checkpoint,
    )
    result.update(
        backbone=args.backbone,
        model_name=model_spec["display_name"],
        dataset_kind=args.dataset_kind,
        pretrained_weights=Path(cfg.weights_path).name,
        pretrained_weights_sha256=model_spec["sha256"],
    )
    run_summary = Path(cfg.runs_dir) / cfg.run_name / "run_summary.json"
    run_summary.parent.mkdir(parents=True, exist_ok=True)
    run_summary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "PIPELINE COMPLETE | "
        f"backbone={args.backbone} | dataset={args.dataset_kind} | "
        f"summary={run_summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
