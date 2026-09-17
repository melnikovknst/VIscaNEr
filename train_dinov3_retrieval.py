#!/usr/bin/env python3
"""Headless CLI for the DINOv3 wine-retrieval pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dinov3_retrieval import (
    PipelineConfig,
    benchmark_model,
    choose_device,
    create_eval_loader,
    evaluate_splits,
    export_gallery_embeddings,
    load_trained_model,
    prepare_retrieval_index,
    train_pipeline,
)


def load_config(path: str | Path, overrides: dict[str, Any]) -> PipelineConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    valid = {field.name for field in fields(PipelineConfig)}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return PipelineConfig(**payload).resolved()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "all", "evaluate", "benchmark"))
    parser.add_argument("--config", default="configs/dinov3_retrieval.yaml")
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
    parser.add_argument("--skip-file-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    if args.command in {"prepare", "all"}:
        _, summary = prepare_retrieval_index(cfg, validate_files=not args.skip_file_validation)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.command in {"train", "all"}:
        result = train_pipeline(cfg, quick_smoke=args.quick_smoke)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command in {"evaluate", "benchmark"}:
        index = pd.read_csv(cfg.index_path)
        device = choose_device(cfg.device)
        model, checkpoint = load_trained_model(Path(cfg.models_dir) / "best.pt", cfg.weights_path, device)
        run_dir = Path(cfg.runs_dir) / cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "evaluate":
            metrics, details = evaluate_splits(model, index, cfg, device)
            (run_dir / "final_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            export_gallery_embeddings(details, run_dir / "gallery_embeddings.pt")
            print(json.dumps({"checkpoint": checkpoint.get("metrics"), "metrics": metrics}, indent=2))
        else:
            records = index[index["split"].eq("gallery")].head(max(cfg.eval_batch_size, 128))
            timings = benchmark_model(model, create_eval_loader(records, cfg), device)
            (run_dir / "benchmark.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
            print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
